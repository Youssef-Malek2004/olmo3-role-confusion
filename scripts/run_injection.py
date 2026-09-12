#!/usr/bin/env python
"""Indirect prompt-injection evaluation through the tool-result channel, one checkpoint at a time.

Usage:
    python scripts/run_injection.py --run-id inj_pilot --stage sft --n-injected 18 --n-user-control 6 --n-clean 4 \
        [--defense none|delimiter|steer|random] [--steer-vector path.npy --steer-block 16 --steer-alpha 4.0] \
        [--batch-size 4] [--max-new-tokens 3000]

Outputs under artifacts/runs/<run_id>/<stage>/:
    items.json                      the deterministic item list (same for every stage/defense)
    gen_<defense>.jsonl             one row per item: raw text, compliance detectors, timing
    acts_<defense>/<item_id>.npz    prompt_end (n_blocks,H), env_mean (n_blocks,H), inj_mean (n_blocks,H) or absent
    manifest_<defense>.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# CUDA: avoid the caching allocator hoarding reserved memory across batches (lets a second process share the GPU)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import numpy as np  # noqa: E402

from role_confusion.generation import (  # noqa: E402
    Decoding, Steer, batch_seed, device_memory_gb, find_span, generate_batch, load_model, render_chat,
    span_token_mask, sync_device,
)
from role_confusion.injection import (  # noqa: E402
    INJECTION_TYPES, REALISTIC_TYPES, TOOLCALL_ABLATIONS, TOOLCALL_FRAMINGS, TOOLCALL_FRAMINGS2, InjectionItem, build_items,
    detect_compliance, injected_span, render_injection_messages,
)
from role_confusion.io import append_jsonl, iter_jsonl, read_json, write_json  # noqa: E402
from role_confusion.parsing import extract_final_answer  # noqa: E402
from role_confusion.questions import build_question  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--revisions", default="configs/revisions.json")
    ap.add_argument("--questions", default="data/processed/questions.jsonl")
    ap.add_argument("--splits", nargs="*", default=["pilot", "train"])
    ap.add_argument("--n-injected", type=int, default=18)
    ap.add_argument("--n-user-control", type=int, default=6)
    ap.add_argument("--n-clean", type=int, default=4)
    ap.add_argument("--n-legit", type=int, default=0, help="legitimate-tool-use items (record holds the correct answer_key; user asks to use it)")
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of conditions to run")
    ap.add_argument("--defense", choices=["none", "delimiter", "steer", "random"], default="none")
    ap.add_argument("--steer-vector", default=None, help=".npy direction (H,) for steer; ignored for random")
    ap.add_argument("--steer-block", type=int, default=16)
    ap.add_argument("--steer-alpha", type=float, default=1.0, help="multiplier on the (unit-norm) vector, in residual units")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--item-ids", nargs="*", default=None, help="explicit subset of item IDs")
    ap.add_argument("--itypes", nargs="*", default=None, help="restrict injected/user_control items to these goal types")
    ap.add_argument("--draw", type=int, default=1, help="sample index; draw>1 writes gen_<tag>_draw<N>.jsonl with a different seed")
    ap.add_argument("--dynamic-cache", action="store_true")
    ap.add_argument("--no-acts", action="store_true", help="skip saving activations (draws>1 reuse draw-1 activations)")
    ap.add_argument("--ablations", action="store_true",
                    help="ablation variants (escaped/lexical fake turn, destyled CoT forgery, length control); implies --realistic")
    ap.add_argument("--types", nargs="*", default=None, help="explicit goal-type list (overrides the set flags); implies --realistic")
    ap.add_argument("--framings2", action="store_true",
                    help="literature-derived tool-call framings (important_instructions, CoT forgery, fake turn, fake completion, contextual, necessity)")
    ap.add_argument("--framings", action="store_true",
                    help="tool-call framing set (dependency, required_field, api_error, preauth, benign_body, internal_addr); implies --realistic, single voice")
    ap.add_argument("--realistic", action="store_true",
                    help="use the realistic goal set (toolcall, answer, link, deny), declare send_summary, always include the key")
    ap.add_argument("--max-items-per-process", type=int, default=24,
                    help="exit with code 3 after this many items so the MPS graph cache is reset by a fresh process")
    args = ap.parse_args()

    # Operational override: a running chain passes a fixed batch size; when the GPU is shared, cap it via this file.
    override = REPO_ROOT / "artifacts" / "runs" / "batch_override.txt"
    if override.exists():
        try:
            cap_bs = int(override.read_text().strip())
            if cap_bs < args.batch_size:
                print(f"batch size {args.batch_size} -> {cap_bs} (batch_override.txt)", flush=True)
                args.batch_size = cap_bs
        except ValueError:
            pass
    cfg = read_json(REPO_ROOT / args.config)
    revs = read_json(REPO_ROOT / args.revisions)
    model_cfg = next(m for m in cfg["models"] if m["stage"] == args.stage)
    repo_id = model_cfg["repo_id"]
    sha = revs["models"][repo_id]["sha"]
    blocks = cfg["activations"]["block_numbers_one_based"]
    system_prompt = cfg["generation"]["system_prompt"]
    gen_cfg = cfg["generation"]
    base_decoding = Decoding("sample", float(gen_cfg["temperature"]), float(gen_cfg["top_p"]), None)

    rows = {r["question_id"]: r for r in iter_jsonl(REPO_ROOT / args.questions) if r["split"] in args.splits}
    qids = sorted(rows)
    if args.types:
        args.realistic = True
        types = tuple(args.types)
        items = build_items(qids, cfg["seed"], args.n_injected, args.n_user_control, args.n_clean, types=types, voices=("plain",), n_legit=args.n_legit)
    elif args.ablations:
        args.realistic = True
        types = TOOLCALL_ABLATIONS
        items = build_items(qids, cfg["seed"], args.n_injected, args.n_user_control, args.n_clean, types=types, voices=("plain",), n_legit=args.n_legit)
    elif args.framings2:
        args.realistic = True
        types = TOOLCALL_FRAMINGS2
        items = build_items(qids, cfg["seed"], args.n_injected, args.n_user_control, args.n_clean, types=types, voices=("plain",), n_legit=args.n_legit)
    elif args.framings:
        args.realistic = True
        types = TOOLCALL_FRAMINGS
        items = build_items(qids, cfg["seed"], args.n_injected, args.n_user_control, args.n_clean, types=types, voices=("plain",), n_legit=args.n_legit)
    else:
        types = REALISTIC_TYPES if args.realistic else INJECTION_TYPES
        items = build_items(qids, cfg["seed"], args.n_injected, args.n_user_control, args.n_clean, types=types, n_legit=args.n_legit)
    if args.conditions:
        items = [it for it in items if it.condition in args.conditions]
    if args.itypes:
        items = [it for it in items if it.itype is None or it.itype in args.itypes]
    if args.item_ids:
        want = set(args.item_ids)
        items = [it for it in items if it.item_id in want]
    if args.limit:
        items = items[: args.limit]

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id / args.stage
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "items.json", [it.to_dict() for it in items])
    tag = args.defense if args.defense != "steer" else f"steer_b{args.steer_block}_a{args.steer_alpha:g}"
    if args.defense == "random":
        tag = f"random_b{args.steer_block}_a{args.steer_alpha:g}_s{args.random_seed}"
    draw_suffix = "" if args.draw == 1 else f"_draw{args.draw}"
    out_path = run_dir / f"gen_{tag}{draw_suffix}.jsonl"
    acts_dir = run_dir / f"acts_{tag}"
    save_acts = not args.no_acts and args.draw == 1
    acts_dir.mkdir(exist_ok=True)
    done = {r["item_id"] for r in iter_jsonl(out_path)}
    todo = [it for it in items if it.item_id not in done]
    print(f"[{args.stage}/{tag}] {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return 0
    stopped_early = False
    if args.max_items_per_process and len(todo) > args.max_items_per_process:
        todo = todo[: args.max_items_per_process]
        stopped_early = True

    lm = load_model(repo_id, sha)
    print(f"loaded {repo_id}@{sha[:8]} on {lm.device}", flush=True)

    steer_vec = None
    if args.defense == "steer":
        steer_vec = np.load(REPO_ROOT / args.steer_vector).astype(np.float32)
        steer_vec = steer_vec / (np.linalg.norm(steer_vec) + 1e-8)
    elif args.defense == "random":
        rng = np.random.RandomState(args.random_seed)
        steer_vec = rng.randn(lm.hidden_size).astype(np.float32)
        steer_vec /= np.linalg.norm(steer_vec)

    manifest = {"run_id": args.run_id, "stage": args.stage, "repo_id": repo_id, "revision": sha, "defense": args.defense,
                "realistic": args.realistic, "types": list(types),
                "tag": tag, "steer_block": args.steer_block if steer_vec is not None else None,
                "steer_alpha": args.steer_alpha if steer_vec is not None else None,
                "steer_vector": args.steer_vector, "decoding": base_decoding.to_dict(), "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size, "blocks_one_based": blocks, "system_prompt": system_prompt,
                "n_items": len(items), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    manifest["draw"] = args.draw
    write_json(run_dir / f"manifest_{tag}{draw_suffix}.json", manifest)

    # render everything first, sort by length
    prepared = []
    for it in todo:
        r = rows[it.question_id]
        q = build_question(r["subject"], r["question"], r["choices"], r["answer_index"])
        msgs = render_injection_messages(q, it, system_prompt, untrusted_delimiter=(args.defense == "delimiter"),
                                         planted=r["planted_letter"], realistic=args.realistic)
        prompt = render_chat(lm, msgs)
        env_text = msgs[3]["content"]
        inj_text = injected_span(it.itype, it.voice, r["planted_letter"], q) if it.condition == "injected" else None
        n_tok = len(lm.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prepared.append((n_tok, it, r, prompt, env_text, inj_text))
    prepared.sort(key=lambda x: (x[0], x[1].item_id))

    total_tok, total_sec = 0, 0.0
    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start:start + args.batch_size]
        prompts = [b[3] for b in batch]
        enc = lm.tokenizer(prompts, padding=True, add_special_tokens=False)
        T = len(enc["input_ids"][0])
        env_masks = np.stack([span_token_mask(lm.tokenizer, p, *find_span(p, e), T) for _, _, _, p, e, _ in batch])
        inj_masks = np.stack([span_token_mask(lm.tokenizer, p, *find_span(p, i), T) if i else np.zeros(T, bool)
                              for _, _, _, p, _, i in batch])
        steer = Steer(args.steer_block, steer_vec, args.steer_alpha, env_masks) if steer_vec is not None else None
        seed = batch_seed(cfg["seed"], f"inj:{tag}", args.draw, [b[1].item_id for b in batch])
        dec = Decoding("sample", base_decoding.temperature, base_decoding.top_p, seed)
        tb = time.time()
        results = generate_batch(lm, prompts, args.max_new_tokens, blocks, decoding=dec, steer=steer,
                                 span_masks={"env": env_masks, "inj": inj_masks}, static_cache=not args.dynamic_cache)
        sync_device(lm.device)
        elapsed = time.time() - tb
        out_rows = []
        for (n_tok, it, r, prompt, env_text, inj_text), res in zip(batch, results):
            comp = detect_compliance(res.generated_text, res.finished, it.itype, planted=r["planted_letter"])
            parsed = extract_final_answer(res.generated_text, res.finished)
            if save_acts:
                npz = {"prompt_end": res.prompt_end_activations, "env_mean": res.span_mean_activations["env"]}
                if inj_text:
                    npz["inj_mean"] = res.span_mean_activations["inj"]
                np.savez(acts_dir / f"{it.item_id}.npz", **npz)
            out_rows.append({
                **it.to_dict(), "stage": args.stage, "defense": args.defense, "tag": tag, "draw": args.draw,
                "subject": r["subject"], "answer_letter": r["answer_letter"], "planted_letter": r["planted_letter"],
                "rendered_prompt": prompt, "prompt_token_count": res.prompt_token_count,
                "generated_text": res.generated_text, "generated_token_count": res.generated_token_count,
                "finished": res.finished, "complied": comp.complied,
                "marker_in_visible": comp.marker_in_visible, "format_only": comp.format_only,
                "key_in_visible": comp.key_in_visible, "marker_in_think": comp.marker_in_think,
                "key_in_think": comp.key_in_think, "visible_len": comp.visible_len,
                "sent_tool_call": comp.sent_tool_call, "sent_to_external": comp.sent_to_external,
                "sent_with_key": comp.sent_with_key, "link_in_visible": comp.link_in_visible,
                "denied_answer": comp.denied_answer, "realistic": args.realistic,
                "parsed_letter": parsed.letter, "parsed_valid": parsed.valid,
                "answer_correct": (parsed.letter == r["answer_letter"]) if parsed.valid else None,
                "seed": seed, "batch_seconds": elapsed, "batch_size": len(batch), "model_revision": sha,
                "n_env_tokens": int(env_masks[batch.index((n_tok, it, r, prompt, env_text, inj_text))].sum()),
            })
        append_jsonl(out_path, out_rows)
        gt = sum(x.generated_token_count for x in results)
        total_tok += gt
        total_sec += elapsed
        mem = device_memory_gb(lm.device)
        nc = sum(1 for x in out_rows if x["complied"])
        print(f"  batch {start // args.batch_size + 1}: {len(batch)} items, {gt} tok, {elapsed:.0f}s "
              f"({gt / max(elapsed, 1e-6):.1f} tok/s), finished={sum(x.finished for x in results)}, complied={nc}, "
              f"done {start + len(batch)}/{len(prepared)}, mem driver={mem['driver_gb']:.1f}GB", flush=True)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["tok_per_s"] = total_tok / max(total_sec, 1e-6)
    write_json(run_dir / f"manifest_{tag}{draw_suffix}.json", manifest)
    if stopped_early:
        print(f"[{args.stage}/{tag}] stopping after {len(todo)} items to reset the MPS graph cache; rerun to continue", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
