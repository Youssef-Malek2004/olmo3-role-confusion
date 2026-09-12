#!/usr/bin/env python
"""Restartable paired generation for one checkpoint at a time.

Usage:
    python scripts/run_generation.py --run-id pilot01 --stage sft --split pilot \
        --variants clean hinted --config configs/pilot.json [--batch-size 8] [--limit N]

Outputs under artifacts/runs/<run_id>/<stage>/:
    <variant>.jsonl                 one row per question, appended as batches finish
    acts/<variant>/<question_id>.npy prompt-end activations, shape (n_blocks, hidden), float32
    manifest.json                   model/library/device/template/settings record

Re-running skips questions already present in the JSONL. Raw generated text is never modified.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Fail fast instead of swapping: cap MPS allocations at a fraction of the recommended working set.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import numpy as np  # noqa: E402

from role_confusion.generation import Decoding, batch_seed, device_memory_gb, generate_batch, load_model, render_chat, sync_device  # noqa: E402
from role_confusion.io import append_jsonl, iter_jsonl, read_json, write_json  # noqa: E402
from role_confusion.parsing import extract_final_answer  # noqa: E402
from role_confusion.prompts import render_messages, render_user_message  # noqa: E402
from role_confusion.questions import build_question  # noqa: E402


def git_rev() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def lib_versions() -> dict:
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "python": platform.python_version()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variants", nargs="+", default=["clean", "hinted"])
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--revisions", default="configs/revisions.json")
    ap.add_argument("--questions", default="data/processed/questions.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N question IDs (sorted)")
    ap.add_argument("--question-ids", nargs="*", default=None, help="explicit subset of question IDs")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="override config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dynamic-cache", action="store_true", help="use the default growing cache instead of static")
    ap.add_argument("--draw", type=int, default=1,
                    help="sample index; draw>1 writes <variant>_draw<N>.jsonl with a different seed and reuses draw-1 activations")
    ap.add_argument("--decoding", choices=["config", "greedy", "sample"], default="config")
    args = ap.parse_args()

    cfg = read_json(REPO_ROOT / args.config)
    revs = read_json(REPO_ROOT / args.revisions)
    model_cfg = next(m for m in cfg["models"] if m["stage"] == args.stage)
    repo_id = model_cfg["repo_id"]
    sha = revs["models"][repo_id]["sha"]
    max_new = args.max_new_tokens or cfg["generation"]["max_new_tokens"]
    blocks = cfg["activations"]["block_numbers_one_based"]
    system_prompt = cfg["generation"].get("system_prompt")
    gen_cfg = cfg["generation"]
    mode = gen_cfg.get("decoding", "sample" if gen_cfg.get("do_sample") else "greedy") if args.decoding == "config" else args.decoding
    if mode == "sample":
        base_decoding = Decoding("sample", float(gen_cfg["temperature"]), float(gen_cfg["top_p"]), None)
    else:
        base_decoding = Decoding("greedy")
    if args.draw > 1 and mode != "sample":
        print("draw>1 only makes sense with sampling", file=sys.stderr)
        return 2

    rows = [r for r in iter_jsonl(REPO_ROOT / args.questions) if r["split"] == args.split]
    rows.sort(key=lambda r: r["question_id"])
    if args.question_ids:
        want = set(args.question_ids)
        rows = [r for r in rows if r["question_id"] in want]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no questions selected", file=sys.stderr)
        return 1

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id / args.stage
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {repo_id}@{sha[:8]} ...", flush=True)
    t0 = time.time()
    lm = load_model(repo_id, sha, device=args.device)
    attn_impl = getattr(lm.model.config, "_attn_implementation", "unknown")
    print(f"loaded in {time.time() - t0:.0f}s on {lm.device}; layers={lm.num_layers} hidden={lm.hidden_size} attn={attn_impl}",
          flush=True)

    manifest_path = run_dir / "manifest.json"
    example_q = build_question(rows[0]["subject"], rows[0]["question"], rows[0]["choices"],
                               rows[0]["answer_index"])
    manifest = {
        "run_id": args.run_id, "stage": args.stage, "split": args.split, "repo_id": repo_id,
        "revision": sha, "config_path": args.config, "questions_path": args.questions,
        "code_revision": git_rev(), "libraries": lib_versions(), "device": lm.device,
        "dtype": lm.dtype, "num_layers": lm.num_layers, "hidden_size": lm.hidden_size,
        "blocks_one_based": blocks, "hidden_state_index_convention":
            "hidden_states[0]=embeddings; hidden_states[k]=output of block k (one-based)",
        "activation_position": "last non-padding input token (left padding, position -1)",
        "max_new_tokens": max_new, "do_sample": base_decoding.mode == "sample", "batch_size": args.batch_size,
        "padding_side": "left", "system_prompt": system_prompt, "cache_implementation": "dynamic" if args.dynamic_cache else "static", "attn_implementation": attn_impl,
        "decoding": base_decoding.to_dict(), "draw": args.draw, "base_seed": cfg["seed"],
        "seed_rule": "per batch: sha256(base_seed|variant|draw|comma-joined question IDs)[:8] & 0x7FFFFFFF; torch.manual_seed before generate",
        "chat_template_sha1": lm.chat_template_sha1, "eos_token_ids": list(lm.eos_token_ids),
        "example_rendered_prompt_clean": render_chat(lm, render_messages(example_q, "clean", None, system_prompt)),
        "example_rendered_prompts_by_variant": {
            v: render_chat(lm, render_messages(example_q, v, rows[0]["planted_letter"], system_prompt))
            for v in args.variants},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(manifest_path, manifest)

    suffix = "" if args.draw == 1 else f"_draw{args.draw}"
    save_acts = args.draw == 1
    items = []  # (n_tokens, variant, row, prompt)
    for variant in args.variants:
        out_path = run_dir / f"{variant}{suffix}.jsonl"
        (run_dir / "acts" / variant).mkdir(parents=True, exist_ok=True)
        done = {r["question_id"] for r in iter_jsonl(out_path)}
        todo = [r for r in rows if r["question_id"] not in done]
        print(f"[{args.stage}/{variant}] {len(done)} done, {len(todo)} to go", flush=True)
        for r in todo:
            q = build_question(r["subject"], r["question"], r["choices"], r["answer_index"])
            prompt = render_chat(lm, render_messages(q, variant, r["planted_letter"], system_prompt))
            n_tok = len(lm.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            items.append((n_tok, variant, r, prompt))
    # Sort by prompt length across variants so batches are full; each row is routed to its variant file.
    items.sort(key=lambda x: (x[0], x[1], x[2]["question_id"]))

    total_gen_tokens = 0
    total_seconds = 0.0
    for start in range(0, len(items), args.batch_size):
        batch = items[start:start + args.batch_size]
        prompts = [p for _, _, _, p in batch]
        tb = time.time()
        qids = [f"{v}:{r['question_id']}" for _, v, r, _ in batch]
        decoding = base_decoding
        if base_decoding.mode == "sample":
            decoding = Decoding("sample", base_decoding.temperature, base_decoding.top_p,
                                batch_seed(cfg["seed"], "mixed", args.draw, qids))
        results = generate_batch(lm, prompts, max_new, blocks, static_cache=not args.dynamic_cache,
                                 decoding=decoding)
        sync_device(lm.device)
        elapsed = time.time() - tb
        by_variant: dict[str, list[dict]] = {}
        for (_, variant, r, _), res in zip(batch, results):
            acts_dir = run_dir / "acts" / variant
            parsed = extract_final_answer(res.generated_text, res.finished)
            if save_acts:
                np.save(acts_dir / f"{r['question_id']}.npy", res.prompt_end_activations)
            by_variant.setdefault(variant, []).append({
                "question_id": r["question_id"], "stage": args.stage, "variant": variant, "draw": args.draw,
                "decoding": decoding.to_dict(),
                "split": args.split, "subject": r["subject"], "answer_letter": r["answer_letter"],
                "planted_letter": r["planted_letter"], "rendered_prompt": res.rendered_prompt,
                "prompt_token_count": res.prompt_token_count,
                "generated_text": res.generated_text,
                "generated_token_count": res.generated_token_count, "finished": res.finished,
                "parsed_letter": parsed.letter, "parsed_valid": parsed.valid,
                "parsed_reason": parsed.reason, "parsed_n_matches": parsed.n_matches,
                "parsed_distinct_letters": list(parsed.distinct_letters),
                "parsed_region": parsed.answer_region, "batch_seconds": elapsed,
                "batch_size": len(batch), "model_revision": sha,
                "activation_file": str((acts_dir / f"{r['question_id']}.npy").relative_to(REPO_ROOT)),
            })
        for variant, out_rows in by_variant.items():
            append_jsonl(run_dir / f"{variant}{suffix}.jsonl", out_rows)
        gen_toks = sum(res.generated_token_count for res in results)
        total_gen_tokens += gen_toks
        total_seconds += elapsed
        n_fin = sum(res.finished for res in results)
        n_valid = sum(1 for rows_ in by_variant.values() for row in rows_ if row["parsed_valid"])
        mem = device_memory_gb(lm.device)
        print(f"  batch {start // args.batch_size + 1}: {len(batch)} items, {gen_toks} gen tokens, "
              f"{elapsed:.0f}s ({gen_toks / max(elapsed, 1e-6):.1f} tok/s), finished={n_fin}, "
              f"valid={n_valid}, done {start + len(batch)}/{len(items)}, "
              f"mem alloc={mem['allocated_gb']:.1f}GB driver={mem['driver_gb']:.1f}GB", flush=True)
    if total_seconds:
        print(f"[{args.stage}] total {total_gen_tokens} gen tokens in {total_seconds:.0f}s "
              f"= {total_gen_tokens / total_seconds:.1f} tok/s aggregate", flush=True)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
