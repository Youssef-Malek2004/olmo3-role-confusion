#!/usr/bin/env python
"""Fast compliance matrix with vLLM (no steering hooks, no activations). Same items, prompts, seeds-per-item, detectors.

Usage (inside .venv-vllm on the pod):
    python scripts/run_injection_vllm.py --run-id inj_vllm --stage rlvr --types tc2_fake_turn tc2_cot_forgery ... \
        --n-injected 100 --n-clean 12 --draws 3 --max-new-tokens 3000

Writes artifacts/runs/<run_id>/<stage>/gen_none[_drawN].jsonl in the same row format as run_injection.py (minus activations),
so summarize_injection.py works unchanged. Per-item seeds: seed = sha256(base|item|draw). Special tokens inside tool content
are tokenized as special tokens (skip_special_tokens=False on input), matching the HF runner.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from role_confusion.injection import (  # noqa: E402
    ALL_TYPES, INJECTION_TYPES, REALISTIC_TYPES, TOOLCALL_ABLATIONS, TOOLCALL_FRAMINGS, TOOLCALL_FRAMINGS2,
    build_items, detect_compliance, render_injection_messages,
)
from role_confusion.io import append_jsonl, iter_jsonl, read_json, write_json  # noqa: E402
from role_confusion.parsing import extract_final_answer  # noqa: E402
from role_confusion.questions import build_question  # noqa: E402

SETS = {"base": INJECTION_TYPES, "realistic": REALISTIC_TYPES, "framings": TOOLCALL_FRAMINGS,
        "framings2": TOOLCALL_FRAMINGS2, "ablations": TOOLCALL_ABLATIONS, "all_tool": TOOLCALL_FRAMINGS + TOOLCALL_FRAMINGS2 + TOOLCALL_ABLATIONS}


def item_seed(base: int, item_id: str, draw: int) -> int:
    return int(hashlib.sha256(f"{base}|{item_id}|{draw}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--set", choices=list(SETS), default=None)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--voices", nargs="*", default=None, help="default: plain only for tool framings, all three for base/realistic")
    ap.add_argument("--n-injected", type=int, default=100)
    ap.add_argument("--n-user-control", type=int, default=0)
    ap.add_argument("--n-clean", type=int, default=12)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--delimiter", action="store_true", help="prompt-based defense arm")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    cfg = read_json(REPO_ROOT / "configs/pilot.json")
    revs = read_json(REPO_ROOT / "configs/revisions.json")
    repo = next(m for m in cfg["models"] if m["stage"] == args.stage)["repo_id"]
    sha = revs["models"][repo]["sha"]
    system_prompt = cfg["generation"]["system_prompt"]
    gen = cfg["generation"]

    types = tuple(args.types) if args.types else SETS[args.set or "framings2"]
    realistic = any(t not in INJECTION_TYPES for t in types)
    voices = tuple(args.voices) if args.voices else (("plain",) if realistic and any(t.startswith("tc") for t in types) else ("plain", "user", "system"))
    rows = {r["question_id"]: r for r in iter_jsonl(REPO_ROOT / "data/processed/questions.jsonl") if r["split"] in ("pilot", "train")}
    qids = sorted(rows)
    items = build_items(qids, cfg["seed"], args.n_injected, args.n_user_control, args.n_clean, types=types, voices=voices)

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id / args.stage
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "items.json", [it.to_dict() for it in items])
    tag = "delimiter" if args.delimiter else "none"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=sha)
    llm = LLM(model=repo, revision=sha, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_new_tokens + 1200, enable_prefix_caching=True, seed=cfg["seed"])
    write_json(run_dir / f"manifest_{tag}_vllm.json", {
        "run_id": args.run_id, "stage": args.stage, "repo_id": repo, "revision": sha, "engine": "vllm",
        "types": list(types), "voices": list(voices), "draws": args.draws, "max_new_tokens": args.max_new_tokens,
        "decoding": {"mode": "sample", "temperature": gen["temperature"], "top_p": gen["top_p"], "seed_rule": "sha256(base|item_id|draw)"},
        "system_prompt": system_prompt, "defense": tag, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    for draw in range(1, args.draws + 1):
        suffix = "" if draw == 1 else f"_draw{draw}"
        out_path = run_dir / f"gen_{tag}{suffix}.jsonl"
        done = {r["item_id"] for r in iter_jsonl(out_path)}
        todo = [it for it in items if it.item_id not in done]
        print(f"[{args.stage}/{tag}/draw{draw}] {len(done)} done, {len(todo)} to go", flush=True)
        if not todo:
            continue
        prompts, metas = [], []
        for it in todo:
            r = rows[it.question_id]
            q = build_question(r["subject"], r["question"], r["choices"], r["answer_index"])
            msgs = render_injection_messages(q, it, system_prompt, untrusted_delimiter=args.delimiter,
                                             planted=r["planted_letter"], realistic=realistic)
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, add_special_tokens=False)["input_ids"]  # special tokens in content stay special (matches HF runner)
            prompts.append(ids)
            metas.append((it, r, prompt))
        params = [SamplingParams(temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), max_tokens=args.max_new_tokens,
                                 seed=item_seed(cfg["seed"], it.item_id, draw), skip_special_tokens=False,
                                 stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id])
                  for it, _, _ in metas]
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
        elapsed = time.time() - t0
        out_rows = []
        for (it, r, prompt), o in zip(metas, outs):
            c = o.outputs[0]
            text = c.text
            finished = c.finish_reason == "stop"
            n_gen = len(c.token_ids)
            comp = detect_compliance(text, finished, it.itype, planted=r["planted_letter"])
            parsed = extract_final_answer(text, finished)
            out_rows.append({**it.to_dict(), "stage": args.stage, "defense": tag, "tag": tag, "draw": draw, "engine": "vllm",
                             "subject": r["subject"], "answer_letter": r["answer_letter"], "planted_letter": r["planted_letter"],
                             "rendered_prompt": prompt, "prompt_token_count": len(o.prompt_token_ids),
                             "generated_text": text, "generated_token_count": n_gen, "finished": finished,
                             "complied": comp.complied, "marker_in_visible": comp.marker_in_visible, "format_only": comp.format_only,
                             "key_in_visible": comp.key_in_visible, "marker_in_think": comp.marker_in_think, "key_in_think": comp.key_in_think,
                             "visible_len": comp.visible_len, "sent_tool_call": comp.sent_tool_call, "sent_to_external": comp.sent_to_external,
                             "sent_with_key": comp.sent_with_key, "link_in_visible": comp.link_in_visible, "denied_answer": comp.denied_answer,
                             "realistic": realistic, "parsed_letter": parsed.letter, "parsed_valid": parsed.valid,
                             "answer_correct": (parsed.letter == r["answer_letter"]) if parsed.valid else None,
                             "seed": item_seed(cfg["seed"], it.item_id, draw), "batch_seconds": elapsed, "batch_size": len(metas),
                             "model_revision": sha, "n_env_tokens": None, "activation_file": None})
        append_jsonl(out_path, out_rows)
        tot = sum(r["generated_token_count"] for r in out_rows)
        print(f"  draw {draw}: {len(out_rows)} items, {tot} tokens, {elapsed:.0f}s ({tot / max(elapsed, 1e-6):.0f} tok/s), "
              f"finished={sum(r['finished'] for r in out_rows)}, complied={sum(1 for r in out_rows if r['complied'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
