#!/usr/bin/env python
"""Prompt-only pass: project each injected span (and the whole tool turn) onto the role direction for a vLLM run.

Usage:
    python scripts/score_spans.py --run-id inj_matrix --stage sft --roles-run roles_gpu [--block 16] [--batch-size 16]

Reads artifacts/runs/<run_id>/<stage>/items.json (types/voices) and gen_none.jsonl (for planted letters and outcomes),
renders the same prompts, runs a prefill with hidden states, and writes results/generated/<run_id>/span_scores_<stage>.jsonl
with per-item: inj_proj (mean over injected span), env_proj (mean over tool turn), inj_frac_user (fraction of injected
tokens whose projection is closer to the user mean than the tool mean), plus per-draw compliance from all gen files.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

from role_confusion.generation import find_span, load_model, prefill_hidden_states, render_chat, span_token_mask  # noqa: E402
from role_confusion.injection import InjectionItem, detect_compliance, injected_span, render_injection_messages  # noqa: E402
from role_confusion.io import iter_jsonl, read_json  # noqa: E402
from role_confusion.questions import build_question  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--roles-run", default="roles_gpu")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cap", type=int, default=3000, help="effective cap for compliance labels")
    ap.add_argument("--direction", default=None, help="optional .npy unit direction to use instead of the role direction")
    ap.add_argument("--tag", default=None, help="output filename tag (default: role)")
    ap.add_argument("--types", nargs="*", default=None, help="restrict to these injected types")
    args = ap.parse_args()

    cfg = read_json(REPO_ROOT / "configs/pilot.json")
    revs = read_json(REPO_ROOT / "configs/revisions.json")
    repo = next(m for m in cfg["models"] if m["stage"] == args.stage)["repo_id"]
    sha = revs["models"][repo]["sha"]
    system_prompt = cfg["generation"]["system_prompt"]
    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id / args.stage
    items = [InjectionItem(**d) for d in read_json(run_dir / "items.json")]
    rows = {r["question_id"]: r for r in iter_jsonl(REPO_ROOT / "data/processed/questions.jsonl")}
    if args.direction:
        direction = np.load(REPO_ROOT / args.direction).astype(np.float32)
        ref = {"user": float("nan"), "env": float("nan")}
        mid = 0.0
    else:
        direction = np.load(REPO_ROOT / "results" / "generated" / args.roles_run / f"dir_{args.stage}_b{args.block}.npy")
        # reference means for user vs tool text at this block from the roles run
        li = [8, 12, 16, 20, 24].index(args.block)
        roles_dir = REPO_ROOT / "artifacts" / "runs" / args.roles_run / args.stage / "roles"
        ref = {}
        for role in ("user", "env"):
            vals = []
            for f in sorted(roles_dir.glob(f"*_{role}.npz"))[:200]:
                a = np.load(f)["acts"].astype(np.float32)
                vals.extend((a[:, li, :] @ direction).tolist())
            ref[role] = float(np.mean(vals))
        mid = (ref["user"] + ref["env"]) / 2

    # compliance per item across draws
    comp: dict[str, list] = collections.defaultdict(list)
    for f in sorted(run_dir.glob("gen_none*.jsonl")):
        for r in iter_jsonl(f):
            fin = r["finished"] and r["generated_token_count"] <= args.cap
            c = detect_compliance(r["generated_text"], fin, r["itype"], planted=r.get("planted_letter")).complied if fin else None
            comp[r["item_id"]].append(c)

    lm = load_model(repo, sha)
    realistic = any(it.itype not in ("marker", "format", "exfil") for it in items)
    jobs = []
    for it in items:
        if it.condition != "injected" or (args.types and it.itype not in args.types):
            continue
        r = rows[it.question_id]
        q = build_question(r["subject"], r["question"], r["choices"], r["answer_index"])
        msgs = render_injection_messages(q, it, system_prompt, planted=r["planted_letter"], realistic=realistic)
        prompt = render_chat(lm, msgs)
        jobs.append((it, prompt, msgs[3]["content"], injected_span(it.itype, it.voice, r["planted_letter"], q)))
    jobs.sort(key=lambda j: len(j[1]))
    tag = args.tag or ("role" if not args.direction else "custom")
    out_path = REPO_ROOT / "results" / "generated" / args.run_id / f"span_scores_{args.stage}_b{args.block}_{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as fo:
        for s in range(0, len(jobs), args.batch_size):
            batch = jobs[s:s + args.batch_size]
            hidden, _ = prefill_hidden_states(lm, [j[1] for j in batch], [args.block])  # (B, T, 1, H)
            T = hidden.shape[1]
            for i, (it, prompt, env_text, inj_text) in enumerate(batch):
                h = hidden[i, :, 0, :].astype(np.float32)
                env_m = span_token_mask(lm.tokenizer, prompt, *find_span(prompt, env_text), T)
                inj_m = span_token_mask(lm.tokenizer, prompt, *find_span(prompt, inj_text), T)
                pi = h[inj_m] @ direction
                pe = h[env_m] @ direction
                cs = comp.get(it.item_id, [])
                fo.write(json.dumps({"item_id": it.item_id, "itype": it.itype, "voice": it.voice, "block": args.block,
                                     "inj_proj": float(pi.mean()), "env_proj": float(pe.mean()),
                                     "inj_frac_user": float((pi < mid).mean()), "n_inj_tokens": int(inj_m.sum()),
                                     "ref_user": ref["user"], "ref_env": ref["env"],
                                     "complied_draws": cs, "n_complied": sum(1 for c in cs if c), "n_scorable": sum(1 for c in cs if c is not None)}) + "\n")
                n += 1
            print(f"  {n}/{len(jobs)}", flush=True)
    # per-type summary
    by = collections.defaultdict(list)
    for r in iter_jsonl(out_path):
        by[r["itype"]].append(r)
    print(f"\nref user {ref['user']:.2f} | ref tool {ref['env']:.2f} (block {args.block})")
    for t, rs in sorted(by.items(), key=lambda kv: np.mean([r["inj_proj"] for r in kv[1]])):
        nc = sum(r["n_complied"] for r in rs); ns = sum(r["n_scorable"] for r in rs)
        print(f"  {t:<24} inj_proj {np.mean([r['inj_proj'] for r in rs]):6.2f}  frac_user {np.mean([r['inj_frac_user'] for r in rs]):.2f}  complied {nc}/{ns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
