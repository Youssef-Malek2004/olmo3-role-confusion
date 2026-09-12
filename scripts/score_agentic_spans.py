#!/usr/bin/env python
"""Project the injected payload span of agentic round-2 prompts onto a direction; report per-type means and compliance.

Usage: python scripts/score_agentic_spans.py --run-id agentic_pilot --stage instruct --tag webpage --direction results/generated/roles_mac/dir_instruct_b16.npy [--block 16]
Reads gen_<tag>[_drawN].jsonl rows that have r2_prompt (items that fetched), finds the payload text, prefill at --block, writes
results/generated/<run_id>/agentic_span_scores_<stage>_<tag>_b<block>_<name>.jsonl and prints per-type inj_proj vs exfil rate.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, sys
from pathlib import Path
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface")); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75"); os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
from role_confusion.agentic import payload_text  # noqa: E402
from role_confusion.generation import find_span, load_model, prefill_hidden_states, span_token_mask  # noqa: E402
from role_confusion.io import read_json  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True); ap.add_argument("--stage", required=True); ap.add_argument("--tag", default="webpage")
    ap.add_argument("--direction", required=True); ap.add_argument("--block", type=int, default=16); ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    cfg = read_json(REPO_ROOT / "configs/pilot.json"); revs = read_json(REPO_ROOT / "configs/revisions.json")
    repo = next(m for m in cfg["models"] if m["stage"] == a.stage)["repo_id"]; sha = revs["models"][repo]["sha"]
    d = np.load(REPO_ROOT / a.direction).astype(np.float32); d /= np.linalg.norm(d) + 1e-8
    name = a.name or Path(a.direction).stem
    rows = []
    for f in sorted(glob.glob(str(REPO_ROOT / "artifacts/runs" / a.run_id / a.stage / f"gen_{a.tag}*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r.get("r2_prompt") and r["condition"] == "injected"]
    if not rows:
        print("no round-2 injected rows"); return 1
    ctx = rows[0]["context"]
    lm = load_model(repo, sha)
    jobs = sorted(rows, key=lambda r: len(r["r2_prompt"]))
    out_path = REPO_ROOT / "results/generated" / a.run_id / f"agentic_span_scores_{a.stage}_{a.tag}_b{a.block}_{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by = collections.defaultdict(list)
    with open(out_path, "w") as fo:
        for s in range(0, len(jobs), a.batch_size):
            batch = jobs[s:s + a.batch_size]
            hidden, _ = prefill_hidden_states(lm, [r["r2_prompt"] for r in batch], [a.block]); T = hidden.shape[1]
            for i, r in enumerate(batch):
                span = payload_text(r["itype"], ctx)
                h = hidden[i, :, 0, :].astype(np.float32)
                try:
                    m = span_token_mask(lm.tokenizer, r["r2_prompt"], *find_span(r["r2_prompt"], span), T)
                except ValueError:
                    continue
                proj = float((h[m] @ d).mean())
                rec = {"item_id": r["item_id"], "itype": r["itype"], "draw": r.get("draw", 1), "inj_proj": proj, "n_tokens": int(m.sum()),
                       "complied": r.get("complied"), "sent_tool_call": r.get("sent_tool_call"), "finished": r.get("finished")}
                fo.write(json.dumps(rec) + "\n"); by[r["itype"]].append(rec)
            print(f"  {min(s + a.batch_size, len(jobs))}/{len(jobs)}", flush=True)
    print(f"\ndirection {name}, block {a.block}, {ctx}")
    for t, rs in sorted(by.items(), key=lambda kv: np.mean([r["inj_proj"] for r in kv[1]])):
        fin = [r for r in rs if r["finished"]]
        print(f"  {t:<24} inj_proj {np.mean([r['inj_proj'] for r in rs]):6.2f}  exfil {sum(1 for r in fin if r['complied'])}/{len(fin)}")
    fin = [r for rs in by.values() for r in rs if r["finished"]]
    if len({r["complied"] for r in fin}) == 2:
        from sklearn.metrics import roc_auc_score
        print(f"  item-level AUROC(-inj_proj -> exfil) = {roc_auc_score([bool(r['complied']) for r in fin], [-r['inj_proj'] for r in fin]):.3f}  (n={len(fin)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
