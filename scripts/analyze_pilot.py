#!/usr/bin/env python
"""Pilot-sized probe sanity analysis. Exploratory by construction: ~40 questions per stage.

Usage:
    python scripts/analyze_pilot.py --run-id pilot01 [--stages sft rlvr]

Requires results/generated/<run_id>/labels_<stage>.jsonl from summarize_run.py. Writes
results/generated/<run_id>/probe_pilot.json. Reports, per stage and block:
  * out-of-fold CV AUROC of the hinted-activation probe for each C
  * the same with shuffled labels (several seeds)
  * simple-feature baseline (subject, planted letter, answer letter, prompt length)
  * clean-activation + planted-letter one-hot control
  * frozen cross-stage transfer between the two stages (fit on all of A, score all of B)
Nothing here is a held-out result; it only tells us whether the pipeline has any signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from role_confusion.io import read_jsonl, write_json  # noqa: E402
from role_confusion.probe import (  # noqa: E402
    cv_auroc, evaluate, fit_probe, load_stage, planted_onehot, safe_auroc, simple_features,
)

C_GRID = [0.001, 0.01, 0.1, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stages", nargs="*", default=["sft", "rlvr"])
    ap.add_argument("--shuffle-seeds", type=int, default=5)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id
    res_dir = REPO_ROOT / "results" / "generated" / args.run_id
    out: dict = {"run_id": args.run_id, "exploratory": True, "stages": {}}
    data = {}
    for stage in args.stages:
        labels_path = res_dir / f"labels_{stage}.jsonl"
        if not labels_path.exists():
            print(f"skip {stage}: no labels", file=sys.stderr)
            continue
        hinted = load_stage(run_dir / stage, labels_path, "hinted")
        clean = load_stage(run_dir / stage, labels_path, "clean")
        assert hinted.question_ids == clean.question_ids
        data[stage] = (hinted, clean)
        gen_rows = {r["question_id"]: r for r in read_jsonl(run_dir / stage / "hinted.jsonl")}
        ptoks = {q: gen_rows[q]["prompt_token_count"] for q in hinted.question_ids}
        y = hinted.y
        st: dict = {"n": int(len(y)), "n_pos": int(y.sum()), "blocks": {}}
        if y.sum() == 0 or (y == 0).sum() == 0:
            st["note"] = "one class only; AUROC undefined"
        for b in hinted.blocks:
            Xh, Xc = hinted.block(b), clean.block(b)
            blk = {"hinted_cv": {}, "hinted_cv_shuffled": {}, "clean_plus_letter_cv": {}}
            for C in C_GRID:
                blk["hinted_cv"][str(C)] = cv_auroc(Xh, y, C)
                blk["clean_plus_letter_cv"][str(C)] = cv_auroc(np.hstack([Xc, planted_onehot(hinted.meta)]), y, C)
                shuf = [cv_auroc(Xh, y, C, seed=s, shuffle_labels=True)["auroc"] for s in range(args.shuffle_seeds)]
                blk["hinted_cv_shuffled"][str(C)] = [None if v is None else round(v, 3) for v in shuf]
            # activation geometry: how different are hinted and clean prompt-end vectors?
            cos = np.sum(Xh * Xc, axis=1) / (np.linalg.norm(Xh, axis=1) * np.linalg.norm(Xc, axis=1) + 1e-9)
            blk["cos_hinted_vs_clean_mean"] = float(cos.mean())
            blk["norm_hinted_mean"] = float(np.linalg.norm(Xh, axis=1).mean())
            st["blocks"][str(b)] = blk
        sf = simple_features(hinted.meta, ptoks)
        st["simple_features_cv"] = {str(C): cv_auroc(sf, y, C) for C in C_GRID}
        st["always_negative_balanced_accuracy"] = 0.5 if len(np.unique(y)) == 2 else None
        out["stages"][stage] = st

    # frozen transfer in both directions (all-of-A -> all-of-B), per block and C; exploratory
    if len(data) == 2:
        (a, (ha, ca)), (b, (hb, cb)) = list(data.items())
        out["frozen_transfer"] = {}
        for src, dst, hs, hd in ((a, b, ha, hb), (b, a, hb, ha)):
            if len(np.unique(hs.y)) < 2:
                out["frozen_transfer"][f"{src}->{dst}"] = {"note": f"source {src} has one class; cannot fit"}
                continue
            res = {}
            for blk in hs.blocks:
                for C in C_GRID:
                    p = fit_probe(hs.block(blk), hs.y, C, blk)
                    res[f"block{blk}_C{C}"] = {"target_auroc": safe_auroc(hd.y, p.scores(hd.block(blk))),
                                               "target_n_pos": int(hd.y.sum()), "target_n": int(len(hd.y))}
            out["frozen_transfer"][f"{src}->{dst}"] = res
        common = sorted(set(ha.question_ids) & set(hb.question_ids))
        la = {q: int(y) for q, y in zip(ha.question_ids, ha.y)}
        lb = {q: int(y) for q, y in zip(hb.question_ids, hb.y)}
        out["label_agreement_common_scorable"] = {
            "n": len(common), "both_pos": sum(la[q] and lb[q] for q in common),
            f"{a}_only": sum(la[q] and not lb[q] for q in common), f"{b}_only": sum(lb[q] and not la[q] for q in common),
        }

    write_json(res_dir / "probe_pilot.json", out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
