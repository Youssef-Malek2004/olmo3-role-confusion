#!/usr/bin/env python
"""Fit user-vs-environment role probes per stage and layer; save difference-of-means directions.

Usage:
    python scripts/fit_role_probe.py --run-id roles01 --stages sft rlvr [--inj-run inj_pilot]

Per stage and block:
  * token-level logistic regression (passage-level split, 5-fold) -> accuracy / AUROC
  * mean-pooled per passage probe -> accuracy
  * same-text control is built in: every passage appears in both roles, so content cannot carry the label
  * difference-of-means direction (env - user), unit norm, saved to results/generated/<run_id>/dir_<stage>_b<block>.npy
  * cross-stage: SFT direction applied to RLVR activations and vice versa (cosine + transfer accuracy)
If --inj-run is given, projects each injected item's inj_mean and env_mean activations onto the direction
("role confusion score": more negative = more user-like) and reports the score split by compliance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from role_confusion.io import read_json, read_jsonl, write_json  # noqa: E402


def load_stage(run_dir: Path):
    passages = read_json(run_dir / "passages.json")
    X_tok, y_tok, g_tok, kind_tok = [], [], [], []
    X_mean, y_mean, g_mean, kind_mean = [], [], [], []
    blocks = None
    for i, p in enumerate(passages):
        for role, label in (("user", 0), ("env", 1)):
            f = run_dir / "roles" / f"{p['passage_id']}_{role}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            acts = z["acts"].astype(np.float32)  # (n_tok, L, H)
            blocks = list(z["blocks"])
            if acts.shape[0] == 0:
                continue
            X_tok.append(acts); y_tok.append(np.full(acts.shape[0], label)); g_tok.append(np.full(acts.shape[0], i))
            kind_tok.append(np.array([p["kind"]] * acts.shape[0]))
            X_mean.append(acts.mean(0)); y_mean.append(label); g_mean.append(i); kind_mean.append(p["kind"])
    return (np.concatenate(X_tok), np.concatenate(y_tok), np.concatenate(g_tok), np.concatenate(kind_tok),
            np.stack(X_mean), np.array(y_mean), np.array(g_mean), np.array(kind_mean), blocks)


def cv_probe(X, y, groups, C=0.1, k=5):
    gkf = GroupKFold(n_splits=k)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    return {"auroc": float(roc_auc_score(y, oof)), "acc": float(((oof > 0).astype(int) == y).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stages", nargs="*", default=["sft", "rlvr"])
    ap.add_argument("--inj-run", default=None)
    ap.add_argument("--max-tokens-per-stage", type=int, default=40000)
    args = ap.parse_args()

    res_dir = REPO_ROOT / "results" / "generated" / args.run_id
    res_dir.mkdir(parents=True, exist_ok=True)
    out: dict = {"run_id": args.run_id, "stages": {}, "cross_stage": {}}
    dirs: dict[tuple[str, int], np.ndarray] = {}
    data = {}
    rng = np.random.RandomState(0)
    for stage in args.stages:
        run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id / stage
        if not (run_dir / "passages.json").exists():
            print(f"skip {stage}", file=sys.stderr)
            continue
        Xt, yt, gt, kt, Xm, ym, gm, km, blocks = load_stage(run_dir)
        # subsample tokens for tractable CV
        if len(yt) > args.max_tokens_per_stage:
            idx = rng.choice(len(yt), args.max_tokens_per_stage, replace=False)
            Xt, yt, gt, kt = Xt[idx], yt[idx], gt[idx], kt[idx]
        data[stage] = (Xt, yt, gt, kt, Xm, ym, gm, km, blocks)
        st = {"n_passages": int(len(ym) // 2), "n_tokens": int(len(yt)), "blocks": {}}
        for bi, b in enumerate(blocks):
            Xb_t, Xb_m = Xt[:, bi, :], Xm[:, bi, :]
            tok = cv_probe(Xb_t, yt, gt)
            mean = cv_probe(Xb_m, ym, gm)
            mu_env, mu_user = Xb_m[ym == 1].mean(0), Xb_m[ym == 0].mean(0)
            d = mu_env - mu_user
            dn = d / (np.linalg.norm(d) + 1e-8)
            dirs[(stage, int(b))] = dn
            np.save(res_dir / f"dir_{stage}_b{b}.npy", dn)
            proj = Xb_m @ dn
            sep = (proj[ym == 1].mean() - proj[ym == 0].mean()) / (proj.std() + 1e-8)
            # per passage kind: token-level accuracy using the mean-diff direction with midpoint threshold
            thr = (proj[ym == 1].mean() + proj[ym == 0].mean()) / 2
            kinds = {}
            for kind in sorted(set(km)):
                m = km == kind
                kinds[kind] = float((((proj[m] > thr).astype(int)) == ym[m]).mean())
            st["blocks"][str(b)] = {"token_probe_cv": tok, "mean_probe_cv": mean,
                                    "diffmean_norm": float(np.linalg.norm(d)), "diffmean_separation_sd": float(sep),
                                    "diffmean_midpoint_acc_by_kind": kinds,
                                    "resid_norm_mean": float(np.linalg.norm(Xb_m, axis=1).mean())}
        out["stages"][stage] = st
        print(stage, json.dumps({b: {"tok": v["token_probe_cv"], "mean": v["mean_probe_cv"], "sep_sd": round(v["diffmean_separation_sd"], 2)}
                                 for b, v in st["blocks"].items()}, indent=None))

    stages = [s for s in args.stages if s in data]
    if len(stages) >= 2:
        for a in stages:
            for b_ in stages:
                if a == b_:
                    continue
                blocks = data[a][8]
                res = {}
                for bi, b in enumerate(blocks):
                    da, db = dirs[(a, b)], dirs[(b_, b)]
                    Xm, ym = data[b_][4][:, bi, :], data[b_][5]
                    proj = Xm @ da
                    thr = (proj[ym == 1].mean() + proj[ym == 0].mean()) / 2  # threshold refit on target (cheap repair)
                    res[str(b)] = {"cosine": float(da @ db), "transfer_acc_midpoint": float((((proj > thr).astype(int)) == ym).mean()),
                                   "transfer_auroc": float(roc_auc_score(ym, proj))}
                out["cross_stage"][f"{a}->{b_}"] = res
                print(f"{a}->{b_}", json.dumps(res))

    if args.inj_run:
        out["injection_scores"] = {}
        for stage in stages:
            inj_dir = REPO_ROOT / "artifacts" / "runs" / args.inj_run / stage
            gen = [r for r in read_jsonl(inj_dir / "gen_none.jsonl")] if (inj_dir / "gen_none.jsonl").exists() else []
            blocks_inj = read_json(next(inj_dir.glob("manifest_none.json")))["blocks_one_based"] if gen else []
            rows = []
            for r in gen:
                z = np.load(inj_dir / "acts_none" / f"{r['item_id']}.npz")
                rec = {"item_id": r["item_id"], "condition": r["condition"], "itype": r["itype"], "voice": r["voice"],
                       "complied": r["complied"], "finished": r["finished"]}
                for bi, b in enumerate(blocks_inj):
                    if (stage, int(b)) not in dirs:
                        continue
                    dn = dirs[(stage, int(b))]
                    rec[f"env_proj_b{b}"] = float(z["env_mean"][bi] @ dn)
                    if "inj_mean" in z:
                        rec[f"inj_proj_b{b}"] = float(z["inj_mean"][bi] @ dn)
                rows.append(rec)
            summary = {}
            for b in blocks_inj:
                key = f"inj_proj_b{b}"
                inj = [r for r in rows if r["condition"] == "injected" and key in r and r["complied"] is not None]
                if inj:
                    c = [r[key] for r in inj if r["complied"]]
                    n = [r[key] for r in inj if not r["complied"]]
                    summary[str(b)] = {"n_complied": len(c), "n_not": len(n),
                                       "mean_proj_complied": float(np.mean(c)) if c else None,
                                       "mean_proj_not": float(np.mean(n)) if n else None,
                                       "auroc_lower_proj_predicts_compliance": (float(roc_auc_score([r["complied"] for r in inj], [-r[key] for r in inj]))
                                                                                 if c and n else None),
                                       "by_voice_mean_proj": {v: float(np.mean([r[key] for r in inj if r["voice"] == v])) for v in sorted({r["voice"] for r in inj})}}
            out["injection_scores"][stage] = {"rows": rows, "summary": summary}
            print(stage, "injection scores", json.dumps(summary))

    write_json(res_dir / "role_probe.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
