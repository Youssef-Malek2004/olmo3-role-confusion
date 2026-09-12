"""Probe fitting and evaluation on prompt-end activations.

Design rules (docs/archive/PROJECT_PLAN.md sections 10-12):
  * Rows are joined on question_id; alignment is asserted, never assumed.
  * Scaling is fit on training rows only and frozen with the probe.
  * Regularization C and the block are chosen on validation (or by CV in pilot mode) only.
  * AUROC is undefined with one class; it is reported as None, never 0.5.
  * Shuffled-label and simple-feature controls use the same pipeline as the real probe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

LETTERS = ("A", "B", "C", "D")


@dataclass
class Dataset:
    question_ids: list[str]
    X: np.ndarray  # (n, n_blocks, hidden)
    y: np.ndarray  # (n,) int {0,1}
    blocks: list[int]
    meta: list[dict] = field(default_factory=list)  # per-row label rows (subject, letters, ...)

    def block(self, b: int) -> np.ndarray:
        return self.X[:, self.blocks.index(b), :]


def load_stage(run_dir: Path, labels_path: Path, variant: str = "hinted",
               scorable_only: bool = True) -> Dataset:
    """Join label rows with activation files for one stage/variant, checking alignment explicitly."""
    label_rows = [json.loads(l) for l in open(labels_path, encoding="utf-8")]
    if scorable_only:
        label_rows = [r for r in label_rows if r["scorable"]]
    manifest = json.load(open(run_dir / "manifest.json", encoding="utf-8"))
    blocks = list(manifest["blocks_one_based"])
    acts_dir = run_dir / "acts" / variant
    X, y, ids, meta = [], [], [], []
    for r in label_rows:
        f = acts_dir / f"{r['question_id']}.npy"
        if not f.exists():
            raise FileNotFoundError(f"activation missing for {r['question_id']} ({variant})")
        a = np.load(f)
        if a.shape != (len(blocks), manifest["hidden_size"]):
            raise ValueError(f"bad activation shape {a.shape} for {r['question_id']}")
        X.append(a)
        y.append(int(bool(r["positive"])))
        ids.append(r["question_id"])
        meta.append(r)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question IDs in label rows")
    return Dataset(ids, np.stack(X) if X else np.zeros((0, len(blocks), manifest["hidden_size"])),
                   np.asarray(y, dtype=int), blocks, meta)


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


@dataclass
class FittedProbe:
    block: int
    C: float
    scaler: StandardScaler
    clf: LogisticRegression
    threshold: float

    def scores(self, X_block: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(self.scaler.transform(X_block))

    def predict(self, X_block: np.ndarray) -> np.ndarray:
        return (self.scores(X_block) >= self.threshold).astype(int)


def fit_probe(X: np.ndarray, y: np.ndarray, C: float, block: int) -> FittedProbe:
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, penalty="l2", max_iter=5000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)
    return FittedProbe(block, C, scaler, clf, threshold=0.0)


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy on the given (validation) scores."""
    cands = np.unique(scores)
    if len(cands) == 0:
        return 0.0
    mids = np.concatenate([[cands[0] - 1e-6], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-6]])
    best, best_t = -1.0, 0.0
    for t in mids:
        ba = balanced_accuracy_score(y, (scores >= t).astype(int))
        if ba > best:
            best, best_t = ba, float(t)
    return best_t


def evaluate(probe: FittedProbe, X_block: np.ndarray, y: np.ndarray) -> dict:
    s = probe.scores(X_block)
    pred = (s >= probe.threshold).astype(int)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "auroc": safe_auroc(y, s)}
    if len(np.unique(y)) == 2:
        out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
        out["recall"] = float(pred[y == 1].mean())
        out["fpr"] = float(pred[y == 0].mean())
    else:
        out["balanced_accuracy"] = None
        out["recall"] = float(pred[y == 1].mean()) if (y == 1).any() else None
        out["fpr"] = float(pred[y == 0].mean()) if (y == 0).any() else None
    return out


def cv_auroc(X: np.ndarray, y: np.ndarray, C: float, n_splits: int = 5, seed: int = 0,
             shuffle_labels: bool = False) -> dict:
    """Out-of-fold AUROC by stratified K-fold; the scaler is refit inside each fold.

    Pilot-sized sanity check only. With ~40 rows and few positives the interval is wide, and
    the same folds are reused for every C so the C choice is itself optimistic.
    """
    y = y.copy()
    rng = np.random.RandomState(seed)
    if shuffle_labels:
        rng.shuffle(y)
    n_pos = int(y.sum())
    k = min(n_splits, n_pos, int((y == 0).sum()))
    if k < 2:
        return {"auroc": None, "reason": f"too few of one class for CV (pos={n_pos}, neg={int((y == 0).sum())})"}
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        p = fit_probe(X[tr], y[tr], C, block=-1)
        oof[te] = p.scores(X[te])
    return {"auroc": safe_auroc(y, oof), "k": k, "n_pos": n_pos}


def simple_features(meta: list[dict], prompt_token_counts: dict[str, int] | None = None) -> np.ndarray:
    """Subject one-hot + planted letter one-hot + answer letter one-hot (+ prompt length if given)."""
    subjects = sorted({m["subject"] for m in meta})
    rows = []
    for m in meta:
        v = [1.0 if m["subject"] == s else 0.0 for s in subjects]
        v += [1.0 if m["planted_letter"] == L else 0.0 for L in LETTERS]
        v += [1.0 if m["answer_letter"] == L else 0.0 for L in LETTERS]
        if prompt_token_counts is not None:
            v.append(float(prompt_token_counts.get(m["question_id"], 0)))
        rows.append(v)
    return np.asarray(rows)


def planted_onehot(meta: list[dict]) -> np.ndarray:
    return np.asarray([[1.0 if m["planted_letter"] == L else 0.0 for L in LETTERS] for m in meta])
