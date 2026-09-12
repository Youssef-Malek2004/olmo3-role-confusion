import json

import numpy as np
import pytest

from role_confusion.probe import (
    best_threshold, cv_auroc, evaluate, fit_probe, load_stage, planted_onehot, safe_auroc, simple_features,
)


def test_safe_auroc_one_class_is_none():
    assert safe_auroc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None
    assert safe_auroc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0


def test_probe_learns_planted_direction_and_cv_detects_it():
    rng = np.random.RandomState(0)
    n, d = 80, 32
    direction = rng.randn(d)
    X = rng.randn(n, d)
    y = (X @ direction + 0.3 * rng.randn(n) > 0).astype(int)
    p = fit_probe(X[:60], y[:60], C=1.0, block=8)
    p.threshold = best_threshold(p.scores(X[:60]), y[:60])
    ev = evaluate(p, X[60:], y[60:])
    assert ev["auroc"] is not None and ev["auroc"] > 0.85
    assert cv_auroc(X, y, C=1.0)["auroc"] > 0.85
    shuffled = cv_auroc(X, y, C=1.0, shuffle_labels=True, seed=1)["auroc"]
    assert shuffled < 0.75  # arbitrary labels should not look impressive


def test_cv_refuses_when_too_few_positives():
    X = np.random.RandomState(0).randn(20, 4)
    y = np.zeros(20, dtype=int)
    y[0] = 1
    assert cv_auroc(X, y, C=1.0)["auroc"] is None


def test_simple_features_shapes():
    meta = [{"question_id": "q1", "subject": "a", "planted_letter": "B", "answer_letter": "A"},
            {"question_id": "q2", "subject": "b", "planted_letter": "C", "answer_letter": "D"}]
    f = simple_features(meta, {"q1": 100, "q2": 120})
    assert f.shape == (2, 2 + 4 + 4 + 1)
    assert planted_onehot(meta).tolist() == [[0, 1, 0, 0], [0, 0, 1, 0]]


def test_load_stage_alignment_and_shape_checks(tmp_path):
    run = tmp_path / "sft"
    (run / "acts" / "hinted").mkdir(parents=True)
    json.dump({"blocks_one_based": [8, 16], "hidden_size": 4}, open(run / "manifest.json", "w"))
    labels = tmp_path / "labels_sft.jsonl"
    rows = [{"question_id": "q1", "scorable": True, "positive": True, "subject": "s", "planted_letter": "B", "answer_letter": "A"},
            {"question_id": "q2", "scorable": True, "positive": False, "subject": "s", "planted_letter": "C", "answer_letter": "A"},
            {"question_id": "q3", "scorable": False, "positive": None, "subject": "s", "planted_letter": "C", "answer_letter": "A"}]
    labels.write_text("\n".join(json.dumps(r) for r in rows))
    np.save(run / "acts" / "hinted" / "q1.npy", np.ones((2, 4)))
    np.save(run / "acts" / "hinted" / "q2.npy", np.zeros((2, 4)))
    ds = load_stage(run, labels, "hinted")
    assert ds.question_ids == ["q1", "q2"] and ds.y.tolist() == [1, 0] and ds.X.shape == (2, 2, 4)
    assert ds.block(16).tolist() == [[1, 1, 1, 1], [0, 0, 0, 0]]
    np.save(run / "acts" / "hinted" / "q2.npy", np.zeros((3, 4)))
    with pytest.raises(ValueError):
        load_stage(run, labels, "hinted")
    (run / "acts" / "hinted" / "q1.npy").unlink()
    with pytest.raises(FileNotFoundError):
        load_stage(run, labels, "hinted")
