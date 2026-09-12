#!/usr/bin/env python
"""Build the frozen question table: IDs, dedup, stratified splits, and planted wrong letters.

Usage:
    python scripts/prepare_questions.py --config configs/pilot.json --out data/processed/questions.jsonl

Reads MMLU from the local HF cache (pinned revision in configs/revisions.json). Produces one
JSONL row per selected question plus a summary JSON next to it. Nothing here touches a model.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

from role_confusion.io import read_json, write_json  # noqa: E402
from role_confusion.questions import deduplicate, load_mmlu, prepare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--revisions", default="configs/revisions.json")
    ap.add_argument("--out", default="data/processed/questions.jsonl")
    args = ap.parse_args()

    cfg = read_json(REPO_ROOT / args.config)
    revs = read_json(REPO_ROOT / args.revisions)
    ds = cfg["dataset"]
    subjects = ds["subjects"]
    per_subject = ds["per_subject_counts"]
    seed = cfg["seed"]

    all_q = load_mmlu(ds["configuration"], ds["source_split"], revision=revs["dataset"]["sha"])
    kept, dropped = deduplicate(all_q)
    prepared = prepare(all_q, per_subject, seed, subjects)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        for p in sorted(prepared, key=lambda p: (p.split, p.question.subject, p.question.question_id)):
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    split_counts = collections.Counter(p.split for p in prepared)
    letter_counts = collections.Counter(p.planted_letter for p in prepared)
    answer_counts = collections.Counter(p.question.answer_letter for p in prepared)
    summary = {
        "dataset_revision": revs["dataset"]["sha"],
        "seed": seed,
        "subjects": subjects,
        "per_subject_counts": per_subject,
        "rows_loaded": len(all_q),
        "rows_after_dedup": len(kept),
        "exact_duplicates_dropped": len(dropped),
        "duplicates_in_selected_subjects": sum(1 for d, k in dropped if any(
            f"-{s}-" in d for s in subjects)),
        "selected": len(prepared),
        "split_counts": dict(split_counts),
        "planted_letter_counts": dict(sorted(letter_counts.items())),
        "answer_letter_counts": dict(sorted(answer_counts.items())),
        "mean_question_chars": sum(len(p.question.question) for p in prepared) / max(1, len(prepared)),
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    print(f"wrote {len(prepared)} questions to {out_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
