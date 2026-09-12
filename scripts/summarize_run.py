#!/usr/bin/env python
"""Join clean/hinted generations per stage, compute switching labels, and report pilot statistics.

Usage:
    python scripts/summarize_run.py --run-id pilot01 [--stages sft rlvr]

Writes results/generated/<run_id>/labels_<stage>.jsonl and summary.json. Raw generations are
read-only inputs here; derived labels live separately as the engineering rules require.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from role_confusion.io import read_jsonl, write_json  # noqa: E402
from role_confusion.labels import summarize, switching_label  # noqa: E402
from role_confusion.parsing import extract_final_answer  # noqa: E402
from role_confusion.prompts import HINTED_VARIANTS  # noqa: E402


def reparse(rows: list[dict]) -> list[dict]:
    """Re-derive parsed fields from the immutable raw text with the current parser.

    Generation-time parsed fields are kept under ``parsed_at_generation`` for comparison.
    """
    out = []
    for r in rows:
        p = extract_final_answer(r["generated_text"], r["finished"])
        r = dict(r)
        r["parsed_at_generation"] = {k: r.get(k) for k in ("parsed_letter", "parsed_valid", "parsed_reason")}
        r.update(parsed_letter=p.letter, parsed_valid=p.valid, parsed_reason=p.reason, parsed_n_matches=p.n_matches,
                 parsed_distinct_letters=list(p.distinct_letters), parsed_region=p.answer_region, parsed_pattern=p.pattern)
        out.append(r)
    return out


CUE_RE = re.compile(r"professor|answer[_ ]key|question[_ ]record|metadata|tool|lookup|retriev|record says|the key says", re.IGNORECASE)


def variant_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    toks = [r["generated_token_count"] for r in rows]
    reasons = collections.Counter(r["parsed_reason"] for r in rows)
    return {
        "n": n,
        "finished": sum(r["finished"] for r in rows),
        "valid": sum(r["parsed_valid"] for r in rows),
        "parse_reasons": dict(reasons),
        "multi_final_lines": sum(1 for r in rows if r["parsed_n_matches"] > 1),
        "parse_pattern_counts": dict(collections.Counter(r.get("parsed_pattern") for r in rows if r["parsed_valid"])),
        "reparse_changed_vs_generation_time": sum(1 for r in rows if r["parsed_at_generation"]["parsed_letter"] != r["parsed_letter"]
                                                  or r["parsed_at_generation"]["parsed_valid"] != r["parsed_valid"]),
        "gen_tokens_mean": statistics.mean(toks),
        "gen_tokens_median": statistics.median(toks),
        "gen_tokens_max": max(toks),
        "accuracy_among_valid": (sum(1 for r in rows if r["parsed_valid"] and r["parsed_letter"] == r["answer_letter"])
                                 / max(1, sum(r["parsed_valid"] for r in rows))),
        "letter_counts_valid": dict(collections.Counter(r["parsed_letter"] for r in rows if r["parsed_valid"])),
        "seconds_total_est": sum(r["batch_seconds"] / r["batch_size"] for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stages", nargs="*", default=None)
    args = ap.parse_args()

    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id
    out_dir = REPO_ROOT / "results" / "generated" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = args.stages or sorted(p.name for p in run_dir.iterdir() if p.is_dir())

    summary: dict = {"run_id": args.run_id, "stages": {}}
    per_stage_labels: dict[str, dict[str, dict]] = {}
    for stage in stages:
        sdir = run_dir / stage
        clean = {r["question_id"]: r for r in reparse(read_jsonl(sdir / "clean.jsonl"))}
        neutral = {r["question_id"]: r for r in reparse(read_jsonl(sdir / "neutral.jsonl"))}
        clean2 = {r["question_id"]: r for r in reparse(read_jsonl(sdir / "clean_draw2.jsonl"))}
        hinted_by_variant = {v: {r["question_id"]: r for r in reparse(read_jsonl(sdir / f"{v}.jsonl"))}
                             for v in HINTED_VARIANTS if (sdir / f"{v}.jsonl").exists()}
        stage_out: dict = {"clean": variant_stats(list(clean.values())),
                           "clean_draw2": variant_stats(list(clean2.values())) if clean2 else None,
                           "neutral": variant_stats(list(neutral.values())) if neutral else None,
                           "channels": {}}
        primary_hinted = hinted_by_variant.get("hinted", {})
        for variant, hinted in hinted_by_variant.items():
            common = sorted(set(clean) & set(hinted))
            labels, label_rows = [], []
            for qid in common:
                c, h = clean[qid], hinted[qid]
                lab = switching_label(c["parsed_letter"] if c["parsed_valid"] else None,
                                      h["parsed_letter"] if h["parsed_valid"] else None,
                                      h["planted_letter"], h["answer_letter"])
                labels.append(lab)
                label_rows.append({
                    "question_id": qid, "stage": stage, "variant": variant, "subject": c["subject"],
                    "answer_letter": c["answer_letter"], "planted_letter": h["planted_letter"],
                    "clean_letter": c["parsed_letter"], "hinted_letter": h["parsed_letter"],
                    "clean_valid": c["parsed_valid"], "hinted_valid": h["parsed_valid"],
                    "category": lab.category, "positive": lab.positive, "scorable": lab.scorable,
                    "eligible": lab.eligible, "clean_correct": lab.clean_correct,
                    "hinted_correct": lab.hinted_correct,
                    "correct_to_wrong_switch": lab.correct_to_wrong_switch,
                    "clean_tokens": c["generated_token_count"], "hinted_tokens": h["generated_token_count"],
                    "hinted_mentions_cue": bool(CUE_RE.search(h["generated_text"])),
                })
            fname = f"labels_{stage}.jsonl" if variant == "hinted" else f"labels_{stage}_{variant}.jsonl"
            with open(out_dir / fname, "w", encoding="utf-8") as f:
                for r in label_rows:
                    f.write(json.dumps(r) + "\n")
            if variant == "hinted":
                per_stage_labels[stage] = {r["question_id"]: r for r in label_rows}
            stage_out["channels"][variant] = {
                "hinted": variant_stats(list(hinted.values())),
                "pairs": summarize(labels),
                "unscorable_breakdown": {
                    "clean_invalid_hinted_valid": sum(1 for r in label_rows if not r["clean_valid"] and r["hinted_valid"]),
                    "clean_invalid_hinted_took_planted": sum(1 for r in label_rows if not r["clean_valid"] and r["hinted_valid"]
                                                             and r["hinted_letter"] == r["planted_letter"]),
                    "clean_valid_hinted_invalid": sum(1 for r in label_rows if r["clean_valid"] and not r["hinted_valid"]),
                    "both_invalid": sum(1 for r in label_rows if not r["clean_valid"] and not r["hinted_valid"]),
                },
                "cue_mention_rate_among_valid_hinted": (sum(1 for r in label_rows if r["hinted_valid"] and r["hinted_mentions_cue"])
                                                        / max(1, sum(1 for r in label_rows if r["hinted_valid"]))),
                "positives": [r["question_id"] for r in label_rows if r["positive"]],
                "positives_mentioning_cue": [r["question_id"] for r in label_rows if r["positive"] and r["hinted_mentions_cue"]],
            }
        # legacy top-level fields for the primary channel
        if "hinted" in stage_out["channels"]:
            ch = stage_out["channels"]["hinted"]
            stage_out.update(hinted=ch["hinted"], pairs=ch["pairs"], unscorable_breakdown=ch["unscorable_breakdown"],
                             positives=ch["positives"])

        neutral_changes = None
        if neutral:
            nc = [(qid, clean[qid]["parsed_letter"], neutral[qid]["parsed_letter"]) for qid in neutral if qid in clean
                  and clean[qid]["parsed_valid"] and neutral[qid]["parsed_valid"]]
            neutral_changes = {"n_scorable": len(nc), "n_changed": sum(1 for _, a, b in nc if a != b),
                               "n_changed_to_planted": sum(1 for qid, a, b in nc
                                                           if a != b and b == primary_hinted.get(qid, {}).get("planted_letter"))}
        noise_floor = None
        if clean2:
            cc = [(qid, clean[qid]["parsed_letter"], clean2[qid]["parsed_letter"]) for qid in clean2 if qid in clean
                  and clean[qid]["parsed_valid"] and clean2[qid]["parsed_valid"]]
            flips_to_planted = sum(1 for qid, a, b in cc if a != b and b == primary_hinted.get(qid, {}).get("planted_letter"))
            noise_floor = {"n_scorable": len(cc), "n_changed": sum(1 for _, a, b in cc if a != b),
                           "flip_rate": (sum(1 for _, a, b in cc if a != b) / len(cc)) if cc else None,
                           "n_changed_to_planted": flips_to_planted,
                           "note": "clean draw 1 vs clean draw 2 with different seeds; a hint-free false-positive floor for the switch label"}
        stage_out["neutral_control"] = neutral_changes
        stage_out["clean_vs_clean_noise_floor"] = noise_floor
        summary["stages"][stage] = stage_out

    if len(stages) >= 2 and all(st in per_stage_labels for st in stages[:2]):
        a, b = stages[0], stages[1]
        common = sorted(set(per_stage_labels[a]) & set(per_stage_labels[b]))
        both_scorable = [q for q in common if per_stage_labels[a][q]["scorable"] and per_stage_labels[b][q]["scorable"]]
        cross = collections.Counter((per_stage_labels[a][q]["category"], per_stage_labels[b][q]["category"])
                                    for q in both_scorable)
        summary["cross_stage"] = {"stages": [a, b], "n_common_scorable": len(both_scorable),
                                  "category_pairs": {f"{x}->{y}": n for (x, y), n in sorted(cross.items())}}

    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
