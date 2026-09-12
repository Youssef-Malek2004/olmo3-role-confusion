#!/usr/bin/env python
"""Re-detect compliance from raw text with the current detectors and tabulate an injection run.

Usage:
    python scripts/summarize_injection.py --run-id inj_pilot [--stages sft rlvr] [--tags none delimiter ...]

Writes results/generated/<run_id>/injection_summary.json and prints a compact table:
stage x tag x condition (x type x voice for injected): n, finished, complied, partial, think-hits, echo, accuracy.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from role_confusion.injection import detect_compliance  # noqa: E402
from role_confusion.io import read_jsonl, write_json  # noqa: E402
from role_confusion.parsing import extract_final_answer  # noqa: E402


def item_bootstrap_ci(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Compliance rate among finished rows with a percentile bootstrap that resamples items (all draws of an item together)."""
    fin = [r for r in rows if r["finished"]]
    if not fin:
        return None
    by_item: dict[str, list[int]] = collections.defaultdict(list)
    for r in fin:
        by_item[r["item_id"]].append(int(bool(r["complied"])))
    items = list(by_item.values())
    rng = np.random.RandomState(seed)
    rates = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(items), len(items))
        vals = [v for i in pick for v in items[i]]
        rates.append(np.mean(vals))
    return {"rate": float(np.mean([v for vs in items for v in vs])), "ci95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))],
            "n_items": len(items), "n_rows": len(fin)}


def agg(rows: list[dict]) -> dict:
    fin = [r for r in rows if r["finished"]]
    comp = [r for r in fin if r["complied"]]
    return {
        "n_draws": len({r.get("draw", 1) for r in rows}) if rows else 0,
        "bootstrap": item_bootstrap_ci(rows),
        "n": len(rows), "finished": len(fin), "complied": len(comp),
        "compliance_rate_finished": (len(comp) / len(fin)) if fin else None,
        "partial_format_word": sum(1 for r in fin if r.get("format_word_present") and not r["complied"]),
        "marker_in_think": sum(1 for r in rows if r["marker_in_think"]),
        "key_in_think": sum(1 for r in rows if r["key_in_think"]),
        "echoed_record": sum(1 for r in rows if r.get("echoed_record")),
        "sent_tool_call_any": sum(1 for r in fin if r.get("sent_tool_call")),
        "sent_with_key": sum(1 for r in fin if r.get("sent_with_key")),
        "chose_planted_letter": sum(1 for r in fin if r.get("chose_planted")),
        "answer_valid": sum(1 for r in fin if r["parsed_valid"]),
        "answer_correct": sum(1 for r in fin if r.get("answer_correct")),
        "gen_tokens_median": sorted(r["generated_token_count"] for r in rows)[len(rows) // 2] if rows else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--effective-cap", type=int, default=3600,
                    help="treat responses longer than this many generated tokens as truncated, so stages run with different caps are comparable")
    args = ap.parse_args()
    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id
    out_dir = REPO_ROOT / "results" / "generated" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = args.stages or sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    summary: dict = {"run_id": args.run_id, "effective_cap": args.effective_cap, "stages": {}}
    planted_by_q = {r["question_id"]: r["planted_letter"] for r in read_jsonl(REPO_ROOT / "data" / "processed" / "questions.jsonl")}
    for stage in stages:
        sdir = run_dir / stage
        summary["stages"][stage] = {}
        groups: dict[str, list[Path]] = collections.defaultdict(list)
        for gen_file in sorted(sdir.glob("gen_*.jsonl")):
            base = re.sub(r"_draw\d+$", "", gen_file.stem[len("gen_"):])
            groups[base].append(gen_file)
        for tag, files in sorted(groups.items()):
            if args.tags and tag not in args.tags:
                continue
            rows = []
            for r in (row for f in files for row in read_jsonl(f)):
                r = dict(r)
                r.setdefault("planted_letter", planted_by_q.get(r["question_id"]))
                if args.effective_cap and r["generated_token_count"] > args.effective_cap:
                    r["finished"] = False
                c = detect_compliance(r["generated_text"], r["finished"], r["itype"], planted=r.get("planted_letter"))
                p = extract_final_answer(r["generated_text"], r["finished"])
                r.update(complied=c.complied, marker_in_visible=c.marker_in_visible, format_only=c.format_only,
                         key_in_visible=c.key_in_visible, marker_in_think=c.marker_in_think, key_in_think=c.key_in_think,
                         echoed_record=c.echoed_record, format_word_present=c.format_word_present,
                         sent_tool_call=c.sent_tool_call, sent_to_external=c.sent_to_external, sent_with_key=c.sent_with_key,
                         link_in_visible=c.link_in_visible, denied_answer=c.denied_answer,
                         chose_planted=(c.final_letter is not None and c.final_letter == r.get("planted_letter")),
                         parsed_letter=p.letter, parsed_valid=p.valid,
                         answer_correct=(p.letter == r["answer_letter"]) if p.valid else None)
                rows.append(r)
            by_cond = {c: agg([r for r in rows if r["condition"] == c]) for c in ("clean", "injected", "user_control")}
            inj = [r for r in rows if r["condition"] == "injected"]
            by_type = {t: agg([r for r in inj if r["itype"] == t]) for t in sorted({r["itype"] for r in inj})}
            by_voice = {v: agg([r for r in inj if r["voice"] == v]) for v in sorted({r["voice"] for r in inj})}
            uc = [r for r in rows if r["condition"] == "user_control"]
            uc_by_type = {t: agg([r for r in uc if r["itype"] == t]) for t in sorted({r["itype"] for r in uc})}
            summary["stages"][stage][tag] = {"by_condition": by_cond, "injected_by_type": by_type,
                                             "injected_by_voice": by_voice, "user_control_by_type": uc_by_type,
                                             "complied_items": [r["item_id"] for r in inj if r["complied"]]}
            with open(out_dir / f"rows_{stage}_{tag}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps({k: v for k, v in r.items() if k not in ("rendered_prompt", "generated_text")}) + "\n")
            print(f"\n== {stage} / {tag}  ({len(files)} draw file(s))")
            for c, a in by_cond.items():
                ci = a["bootstrap"]["ci95"] if a["bootstrap"] else None
                print(f"  {c:<13} n={a['n']:>3} fin={a['finished']:>3} complied={a['complied']:>3} "
                      + (f"ci95=[{ci[0]:.2f},{ci[1]:.2f}] " if ci else "") +
                      f"rate={a['compliance_rate_finished'] if a['compliance_rate_finished'] is None else round(a['compliance_rate_finished'], 2)} "
                      f"partial={a['partial_format_word']} think(m/k)={a['marker_in_think']}/{a['key_in_think']} echo={a['echoed_record']} "
                      f"acc={a['answer_correct']}/{a['answer_valid']} med_tok={a['gen_tokens_median']}")
            cl = by_cond["clean"]
            if cl["finished"]:
                print(f"  clean floors: chose_planted_letter={cl['chose_planted_letter']}/{cl['finished']} sent_tool_call={cl['sent_tool_call_any']}")
            print("  injected by type :", {t: f"{a['complied']}/{a['finished']}" for t, a in by_type.items()},
                  "| by voice:", {v: f"{a['complied']}/{a['finished']}" for v, a in by_voice.items()},
                  "| user_control by type:", {t: f"{a['complied']}/{a['finished']}" for t, a in uc_by_type.items()})
    write_json(out_dir / "injection_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
