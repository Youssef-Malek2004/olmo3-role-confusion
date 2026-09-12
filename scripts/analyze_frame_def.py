#!/usr/bin/env python
"""Defense arms vs role-imitation framings (inj_frame_def, inj_frame_def_style, inj_frame_def_sum).

Per stage x arm x framing: exfil to the attacker's address (the compliance metric), any send_summary call (unrequested
action of any kind), sends carrying the internal key, valid/correct answers, finished. Plus paired flips vs the no-defense
arm on the same items. Usage: python scripts/analyze_frame_def.py [--cap 3000] [--md out.md]
"""
from __future__ import annotations
import argparse, collections, glob, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from role_confusion.injection import detect_compliance  # noqa: E402

ARM_LABEL = {"none": "no defense", "delimiter": "prompt warning"}
def arm_label(run, tag):
    if tag in ARM_LABEL: return ARM_LABEL[tag]
    a = float(re.search(r"_a([0-9.]+)", tag).group(1))
    fam = {"inj_frame_def": "role", "inj_frame_def_style": "style-neg", "inj_frame_def_sum": "role+style"}[run]
    if tag.startswith("random"): fam = "random"
    return f"{fam} a={a:g}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=3000); ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = []  # (stage, arm, framing, item_id, fin, exfil, send, key, valid, correct)
    for f in sorted(glob.glob(str(ROOT / "artifacts/runs/inj_frame_def*/*/gen_*.jsonl"))):
        p = Path(f); run, stage = p.parts[-3], p.parts[-2]; tag = p.stem[4:]
        if "_draw" in tag: continue
        arm = arm_label(run, tag)
        for r in map(json.loads, open(f)):
            t = r["itype"].replace("tc2_", "") if r["condition"] == "injected" else "clean"
            fin = r["finished"] and r["generated_token_count"] <= args.cap
            d = detect_compliance(r["generated_text"], fin, r["itype"], planted=r.get("planted_letter")) if fin else None
            rows.append((stage, arm, t, r["item_id"], fin, bool(d and d.complied), bool(d and d.sent_tool_call), bool(d and d.sent_with_key),
                         bool(d and d.final_letter), bool(fin and r.get("answer_correct"))))
    out = []
    for stage in ("rlvr", "sft"):
        arms = sorted({a for s, a, *_ in rows if s == stage}, key=lambda a: (a != "no defense", a))
        if not arms: continue
        out.append(f"\n### {stage.upper()} (cap {args.cap}; cells = exfil-to-attacker / any-send / finished)\n")
        frs = ["fake_turn", "cot_forgery", "fake_completion", "important", "clean"]
        out.append("| arm | " + " | ".join(frs) + " | valid | correct | paired: baseline exfil off / new on |"); out.append("|---" * (len(frs) + 4) + "|")
        base = {(t, i): e for s, a, t, i, fin, e, *_ in rows if s == stage and a == "no defense" and fin}
        for arm in arms:
            cells = []
            for t in frs:
                rs = [x for x in rows if x[0] == stage and x[1] == arm and x[2] == t]
                fin = sum(x[4] for x in rs); ex = sum(x[5] for x in rs); sd = sum(x[6] for x in rs)
                cells.append(f"{ex}/{sd}/{fin}" if rs else "-")
            rs = [x for x in rows if x[0] == stage and x[1] == arm and x[2] != "clean" and x[4]]
            valid = f"{sum(x[8] for x in rs)}/{len(rs)}"; corr = f"{sum(x[9] for x in rs)}/{len(rs)}"
            off = on = 0
            for x in rs:
                b = base.get((x[2], x[3]))
                if b is None: continue
                off += b and not x[5]; on += (not b) and x[5]
            out.append(f"| {arm} | " + " | ".join(cells) + f" | {valid} | {corr} | {off} / {on} |")
        keys = collections.Counter((a, t) for s, a, t, i, fin, e, sd, k, *_ in rows if s == stage and k)
        if keys: out.append(f"\nsends carrying the internal key: " + ", ".join(f"{a} {t} {n}" for (a, t), n in sorted(keys.items())))
    text = "\n".join(out); print(text)
    if args.md: Path(args.md).write_text("# Defense arms vs imitation framings\n" + text + "\n")

if __name__ == "__main__":
    main()
