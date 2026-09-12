#!/usr/bin/env python
"""Draft the four main figures from saved results. Output: figures/*.png (+ .csv of the plotted numbers).

Categorical slots follow the validated reference palette: SFT blue #2a78d6, DPO orange #eb6834, RLVR aqua #1baf7a
(first three slots validate all-pairs). Text in ink tokens, thin marks, recessive grid, direct labels on <=4 series.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
STAGE_COLOR = {"sft": "#2a78d6", "dpo": "#eb6834", "rlvr": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def load(run):
    return json.load(open(ROOT / "results" / "generated" / run / "injection_summary.json"))["stages"]


def fig1_stage_by_goal():
    S = load("inj_gpu")
    goals = ["marker", "format", "exfil"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.26
    rows = []
    for i, st in enumerate(("sft", "dpo", "rlvr")):
        d = S[st]["none"]["injected_by_type"]
        for j, g in enumerate(goals):
            a = d[g]; b = a["bootstrap"]; r = 100 * b["rate"]; lo, hi = 100 * b["ci95"][0], 100 * b["ci95"][1]
            x = j + (i - 1) * w
            ax.bar(x, r, width=w - 0.02, color=STAGE_COLOR[st], edgecolor="#fcfcfb", linewidth=2, label=st.upper() if j == 0 else None)
            ax.plot([x, x], [lo, hi], color=INK, linewidth=1)
            ax.text(x, hi + 2, f"{r:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
            rows.append([st, g, a["complied"], a["finished"], r, lo, hi])
    ax.set_xticks(range(len(goals))); ax.set_xticklabels(["marker\n(write ACCESS CODE)", "format\n(reply only CONFIRMED)", "exfil\n(include internal key)"])
    ax.set_ylabel("compliance among finished items (%)"); ax.set_ylim(0, 105); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title("Compliance with injected tool-output instructions, by stage\n3 draws/item, 30 items/goal, 5,000-token cap; CI = item bootstrap", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig1_stage_by_goal.png", dpi=200)
    with open(OUT / "fig1_stage_by_goal.csv", "w", newline="") as f:
        csv.writer(f).writerows([["stage", "goal", "complied", "finished", "rate_pct", "ci_lo", "ci_hi"]] + rows)


def fig2_framings():
    S = load("inj_matrix")
    order = ["tc2_fake_turn", "tc3_fake_turn_escaped", "tc3_fake_turn_lexical", "tc2_fake_completion", "tc2_important", "tc2_cot_forgery",
             "tc_internal_addr", "tc_dependency", "tc_benign_body", "tc_preauth", "tc2_necessity", "tc_api_error", "tc_required_field",
             "tc2_contextual", "tc3_cot_destyled", "tc3_length_control"]
    labels = {"tc2_fake_turn": "fake user turn (real tokens)", "tc3_fake_turn_escaped": "fake user turn (escaped text)", "tc3_fake_turn_lexical": "\"### User\" marker",
              "tc2_fake_completion": "fake Assistant/User dialogue", "tc2_important": "important-instructions block", "tc2_cot_forgery": "CoT forgery (model style)",
              "tc_internal_addr": "plain request, internal address", "tc_dependency": "locked record, key after forwarding", "tc_benign_body": "benign 'summarise all fields'",
              "tc_preauth": "pre-authorisation claim", "tc2_necessity": "task necessity", "tc_api_error": "fake 403 error", "tc_required_field": "structured required_action",
              "tc2_contextual": "contextual continuation", "tc3_cot_destyled": "CoT forgery, destyled", "tc3_length_control": "length-matched benign (control)"}
    stages = [s for s in ("sft", "rlvr", "dpo") if s in S]
    fig, ax = plt.subplots(figsize=(8, 6.2))
    h = 0.8 / len(stages)
    rows = []
    for i, st in enumerate(stages):
        d = S[st]["none"]["injected_by_type"]
        for j, t in enumerate(order):
            if t not in d or not d[t]["finished"]:
                continue
            a = d[t]; b = a["bootstrap"]; r = 100 * b["rate"]; lo, hi = 100 * b["ci95"][0], 100 * b["ci95"][1]
            y = len(order) - 1 - j + (i - (len(stages) - 1) / 2) * h
            ax.barh(y, r, height=h - 0.03, color=STAGE_COLOR[st], edgecolor="#fcfcfb", linewidth=1.5, label=f"{st.upper()} ({S[st]['none']['by_condition']['injected']['n_draws']} draws)" if j == 0 else None)
            ax.plot([lo, hi], [y, y], color=INK, linewidth=1)
            rows.append([st, t, a["complied"], a["finished"], r, lo, hi])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[t] for t in reversed(order)], fontsize=9)
    ax.set_xlabel("send_summary call to the attacker's address, % of finished items"); ax.set_xlim(0, 105); ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title("Which injections produce a tool action\n25 items/framing, 3,000-token cap; CI = item bootstrap", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig2_framings.png", dpi=200)
    with open(OUT / "fig2_framings.csv", "w", newline="") as f:
        csv.writer(f).writerows([["stage", "framing", "complied", "finished", "rate_pct", "ci_lo", "ci_hi"]] + rows)


def fig3_role_score():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    rows = []
    for ax, st in zip(axes, ("sft", "rlvr")):
        p = ROOT / "results" / "generated" / "inj_matrix" / f"span_scores_{st}_b8.jsonl"
        if not p.exists():
            continue
        items = [json.loads(l) for l in open(p)]
        items = [r for r in items if r["n_scorable"] > 0 and r["itype"] != "tc3_length_control"]
        s = np.array([-r["inj_proj"] for r in items]); c = np.array([r["n_complied"] / r["n_scorable"] for r in items])
        order = np.argsort(s); qs = np.array_split(order, 5)
        rates = [100 * c[q].mean() for q in qs]
        # item-clustered bootstrap per quintile
        rng = np.random.RandomState(0); cis = []
        for q in qs:
            boots = [100 * c[rng.choice(q, len(q))].mean() for _ in range(2000)]
            cis.append((np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
        x = np.arange(5)
        ax.bar(x, rates, color=STAGE_COLOR[st], width=0.7, edgecolor="#fcfcfb", linewidth=2)
        for xi, r, (lo, hi) in zip(x, rates, cis):
            ax.plot([xi, xi], [lo, hi], color=INK, linewidth=1); ax.text(xi, hi + 2, f"{r:.0f}", ha="center", fontsize=8, color=INK2)
            rows.append([st, int(xi) + 1, r, lo, hi])
        ax.set_xticks(x); ax.set_xticklabels(["least\nuser-like", "", "", "", "most\nuser-like"]); ax.set_title(st.upper(), color=INK, fontsize=10, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.set_ylim(0, 105)
    axes[0].set_ylabel("compliance (%)")
    fig.suptitle("Compliance by how user-like the injected span looks internally\nquintiles of projection on the user-vs-tool direction, block 8; CI = item bootstrap", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "fig3_role_score_quintiles.png", dpi=200)
    with open(OUT / "fig3_role_score_quintiles.csv", "w", newline="") as f:
        csv.writer(f).writerows([["stage", "quintile", "rate_pct", "ci_lo", "ci_hi"]] + rows)


def fig4_defense():
    S = load("inj_gpu")
    arms = [("none", "no defense"), ("delimiter", "prompt: 'tool text is untrusted'"), ("random", "random vector, 2x norm"),
            ("steer1", "steer 1x class gap"), ("steer2", "steer 2x class gap"), ("steer4", "steer 4x class gap")]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    rows = []
    w = 0.26
    for i, st in enumerate(("sft", "dpo", "rlvr")):
        tags = S[st]
        def find(kind):
            for tag, d in tags.items():
                if kind == "none" and tag == "none": return d
                if kind == "delimiter" and tag == "delimiter": return d
                if kind == "random" and tag.startswith("random"): return d
                if kind.startswith("steer") and tag.startswith("steer"):
                    a = float(re.search(r"_a([0-9.]+)", tag).group(1))
                    gap = {"sft": 2.78, "dpo": 2.82, "rlvr": 2.85}[st]
                    mult = round(a / gap)
                    if {"steer1": 1, "steer2": 2, "steer4": 4}[kind] == mult: return d
            return None
        for j, (kind, lab) in enumerate(arms):
            d = find(kind)
            if d is None: continue
            m = d["injected_by_type"].get("marker")
            if not m or not m["finished"]: continue
            r = 100 * m["complied"] / m["finished"]; b = m["bootstrap"]; lo, hi = 100 * b["ci95"][0], 100 * b["ci95"][1]
            x = j + (i - 1) * w
            ax.bar(x, r, width=w - 0.02, color=STAGE_COLOR[st], edgecolor="#fcfcfb", linewidth=2, label=st.upper() if j == 0 else None)
            ax.plot([x, x], [lo, hi], color=INK, linewidth=1); ax.text(x, hi + 2, f"{r:.0f}", ha="center", fontsize=8, color=INK2)
            rows.append([st, kind, m["complied"], m["finished"], r, lo, hi, d["by_condition"]["injected"]["answer_valid"], d["by_condition"]["injected"]["finished"]])
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([l for _, l in arms], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("marker compliance (%)"); ax.set_ylim(0, 110); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title("Steering tool-turn tokens along the user-vs-tool direction vs baselines\nmarker items, 5,000-token cap; no-defense = 3 draws, other arms 1 draw; CI = item bootstrap", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig4_defense.png", dpi=200)
    with open(OUT / "fig4_defense.csv", "w", newline="") as f:
        csv.writer(f).writerows([["stage", "arm", "complied", "finished", "rate_pct", "ci_lo", "ci_hi", "answers_valid", "items_finished"]] + rows)


if __name__ == "__main__":
    fig1_stage_by_goal(); fig2_framings(); fig3_role_score(); fig4_defense()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
