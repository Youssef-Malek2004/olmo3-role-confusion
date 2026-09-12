#!/usr/bin/env python
"""Own-voice direction: the same passage placed inside the assistant's think block vs inside a tool turn.

Usage: python scripts/capture_voice.py --stage sft [--n 60]

Passages: OLMo-style reasoning openers ("Okay, so ...") harvested from real think blocks in inj_gpu, plus neutral
prose. Each is rendered (a) as the start of the assistant's <think> (prefix continuation) and (b) as environment
content after a tool call. Saves per-passage mean activations at blocks 8/16/24 for both placements, fits the
difference-of-means "assistant-voice" direction, and scores the styled vs destyled CoT-forgery payloads and the
other framing texts on it. Prefill-only; a few minutes on the Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import numpy as np  # noqa: E402

from role_confusion.generation import find_span, load_model, prefill_hidden_states, span_token_mask  # noqa: E402
from role_confusion.injection import ALL_TYPES, goal_text  # noqa: E402
from role_confusion.io import iter_jsonl, read_json, write_json  # noqa: E402
from role_confusion.prompts import INSTRUCTION, TOOL_FUNCTIONS, TOOL_NAME, question_block  # noqa: E402
from role_confusion.questions import build_question  # noqa: E402

BLOCKS = [8, 16, 24]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sft")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()
    cfg = read_json(REPO_ROOT / "configs/pilot.json")
    revs = read_json(REPO_ROOT / "configs/revisions.json")
    repo = next(m for m in cfg["models"] if m["stage"] == args.stage)["repo_id"]
    sha = revs["models"][repo]["sha"]
    SP = cfg["generation"]["system_prompt"]

    # harvest real reasoning openers (first ~60 words of think blocks) from the main run; neutral prose from visible answers
    think_snips, prose_snips = [], []
    for r in iter_jsonl(REPO_ROOT / "artifacts/runs/inj_gpu" / args.stage / "gen_none.jsonl"):
        th, sep, vis = r["generated_text"].rpartition("</think>")
        if sep and len(th.split()) > 80:
            think_snips.append(" ".join(th.split()[:60]))
        if sep and len(vis.split()) > 40:
            prose_snips.append(" ".join(vis.split()[:60]))
    random.seed(0)
    think_snips = random.sample(think_snips, min(args.n, len(think_snips)))
    prose_snips = random.sample(prose_snips, min(args.n, len(prose_snips)))
    passages = [("think", s) for s in think_snips] + [("prose", s) for s in prose_snips]

    rows = {r["question_id"]: r for r in iter_jsonl(REPO_ROOT / "data/processed/questions.jsonl") if r["split"] == "pilot"}
    r0 = rows[sorted(rows)[0]]
    q = build_question(r0["subject"], r0["question"], r0["choices"], r0["answer_index"])
    call = json.dumps([{"name": TOOL_NAME, "arguments": {"question_id": q.question_id}}])
    lm = load_model(repo, sha)
    tok = lm.tokenizer

    def render_as_think(text):
        msgs = [{"role": "system", "content": SP}, {"role": "user", "content": question_block(q) + "\n\n" + INSTRUCTION}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + text

    def render_as_tool(text):
        msgs = [{"role": "system", "content": SP, "functions": TOOL_FUNCTIONS},
                {"role": "user", "content": question_block(q) + "\n\n" + INSTRUCTION},
                {"role": "assistant", "content": None, "function_calls": call},
                {"role": "environment", "content": text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    jobs = []
    for kind, text in passages:
        jobs.append((kind, "think_pos", render_as_think(text), text))
        jobs.append((kind, "tool_pos", render_as_tool(text), text))
    # framing payloads to score (placed in the tool turn, as in the experiments)
    payload_types = [t for t in ALL_TYPES if t.startswith("tc")]
    for t in payload_types:
        text = goal_text(t, "C")
        if text.startswith("__"):
            continue
        jobs.append(("payload:" + t, "tool_pos", render_as_tool(text), text))

    feats = {}
    jobs.sort(key=lambda j: len(j[2]))
    for s in range(0, len(jobs), args.batch_size):
        batch = jobs[s:s + args.batch_size]
        hidden, _ = prefill_hidden_states(lm, [j[2] for j in batch], BLOCKS)
        T = hidden.shape[1]
        for i, (kind, pos, prompt, text) in enumerate(batch):
            m = span_token_mask(tok, prompt, *find_span(prompt, text), T)
            feats.setdefault((kind, pos), []).append(hidden[i][m].astype(np.float32).mean(0))  # (L, H)
        print(f"  {s + len(batch)}/{len(jobs)}", flush=True)

    out = {"stage": args.stage, "blocks": BLOCKS, "n_think": len(think_snips), "n_prose": len(prose_snips), "results": {}}
    res_dir = REPO_ROOT / "results" / "generated" / "voice01"
    res_dir.mkdir(parents=True, exist_ok=True)
    for bi, b in enumerate(BLOCKS):
        A = np.stack([v[bi] for v in feats[("think", "think_pos")]])   # model-style text in the think position
        Bt = np.stack([v[bi] for v in feats[("think", "tool_pos")]])   # same text in the tool turn
        P_tool = np.stack([v[bi] for v in feats[("prose", "tool_pos")]])
        # direction 1: placement (think position minus tool position) for identical text
        d_place = (A.mean(0) - Bt.mean(0)); d_place /= np.linalg.norm(d_place) + 1e-8
        # direction 2: style within the tool turn (model-style think text minus neutral prose, both in tool position)
        d_style = (Bt.mean(0) - P_tool.mean(0)); d_style /= np.linalg.norm(d_style) + 1e-8
        np.save(res_dir / f"dir_place_{args.stage}_b{b}.npy", d_place)
        np.save(res_dir / f"dir_style_{args.stage}_b{b}.npy", d_style)
        role_dir = np.load(REPO_ROOT / "results/generated/roles_gpu" / f"dir_{args.stage}_b{b}.npy") if b in (8, 16, 24) else None
        entry = {"cos_style_vs_role_dir": float(d_style @ role_dir) if role_dir is not None else None,
                 "ref_style_think_in_tool": float((Bt @ d_style).mean()), "ref_style_prose_in_tool": float((P_tool @ d_style).mean()),
                 "payloads": {}}
        for (kind, pos), vs in feats.items():
            if kind.startswith("payload:"):
                v = vs[0][bi]
                entry["payloads"][kind[8:]] = {"style_proj": float(v @ d_style), "place_proj": float(v @ d_place),
                                               "role_proj": float(v @ role_dir) if role_dir is not None else None}
        out["results"][str(b)] = entry
        print(f"\nblock {b}: cos(style dir, role dir)={entry['cos_style_vs_role_dir']:.3f}; style score think-text {entry['ref_style_think_in_tool']:.2f} vs prose {entry['ref_style_prose_in_tool']:.2f}")
        for t, v in sorted(entry["payloads"].items(), key=lambda kv: -kv[1]["style_proj"]):
            print(f"  {t:<24} style {v['style_proj']:6.2f}   role {v['role_proj']:6.2f}")
    write_json(res_dir / f"voice_{args.stage}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
