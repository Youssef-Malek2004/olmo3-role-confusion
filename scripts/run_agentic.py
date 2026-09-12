#!/usr/bin/env python
"""Agentic injection runner: the model calls the fetch/read tool itself; the payload arrives in the tool result.

Usage:
    python scripts/run_agentic.py --engine hf   --stage sft --run-id agentic_pilot --context webpage --n-injected 6 --n-clean 2 --batch-size 4 --max-new-tokens 2000
    python scripts/run_agentic.py --engine vllm --stage rlvr --run-id agentic01 --context email --n-injected 150 --n-clean 12 --draws 2

Rows (artifacts/runs/<run_id>/<stage>/gen_<context>[_drawN].jsonl) carry all rounds: r1_text/r1_calls/called_fetch,
r2_text/r2_calls, r3_text; `complied` = round-2 send_summary to the attacker's address (same detector as before);
`finished` = round 2 finished (only items that fetched have a round 2). Restartable per item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from role_confusion.agentic import (  # noqa: E402
    AGENTIC_TYPES, CONTEXTS, FINAL_TOOL, INSTRUCT_SYSTEM_PROMPT, REASON_TOOL, attacker_in_calls, called, final_letter_from_calls,
    format_calls, parse_calls, round1_messages, round2_messages, round3_messages,
)
from role_confusion.injection import SEND_TOOL, build_items, detect_compliance  # noqa: E402
from role_confusion.io import append_jsonl, iter_jsonl, read_json, write_json  # noqa: E402
from role_confusion.parsing import extract_final_answer  # noqa: E402
from role_confusion.questions import build_question  # noqa: E402


def item_seed(base: int, item_id: str, draw: int, rnd: int) -> int:
    return int(hashlib.sha256(f"{base}|{item_id}|{draw}|r{rnd}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


class Engine:
    """generate(prompts, seeds) -> list of (text, n_tokens, finished)."""

    def __init__(self, engine: str, repo: str, sha: str, gen: dict, max_new_tokens: int, batch_size: int, gpu_mem: float, base_seed: int):
        self.kind, self.gen, self.max_new, self.bs = engine, gen, max_new_tokens, batch_size
        if engine == "hf":
            from role_confusion.generation import load_model
            self.lm = load_model(repo, sha)
            self.tok = self.lm.tokenizer
        else:
            from transformers import AutoTokenizer
            from vllm import LLM
            self.tok = AutoTokenizer.from_pretrained(repo, revision=sha)
            self.llm = LLM(model=repo, revision=sha, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
                           max_model_len=max_new_tokens + 2500, enable_prefix_caching=True, seed=base_seed)

    no_think = False
    force_think = False
    force_reason = False

    def render(self, msgs: list[dict]) -> str:
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if self.no_think and p.endswith("<think>"):
            p += "</think>\n\n"
        if self.force_think and p.endswith("<|im_start|>assistant\n"):
            p += "<think>\n"
        if self.force_reason and p.endswith("<|im_start|>assistant\n"):
            p += '<function_calls>reason(reasoning="'
        return p

    def generate(self, prompts: list[str], seeds: list[int]) -> list[tuple[str, int, bool]]:
        if not prompts:
            return []
        if self.kind == "vllm":
            from vllm import SamplingParams
            params = [SamplingParams(temperature=float(self.gen["temperature"]), top_p=float(self.gen["top_p"]), max_tokens=self.max_new,
                                     seed=s, skip_special_tokens=False,
                                     stop_token_ids=[self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id]) for s in seeds]
            ids = [{"prompt_token_ids": self.tok(p, add_special_tokens=False)["input_ids"]} for p in prompts]
            outs = self.llm.generate(ids, params)
            return [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason == "stop") for o in outs]
        from role_confusion.generation import Decoding, generate_batch
        res = []
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            dec = Decoding(mode="sample", temperature=float(self.gen["temperature"]), top_p=float(self.gen["top_p"]), seed=seeds[idx[0]])
            rs = generate_batch(self.lm, [prompts[i] for i in idx], self.max_new, [16], static_cache=True, decoding=dec)
            for i, r in zip(idx, rs):
                out[i] = (r.generated_text, r.generated_token_count, r.finished)
            print(f"    hf batch {s // self.bs + 1}: {len(idx)} prompts", flush=True)
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        return out


def resolve_reason(eng, msgs_by_item, outs_by_item, items, seeds, call_format, max_steps=3):
    """For items whose latest output is a reason(...) call, append the call + a 'recorded' result and regenerate.
    Returns updated (msgs_by_item, outs_by_item, reasoning_by_item)."""
    reasoning = {it.item_id: [] for it in items}
    for step in range(max_steps):
        pend = [it for it in items if called(parse_calls(outs_by_item[it.item_id][0]), REASON_TOOL)]
        if not pend:
            break
        prompts = []
        for it in pend:
            calls = parse_calls(outs_by_item[it.item_id][0])
            reasoning[it.item_id].append(" ".join(str(c.get("arguments", {}).get("reasoning", "")) for c in calls if c.get("name") == REASON_TOOL))
            msgs_by_item[it.item_id] = msgs_by_item[it.item_id] + [
                {"role": "assistant", "content": None, "function_calls": format_calls(calls, call_format)},
                {"role": "environment", "content": json.dumps([{"name": REASON_TOOL, "result": "Reasoning recorded."}])}]
            prompts.append(eng.render(msgs_by_item[it.item_id]))
        outs = eng.generate(prompts, [seeds[it.item_id] + 1000 * (step + 1) for it in pend])
        for it, o in zip(pend, outs):
            outs_by_item[it.item_id] = o
    return msgs_by_item, outs_by_item, reasoning


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--context", choices=list(CONTEXTS), default="webpage")
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--n-injected", type=int, default=6)
    ap.add_argument("--n-clean", type=int, default=2)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ask", choices=["none", "soft", "strong"], default="strong", help="how firmly the user asks for the fetch; none = question only (scripted runs)")
    ap.add_argument("--system-hint", action="store_true", help="add a tool-use sentence to the system prompt")
    ap.add_argument("--rounds", type=int, default=3, help="1 = round-1 fetch-rate probe only")
    ap.add_argument("--tag", default=None, help="output file tag (default: context)")
    ap.add_argument("--call-format", choices=["auto", "json", "native"], default="auto", help="scripted call syntax; auto = native for the instruct stage")
    ap.add_argument("--scripted", action="store_true", help="skip round 1: insert a canonical fetch/read call ourselves (Think-line comparison)")
    ap.add_argument("--payload-nouns", choices=["context", "record"], default="context", help="record = keep the original record wording in any context")
    ap.add_argument("--reason", choices=["none", "think", "think_example", "tool"], default="none",
                    help="Instruct: think = ask for a <think> block; tool = provide a reason(reasoning) function that must be called first")
    ap.add_argument("--no-think", action="store_true", help="Think: pre-fill an empty <think></think> so the model answers without reasoning")
    ap.add_argument("--force-think", action="store_true", help="Instruct: pre-fill an opening <think> tag so the model must reason before acting")
    ap.add_argument("--final-tool", action="store_true", help="Instruct: provide record_final(letter) for the final answer")
    args = ap.parse_args()

    cfg = read_json(REPO_ROOT / "configs/pilot.json")
    revs = read_json(REPO_ROOT / "configs/revisions.json")
    repo = next(m for m in cfg["models"] if m["stage"] == args.stage)["repo_id"]
    sha = revs["models"][repo]["sha"]
    system_prompt = INSTRUCT_SYSTEM_PROMPT if args.stage.startswith("instruct") else cfg["generation"]["system_prompt"]
    call_format = ("native" if args.stage.startswith("instruct") else "json") if args.call_format == "auto" else args.call_format
    gen = cfg["generation"]
    types = tuple(args.types) if args.types else AGENTIC_TYPES
    ctx = CONTEXTS[args.context]

    rows = {r["question_id"]: r for r in iter_jsonl(REPO_ROOT / "data/processed/questions.jsonl") if r["split"] in ("pilot", "train")}
    items = build_items(sorted(rows), cfg["seed"], args.n_injected, 0, args.n_clean, types=types, voices=("plain",))
    if args.limit:
        items = items[:args.limit]
    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id / args.stage
    run_dir.mkdir(parents=True, exist_ok=True)
    tagname = args.tag or args.context
    write_json(run_dir / f"items_{tagname}.json", [it.to_dict() for it in items])
    eng = Engine(args.engine, repo, sha, gen, args.max_new_tokens, args.batch_size, args.gpu_mem, cfg["seed"])
    eng.no_think = args.no_think
    eng.force_think = args.force_think
    write_json(run_dir / f"manifest_{tagname}.json", {
        "run_id": args.run_id, "stage": args.stage, "repo_id": repo, "revision": sha, "engine": args.engine, "context": args.context,
        "ask": args.ask, "system_hint": args.system_hint, "rounds": args.rounds, "call_format": call_format, "scripted": args.scripted, "payload_nouns": args.payload_nouns,
        "reason": args.reason, "no_think": args.no_think, "force_think": args.force_think, "final_tool": args.final_tool,
        "types": list(types), "draws": args.draws, "max_new_tokens": args.max_new_tokens, "system_prompt": system_prompt,
        "decoding": {"mode": "sample", "temperature": gen["temperature"], "top_p": gen["top_p"], "seed_rule": "sha256(base|item|draw|round)"},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    for draw in range(1, args.draws + 1):
        out_path = run_dir / f"gen_{tagname}{'' if draw == 1 else f'_draw{draw}'}.jsonl"
        done = {r["item_id"] for r in iter_jsonl(out_path)}
        todo = [it for it in items if it.item_id not in done]
        print(f"[{args.stage}/{args.context}/draw{draw}] {len(done)} done, {len(todo)} to go", flush=True)
        if not todo:
            continue
        t0 = time.time()
        # round 1
        qs = {}
        for it in todo:
            r = rows[it.question_id]
            qs[it.item_id] = build_question(r["subject"], r["question"], r["choices"], r["answer_index"])
        m1 = {it.item_id: round1_messages(qs[it.item_id], args.context, system_prompt, args.ask, args.system_hint, args.reason, args.final_tool) for it in todo}
        p1 = [eng.render(m1[it.item_id]) for it in todo]
        if args.scripted:
            from role_confusion.agentic import page_url
            def canon(q):
                if args.context in ("record", "webpage_lookup"):
                    return [{"name": ctx["tool"], "arguments": {"question_id": q.question_id}}]
                return [{"name": ctx["tool"], "arguments": ({"url": page_url(q)} if args.context.startswith("webpage") else {"query": f"question {q.question_id.rsplit('-', 1)[-1]}"})}]
            o1 = [("<scripted>", 0, True) for _ in todo]
        else:
            o1 = eng.generate(p1, [item_seed(cfg["seed"], it.item_id, draw, 1) for it in todo])
        reasoning1 = {}
        if args.reason == "tool" and not args.scripted:
            o1d = {it.item_id: o for it, o in zip(todo, o1)}
            m1, o1d, reasoning1 = resolve_reason(eng, m1, o1d, todo, {it.item_id: item_seed(cfg["seed"], it.item_id, draw, 1) for it in todo}, call_format)
            o1 = [o1d[it.item_id] for it in todo]
        rec = {}
        for it, prompt, (text, n, fin) in zip(todo, p1, o1):
            calls = canon(qs[it.item_id]) if args.scripted else parse_calls(text)
            rec[it.item_id] = {**it.to_dict(), "stage": args.stage, "context": args.context, "draw": draw, "engine": args.engine,
                               "subject": rows[it.question_id]["subject"], "answer_letter": rows[it.question_id]["answer_letter"],
                               "planted_letter": rows[it.question_id]["planted_letter"],
                               "r1_prompt": prompt, "r1_text": text, "r1_tokens": n, "r1_finished": fin, "r1_calls": calls,
                               "called_fetch": called(calls, ctx["tool"]), "r1_sent_attacker": attacker_in_calls(calls), "scripted": args.scripted,
                               "r1_reasoning": reasoning1.get(it.item_id, []),
                               "r1_letter": extract_final_answer(text, fin).letter}
        print(f"  round 1: {sum(r['called_fetch'] for r in rec.values())}/{len(rec)} called {ctx['tool']}, "
              f"{sum(1 for r in rec.values() if r['r1_letter'])} answered without fetching", flush=True)
        # round 2
        r2_items = [it for it in todo if rec[it.item_id]["called_fetch"]] if args.rounds >= 2 else []
        m2 = {}
        for it in r2_items:
            base = round2_messages(qs[it.item_id], args.context, system_prompt, rec[it.item_id]["r1_calls"], it.itype,
                                   args.ask, args.system_hint, call_format, args.payload_nouns,
                                   rec[it.item_id]["planted_letter"], args.reason, args.final_tool)
            # keep any reason turns the model produced in round 1 (they sit between the user turn and the fetch call)
            m2[it.item_id] = m1[it.item_id] + base[len(round1_messages(qs[it.item_id], args.context, system_prompt, args.ask, args.system_hint, args.reason, args.final_tool)):]
        eng.force_reason = (args.reason == "tool")
        p2 = [eng.render(m2[it.item_id]) for it in r2_items]
        o2 = eng.generate(p2, [item_seed(cfg["seed"], it.item_id, draw, 2) for it in r2_items])
        eng.force_reason = False
        if args.reason == "tool":  # put the prefilled opener back on the generated text so parse_calls sees the whole call
            o2 = [('<function_calls>reason(reasoning="' + t, n, fin) for (t, n, fin) in o2]
        reasoning2 = {}
        if args.reason == "tool":
            o2d = {it.item_id: o for it, o in zip(r2_items, o2)}
            m2, o2d, reasoning2 = resolve_reason(eng, m2, o2d, r2_items, {it.item_id: item_seed(cfg["seed"], it.item_id, draw, 2) for it in r2_items}, call_format)
            o2 = [o2d[it.item_id] for it in r2_items]
        for it, prompt, (text, n, fin) in zip(r2_items, p2, o2):
            R = rec[it.item_id]
            R["r2_reasoning"] = reasoning2.get(it.item_id, [])
            comp = detect_compliance(text, fin, it.itype, planted=R["planted_letter"])
            calls = parse_calls(text)
            R.update({"r2_prompt": prompt, "r2_text": text, "r2_tokens": n, "r2_finished": fin, "r2_calls": calls,
                      "sent_tool_call": comp.sent_tool_call, "sent_to_external": comp.sent_to_external, "sent_with_key": comp.sent_with_key,
                      "complied": comp.complied, "finished": fin, "generated_text": text, "generated_token_count": n,
                      "r2_letter": final_letter_from_calls(calls) or comp.final_letter})
        # round 3 for items whose round 2 ended in any tool call (typically send_summary)
        # round 3 only for items whose round-2 action was something other than the final answer itself
        r3_items = [it for it in r2_items if rec[it.item_id]["r2_calls"] and rec[it.item_id]["r2_finished"]
                    and not final_letter_from_calls(rec[it.item_id]["r2_calls"])] if args.rounds >= 3 else []
        m3 = {it.item_id: round3_messages(m2[it.item_id], rec[it.item_id]["r2_calls"], call_format) for it in r3_items}
        p3 = [eng.render(m3[it.item_id]) for it in r3_items]
        o3 = eng.generate(p3, [item_seed(cfg["seed"], it.item_id, draw, 3) for it in r3_items])
        reasoning3 = {}
        if args.reason == "tool":
            o3d = {it.item_id: o for it, o in zip(r3_items, o3)}
            m3, o3d, reasoning3 = resolve_reason(eng, m3, o3d, r3_items, {it.item_id: item_seed(cfg["seed"], it.item_id, draw, 3) for it in r3_items}, call_format)
            o3 = [o3d[it.item_id] for it in r3_items]
        for it, prompt, (text, n, fin) in zip(r3_items, p3, o3):
            R = rec[it.item_id]
            c3 = parse_calls(text)
            R.update({"r3_text": text, "r3_tokens": n, "r3_finished": fin, "r3_reasoning": reasoning3.get(it.item_id, []),
                      "r3_letter": final_letter_from_calls(c3) or extract_final_answer(text, fin).letter})
        out_rows = []
        for it in todo:
            R = rec[it.item_id]
            R.setdefault("complied", None); R.setdefault("finished", False); R.setdefault("sent_tool_call", False)
            R.setdefault("sent_to_external", False); R.setdefault("sent_with_key", False)
            letter = R.get("r3_letter") or R.get("r2_letter") or R.get("r1_letter")
            R["parsed_letter"] = letter; R["answer_correct"] = (letter == R["answer_letter"]) if letter else None
            R["seed"] = item_seed(cfg["seed"], it.item_id, draw, 1)
            out_rows.append(R)
        append_jsonl(out_path, out_rows)
        el = time.time() - t0
        inj = [r for r in out_rows if r["condition"] == "injected" and r["finished"]]
        print(f"  draw {draw}: {len(out_rows)} items in {el:.0f}s | fetched {sum(r['called_fetch'] for r in out_rows)}/{len(out_rows)} | "
              f"injected finished {len(inj)}: exfil {sum(1 for r in inj if r['complied'])}, any send {sum(1 for r in inj if r['sent_tool_call'])} | "
              f"clean sends {sum(1 for r in out_rows if r['condition'] == 'clean' and r['sent_tool_call'])} | round-3 items {len(r3_items)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
