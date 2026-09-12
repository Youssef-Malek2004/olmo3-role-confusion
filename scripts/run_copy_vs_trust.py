#!/usr/bin/env python
"""Copy-versus-trust pilot: can a steered model still use tool content when the USER asks for it?

See docs/COPY_VS_TRUST.md. Each question carries a verification code that exists only in the
tool result, so no condition can be passed from prior knowledge.

Smoke test first (no steering vector needed, ~2 minutes on the Mac):

    python scripts/run_copy_vs_trust.py --run-id ct_smoke --stage sft --n 2 --max-new-tokens 1200

Full no-defense arm, then the steered arms once a direction exists:

    python scripts/run_copy_vs_trust.py --run-id ct01 --stage sft --n 12 --arm none
    python scripts/capture_roles.py --run-id roles_ct --stage sft --batch-size 8   # ~20 min, writes the .npy
    python scripts/fit_role_probe.py --run-id roles_ct --stages sft
    python scripts/run_copy_vs_trust.py --run-id ct01 --stage sft --n 12 --arm steer \
        --steer-vector results/generated/roles_ct/dir_sft_b16.npy --steer-alpha 11.12
    python scripts/run_copy_vs_trust.py --run-id ct01 --stage sft --n 12 --arm random --steer-alpha 11.12

Then read the summary with --report:

    python scripts/run_copy_vs_trust.py --run-id ct01 --stage sft --report

Writes artifacts/runs/<run_id>/<stage>/gen_<tag>.jsonl (raw text kept immutable) and
results/generated/<run_id>/rows_<stage>_<tag>.jsonl (derived outcomes).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.75")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import numpy as np  # noqa: E402

from role_confusion.copy_trust import (  # noqa: E402
    CONDITIONS, build_items, detect, render_messages, verification_code,
)
from role_confusion.generation import (  # noqa: E402
    Decoding, Steer, batch_seed, find_span, generate_batch, load_model, render_chat, span_token_mask,
    sync_device,
)
from role_confusion.io import append_jsonl, iter_jsonl, read_json, write_json  # noqa: E402
from role_confusion.parsing import extract_final_answer  # noqa: E402
from role_confusion.questions import build_question  # noqa: E402

METRICS = ("code_in_visible", "code_in_visible_raw", "sent_tool_call", "sent_to_requested",
           "sent_to_attacker", "code_in_send", "key_in_send", "retrieved_send_ok", "exfiltrated")


def rescore(run_id: str, stage: str, questions: str) -> None:
    """Rebuild every rows_<stage>_<tag>.jsonl from gen_<tag>.jsonl so detector changes apply to finished arms."""
    qrows = {r["question_id"]: r for r in iter_jsonl(REPO_ROOT / questions)}
    run_dir = REPO_ROOT / "artifacts" / "runs" / run_id / stage
    for gen_path in sorted(run_dir.glob("gen_*.jsonl")):
        tag = gen_path.stem[len("gen_"):]
        arm = "none" if tag == "none" else tag.split("_")[0]
        out = []
        for g in iter_jsonl(gen_path):
            r = qrows[g["question_id"]]
            d = detect(g["generated_text"], g["finished"], g["code"])
            parsed = extract_final_answer(g["generated_text"], g["finished"])
            out.append({"item_id": g["item_id"], "question_id": g["question_id"], "condition": g["condition"],
                        "stage": stage, "tag": tag, "arm": arm, "code": g["code"], "subject": r["subject"],
                        "answer_letter": r["answer_letter"], "generated_token_count": g["generated_token_count"],
                        **d, "parsed_letter": parsed.letter, "parsed_valid": parsed.valid,
                        "answer_correct": (parsed.letter == r["answer_letter"]) if parsed.valid else None,
                        "seed": g["seed"], "model_revision": g["model_revision"]})
        res_path = REPO_ROOT / "results" / "generated" / run_id / f"rows_{stage}_{tag}.jsonl"
        res_path.write_text("".join(json.dumps(o) + "\n" for o in out))
        print(f"rescored {len(out)} rows -> {res_path.relative_to(REPO_ROOT)}")


def report(run_id: str, stage: str) -> int:
    """Print every arm found on disk, condition by condition."""
    res_dir = REPO_ROOT / "results" / "generated" / run_id
    files = sorted(res_dir.glob(f"rows_{stage}_*.jsonl"))
    if not files:
        print(f"no rows under {res_dir} for stage {stage}")
        return 1
    for f in files:
        arm = f.stem.replace(f"rows_{stage}_", "")
        by = collections.defaultdict(list)
        for r in iter_jsonl(f):
            by[r["condition"]].append(r)
        print(f"\n=== {arm} ===")
        for cond in CONDITIONS:
            rs = [r for r in by.get(cond, []) if r["finished"]]
            if not rs:
                continue
            n = len(rs)
            if cond == "retrieve_print":
                hit = sum(r["code_in_visible"] for r in rs)
                extra = f"code in answer {hit}/{n}  (incl. echoes {sum(r['code_in_visible_raw'] for r in rs)}/{n})"
            elif cond == "retrieve_send":
                extra = (f"sent+code {sum(r['retrieved_send_ok'] for r in rs)}/{n}  "
                         f"any send {sum(r['sent_tool_call'] for r in rs)}  "
                         f"right address {sum(r['sent_to_requested'] for r in rs)}  "
                         f"code in call {sum(r['code_in_send'] for r in rs)}")
            elif cond == "attack":
                extra = (f"to attacker {sum(r['sent_to_attacker'] for r in rs)}/{n}  "
                         f"with key {sum(r['exfiltrated'] for r in rs)}  "
                         f"any send {sum(r['sent_tool_call'] for r in rs)}")
            else:
                extra = (f"spontaneous sends {sum(r['sent_tool_call'] for r in rs)}/{n}  "
                         f"code volunteered {sum(r['code_in_visible'] for r in rs)}")
            valid = sum(1 for r in rs if r["parsed_valid"])
            read = sum(r.get("code_in_reasoning", False) for r in rs)
            print(f"  {cond:15s} completed {n:3d}  {extra}   code read in reasoning {read}/{n}   valid answers {valid}")
    print("\nReading: retrieve_print holding while retrieve_send collapses means steering breaks "
          "tool-content copying into call arguments specifically.\nBoth holding means the defense is "
          "selective. Both collapsing means it suppresses tool content in general.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config", default="configs/pilot.json")
    ap.add_argument("--revisions", default="configs/revisions.json")
    ap.add_argument("--questions", default="data/processed/questions.jsonl")
    ap.add_argument("--splits", nargs="*", default=["validation"],
                    help="validation is unused by the framing matrices, so there is no reuse concern")
    ap.add_argument("--n", type=int, default=12, help="questions per condition (all conditions share them)")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--arm", choices=["none", "steer", "random"], default="none")
    ap.add_argument("--steer-vector", default=None)
    ap.add_argument("--steer-block", type=int, default=16)
    ap.add_argument("--steer-alpha", type=float, default=0.0)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=2000)
    ap.add_argument("--report", action="store_true", help="summarise existing rows and exit")
    ap.add_argument("--rescore", action="store_true",
                    help="re-derive results/.../rows_* from the immutable artifacts/.../gen_* files, then report")
    args = ap.parse_args()

    if args.rescore:
        rescore(args.run_id, args.stage, args.questions)
    if args.report or args.rescore:
        return report(args.run_id, args.stage)

    cfg = read_json(REPO_ROOT / args.config)
    revs = read_json(REPO_ROOT / args.revisions)
    model_cfg = next(m for m in cfg["models"] if m["stage"] == args.stage)
    repo_id = model_cfg["repo_id"]
    sha = revs["models"][repo_id]["sha"]
    system_prompt = cfg["generation"]["system_prompt"]
    gen_cfg = cfg["generation"]
    seed = cfg["seed"]

    rows = {r["question_id"]: r for r in iter_jsonl(REPO_ROOT / args.questions) if r["split"] in args.splits}
    if not rows:
        print(f"no questions in splits {args.splits}", file=sys.stderr)
        return 2
    items = build_items(sorted(rows), seed, args.n, tuple(args.conditions))

    tag = args.arm if args.arm == "none" else (
        f"steer_b{args.steer_block}_a{args.steer_alpha:g}" if args.arm == "steer"
        else f"random_b{args.steer_block}_a{args.steer_alpha:g}_s{args.random_seed}")
    run_dir = REPO_ROOT / "artifacts" / "runs" / args.run_id / args.stage
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "items.json", [it.to_dict() for it in items])
    gen_path = run_dir / f"gen_{tag}.jsonl"
    res_path = REPO_ROOT / "results" / "generated" / args.run_id / f"rows_{args.stage}_{tag}.jsonl"
    res_path.parent.mkdir(parents=True, exist_ok=True)

    done = {r["item_id"] for r in iter_jsonl(gen_path)}
    todo = [it for it in items if it.item_id not in done]
    print(f"[{args.stage}/{tag}] {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return report(args.run_id, args.stage)

    lm = load_model(repo_id, sha)
    print(f"loaded {repo_id}@{sha[:8]} on {lm.device}", flush=True)

    steer_vec = None
    if args.arm == "steer":
        if not args.steer_vector:
            print("--arm steer needs --steer-vector (run capture_roles.py + fit_role_probe.py first; "
                  "no dir_*.npy is in this checkout)", file=sys.stderr)
            return 2
        steer_vec = np.load(REPO_ROOT / args.steer_vector).astype(np.float32)
        steer_vec /= np.linalg.norm(steer_vec) + 1e-8
    elif args.arm == "random":
        rng = np.random.RandomState(args.random_seed)
        steer_vec = rng.randn(lm.hidden_size).astype(np.float32)
        steer_vec /= np.linalg.norm(steer_vec)

    write_json(run_dir / f"manifest_{tag}.json", {
        "run_id": args.run_id, "stage": args.stage, "repo_id": repo_id, "revision": sha, "arm": args.arm,
        "tag": tag, "steer_block": args.steer_block if steer_vec is not None else None,
        "steer_alpha": args.steer_alpha if steer_vec is not None else None,
        "steer_vector": args.steer_vector, "max_new_tokens": args.max_new_tokens,
        "n_per_condition": args.n, "conditions": list(args.conditions), "splits": args.splits,
        "seed": seed, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    prepared = []
    for it in todo:
        r = rows[it.question_id]
        q = build_question(r["subject"], r["question"], r["choices"], r["answer_index"])
        code = verification_code(it.question_id, seed)
        msgs = render_messages(q, it.condition, code)
        msgs[0]["content"] = system_prompt
        prompt = render_chat(lm, msgs)
        n_tok = len(lm.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prepared.append((n_tok, it, r, code, prompt, msgs[3]["content"]))
    prepared.sort(key=lambda x: (x[0], x[1].item_id))

    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start:start + args.batch_size]
        prompts = [b[4] for b in batch]
        enc = lm.tokenizer(prompts, padding=True, add_special_tokens=False)
        T = len(enc["input_ids"][0])
        env_masks = np.stack([span_token_mask(lm.tokenizer, p, *find_span(p, env), T)
                              for _, _, _, _, p, env in batch])
        steer = Steer(args.steer_block, steer_vec, args.steer_alpha, env_masks) if steer_vec is not None else None
        sd = batch_seed(seed, f"ct:{tag}", 1, [b[1].item_id for b in batch])
        dec = Decoding("sample", float(gen_cfg["temperature"]), float(gen_cfg["top_p"]), sd)
        t0 = time.time()
        results = generate_batch(lm, prompts, args.max_new_tokens, [args.steer_block],
                                 decoding=dec, steer=steer, span_masks={"env": env_masks})
        sync_device(lm.device)
        elapsed = time.time() - t0

        gen_rows, out_rows = [], []
        for (_, it, r, code, prompt, _), res in zip(batch, results):
            d = detect(res.generated_text, res.finished, code)
            parsed = extract_final_answer(res.generated_text, res.finished)
            gen_rows.append({**it.to_dict(), "stage": args.stage, "tag": tag, "code": code,
                             "rendered_prompt": prompt, "generated_text": res.generated_text,
                             "generated_token_count": res.generated_token_count, "finished": res.finished,
                             "seed": sd, "model_revision": sha})
            out_rows.append({**it.to_dict(), "stage": args.stage, "tag": tag, "arm": args.arm, "code": code,
                             "subject": r["subject"], "answer_letter": r["answer_letter"],
                             "generated_token_count": res.generated_token_count, **d,
                             "parsed_letter": parsed.letter, "parsed_valid": parsed.valid,
                             "answer_correct": (parsed.letter == r["answer_letter"]) if parsed.valid else None,
                             "seed": sd, "model_revision": sha})
        append_jsonl(gen_path, gen_rows)
        append_jsonl(res_path, out_rows)
        fin = sum(x.finished for x in results)
        print(f"  batch {start // args.batch_size + 1}: {len(batch)} items, {elapsed:.0f}s, finished {fin}/{len(batch)}, "
              f"done {start + len(batch)}/{len(prepared)}", flush=True)

    print()
    return report(args.run_id, args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
