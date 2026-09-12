"""End-to-end check of scripts/summarize_run.py on synthetic generation rows (no model)."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def row(qid, stage, variant, letter, valid=True, planted="C", answer="A", tokens=100, finished=True):
    return {
        "question_id": qid, "stage": stage, "variant": variant, "split": "pilot", "subject": "s",
        "answer_letter": answer, "planted_letter": planted, "rendered_prompt": "p",
        "prompt_token_count": 10, "generated_text": f"FINAL: {letter}" if valid else "partial reasoning",
        "generated_token_count": tokens, "finished": finished, "parsed_letter": letter if valid else None,
        "parsed_valid": valid, "parsed_reason": "ok" if valid else "truncated", "parsed_n_matches": 1,
        "parsed_distinct_letters": [letter] if valid else [], "parsed_region": "whole",
        "batch_seconds": 8.0, "batch_size": 2, "model_revision": "x", "activation_file": "a.npy",
    }


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_summarize_run_joins_and_counts(tmp_path, monkeypatch):
    run_id = "unit_test_run"
    run_dir = REPO / "artifacts" / "runs" / run_id
    out_dir = REPO / "results" / "generated" / run_id
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        # sft: q1 switch, q2 no change, q3 already agreed, q4 unscorable (hinted truncated)
        write(run_dir / "sft" / "clean.jsonl", [row("q1", "sft", "clean", "B"), row("q2", "sft", "clean", "B"),
                                                row("q3", "sft", "clean", "C"), row("q4", "sft", "clean", "B")])
        write(run_dir / "sft" / "hinted.jsonl", [row("q1", "sft", "hinted", "C"), row("q2", "sft", "hinted", "B"),
                                                 row("q3", "sft", "hinted", "C"),
                                                 row("q4", "sft", "hinted", None, valid=False, finished=False)])
        write(run_dir / "sft" / "neutral.jsonl", [row("q1", "sft", "neutral", "B"), row("q2", "sft", "neutral", "D")])
        # rlvr: q1 resists, others as before; q4 valid now
        write(run_dir / "rlvr" / "clean.jsonl", [row("q1", "rlvr", "clean", "B"), row("q2", "rlvr", "clean", "B"),
                                                 row("q3", "rlvr", "clean", "C"), row("q4", "rlvr", "clean", "A")])
        write(run_dir / "rlvr" / "hinted.jsonl", [row("q1", "rlvr", "hinted", "B"), row("q2", "rlvr", "hinted", "B"),
                                                  row("q3", "rlvr", "hinted", "C"), row("q4", "rlvr", "hinted", "C")])
        proc = subprocess.run([sys.executable, str(REPO / "scripts" / "summarize_run.py"), "--run-id", run_id,
                               "--stages", "sft", "rlvr"], capture_output=True, text=True, cwd=REPO)
        assert proc.returncode == 0, proc.stderr
        s = json.load(open(out_dir / "summary.json"))
        sft = s["stages"]["sft"]
        assert sft["pairs"]["n_total"] == 4 and sft["pairs"]["n_scorable"] == 3
        assert sft["pairs"]["n_positive"] == 1 and sft["positives"] == ["q1"]
        assert sft["pairs"]["category_counts"] == {"switch_to_planted": 1, "no_change": 1,
                                                    "already_agreed": 1, "unscorable": 1}
        assert sft["hinted"]["parse_reasons"] == {"ok": 3, "truncated": 1}
        assert sft["neutral_control"] == {"n_scorable": 2, "n_changed": 1, "n_changed_to_planted": 0}
        rl = s["stages"]["rlvr"]
        assert rl["pairs"]["n_positive"] == 1 and rl["positives"] == ["q4"]
        assert rl["pairs"]["n_correct_to_wrong_switch"] == 1
        cross = s["cross_stage"]
        assert cross["n_common_scorable"] == 3  # q4 unscorable at sft
        assert cross["category_pairs"]["switch_to_planted->no_change"] == 1
        labels = [json.loads(l) for l in open(out_dir / "labels_sft.jsonl")]
        assert {l["question_id"] for l in labels} == {"q1", "q2", "q3", "q4"}
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
