#!/usr/bin/env bash
# Unattended pilot chain: one checkpoint at a time, clean+hinted on all pilot questions, neutral on the fixed subset.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_ID="${RUN_ID:-pilot01}"; BS="${BS:-6}"; CAP="${CAP:-4096}"
NEUTRAL_IDS=$(.venv/bin/python -c 'import json;print(" ".join(json.load(open("data/processed/neutral_control_ids.json"))["question_ids"]))')
mkdir -p "artifacts/runs/$RUN_ID"
for stage in sft rlvr; do
  log="artifacts/runs/$RUN_ID/${stage}_generation.log"
  echo "=== $stage clean+hinted start $(date)" >> "$log"
  .venv/bin/python scripts/run_generation.py --run-id "$RUN_ID" --stage "$stage" --split pilot \
      --variants clean hinted --batch-size "$BS" --max-new-tokens "$CAP" >> "$log" 2>&1
  echo "=== $stage clean draw 2 start $(date)" >> "$log"
  .venv/bin/python scripts/run_generation.py --run-id "$RUN_ID" --stage "$stage" --split pilot \
      --variants clean --draw 2 --batch-size "$BS" --max-new-tokens "$CAP" >> "$log" 2>&1
  echo "=== $stage neutral start $(date)" >> "$log"
  .venv/bin/python scripts/run_generation.py --run-id "$RUN_ID" --stage "$stage" --split pilot \
      --variants neutral --batch-size "$BS" --max-new-tokens "$CAP" --question-ids $NEUTRAL_IDS >> "$log" 2>&1
  echo "=== $stage done $(date)" >> "$log"
done
.venv/bin/python scripts/summarize_run.py --run-id "$RUN_ID" > "artifacts/runs/$RUN_ID/summary_stdout.log" 2>&1
echo "=== chain complete $(date)" >> "artifacts/runs/$RUN_ID/chain.log"
