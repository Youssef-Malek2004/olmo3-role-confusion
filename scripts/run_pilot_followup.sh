#!/usr/bin/env bash
# Waits for the pilot chain to finish, then re-runs hinted items that hit the 4,096 cap at 8,192 tokens.
# Budget check only: the re-run is a fresh sampled draw (different batch composition -> different seed).
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_ID="${RUN_ID:-pilot01}"; FOLLOW_ID="${FOLLOW_ID:-pilot01_cap8192}"; BS="${BS:-4}"; CAP="${CAP:-8192}"
until grep -q "chain complete" "artifacts/runs/$RUN_ID/chain.log" 2>/dev/null; do sleep 60; done
mkdir -p "artifacts/runs/$FOLLOW_ID"
for stage in sft rlvr; do
  ids=$(.venv/bin/python -c "
import json,sys
rows=[json.loads(l) for l in open('artifacts/runs/$RUN_ID/$stage/hinted.jsonl')]
print(' '.join(r['question_id'] for r in rows if not r['finished']))")
  log="artifacts/runs/$FOLLOW_ID/${stage}_generation.log"
  echo "=== $stage hinted truncated re-run at $CAP start $(date): $(echo $ids | wc -w) items" >> "$log"
  [ -z "$ids" ] && continue
  .venv/bin/python scripts/run_generation.py --run-id "$FOLLOW_ID" --stage "$stage" --split pilot \
      --variants hinted --batch-size "$BS" --max-new-tokens "$CAP" --question-ids $ids >> "$log" 2>&1
  echo "=== $stage done $(date)" >> "$log"
done
echo "=== followup complete $(date)" >> "artifacts/runs/$FOLLOW_ID/chain.log"
