#!/usr/bin/env bash
# When >= 20 GB is free, score injected spans on the role direction for every stage of inj_matrix that has gen files.
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
until grep -q 'matrix complete' artifacts/runs/vllm_matrix.log 2>/dev/null; do sleep 120; done
while true; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 20000 ] && break; sleep 60; done
for stage in sft rlvr dpo; do
  [ -f artifacts/runs/inj_matrix/$stage/gen_none.jsonl ] || continue
  .venv/bin/python scripts/score_spans.py --run-id inj_matrix --stage $stage --roles-run roles_gpu --block 16 --batch-size 16 >> artifacts/runs/score_spans_$stage.log 2>&1
  .venv/bin/python scripts/score_spans.py --run-id inj_matrix --stage $stage --roles-run roles_gpu --block 8 --batch-size 16 >> artifacts/runs/score_spans_$stage.log 2>&1
done
echo done >> artifacts/runs/score_spans_chain.log
