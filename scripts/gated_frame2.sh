#!/usr/bin/env bash
# Wait until the GPU has >= 24 GB free, then run framing set 2 on rlvr and sft.
cd "$(dirname "$0")/.."
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge 24000 ] && break
  sleep 60
done
echo "started with free=$free MiB at $(date -u)" >> artifacts/runs/inj_frame2_launch.log
for st in rlvr sft; do
  .venv/bin/python scripts/run_injection.py --run-id inj_frame2 --stage $st --framings2 --n-injected 30 --n-user-control 0 --n-clean 4 \
      --batch-size 6 --max-new-tokens 2000 --max-items-per-process 100000 >> artifacts/runs/inj_frame2_$st.log 2>&1
done
echo done >> artifacts/runs/inj_frame2_chain.log
