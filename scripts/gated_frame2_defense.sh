#!/usr/bin/env bash
# After memory frees (>= 24 GB), run defenses on framing-set-2 SFT items: delimiter, steer 2x class gap, random matched norm.
cd "$(dirname "$0")/.."
while true; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 24000 ] && break; sleep 60; done
echo "defense run started free=$free at $(date -u)" >> artifacts/runs/inj_frame2_launch.log
A=$(.venv/bin/python -c "import json;print(round(json.load(open('results/generated/roles_gpu/role_probe.json'))['stages']['sft']['blocks']['16']['diffmean_norm']*2,2))")
COMMON=(--run-id inj_frame2 --stage sft --framings2 --n-injected 30 --n-user-control 0 --n-clean 4 --batch-size 6 --max-new-tokens 2000 --max-items-per-process 100000)
.venv/bin/python scripts/run_injection.py "${COMMON[@]}" --defense delimiter >> artifacts/runs/inj_frame2_sft.log 2>&1
.venv/bin/python scripts/run_injection.py "${COMMON[@]}" --defense steer --steer-vector results/generated/roles_gpu/dir_sft_b16.npy --steer-block 16 --steer-alpha "$A" >> artifacts/runs/inj_frame2_sft.log 2>&1
.venv/bin/python scripts/run_injection.py "${COMMON[@]}" --defense random --steer-block 16 --steer-alpha "$A" --random-seed 0 >> artifacts/runs/inj_frame2_sft.log 2>&1
echo done >> artifacts/runs/inj_frame2_defense_chain.log
