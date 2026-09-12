#!/usr/bin/env bash
# One-shot setup on a rented CUDA pod. Run from the repo root on the pod:
#   bash scripts/gpu_setup.sh
# Creates .venv, installs requirements (Linux pip torch ships CUDA), installs the package, downloads the three
# pinned checkpoints + MMLU into ./.cache/huggingface, and runs a 2-item smoke test to record throughput.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
PY="${PYTHON_BIN:-python3}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
if [ ! -d .venv ]; then $PY -m venv .venv; fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements/research.txt -r requirements/dev.txt
.venv/bin/python -m pip install -q --no-deps -e .
.venv/bin/python -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0)); print("transformers", transformers.__version__)'
.venv/bin/python -m pytest -q tests 2>&1 | tail -1
mkdir -p artifacts/environment logs
.venv/bin/python scripts/record_environment.py || true
# downloads (pinned shas in configs/revisions.json)
.venv/bin/python scripts/pin_and_download.py --config configs/pilot.json --stages sft dpo rlvr 2>&1 | grep -vE "it/s\]|s/it\]" | tail -6
# smoke test: 2 items, short cap, records tok/s in the manifest
GEN_HEARTBEAT=200 .venv/bin/python scripts/run_injection.py --run-id gpu_smoke --stage rlvr --n-injected 54 --n-user-control 12 --n-clean 8 \
    --item-ids inj-000-marker-plain inj-001-marker-user --batch-size 2 --max-new-tokens 400 --max-items-per-process 100000 2>&1 | grep -E "batch|loaded|heartbeat" | tail -4
echo "setup complete $(date -u +%FT%TZ)"
