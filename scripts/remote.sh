#!/usr/bin/env bash
# Thin wrapper around ssh/rsync for the rented GPU. Reads connection details from .env (git-ignored):
#   GPU_HOST=...  GPU_PORT=22  GPU_USER=root  GPU_KEY=~/.ssh/mats_gpu  GPU_DIR=/workspace/olmo3-role-confusion
# Usage:
#   bash scripts/remote.sh check                 # nvidia-smi + disk
#   bash scripts/remote.sh push                  # rsync code + configs + data (no caches/artifacts)
#   bash scripts/remote.sh run "<command>"       # run a command in the remote repo dir
#   bash scripts/remote.sh pull                  # rsync results/ and artifacts/runs/*.log + manifests + jsonl back
#   bash scripts/remote.sh pull-acts <run_id>    # also pull activation files for one run
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
# Two modes: (a) GPU_HOST + GPU_PORT + GPU_USER + GPU_KEY; (b) GPU_ALIAS = a Host entry in ~/.ssh/config (e.g. written by `tnr connect`).
GPU_DIR="${GPU_DIR:-/workspace/olmo3-role-confusion}"
if [ -n "${GPU_ALIAS:-}" ]; then
  TARGET="$GPU_ALIAS"; OPTS=(-o ServerAliveInterval=30)
else
  : "${GPU_HOST:?set GPU_HOST (or GPU_ALIAS) in .env}"; GPU_PORT="${GPU_PORT:-22}"; GPU_USER="${GPU_USER:-root}"; GPU_KEY="${GPU_KEY:-$HOME/.ssh/mats_gpu}"
  TARGET="$GPU_USER@$GPU_HOST"; OPTS=(-i "$GPU_KEY" -p "$GPU_PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
fi
SSH=(ssh "${OPTS[@]}" "$TARGET")
RSYNC_SSH="ssh ${OPTS[*]}"
case "${1:-}" in
  check) "${SSH[@]}" "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader; df -h / /workspace 2>/dev/null | tail -2; python3 --version; nproc; free -g | head -2" ;;
  push)
    "${SSH[@]}" "mkdir -p $GPU_DIR"
    rsync -az -e "$RSYNC_SSH" --exclude .venv --exclude .cache --exclude artifacts --exclude results/generated --exclude '__pycache__' \
      --exclude .git --exclude '*.npz' --exclude '*.npy' ./ "$TARGET:$GPU_DIR/" ;;
  run) shift; "${SSH[@]}" "cd $GPU_DIR && $*" ;;
  pull)
    mkdir -p results/generated artifacts/runs
    rsync -az -e "$RSYNC_SSH" "$TARGET:$GPU_DIR/results/generated/" results/generated/
    rsync -az -e "$RSYNC_SSH" --include '*/' --include '*.log' --include '*.json' --include '*.jsonl' --exclude '*' \
      "$TARGET:$GPU_DIR/artifacts/runs/" artifacts/runs/ ;;
  pull-acts) shift; rsync -az -e "$RSYNC_SSH" "$TARGET:$GPU_DIR/artifacts/runs/$1/" "artifacts/runs/$1/" ;;
  *) echo "usage: remote.sh check|push|run <cmd>|pull|pull-acts <run_id>" >&2; exit 2 ;;
esac
