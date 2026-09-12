#!/usr/bin/env bash
# Diagnostic for the context gap on Think (gated on "tail2 complete"): (a) record context through the agentic runner (pipeline check);
# (b) web page with the original record wording in the payload (narrative vs document type). SFT then RLVR, 1 draw, vLLM.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
PYV=.venv-vllm/bin/python; LOG="artifacts/runs/day2_tail3.log"; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
ag() { local i rc; for i in $(seq 1 3); do $PYV scripts/run_agentic.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_agentic exit $rc, retry $i"; sleep 30; done; return 1; }
while ! grep -q "tail2 complete" artifacts/runs/day2_tail2.log 2>/dev/null; do sleep 60; done
log "tail3 start"
for st in sft rlvr; do
  log "record context via runner $st"
  ag --engine vllm --stage $st --run-id agentic01 --context record --scripted --rounds 3 --n-injected 200 --n-clean 12 --draws 1 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/agentic01_${st}_record.log 2>&1
  log "webpage with record nouns $st"
  ag --engine vllm --stage $st --run-id agentic01 --context webpage --tag webpage_recnoun --payload-nouns record --scripted --rounds 3 --n-injected 200 --n-clean 12 --draws 1 --max-new-tokens 3000 --gpu-mem 0.85 >> artifacts/runs/agentic01_${st}_webpage_recnoun.log 2>&1
done
log "tail3 complete"
