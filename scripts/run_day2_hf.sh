#!/usr/bin/env bash
# Day-2 HF defense chain: role vector vs style vector vs their sum against the working role-imitation framings.
# Items per stage: fake_turn, fake_completion, important, cot_forgery x N_PER items + 4 clean. Arms (interleaved rlvr/sft so
# both stages progress together): none, role 4x, style-neg 2x, sum(role 2x + style 2x), random 4x; rlvr also style-neg 4x.
# Run-ids: inj_frame_def (none/role/random), inj_frame_def_style, inj_frame_def_sum (same seeded items in each).
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; BS="${BS:-10}"; CAP="${CAP:-3000}"; N_PER="${N_PER:-15}"; N_INJ=$((N_PER*4))
TYPES=(tc2_fake_turn tc2_fake_completion tc2_important tc2_cot_forgery)
LOG="artifacts/runs/day2_hf.log"; mkdir -p artifacts/runs; log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
inj() { local i rc; for i in $(seq 1 8); do $PY scripts/run_injection.py "$@"; rc=$?; [ $rc -eq 0 ] && return 0; log "run_injection exit $rc, retry $i"; sleep 15; done; return 1; }
COMMON=(--types "${TYPES[@]}" --n-injected "$N_INJ" --n-user-control 0 --n-clean 4 --batch-size "$BS" --max-new-tokens "$CAP" --max-items-per-process 100000 --no-acts)
declare -A ROLE_GAP=([sft]=2.78 [dpo]=2.82 [rlvr]=2.86) STYLE_GAP=([sft]=3.8 [rlvr]=4.15)
mul() { $PY -c "print(round($1*$2,2))"; }
# sum vectors: unit(role*2gap + style_neg*2gap), alpha = its norm
for st in sft rlvr; do
  $PY - "$st" "${ROLE_GAP[$st]}" "${STYLE_GAP[$st]}" <<'PYEOF'
import sys, numpy as np
st, rg, sg = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = np.load(f"results/generated/roles_gpu/dir_{st}_b16.npy").astype(np.float32); r /= np.linalg.norm(r)
s = np.load(f"results/generated/voice01/dir_style_neg_{st}_b16.npy").astype(np.float32); s /= np.linalg.norm(s)
v = 2*rg*r + 2*sg*s; n = float(np.linalg.norm(v))
np.save(f"results/generated/voice01/dir_sum_{st}_b16.npy", v/n); open(f"results/generated/voice01/sum_alpha_{st}.txt","w").write(f"{n:.2f}")
print(st, "sum alpha", round(n,2), "cos(role,style_neg)", round(float(r@s),3))
PYEOF
done
log "day2 hf start BS=$BS CAP=$CAP N_PER=$N_PER"
for st in rlvr sft; do log "none $st"; inj "${COMMON[@]}" --run-id inj_frame_def --stage $st >> artifacts/runs/inj_frame_def_$st.log 2>&1; done
for st in rlvr sft; do a=$(mul ${ROLE_GAP[$st]} 4); log "role 4x $st alpha $a"
  inj "${COMMON[@]}" --run-id inj_frame_def --stage $st --defense steer --steer-vector results/generated/roles_gpu/dir_${st}_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_$st.log 2>&1; done
for st in rlvr sft; do a=$(mul ${STYLE_GAP[$st]} 2); log "style 2x $st alpha $a"
  inj "${COMMON[@]}" --run-id inj_frame_def_style --stage $st --defense steer --steer-vector results/generated/voice01/dir_style_neg_${st}_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_style_$st.log 2>&1; done
for st in rlvr sft; do a=$(cat results/generated/voice01/sum_alpha_$st.txt); log "sum role2x+style2x $st alpha $a"
  inj "${COMMON[@]}" --run-id inj_frame_def_sum --stage $st --defense steer --steer-vector results/generated/voice01/dir_sum_${st}_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_sum_$st.log 2>&1; done
for st in rlvr sft; do a=$(mul ${ROLE_GAP[$st]} 4); log "random 4x $st alpha $a"
  inj "${COMMON[@]}" --run-id inj_frame_def --stage $st --defense random --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_$st.log 2>&1; done
a=$(mul ${STYLE_GAP[rlvr]} 4); log "style 4x rlvr alpha $a"
inj "${COMMON[@]}" --run-id inj_frame_def_style --stage rlvr --defense steer --steer-vector results/generated/voice01/dir_style_neg_rlvr_b16.npy --steer-block 16 --steer-alpha $a >> artifacts/runs/inj_frame_def_style_rlvr.log 2>&1
log "summaries"
for r in inj_frame_def inj_frame_def_style inj_frame_def_sum; do $PY scripts/summarize_injection.py --run-id $r --effective-cap "$CAP" >> artifacts/runs/${r}_summary.log 2>&1; done
log "day2 hf complete"
