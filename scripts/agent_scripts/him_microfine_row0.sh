#!/usr/bin/env bash
# HIM row0 controlled micro-finetune: warm-start teacher + low lr + frozen std +
# teacher anchor, 300 iter, then strict gate every saved checkpoint.
# Intentionally NOT a 1500-iter blind run.
set -uo pipefail
REPO="$HOME/rc_work/nwwolf_mjlab_training_release"
GATE="$HOME/rc_work/him_part2_gate.py"
OFFICIAL="${BLACKW_OFFICIAL_WARMSTART:-/tmp/official_blackw_obstacle.pt}"
JOB="him_microfine_row0_$(date +%m%d_%H%M)"
LOGDIR="$HOME/rc_work/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$JOB.log"; SUM="$LOGDIR/${JOB}_summary.txt"
export PATH="$HOME/.local/bin:$PATH"; export MUJOCO_GL=egl
export BLACKW_MAX_ROW=0
export BLACKW_HIM_INIT_STD=0.1
export BLACKW_HIM_FREEZE_STD=1
export BLACKW_HIM_ANCHOR_ALPHA=0.01
export BLACKW_REWARD_CLIP=50
exec > >(tee -a "$LOG") 2>&1
cd "$REPO" || exit 1
on_signal() { printf 0 > "$HOME/5060ti_machine_use_flag.txt"; exit "$1"; }
trap 'printf 0 > "$HOME/5060ti_machine_use_flag.txt"' EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
: > "$SUM"
printf 1 > "$HOME/5060ti_machine_use_flag.txt"
exec 9>/tmp/gpu.lock
flock -w 900 9 || exit 3
for _ in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "${used:-99999}" -lt 1500 ] && break
  sleep 10
done
echo "[$JOB] start=$(date -Is) official=$OFFICIAL" | tee -a "$SUM"
# Fail loudly if the strict eval contract is broken before spending GPU time.
for cat in platform slope stairs high_wall trench; do
  .venv/bin/python "$GATE" --task blackw-official-him --category "$cat" --row 0 \
    --check-config-only >> "$LOG" 2>&1 || { echo "preflight gate failed $cat" | tee -a "$SUM"; exit 4; }
done
echo "train $(date -Is)" | tee -a "$SUM"
env BLACKW_OFFICIAL_WARMSTART="$OFFICIAL" \
  .venv/bin/train blackw-official-him \
    --env.scene.num-envs 512 --agent.num-steps-per-env 96 \
    --agent.max-iterations 300 --agent.save-interval 25 \
    --agent.algorithm.learning-rate 1e-4 \
    --agent.algorithm.entropy-coef 0.0 \
    --agent.run-name "$JOB" --agent.logger tensorboard
RC=$?
echo "train rc=$RC" | tee -a "$SUM"
RUN_DIR=$(ls -dt logs/rsl_rl/blackw_velocity_him/*_"$JOB" 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then echo "no run dir" | tee -a "$SUM"; exit 5; fi
CKPTS=$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V)
if [ -z "$CKPTS" ]; then echo "no checkpoints" | tee -a "$SUM"; exit 6; fi
FAIL=0
for ck in $CKPTS; do
  name=$(basename "$ck" .pt)
  for cat in platform slope stairs high_wall trench; do
    out="$LOGDIR/${JOB}_${name}_${cat}.json"
    .venv/bin/python "$GATE" --task blackw-official-him --checkpoint "$ck" \
      --category "$cat" --row 0 --attempts 20 --num-envs 8 --seed 0 \
      --jitter-x 0.05 --jitter-y 0.05 --jitter-yaw 0.05 \
      --vx-min 0.25 --vx-max 1.0 --vy 0.15 --wz 0.2 \
      --max-steps 1000 --out "$out" >/dev/null 2>&1
    grc=$?
    if [ "$grc" -ne 0 ] || [ ! -f "$out" ]; then
      echo "gate $name $cat FAILED rc=$grc" | tee -a "$SUM"; FAIL=1; continue
    fi
    .venv/bin/python - "$out" "$name" "$cat" << 'PY' >> "$SUM"
import json, sys
d=json.load(open(sys.argv[1]))
print(f"gate {sys.argv[2]} {sys.argv[3]} done={d['attempts_done']} success={d['success']} "
      f"fail={d['failure']} out={d['out_of_course']} screen={d['screen_pass']}")
PY
  done
done
echo "[$JOB] train_rc=$RC gate_fail=$FAIL done=$(date -Is)" | tee -a "$SUM"
exit "$((RC != 0 || FAIL != 0))"
