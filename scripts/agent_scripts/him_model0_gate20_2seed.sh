#!/usr/bin/env bash
# P0: strict gate of the HIM warm-start checkpoint (model_0), 5 categories,
# 20 attempts x 2 seeds, both training and deployment physics. Fails loud if
# any gate invocation produces no JSON or non-zero rc.
set -uo pipefail
REPO="$HOME/rc_work/nwwolf_mjlab_training_release"
GATE="$HOME/rc_work/him_part2_gate.py"
RUN_DIR=$(ls -dt "$REPO"/logs/rsl_rl/blackw_velocity_him/*_him_row0_mixed_gate_* 2>/dev/null | head -1)
CKPT="$RUN_DIR/model_0.pt"
JOB="him_model0_gate20_$(date +%m%d_%H%M)"
LOGDIR="$HOME/rc_work/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$JOB.log"; SUM="$LOGDIR/${JOB}_summary.txt"
export PATH="$HOME/.local/bin:$PATH"; export MUJOCO_GL=egl
exec > >(tee -a "$LOG") 2>&1
cd "$REPO" || exit 1
on_signal() { printf 0 > "$HOME/5060ti_machine_use_flag.txt"; exit "$1"; }
trap 'printf 0 > "$HOME/5060ti_machine_use_flag.txt"' EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
: > "$SUM"
if [ ! -f "$CKPT" ] || [ ! -f "$GATE" ]; then
  echo "missing checkpoint/gate: ckpt=$CKPT gate=$GATE" | tee -a "$SUM"; exit 2
fi
printf 1 > "$HOME/5060ti_machine_use_flag.txt"
exec 9>/tmp/gpu.lock
flock -w 900 9 || exit 3
for _ in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "${used:-99999}" -lt 1500 ] && break
  sleep 10
done
echo "[$JOB] ckpt=$CKPT start=$(date -Is)" | tee -a "$SUM"
FAIL=0
for seed in 0 1; do
  for physics in train deploy; do
    for cat in platform slope stairs high_wall trench; do
      out="$LOGDIR/${JOB}_s${seed}_${physics}_${cat}.json"
      .venv/bin/python "$GATE" \
        --task blackw-official-him --checkpoint "$CKPT" \
        --category "$cat" --row 0 --attempts 20 --num-envs 8 --seed "$seed" \
        --jitter-x 0.05 --jitter-y 0.05 --jitter-yaw 0.05 \
        --physics "$physics" --vx-min 0.25 --vx-max 1.0 --vy 0.15 --wz 0.2 \
        --max-steps 1000 --out "$out" >/dev/null 2>&1
      rc=$?
      if [ "$rc" -ne 0 ] || [ ! -f "$out" ]; then
        echo "gate s=$seed p=$physics c=$cat FAILED rc=$rc" | tee -a "$SUM"
        FAIL=1
        continue
      fi
      .venv/bin/python - "$out" "$seed" "$physics" "$cat" << 'PY' >> "$SUM"
import json, sys
d=json.load(open(sys.argv[1]))
print(f"gate s={sys.argv[2]} p={sys.argv[3]} c={sys.argv[4]} "
      f"done={d['attempts_done']} success={d['success']} fail={d['failure']} "
      f"out={d['out_of_course']} screen={d['screen_pass']} rate={d['success_rate']:.3f}")
PY
    done
  done
done
echo "[$JOB] FAIL=$FAIL done=$(date -Is)" | tee -a "$SUM"
exit "$FAIL"
