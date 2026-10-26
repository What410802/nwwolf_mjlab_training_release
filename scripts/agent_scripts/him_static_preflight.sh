#!/usr/bin/env bash
# No-GPU static gate for HIM scripts and (optional) HIM training code.
# Usage: HIM_CODE_ROOT=/path/to/training_repo/src ./him_static_preflight.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$ROOT/him_part2_gate.py"
for f in "$GATE" "$ROOT/him_model0_gate20_2seed.sh" "$ROOT/him_microfine_row0.sh" "$ROOT/him_regate_job12.sh"; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
python3 -m pyflakes "$GATE"
python3 -m py_compile "$GATE"
grep -q "load_env_cfg(args.task, play=True)" "$GATE" && { echo "gate must not use play=True"; exit 1; } || true
grep -q "screen_pass" "$GATE" || { echo "gate missing screen_pass"; exit 1; }
grep -q "acceptance_eligible" "$GATE" || { echo "gate missing acceptance_eligible"; exit 1; }
if grep -R -q "gate_pass" "$ROOT"/him_model0_gate20_2seed.sh "$ROOT"/him_microfine_row0.sh "$GATE"; then
  echo "stale gate_pass reference"; exit 1
fi
for f in "$ROOT"/him_*.sh; do bash -n "$f"; done
if [ -n "${HIM_CODE_ROOT:-}" ]; then
  python3 -m pyflakes "$HIM_CODE_ROOT"/alldog_mjlab/algorithms/him/*.py \
    "$HIM_CODE_ROOT"/alldog_mjlab/tasks/velocity/blackw/*.py \
    "$HIM_CODE_ROOT"/alldog_mjlab/tasks/velocity/black/him.py
  python3 -m compileall -q "$HIM_CODE_ROOT"/alldog_mjlab/algorithms/him \
    "$HIM_CODE_ROOT"/alldog_mjlab/tasks/velocity/blackw
fi
echo "HIM static preflight OK"
