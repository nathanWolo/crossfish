#!/usr/bin/env bash
# Replaces run_extra2.sh (after fns): the equal-lr arms (shipped parameters at
# lr 3e-3 too), the 256-centroid packing check on the H8 models, then the slower
# D32/H32 and side-to-move arms if time allows.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$ROOT/toolchains/py312-dml/Scripts/python.exe
OUT=$ROOT/datasets/eval2/capacity
Q=$OUT/queue.log
run() {  # name, args...
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name" >> "$Q"
  "$PY" "$ROOT/tools/experiments/capacity/capacity_probe.py" "$@" > "$OUT/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] end $name (exit $?): $(grep RESULT "$OUT/$name.log")" >> "$Q"
}
run a2_lr3e3 --arm a2 --lr 3e-3 --tag lr3e3
run e_lr3e3 --arm e --lr 3e-3 --tag lr3e3
echo "[$(date +%H:%M:%S)] start quant_eval" >> "$Q"
"$PY" "$ROOT/tools/experiments/capacity/quant_eval.py" ffull a_lr3e3 a2_lr3e3 e_lr3e3 \
  --export a2_lr3e3 > "$OUT/quant_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] end quant_eval (exit $?)" >> "$Q"
run d_lr3e3 --arm d --lr 3e-3 --tag lr3e3
run a2s_lr3e3 --arm a2s --lr 3e-3 --tag lr3e3
echo "[$(date +%H:%M:%S)] extra3 done" >> "$Q"
