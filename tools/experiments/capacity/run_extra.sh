#!/usr/bin/env bash
# Second batch: lr controls and side-to-move ablations, then the 256-centroid
# packing check. Waits for run_all.sh's "queue done" line first.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$ROOT/toolchains/py312-dml/Scripts/python.exe
OUT=$ROOT/datasets/eval2/capacity
Q=$OUT/queue.log
until grep -q "queue done" "$Q"; do sleep 10; done
run() {  # name, args...
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name" >> "$Q"
  "$PY" "$ROOT/tools/experiments/capacity/capacity_probe.py" "$@" > "$OUT/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] end $name (exit $?): $(grep RESULT "$OUT/$name.log")" >> "$Q"
}
run a_lr3e3 --arm a --lr 3e-3 --tag lr3e3
run fns --arm fns
run a2_lr3e3 --arm a2 --lr 3e-3 --tag lr3e3
run a2s --arm a2s
echo "[$(date +%H:%M:%S)] start quant_eval" >> "$Q"
"$PY" "$ROOT/tools/experiments/capacity/quant_eval.py" ffull a2 a_lr3e3 a2_lr3e3 c e b d --export a2 > "$OUT/quant_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] end quant_eval (exit $?)" >> "$Q"
echo "[$(date +%H:%M:%S)] extra done" >> "$Q"
