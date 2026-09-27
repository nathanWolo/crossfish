#!/usr/bin/env bash
# Third batch, after a_lr3e3 showed F_full was lr-limited: the dedup, second-layer
# and D32/H32 arms again with the shipped parameters at lr 3e-3 too, then the
# 256-centroid packing check. Waits for the running fns arm to finish.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$ROOT/toolchains/py312-dml/Scripts/python.exe
OUT=$ROOT/datasets/eval2/capacity
Q=$OUT/queue.log
until grep -q "RESULT\|Traceback" "$OUT/fns.log"; do sleep 5; done
echo "[$(date +%H:%M:%S)] end fns: $(grep RESULT "$OUT/fns.log")" >> "$Q"
run() {  # name, args...
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name" >> "$Q"
  "$PY" "$ROOT/tools/experiments/capacity/capacity_probe.py" "$@" > "$OUT/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] end $name (exit $?): $(grep RESULT "$OUT/$name.log")" >> "$Q"
}
run a2_lr3e3 --arm a2 --lr 3e-3 --tag lr3e3
run e_lr3e3 --arm e --lr 3e-3 --tag lr3e3
run d_lr3e3 --arm d --lr 3e-3 --tag lr3e3
echo "[$(date +%H:%M:%S)] start quant_eval" >> "$Q"
"$PY" "$ROOT/tools/experiments/capacity/quant_eval.py" ffull a_lr3e3 a2_lr3e3 e_lr3e3 a2 d_lr3e3 \
  --export a2_lr3e3 > "$OUT/quant_eval.log" 2>&1
echo "[$(date +%H:%M:%S)] end quant_eval (exit $?)" >> "$Q"
echo "[$(date +%H:%M:%S)] extra2 done" >> "$Q"
