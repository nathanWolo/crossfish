#!/usr/bin/env bash
# Runs the capacity-probe arms one after another on the GPU (DirectML).
# Per-arm logs: datasets/eval2/capacity/<arm>.log; queue progress: queue.log.
# Usage: tools/experiments/capacity/run_all.sh [arm ...]   (default: a a2 f g c d b e fj)
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$ROOT/toolchains/py312-dml/Scripts/python.exe
OUT=$ROOT/datasets/eval2/capacity
mkdir -p "$OUT"
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(a a2 f g c d b e fj)
for arm in "${ARMS[@]}"; do
  echo "[$(date +%H:%M:%S)] start $arm" >> "$OUT/queue.log"
  "$PY" "$ROOT/tools/experiments/capacity/capacity_probe.py" --arm "$arm" > "$OUT/$arm.log" 2>&1
  echo "[$(date +%H:%M:%S)] end $arm (exit $?): $(grep RESULT "$OUT/$arm.log")" >> "$OUT/queue.log"
done
echo "[$(date +%H:%M:%S)] queue done" >> "$OUT/queue.log"
