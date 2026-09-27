#!/usr/bin/env bash
# Export a trained pattern-generator net (gen_nnue.py / gen_nnue_stream.py checkpoint) for the engine, build
# its candidate, check its quantized parity and record the exponents the engine's quantizer picks. The
# full-data and long-training rounds exported every net this way (then export_stream_net.sh and
# export_long_net.sh).
#
#   export_net.sh NAME KIND [--cand DIR] [--log FILE] [--force]        KIND: b64 | b128 (make_cand_b.py --kind)
#
#   1. export_bgn.py export NAME datasets/nnue2/fast/NAME_perm.bin --perm   (lane-paired BGN1; skipped when
#      the file exists, unless --force)
#   2. export_bgn.py verify: the file's baked float tables reproduce PyTorch Gen.forward within 0.05 eval
#      units. Gen.forward's own float32 drifts by ~0.1 on nets with large first-layer rows, so on a failure
#      the file is checked against Gen.forward in float64 (verify64.py) and accepted within the same 0.05.
#   3. the candidate DIR (default cpp_impl/bin/cand_full_NAME; refused if it exists, unless --force):
#      build_cand.sh DIR --kind KIND --net NAME_perm.bin --no-tools
#   4. compare_bgn.py NAME: the candidate's quantized static eval (datagen label, from-scratch path) against
#      PyTorch float on parity_in.cfdg's 20,000 positions; then the same with FASTNNUE_BAKE=1 (tables
#      re-baked in C++ from the file's generator section)
#   5. the engine's loader line on 10 positions: the net's CRC-32 and the exponents QA, PSQT, QB, Q2, QO
#      (the CodinGame port's quantizer takes them as --qexp QA,QPS,QB,Q2,QO)
# Log (appended, timestamped): --log FILE, default datasets/nnue2/fast/export.log.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FN="$ROOT/tools/experiments/fast_nnue"
F="$ROOT/datasets/nnue2/fast"
PY="${PY:-$ROOT/toolchains/py312-dml/Scripts/python.exe}"
[ $# -ge 2 ] || { sed -n 2,22p "$0"; exit 2; }
NAME=$1; KIND=$2; shift 2
CAND="$ROOT/cpp_impl/bin/cand_full_$NAME"
LOG="$F/export.log"
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cand) CAND="$2"; shift ;;
        --log) LOG="$2"; shift ;;
        --force) FORCE=1 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
case "$CAND" in /*|[A-Za-z]:*) ;; *) CAND="$ROOT/$CAND" ;; esac
NET="$F/${NAME}_perm.bin"
export PATH="$ROOT/toolchains/llvm-mingw-20260616-ucrt-x86_64/bin:$PATH"
unset FASTNNUE_PATH FASTNNUE_BAKE
mkdir -p "$F" "$(dirname "$LOG")"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() {  # a step whose output (minus eval_candidate's noise) goes to the log; stop on failure
    "$@" 2>&1 | grep -v "HCE weights\|patches from\|^Prev from" | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then say "FAILED (exit $rc): $*"; exit 1; fi
}
cd "$ROOT"
if [ -e "$CAND" ] && [ $FORCE = 0 ]; then say "FAILED: $CAND exists (--force to rebuild it)"; exit 1; fi
say "start export $NAME ($KIND) -> $NET, candidate $CAND"
if [ -f "$NET" ] && [ $FORCE = 0 ]; then
    say "$NET exists; not exported again"
else
    run "$PY" "$FN/export_bgn.py" export "$NAME" "$NET" --perm
fi
"$PY" "$FN/export_bgn.py" verify "$NET" "$NAME" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    say "export_bgn.py verify failed against the float32 reference; checking the file against float64 (verify64.py)"
    V64=$("$PY" "$ROOT/tools/experiments/nnue2/verify64.py" "$NAME" --bgn "$NET" 2>&1 | grep -v -i warn)
    echo "$V64" | tee -a "$LOG"
    m64=$(echo "$V64" | grep "file vs torch float64" | sed 's/.*max |d| \([0-9.e+-]*\),.*/\1/')
    if ! awk -v x="$m64" 'BEGIN { exit !(x != "" && x + 0 <= 0.05) }'; then say "FAILED: file vs float64 max |d| $m64"; exit 1; fi
    say "file vs float64 max |d| $m64 <= 0.05: accepted"
fi
run bash "$FN/build_cand.sh" "$CAND" --kind "$KIND" --net "$NET" --no-tools
run "$PY" "$FN/compare_bgn.py" "$NAME" --net "$NET" --cand "$CAND"
run "$PY" "$FN/compare_bgn.py" "$NAME" --net "$NET" --cand "$CAND" --bake
TMP="$(mktemp -d)"
head -c 1280 "$ROOT/datasets/nnue2/fnn1/parity_in.cfdg" > "$TMP/in.cfdg"
"$CAND/datagen.exe" label "$TMP/in.cfdg" "$TMP/out.cfdg" 0 1 2> "$TMP/err" > /dev/null
rc=$?
grep "fast_nnue" "$TMP/err" | tee -a "$LOG"
rm -rf "$TMP"
if [ $rc -ne 0 ]; then say "FAILED (exit $rc): loader line"; exit 1; fi
say "exit 0 ($NAME)"
