#!/usr/bin/env bash
# Correctness checks of a fast-NNUE candidate (stages 1-3, test (a)): with the -DFASTNNUE_CHECK builds that
# build_cand.sh makes, every evaluation of real searches checks that the incremental accumulators (lanes and
# PSQT) equal a scalar int32 from-scratch sum, that the AVX2 head equals the scalar dense reference, that the
# bound's premises and the int16 range hold, and that every eval-cache hit equals a from-scratch evaluation.
# Any mismatch aborts the run; the counters print at exit ("fast_nnue check: ...").
#
#   run_checks.sh CAND_DIR NET TAG [N_POS]
#
#   1. fast_check on N_POS (default 20000) parity positions: fixed-depth 6/8/10 searches on fresh engines,
#      self-play games at 3 ms per move with reused engines, from-scratch evals
#   2. bench_ab_check nodes 60 10 (search_fixed_depth on bench_ab's random positions)
#   3. test_bots_check depth-prune 8 and 20 (20 ms per move: getMove, iterative deepening, aspiration
#      re-searches, time-outs unwinding mid-tree), CHECK_GAMES games each (default 112; stage 3 ran 200)
#   4. datagen_check label (the from-scratch path datagen uses)
#   5. fallback_repro and stack_repro (searches entered without refresh_root), release and check builds
# NET is passed as FASTNNUE_PATH (it overrides a compiled-in net). CHECK_ONLY_FAST=1 runs step 1 only;
# CHECK_THREADS (default 14) threads for fast_check and depth-prune 8, 6 for the 20 ms games.
# Logs: CHECK_LOG_DIR (default datasets/nnue2/fast)/check_b_<TAG>_<step>.log, each with "[HH:MM:SS] start"
# and "[HH:MM:SS] exit N" lines; a summary of the counters on stdout.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
[ $# -ge 3 ] || { sed -n 2,21p "$0"; exit 2; }
CAND="$1"; NET="$2"; TAG="$3"; NPOS="${4:-20000}"
case "$CAND" in /*|[A-Za-z]:*) ;; *) CAND="$ROOT/$CAND" ;; esac
case "$NET" in /*|[A-Za-z]:*) ;; *) NET="$ROOT/$NET" ;; esac
OUT="${CHECK_LOG_DIR:-$ROOT/datasets/nnue2/fast}"
POS="$ROOT/datasets/nnue2/fnn1/parity_in.cfdg"
THREADS="${CHECK_THREADS:-14}"
GAMES="${CHECK_GAMES:-112}"
mkdir -p "$OUT"
export PATH="$ROOT/toolchains/llvm-mingw-20260616-ucrt-x86_64/bin:$PATH"
export FASTNNUE_PATH="$NET"
unset FASTNNUE_BAKE
ts() { date +%H:%M:%S; }
FAILS=0
run() {  # NAME CMD...: timestamped log with the exit status as the last line
    local name="$1"; shift
    local log="$OUT/check_b_${TAG}_$name.log"
    echo "[$(ts)] start $TAG $name: $* (net $(basename "$NET"))" | tee "$log"
    (cd "$ROOT/cpp_impl" && "$@") >> "$log" 2>&1
    local rc=$?
    echo "[$(ts)] exit $rc" | tee -a "$log"
    [ $rc -eq 0 ] || FAILS=$((FAILS + 1))
    grep -h "fast_nnue check:\|fast_nnue_b check:\|fast_check:\|CHECK\|nodes depth\|nps=\|Elo diff\|_repro:" "$log" \
        | tail -4 | sed 's/^/    /'
    return 0
}
# 1. parity positions: fixed depth 6/8/10 searches, self-play games at 3 ms, scratch evals
run fast_check "$CAND/fast_check.exe" "$POS" "$NPOS" "$THREADS"
if [ "${CHECK_ONLY_FAST:-0}" != 1 ]; then
    # 2. bench_ab's random positions at depth 10 (search_fixed_depth on fresh engines)
    run bench_ab_d10 "$CAND/bench_ab_check.exe" nodes 60 10
    # 3. real match play: fixed depth 8 with pruning, and 20 ms timed
    SPRT_MAX_GAMES=$GAMES SPRT_LLR_BOUND=100 SPRT_GAME_OFFSET=45000 SPRT_THREADS=$THREADS \
        run test_bots_depth-prune_8 "$CAND/test_bots_check.exe" depth-prune 8
    SPRT_MAX_GAMES=$GAMES SPRT_LLR_BOUND=100 SPRT_GAME_OFFSET=45000 SPRT_THREADS=6 \
        run test_bots_20ms "$CAND/test_bots_check.exe" 20
    # 4. datagen label statics (evaluate(GlobalBoard&): scratch AVX2 vs scalar)
    L0="$OUT/check_b_${TAG}_label0.cfdg"
    rm -f "$L0"
    run datagen_label0 "$CAND/datagen_check.exe" label "$POS" "$L0" 0 2
    rm -f "$L0"
    # 5. searches entered without refresh_root (exit status 3 when an evaluation is wrong)
    for e in fallback_repro fallback_repro_check stack_repro stack_repro_check; do
        run "$e" "$CAND/$e.exe" "$POS"
    done
fi
echo "summary ($TAG):"
for f in "$OUT"/check_b_"${TAG}"_*.log; do
    echo "  $(basename "$f"): $(tail -1 "$f") | $(grep -h "fast_nnue check:\|fast_nnue_b check:" "$f" | grep -o \
        "evals=[0-9]*\|acc_mismatch=[0-9]*\|out_mismatch=[0-9]*\|fallback refreshes=[0-9]*\|cache mismatches=[0-9]*\|range violations=[0-9]*" \
        | tr '\n' ' ')"
done
echo "$FAILS run(s) failed"
exit $((FAILS > 0))
