#!/usr/bin/env bash
# Build, play, check and rate a round robin of fast-NNUE candidates (planned with tools/round_robin.py plan;
# 20 ms, 2,000 or 4,000 games per pair in the recorded rounds). The rounds' tournaments nnue_fast_20ms,
# nnue_full_20ms and nnue_long_20ms were played this way (then run_rr_s4.sh, run_rr_full.sh, run_long_rr.sh
# and run_long_play_rr.sh).
#
#   run_rr.sh NAME [--anchor noop] [--root DIR] [rr_parallel.py run / fast_pair.py rr-run / round_robin.py run options...]
#   e.g. run_rr.sh nnue_long_20ms --local-threads 6 --worker --worker-threads 4 --worker-exclude '^B128_'
#
#   1. rr_parallel.py build NAME: every pairing not up to date is built (fast_pair.py pair, 6 at a time)
#   2. rr_parallel.py run NAME OPTIONS: fast_pair.py rr-run (stamps checked, FASTNNUE_* dropped, round_robin's
#      run with progress and ETAs in datasets/eval2/rr/NAME/run.log; with --worker, the Linux worker's pairs
#      go through fast_worker.py, which needs CROSSFISH_WORKER=user@host; --worker-exclude keeps matching
#      engines off the worker). An interrupted run resumes where it stopped.
#   3. fast_pair.py check-logs NAME: every pairing log shows each fast side loading its own planned net
#   4. round_robin.py rate NAME --anchor ANCHOR (ratings.json)
# --root DIR keeps the tournament in DIR instead of datasets/eval2/rr (round_robin.py --root).
# Queue lines: <root>/NAME/rr_queue.log.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-$ROOT/toolchains/py312-dml/Scripts/python.exe}"
FN="$ROOT/tools/experiments/fast_nnue"
[ $# -ge 1 ] || { sed -n 2,18p "$0"; exit 2; }
NAME=$1; shift
ANCHOR=noop
RR_ROOT_DIR="$ROOT/datasets/eval2/rr"
while [ $# -gt 0 ]; do
    case "$1" in
        --anchor) ANCHOR=$2; shift 2 ;;
        --root) RR_ROOT_DIR=$2; shift 2 ;;
        *) break ;;
    esac
done
case "$RR_ROOT_DIR" in /*|[A-Za-z]:*) ;; *) RR_ROOT_DIR="$ROOT/$RR_ROOT_DIR" ;; esac
export PATH="$ROOT/toolchains/llvm-mingw-20260616-ucrt-x86_64/bin:$PATH"
cd "$ROOT"
Q="$RR_ROOT_DIR/$NAME/rr_queue.log"
mkdir -p "$(dirname "$Q")"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }
say "start $NAME: build, then rr_parallel.py run $NAME $*"
"$PY" "$FN/rr_parallel.py" --root "$RR_ROOT_DIR" build "$NAME" 2>&1 | tee -a "$Q"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "build failed"; exit 1; }
"$PY" "$FN/rr_parallel.py" --root "$RR_ROOT_DIR" run "$NAME" "$@"
rc=$?
say "round robin $NAME exit $rc"
"$PY" "$FN/fast_pair.py" --root "$RR_ROOT_DIR" check-logs "$NAME" 2>&1 | tee -a "$Q"
"$PY" tools/round_robin.py --root "$RR_ROOT_DIR" rate "$NAME" --anchor "$ANCHOR" 2>&1 | tee -a "$Q"
say "exit $rc ($NAME)"
exit $rc
