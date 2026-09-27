#!/usr/bin/env bash
# Build a fast-NNUE candidate (the engine with the fast integer NNUE as Dev, the shipped engine as Prev) and,
# by default, its check and benchmark tools.
#
#   build_cand.sh DIR [--kind any|cell|b64|b128|b64s|b128s] [--net NET] [--keep-dead-hce] [--no-tools]
#
#   1. make_cand_b.py DIR --kind K [--net NET]: DIR/dev_patches.json and the fast_nnue*.hpp headers.
#      --kind fixes the net kind at compile time (the default any dispatches on the net file's magic, 3-5%
#      slower); --net compiles NET's path in (FASTNNUE_PATH still overrides it at run time).
#   2. tools/eval_candidate.py build DIR: DIR/test_bots.exe, datagen.exe, bench_ab.exe.
#   3. Unless --no-tools, compiled in DIR next to its patched sources (8 at a time):
#        fast_check.exe test_bots_check.exe bench_ab_check.exe datagen_check.exe   -DFASTNNUE_CHECK builds:
#            every evaluation checks incremental == from-scratch and AVX2 == scalar (run_checks.sh runs them)
#        fallback_repro[_check].exe  stack_repro[_check].exe   searches entered without refresh_root
#        fast_bench_b.exe            per-call costs of the B nets (not built for --kind cell)
#   4. The AVX2 instructions in bench_ab.exe (objdump), as a check that the kernels are really vectorised.
# Timestamped log: DIR/build_cand.log. Needs the repository toolchain (llvm-mingw) and the Python with
# numpy (PY, default toolchains/py312-dml).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$ROOT/tools/experiments/fast_nnue"
PY="${PY:-$ROOT/toolchains/py312-dml/Scripts/python.exe}"
export PATH="$ROOT/toolchains/llvm-mingw-20260616-ucrt-x86_64/bin:$PATH"
[ $# -ge 1 ] || { sed -n 2,19p "$0"; exit 2; }
DIR="$1"; shift
case "$DIR" in /*|[A-Za-z]:*) ;; *) DIR="$ROOT/$DIR" ;; esac
KIND=any; MAKE_ARGS=(); TOOLS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --kind) KIND="$2"; MAKE_ARGS+=(--kind "$2"); shift ;;
        --net) MAKE_ARGS+=(--net "$2"); shift ;;
        --keep-dead-hce) MAKE_ARGS+=(--keep-dead-hce) ;;
        --no-tools) TOOLS=0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
mkdir -p "$DIR"
LOG="$DIR/build_cand.log"
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
step() {  # a step whose output goes to the log; stop on failure
    "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then say "FAILED (exit $rc): $*"; tail -20 "$LOG" >&2; exit 1; fi
}
cd "$ROOT"
say "start build_cand $DIR (kind $KIND${MAKE_ARGS[*]:+, make_cand_b.py ${MAKE_ARGS[*]}})"
step "$PY" "$HERE/make_cand_b.py" "$DIR" "${MAKE_ARGS[@]}"
step "$PY" "$ROOT/tools/eval_candidate.py" build "$DIR"
say "built test_bots.exe datagen.exe bench_ab.exe"

if [ $TOOLS = 1 ]; then
    # eval_candidate.py's CXXFLAGS; -I DIR first so the tools include the candidate's patched test_bots.cpp
    FLAGS=(-O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread -Wno-unknown-pragmas
           -Wno-ignored-attributes -Wl,--stack,16777216 "-I$DIR" "-I$ROOT/cpp_impl")
    cp "$HERE/fast_check.cpp" "$HERE/fallback_repro.cpp" "$HERE/stack_repro.cpp" "$HERE/fast_bench_b.cpp" "$DIR/"
    JOBS=("-DFASTNNUE_CHECK|fast_check.exe|fast_check.cpp" "-DFASTNNUE_CHECK|test_bots_check.exe|test_bots.cpp"
          "-DFASTNNUE_CHECK|bench_ab_check.exe|bench_ab.cpp" "-DFASTNNUE_CHECK|datagen_check.exe|datagen.cpp"
          "|fallback_repro.exe|fallback_repro.cpp" "-DFASTNNUE_CHECK|fallback_repro_check.exe|fallback_repro.cpp"
          "|stack_repro.exe|stack_repro.cpp" "-DFASTNNUE_CHECK|stack_repro_check.exe|stack_repro.cpp")
    [ "$KIND" != cell ] && JOBS+=("|fast_bench_b.exe|fast_bench_b.cpp")
    compile() {
        local defs out src
        IFS='|' read -r defs out src <<< "$1"
        if g++ "${FLAGS[@]}" $defs -o "$DIR/$out" "$DIR/$src" 2> "$DIR/${out%.exe}.err"; then
            echo "[$(date +%H:%M:%S)] compiled $out${defs:+ ($defs)}" >> "$LOG"; rm -f "$DIR/${out%.exe}.err"
        else
            echo "[$(date +%H:%M:%S)] FAILED $out: $(head -3 "$DIR/${out%.exe}.err")" >> "$LOG"
        fi
    }
    running=0
    for j in "${JOBS[@]}"; do
        compile "$j" &
        running=$((running + 1))
        if [ $running -ge 8 ]; then wait -n; running=$((running - 1)); fi
    done
    wait
    n_fail=$(grep -c "FAILED" "$LOG")
    say "tools: $(grep -c '] compiled ' "$LOG") compiled, $n_fail failed"
    [ "$n_fail" = 0 ] || { grep FAILED "$LOG" >&2; exit 1; }
fi

if command -v objdump > /dev/null; then
    objdump -d --no-show-raw-insn "$DIR/bench_ab.exe" > "$DIR/bench_ab.asm"
    say "AVX2 in bench_ab.exe: vpmaddwd ymm $(grep -c 'vpmaddwd.*ymm' "$DIR/bench_ab.asm"), vpaddw ymm" \
        "$(grep -c 'vpaddw.*ymm' "$DIR/bench_ab.asm"), vpsubw ymm $(grep -c 'vpsubw.*ymm' "$DIR/bench_ab.asm")"
    rm -f "$DIR/bench_ab.asm"
fi
say "exit 0 ($DIR)"
