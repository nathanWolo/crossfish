#!/usr/bin/env bash
# Search speed of a fast-NNUE candidate and of compile-time variants of it, against the shipped engine in the
# same process (bench_ab: Dev = the candidate's engine, Prev = shipped, on the same positions, so the ratio
# is robust to machine load). The stages' speed tables (bench_variants*.log, bench_s3.log, bench_s4.log)
# were measured this way.
#
#   bench_variants.sh CAND_DIR NET [--rounds R] [--args "nodes 400 9"] [LABEL=FLAGS ...]
#   e.g. bench_variants.sh cpp_impl/bin/cand_full_B64_d5M_57ep datasets/nnue2/fast/B64_d5M_57ep_perm.bin \
#            nopf="-DFASTNNUE_B_PREFETCH=0" nocache="-DFASTNNUE_CACHE_BITS=0" --args "walk 16 40 20"
#
# Each LABEL is CAND_DIR/bench_ab.cpp compiled with FLAGS (eval_candidate.py's flags) into
# CAND_DIR/bench_ab_LABEL.exe; "default" is the candidate's own bench_ab.exe. R rounds (default 3) run every
# configuration once per round, interleaved, with FASTNNUE_PATH=NET. ARGS: "nodes N D" (fixed-depth trees on
# fresh engines; the node counts show whether two builds search the same tree) or "walk G P MS" (one engine
# per game, MS per move, persistent TT and eval cache: in-game speed and mean depth). Prints one line per
# run and the median Dev/Prev nodes/s ratio per configuration. Run it on an otherwise idle machine.
# The variant flags are listed in tools/experiments/fast_nnue/README.md.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
[ $# -ge 2 ] || { sed -n 2,17p "$0"; exit 2; }
CAND="$1"; NET="$2"; shift 2
case "$CAND" in /*|[A-Za-z]:*) ;; *) CAND="$ROOT/$CAND" ;; esac
case "$NET" in /*|[A-Za-z]:*) ;; *) NET="$ROOT/$NET" ;; esac
ROUNDS=3; ARGS="nodes 400 9"; LABELS=(default); declare -A VFLAGS=([default]="")
while [ $# -gt 0 ]; do
    case "$1" in
        --rounds) ROUNDS="$2"; shift ;;
        --args) ARGS="$2"; shift ;;
        *=*) LABELS+=("${1%%=*}"); VFLAGS["${1%%=*}"]="${1#*=}" ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
export PATH="$ROOT/toolchains/llvm-mingw-20260616-ucrt-x86_64/bin:$PATH"
unset FASTNNUE_BAKE
FLAGS=(-O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread -Wno-unknown-pragmas
       -Wno-ignored-attributes -Wl,--stack,16777216 "-I$ROOT/cpp_impl")
exe() { if [ "$1" = default ]; then echo "$CAND/bench_ab.exe"; else echo "$CAND/bench_ab_$1.exe"; fi; }
for l in "${LABELS[@]}"; do
    [ "$l" = default ] && continue
    # shellcheck disable=SC2086
    g++ "${FLAGS[@]}" ${VFLAGS[$l]} -o "$(exe "$l")" "$CAND/bench_ab.cpp" || { echo "build of $l failed" >&2; exit 1; }
    echo "[$(date +%H:%M:%S)] built bench_ab_$l.exe (${VFLAGS[$l]})"
done
# bench_ab prints "  prev nodes=N ... nps=X" and "  dev  nodes=N ... [mean_depth=D] ... nps=Y"
PARSE='function val(k,   i) { for (i = 1; i <= NF; i++) if (index($i, k "=") == 1) return substr($i, length(k) + 2); return "" }
$1 == "prev" { pn = val("nps") }
$1 == "dev" { dn = val("nps"); nodes = val("nodes"); d = val("mean_depth") }
END { print (nodes == "" ? "?" : nodes), (d == "" ? "-" : d), (pn > 0 ? sprintf("%.1f", 100 * dn / pn) : "?") }'
declare -A RATIOS NODES
cd "$ROOT/cpp_impl"
for r in $(seq 1 "$ROUNDS"); do
    for l in "${LABELS[@]}"; do
        # shellcheck disable=SC2086
        read -r dev depth ratio < <(FASTNNUE_PATH="$NET" "$(exe "$l")" $ARGS 2>/dev/null | tr -d '\r' | awk "$PARSE")
        echo "[$(date +%H:%M:%S)] round $r/$ROUNDS $l: dev nodes $dev, mean depth $depth, nps ratio dev/prev $ratio%"
        RATIOS[$l]="${RATIOS[$l]:-} $ratio"
        if [ -n "${NODES[$l]:-}" ] && [ "${NODES[$l]}" != "$dev" ] && [ "${ARGS%% *}" = nodes ]; then
            echo "  note: $l searched $dev nodes, round 1 searched ${NODES[$l]} (not the same tree)"
        fi
        NODES[$l]="${NODES[$l]:-$dev}"
    done
done
echo "median Dev/Prev nodes/s ($ARGS, $ROUNDS rounds, net $(basename "$NET")):"
for l in "${LABELS[@]}"; do
    med=$(echo "${RATIOS[$l]}" | tr ' ' '\n' | grep '^[0-9.]*$' | grep . | sort -g \
          | awk '{a[NR] = $1} END {if (NR) print (NR % 2 ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2)}')
    echo "  $l: ${med:-?}%  (dev nodes ${NODES[$l]:-?})"
done
