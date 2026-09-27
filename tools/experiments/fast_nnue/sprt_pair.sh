#!/usr/bin/env bash
# An early-stopping SPRT between two fast-NNUE candidates in one two-net pairing (fast_pair.py pair: Dev = NEW
# with its own compiled-in net, Prev = OPP with its own; every FASTNNUE_* variable unset). test_bots's SPRT
# defaults: H0 0, H1 +5, LLR bounds +/-3, early stop, openings from 0. The full-data play stage's 90 ms SPRT
# (B64_d5M_57ep vs B64_lr1e2: PASS at 864 games, +36.3 +/- 14.3) was run this way (then sprt90_full.sh).
#
#   sprt_pair.sh NEW OPP [MS=90] [THREADS=6]
#
# Candidates: NEW_CAND / OPP_CAND (default cpp_impl/bin/cand_full_NEW and cand_full_OPP, as export_net.sh
# names them; the shipped engine's candidate is cpp_impl/bin/cand_noop). Pairing cpp_impl/bin/pair_sprt_NEW__OPP
# (built fresh). Log datasets/nnue2/matches/NEW__vs_OPP__MS_ms.log (first line "[HH:MM:SS] start ...", last
# "[HH:MM:SS] exit N"), then fast_pair.py check-log on it (each side's net file and CRC-32).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-$ROOT/toolchains/py312-dml/Scripts/python.exe}"
[ $# -ge 2 ] || { sed -n 2,12p "$0"; exit 2; }
new="$1"; opp="$2"; ms="${3:-90}"; threads="${4:-6}"
new_cand="${NEW_CAND:-cpp_impl/bin/cand_full_$new}"
opp_cand="${OPP_CAND:-cpp_impl/bin/cand_full_$opp}"
pair="cpp_impl/bin/pair_sprt_${new}__${opp}"
mdir="$ROOT/datasets/nnue2/matches"
log="$mdir/${new}__vs_${opp}__${ms}_ms.log"
mkdir -p "$mdir"
export PATH="$ROOT/toolchains/llvm-mingw-20260616-ucrt-x86_64/bin:$PATH"
for v in $(env | grep -o '^FASTNNUE_[A-Z_]*'); do unset "$v"; done
unset SPRT_MAX_GAMES SPRT_LLR_BOUND SPRT_GAME_OFFSET SPRT_ELO0 SPRT_ELO1 SPRT_THINK_MS
cd "$ROOT"
"$PY" tools/experiments/fast_nnue/fast_pair.py pair "$new=$new_cand" "$opp=$opp_cand" --out "$pair" \
    > "$mdir/${new}__vs_${opp}__${ms}_ms.build.log" 2>&1 \
    || { echo "build of $pair failed (see ${new}__vs_${opp}__${ms}_ms.build.log)"; exit 1; }
tail -4 "$mdir/${new}__vs_${opp}__${ms}_ms.build.log"
# the nets each side was built for, as check-log arguments (none for a side without a fast NNUE)
mapfile -t nets < <("$PY" -c "import json,sys; d=json.load(open(sys.argv[1]))
for s in ('dev', 'prev'):
    if d[s].get('fast'): print(f'--{s}'); print(d[s]['net'])" "$pair/fast_pair.json" | tr -d '\r')
echo "[$(date +%H:%M:%S)] start $ms: $ms ($new vs $opp, two-net pairing $(basename "$pair"), nets compiled in:" \
     "${nets[*]}; SPRT H0 0 H1 5, default LLR bounds +/-3, early stop, openings from 0, $threads threads)" > "$log"
cd "$ROOT/cpp_impl"
SPRT_THREADS=$threads "$ROOT/$pair/test_bots.exe" "$ms" >> "$log" 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] exit $rc" >> "$log"
cd "$ROOT"
echo "[$(date +%H:%M:%S)] $new vs $opp $ms ms: $(grep '^N:' "$log" | tail -1 | cut -d' ' -f1-15); $(grep '^SPRT ' "$log" | tail -1)"
"$PY" tools/experiments/fast_nnue/fast_pair.py check-log "$log" "${nets[@]}"
exit $rc
