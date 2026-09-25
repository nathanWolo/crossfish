#!/bin/bash
# Data-scaling study (improvement log section 52): the same full NNUE trained
# on nested subsets, each played at equal depth 4 against Prev (400 games).
# Run from the repo root after `python tools/experiments/full_nnue/wire_into_dev.py
# cpp_impl/crossfish_dev.hpp` and copying full_nnue_float.hpp to
# cpp_impl/full_nnue.hpp; revert Dev afterwards. OUT holds checkpoints/results.
set -e
OUT=${OUT:-/tmp/full_nnue_scale}
DATA=${DATA:-"datasets/full_play_d12_base.bin datasets/full_rand_d12_base.bin"}
mkdir -p "$OUT"
make -C cpp_impl "$PWD/cpp_impl/bin/test_bots" >/dev/null
for n in 125000 250000 500000 1000000 0; do
  python3 tools/experiments/full_nnue/train_full_nnue.py --data $DATA --out "$OUT/wdl_$n.pt" \
    --A 256 --epochs 20 --batch 8192 --lr 1e-3 --subsample $n > "$OUT/wdl_$n.log" 2>&1
  res=$(cd cpp_impl && FULLNNUE_PATH="$OUT/wdl_$n.bin" SPRT_LLR_BOUND=100 SPRT_MAX_GAMES=400 \
        SPRT_GAME_OFFSET=30000 ./bin/test_bots depth 4 2>&1 | grep "^N:" | tail -1)
  echo "n=$n | $(grep '^epoch' "$OUT/wdl_$n.log" | tail -1) | $res" | tee -a "$OUT/scale_results.txt"
done
