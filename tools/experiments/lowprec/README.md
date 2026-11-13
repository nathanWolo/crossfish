# Low-precision ("natively quantized") NNUE training

This is a test of whether the recipe in Thinking Machines' [end-to-end 4-bit training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/)
helps CrossFish's pattern-generator NNUE. That recipe uses an NVFP4 forward pass, a high-precision backward pass,
block scales with 4/6 scale selection, and a higher-precision tail. Nothing here ships, and nothing touches
`cpp_impl/`.

## What transfers and what doesn't

The post is about training throughput on Blackwell tensor cores. Its forward GEMMs run in NVFP4, its backward
GEMMs run in MXFP8 with stochastic rounding, and the result is about 8% faster pretraining. None of that applies
to a 51k-parameter net whose 2.4G-row runs take about 80 minutes on a DirectML GPU at 507k rows/s. The bottleneck
there is the loader and the per-step overhead, not GEMM precision, and DirectML has no FP4 or FP8 units.

What does transfer is the forward-pass idea: train with the numerics the net will be deployed in, so that nothing
has to be rounded or calibrated afterwards. CrossFish has three places where a lower-precision net could pay off:

| Where | Today | Low-precision version | What it could buy |
| --- | --- | --- | --- |
| Payload (CodinGame source) | Generator at 12-14 bits per weight with per-row bf16 scales, rounded by GPTQ after training (W1: 42,682 characters) | NVFP4, 4.5 bits per weight | About 25k free characters, enough for a wider net or a book |
| Pattern tables (runtime) | int16, 25.6 MB | int8, one power-of-two step per lane | Half the bytes per row (one cache line instead of two) |
| Head layer 1 (runtime) | int16 x int16 `vpmaddwd` over nonzero pairs | u8 0..127 x int8, `vpmaddubsw` | FINDINGS A20 estimated +1 to +7 Elo "before precision loss" |

## Setup

`lowprec.py` reuses `gen_nnue.Gen` and replaces the forward pass with one that fake-quantizes the chosen parts
through a straight-through estimator. The master weights and the backward pass stay in float32. `check` confirms
that the float forward equals `Gen.fast`.

The runs used the following:

- **Init:** W1 (`r16_x128_l2400_s1601`, unscaled).
- **Training data:** e2b train (4.34M rows) plus a 4M-row slice of sp14 train. The slice is 8 blocks of 500k rows
  spread across the file, kept on the desktop under `datasets/nnue2/lowprec_slice/`.
- **Recipe:** W1's mix (e2b 0.4644 / sp14 0.5356), the WDL filter on sp14, the power-2.5 loss with 0.2 asymmetry,
  PSQT weight 0.06878, D4 augmentation, clip 1.0 and batch 16,384.
- **Length:** 3,000 steps (49M rows) at lr 3e-4, cosine to 1e-5.
- **Hardware:** a 4-core cloud CPU at about 20k rows/s per run.
- **Holdouts:** 200k rows each of V2 (e2b hold) and SPH13 (sp13 hold). W1 trained on neither.

Every QAT run is compared with `ctrl`, a float continuation with the same data, steps and seed. The short
continuation alone moves W1 by +0.3% on V2, so the control is the fair baseline.

## Results

These are the label MSE changes against the matching baseline. The post-training column compares rounded W1 with
W1, and the QAT column compares the QAT run with `ctrl`. Payload characters are a raw estimate: bits / 15 with no
Rice coding. The same formula gives W1 49,198, against its real 42,682.

| Spec | Raw payload chars | Post-training V2 | Post-training SPH13 | QAT 49M V2 | QAT 49M SPH13 |
| --- | ---: | ---: | ---: | ---: | ---: |
| float (W1 / ctrl) | 49,198 | 0 | 0 | 0 | 0 |
| `gen=int14,head=int14` (today's layout) | 49,198 | +0.01% | -0.01% | | |
| `gen=int12` | 42,826 | +0.02% | -0.02% | | |
| `gen=int8` | 30,082 | +3.3% | +3.5% | | |
| `gen=int6` | 23,709 | +16.0% | +16.6% | **+4.8%** | **+5.9%** |
| `gen=int4` (per-row scale) | 17,337 | +161% | +184% | +22.7% | +22.1% |
| `gen=nvfp4` (blocks of 16, E4M3 scales, 4/6) | 17,791 | +65% | +68% | **+15.8%** | **+18.5%** |
| `gen=nvfp4,head=nvfp4` (all 4-bit) | 16,097 | +73% | +79% | +19.4% | +21.3% |
| `tab=int8` (int8 tables) | | +0.8% | +1.1% | | |
| `act=u7,l1=int8p2` (int8 head) | | **+0.06%** | **-0.02%** | | |
| `tab=int8,act=u7,l1=int8p2` | | +1.0% | +1.2% | +1.0% | +1.0% |

Other measurements:

- **Without 4/6 (`nvfp4n`):** post-training rounding came out slightly better than with it (+63.5% against
  +64.8% on V2). The 4/6 selection doesn't help on these weights.
- **4-bit after QAT:** the curves were still falling at 49M rows, but they sit far from usable. Through the round-12
  calibration (6.8 Elo per 1% of loss) +16% is roughly -100 Elo. Block scales beat per-row scales (NVFP4 +15.8%,
  int4 +22.7%), as the post would predict.
- **int8 tables:** QAT recovered none of the loss in 49M rows. The likely cause is the lane step. One power-of-two
  step per lane is set by that lane's largest value across every row and table, so the step is coarse, and the
  rounding noise adds up over the roughly 11 rows in each accumulator.
- **int8 tables, speed:** `make_t8_runtime.py` writes a copy of `nnue_b64.hpp` with int8 T, F and DEC tables (mode
  2) and a control that keeps the same rounded values in int16 (mode 1). Both search identical trees: same node
  counts, and the evals are equal by construction. The test was `bench_ab nodes 300 11 4242`, 12 runs per build,
  alternated and pinned to one core, on a cloud Xeon with 8 MB L2 and 260 MB L3. Mean search time was 1.015 s for
  the int16 control and 1.038 s for int8, so **int8 was about 2% slower**, within noise to slightly worse. The
  widen-and-multiply costs more than the cache line saves on this host. A machine with a small L3 might differ.

## What this means for CrossFish

1. **The 4-bit generator isn't worth pursuing now.** Even NVFP4-native training is +16-19% worse after 49M rows.
   It would need far longer QAT on the GPU, and the payload isn't the binding limit anyway. W1's paste leaves
   17,809 characters, encoder width saturated past 128 (FINDINGS A21), and the book was dropped on play, not size.
   If the payload becomes binding again, int6 with QAT is the realistic middle ground (+5% at 49M rows, about half
   of W1's characters).
2. **int8 tables don't pay off.** They lose about 1% of loss that short QAT doesn't recover, and they weren't faster
   in a same-tree benchmark here.
3. **The int8 head is free in precision.** u8 0..127 activations and int8 first-layer weights with one
   power-of-two scale per output row leave held-out loss unchanged with no retraining (V2 +0.06%, SPH13 -0.02%). A
   per-tensor scale costs about 1%, so the scale has to be per row. That removes the "before precision loss"
   caveat from A20's +1 to +7 Elo estimate. What's left is whether a `vpmaddubsw` kernel beats today's sparse-pair
   `vpmaddwd` loop. That loop already skips about 55% of the pairs. A plain 4-input loop does about the same work,
   so a win needs Stockfish's two-quads-per-`vpmaddwd` accumulation in int16, which in turn needs saturation
   bounds. QAT can enforce those bounds if needed.

## Reproduce

```bash
D=path/with/e2b,sp13,sp14s   # e2b/{train,hold}.*, sp13/hold.*, sp14s.{x,lab,res,flags}.*
NET=datasets/nnue2/r16/nets/r16_x128_l2400_s1601.pt
python tools/experiments/lowprec/lowprec.py check --init $NET --data $D
python tools/experiments/lowprec/lowprec.py eval --init $NET --data $D --hold-n 200000 \
    --q none gen=int12 gen=int6 gen=nvfp4 tab=int8 act=u7,l1=int8p2
python tools/experiments/lowprec/lowprec.py train --init $NET --data $D --name q_gen_nvfp4 --q gen=nvfp4 \
    --steps 3000 --lr 3e-4 --out-dir OUT
python tools/experiments/lowprec/make_t8_runtime.py cpp_impl /tmp/cpp_t8
cd /tmp/cpp_t8 && for m in 1 2; do g++ -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread \
    -DT8MODE=$m -o bench_t8_$m bench_ab.cpp; done
./bench_t8_1 nodes 300 11 4242; ./bench_t8_2 nodes 300 11 4242   # same node counts; compare secs
```

`train` runs on CPU only. On the desktop, a `--device` switch to DirectML would be the first change to make.
