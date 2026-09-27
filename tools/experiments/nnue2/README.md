# Pattern-generator NNUE training (nnue2)

Offline trainers and analyses behind the NNUE that replaced HCE + MiniNet + macro as the engine's
evaluation: the pattern-generator net **B64_d5M_57ep** (35,243 parameters). Nothing here runs in the
engine. The integer inference, candidate builds, two-net matches and the Linux worker are in
[`../fast_nnue`](../fast_nnue/README.md). The round-by-round write-ups these notes are distilled from stay
with the data in `datasets/nnue2/fast/*.md` (gitignored).

Every tool finds its data from the repository root, so it runs from any working directory. Relative
paths given on the command line are relative to the working directory, except the trainers' `--extra`,
which is relative to the root like its default. Training needs PyTorch; the recorded runs used the
DirectML build in `toolchains/py312-dml` (`--device dml`, the default). Pass `--device cpu` or `--device
cuda` elsewhere. `tools/eval_data.py` and `tools/nnue_train_blend.py` supply the record format, the eval2
loader and the game-level holdout.

## Data and holdouts

| set | what | rows |
| --- | --- | ---: |
| eval2 | `datasets/eval2/cf_play_d14.cfdg` + `uttt_sp_d14.cfdg`: positions with depth-14 search labels ([`documentation/eval_data.md`](../../../documentation/eval_data.md)) | 4.82M |
| V2 | holdout: 10% of the eval2 games (`nnue_train_blend.game_holdout`). The shipped static eval's loss on it is 0.046817. | 482,136 |
| d8 | `datasets/nnue2/d8_{a,b,laptop,laptop_b}.cfdg`: self-play that labels itself at depth 8, `datagen play OUT 64000000 d8 THREADS SEED datasets/eval2/uttt_openings.txt` (the laptop files 24M) | 176.0M |
| D8H | holdout: 1% of the d8 games (splitmix64 of (file, game) % 100), never streamed, scored without the games inside d8_a's first 5M records. Shipped loss 0.039486. | 1,708,755 |
| parity | `datasets/nnue2/fnn1/parity_in.cfdg`: 20,000 fixed eval2 positions, the parity sample of every export and check | 20,000 |

The target is sigmoid(search / 1600). The loss is the MSE of that win probability, reported as a change
against the shipped static eval on the same rows. The standard training set ("d5M") is eval2's training
rows plus the labeled rows of d8_a's first 5M records: 9,340,285 rows, 570 steps of 16,384 per epoch.

## Files

| file | role |
| --- | --- |
| `probe.py` | The first round. It trains nets that predict the whole eval: `r10` (199 per-cell features, the round-ten net, A=256, 2A->32->1) and `pat` / `patact` (one 3^9 pattern table per miniboard, 22.7M parameters). It also runs the D4 symmetry checks (`symcheck-prep` / `symcheck-compare`). Shared data code: `load_data`, `compact`, the D4 maps. |
| `gen_nnue.py` | The B design: a small encoder (27->64->64->32, shared) generates each miniboard's pattern rows, then projections per location and for the forced board, A lanes + 1 PSQT lane, clipped ReLU, 2A->16->32->1. `train --run NAME,A,DENSE,LR[,epochs=N,...]` on eval2 + `--extra` (default the first 5M records of d8_a), in RAM. `check` runs the unit checks and `bench` the throughput. |
| `gen_nnue_stream.py` | The same model streamed over all 176M d8 records plus eval2. `survey` scans every record and writes the D8H rows and hashes, `dups` finds duplicates and leaks up to symmetry, `bench` and `check` test the loader, `train` trains (one epoch = one pass; `e2rep=`, `steps=`), and `eval` scores V2 and D8H by ply and on novel rows. |
| `buckets.py` | V2 metrics per bucket (ply, capture available, large labels, calibration) for any run, `shipped` or `hce`. |
| `summarize.py` | One line per run in `datasets/nnue2/probe/*.json`: loss vs shipped and vs F_full, curve. |
| `eval_models.py`, `novelty.py`, `sym_shipped.py` | First-round analyses: the holdout loss on all, non-mate and mate rows; which holdout positions also occur in training (up to symmetry); the shipped eval averaged over the 8 symmetries. |
| `long_gap.py` | The loss on rows a net trained on, next to unseen rows from the same source. This separates overfitting from bad optimization. |
| `quant_loss.py` | The holdout loss of the engine's quantized eval (a candidate's `datagen label`) next to the float net's. |
| `verify64.py` | `export_bgn.py verify` in float64: is a BGN1 file exact when PyTorch float32 itself drifts? |
| `export_fnn1.py` | `probe.py` r10 checkpoint -> FNN1 float file (the per-cell net's engine format), plus a parity check. |

Outputs go to `datasets/nnue2/probe/<NAME>.{pt,json,log}` (checkpoint of the best V2 epoch, history, a
timestamped log with step lines and ETAs), `datasets/nnue2/stream/` (D8H, hashes) and merged result
files in `datasets/nnue2/probe/` (`buckets.json`, `stream_eval*.json`, `quant_loss.json`, ...).

## Reproduce

```bash
PY=toolchains/py312-dml/Scripts/python.exe          # any Python with torch + numpy
N=tools/experiments/nnue2
$PY $N/gen_nnue.py check                            # unit checks (seconds)
# the shipped net: the lr1e2 recipe (AdamW 1e-2, 2% warmup, cosine to 1e-5, batch 16,384, D4 aug, K 1600,
# seed 1) for 57 epochs on d5M (14 min on the desktop's GPU)
$PY $N/gen_nnue.py train --run B64_d5M_57ep,64,16x32,1e-2,epochs=57
$PY $N/gen_nnue.py train --run B64_lr1e2,64,16x32,1e-2 --run B128_lr1e2,128,16x32,1e-2   # 10-epoch baselines
# the per-cell net of stage 1, and its engine file
$PY $N/probe.py train --arm r10 --aug --name r10_aug_d8x5M_lr6e3 --extra datasets/nnue2/d8_a.cfdg \
    --extra-rows 5000000 --epochs 10 --lr 6e-3 --save
$PY $N/export_fnn1.py export datasets/nnue2/probe/r10_aug_d8x5M_lr6e3.pt datasets/nnue2/fnn1/r10_aug_d8x5M_lr6e3.bin
# all the d8 data (survey once: D8H and hashes; 3.4 min)
$PY $N/gen_nnue_stream.py survey && $PY $N/gen_nnue_stream.py dups --out dups_sym
$PY $N/gen_nnue_stream.py check --batches 130 --tag smoke           # stream rows == their records (20 s)
$PY $N/gen_nnue_stream.py train --run B64_all,64,16x32,1e-2 --passes 3
$PY $N/gen_nnue_stream.py train --run B64_all_e2x10,64,16x32,1e-2,e2rep=10,steps=32697
# scoring
$PY $N/gen_nnue_stream.py eval B64_d5M_57ep --out stream_eval.json     # V2 + D8H, by ply, novel rows
$PY $N/buckets.py B64_d5M_57ep B64_lr1e2 && $PY $N/long_gap.py B64_d5M_57ep && $PY $N/summarize.py
$PY $N/quant_loss.py B64_d5M_57ep=cpp_impl/bin/cand_full_B64_d5M_57ep   # after export (fast_nnue)
```

The other recorded runs change only these parts: `epochs=114` / `200` (long), `5e-3` (the lr-5e-3 run),
`--extra "" --extra-rows 0 ...,epochs=246` (the eval2-only control), `--passes 10` (B64_all_10p), and
the 10-epoch lr sweep `1e-3`, `3e-3`, `6e-3`, `2e-2`. Each run's exact arguments are in its
`<NAME>.json` (`args`). The first round's arms are `probe.py train --arm r10|pat [--aug] [--train-rows N]
[--extra ... --extra-rows 5000000]`. Exporting a checkpoint for the engine, its candidate build and its
games are covered in [`../fast_nnue/README.md`](../fast_nnue/README.md). The candidate builds (and so
`quant_loss.py`) patch the pre-NNUE engine; that README says how to get a tree with it.

## Results

Held-out loss against the shipped eval (lower is better), then the correlation with the search label on
non-mate rows. The Elo column is 20 ms per move against B64_d5M_57ep, from the round robins in
`../fast_nnue`.

| net | params | data, schedule | V2 | D8H | Elo vs B64_d5M_57ep |
| --- | ---: | --- | --- | --- | ---: |
| r10 (per-cell, round ten's design) | 67,649 | eval2, 5 epochs | -14.37% | | |
| pat (direct pattern tables) + D4 aug | 22.7M | eval2, 8 epochs | -28.17% | | |
| r10_aug_d8x5M_lr6e3 (stage 1's per-cell net) | 67,649 | d5M, 10 ep, lr 6e-3 | -39.56% | | |
| B64_lr1e2 (first CodinGame net) | 35,243 | d5M, 10 ep | -47.43% / 0.779 | -42.85% / 0.741 | -29.7 ± 5.0 |
| B128_lr1e2 | 61,483 | d5M, 10 ep | -49.89% / 0.785 | -45.33% / 0.747 | |
| **B64_d5M_57ep (shipped)** | 35,243 | d5M, 57 ep | **-51.41% / 0.788** | **-47.28% / 0.752** | 0 |
| B128_d5M_57ep | 61,483 | d5M, 57 ep | -51.77% / 0.789 | -47.44% / 0.753 | -6.3 ± 5.0 |
| B64_all | 35,243 | all 178.6M rows, 3 passes | -47.83% / 0.776 | -46.75% / 0.753 | (-44.0 ± 11.1 h2h) |
| B128_all | 61,483 | all, 3 passes | -50.94% / 0.785 | -50.09% / 0.763 | |
| B64_all_e2x10 | 35,243 | all, eval2 x10 | -49.66% / 0.781 | -47.29% / 0.752 | (-19.8 ± 10.5 h2h) |
| B64_d5M_114ep | 35,243 | d5M, 114 ep | -50.22% / 0.787 | -45.94% / 0.749 | -9.9 ± 5.0 |
| B64_d5M_200ep | 35,243 | d5M, 200 ep | -48.76% / 0.780 | -44.56% / 0.742 | -20.9 ± 5.0 |
| B64_d5M_114ep_lr5e3 | 35,243 | d5M, 114 ep, lr 5e-3 | -51.33% / 0.789 | -46.97% / 0.752 | -1.7 ± 5.0 |
| B64_e2only_114st | 35,243 | eval2 only, equal steps | -49.16% / 0.785 | -40.66% / 0.731 | -18.8 ± 5.1 |
| B128_d5M_114ep | 61,483 | d5M, 114 ep | -49.06% / 0.780 | -44.73% / 0.741 | -27.3 ± 5.0 |

The Elo figures are rating differences in `nnue_long_20ms` (9 engines, 72,000 games). The "h2h" ones are
head-to-head results in `nnue_full_20ms`, whose best engine was also B64_d5M_57ep (+318.1 ± 6.7 against
the shipped eval, 2,000 games per pair).

What the rounds found:
- **The generator beats the per-cell net and the direct tables.** At d5M and 10 epochs, B64 reaches
  -47.4% against -39.6% for the per-cell net, with half its parameters. The 22.7M-parameter pattern
  tables stop at -28%: they have too many rows for the data.
- **Training steps, not data, limited the lr1e2 nets.** Five times the steps on the same 9.34M rows (57
  epochs) gains 2-4 points of V2 and +30 Elo for B64 (+36.3 ± 14.3 in the 90 ms SPRT against B64_lr1e2,
  PASS at 864 games). Streaming all 176M d8 rows for the same steps gains less (B128) or nothing (B64). Ten
  passes over them add nothing either.
- **57 epochs at lr 1e-2 is the best point.** Longer high-lr training damages the net: fewer active
  lanes, first-layer rows 1.5-2.6 times larger, worse loss on its own training rows. It is not
  overfitting. At lr 5e-3 a 114-epoch run lands on B64_d5M_57ep's loss and ties it in play, but it is 11%
  slower per node.
- **The d8 rows matter.** With the same steps on eval2 alone, the net memorizes eval2 (holdout gap 0.00084
  against at most 0.0002) and loses 5 points of D8H.
- **B-128 does not pay at 20 ms.** It gains 0.36 points of V2 over B-64 at 57 epochs but is about 15%
  slower. B128_d5M_57ep rated 6.3 ± 5.0 below B64_d5M_57ep.
- **The quantized engine eval loses nothing measurable.** Its holdout loss equals the float net's to
  within 3e-6 for every net (`quant_loss.py`).

## Notes

- **Duplicates and leakage.** 12.1% of the D8H rows (13.7% up to symmetry) are positions that also occur
  in the d8 training rows, mostly at plies 0-9. 22.9% of the V2 positions occur in the d8 rows up to
  symmetry. The "novel" columns of `gen_nnue_stream.py eval` exclude them, and the rankings do not change.
- **DirectML.** A holdout prediction once failed with "The parameter is incorrect"; `evaluate()` now
  retries. Two small nets share the GPU at about 1.16 times one net's throughput.
- **Reading old logs.** The recorded logs name their queue scripts (`run_*.sh`). Those scripts were folded
  into the commands above; `datasets/nnue2/ship/curate.md` maps every removed file.
