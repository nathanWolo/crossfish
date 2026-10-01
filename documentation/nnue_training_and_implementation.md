# NNUE training and runtime implementation

Since 2026-09-27 (improvement log section 56) Crossfish's whole evaluation is
one small NNUE: a pattern-generator net with 35,243 parameters, integer and
incremental at run time, shared by the local engines (`crossfish_dev.hpp`,
`crossfish_prev.hpp`) and the CodinGame bot. The shipped net is **r13w_20**
(since 2026-10-01, section 64): r12_M2's weights fine-tuned for 2.4G rows on
round thirteen's data. Before it, **r12_M2** (2026-09-28, section 57) was the same
architecture trained on data that the NNUE engine labelled itself, and the
first shipped net was **B64_d5M_57ep**. The NNUE replaced
the hybrid that shipped from round seven: a handcrafted evaluation (HCE) plus
a D16/H8 MiniNet residual plus a macro residual.

- **Part I** (sections 1-10) describes the NNUE: architecture, data,
  training, quantization, payload, runtime, search integration, tests and
  how to ship a new net.
- **Part II** (sections 11-26) describes the hybrid it replaced, as it
  shipped. Parts of it still run: the macro net is the macro correction
  history's prior, datagen still records the HCE, and the old data formats
  and tools remain.

The experiments behind the NNUE have their own READMEs:
[`tools/experiments/nnue2/`](../tools/experiments/nnue2/README.md) (trainers,
data, results) and [`tools/experiments/fast_nnue/`](../tools/experiments/fast_nnue/README.md)
(the integer inference as it was developed, candidate builds, exactness
checks, two-net matches, the Linux worker). Their round-by-round write-ups
stay with the data in `datasets/nnue2/` (not in the repository).

# Part I. The pattern-generator NNUE

## 1. The evaluator in one picture

```text
per perspective P (both are kept; the side to move's comes first in the head)

  live miniboard m, pattern p (3^9)  -> T[m][p]         generated: proj_m(enc(p))
  decided miniboard m (won/lost/drawn) -> DEC[3m + s]
  bias                                -> BIAS
  ------------------------------------------------ stored accumulator, 64 lanes + 1 PSQT lane
  + constraint c (0..8 forced, 9 free) -> CON[c] (P to move) or CON[10 + c]
  + forced board's pattern (c < 9)     -> F[p_c]        generated: fwd(enc(p_c))
  ------------------------------------------------ at evaluation time

  clamp(acc_stm[:64]) ++ clamp(acc_other[:64]) -> 16 -> clamp -> 32 -> clamp -> 1
  eval = 1000 * (out + (psqt_stm - psqt_other) / 2)        side to move's view
```

Qsearch stands pat on this plus the structural correction history; interior
pruning uses it plus all three correction histories (section 7).

## 2. Architecture

**Features.** A miniboard's pattern is `p = sum_i cell_i 3^i` over its nine
squares, with 0 empty, 1 the perspective's stone, 2 the other side's. The
opponent's view swaps 1 and 2. Only the 11,093 patterns with no line and not
full occur on a live board; decided boards contribute their state (won by P,
won by the other, drawn) instead, and their stones are ignored, as they are in
the transposition key.

**The generator.** Rather than learning 9 x 3^9 free rows (a first probe of
such direct tables had 22.7M parameters and stopped at -28% held-out loss:
too many rows for the data), every pattern row is computed by a small
network:

| Part | Shape | Parameters |
| --- | --- | ---: |
| encoder: one-hot 27 (3 states x 9 squares) -> 64 -> 64 -> 32, ReLU between | shared by all rows | 8,032 |
| projection per location: `T[m][p] = enc(p) @ proj_w[m] + proj_b[m]` | 9 x (32 x 65 + 65) | 19,305 |
| forced-board projection: `F[p] = enc(p) @ fwd_w + fwd_b` | 32 x 65 + 65 | 2,145 |
| bias, decided rows (27), constraint rows (20: 10 per side) | 48 x 65 | 3,120 |
| head: 128 -> 16 -> 32 -> 1 | | 2,641 |
| **total** | | **35,243** |

Lane 64 of every row is a PSQT lane: it bypasses the head and enters the
output directly, so the accumulators carry a linear material-style term for
free.

**Why this shape.** The same data and epochs gave -47.4% held-out loss for
B-64 against -39.6% for a per-cell net with twice the parameters (improvement
log section 53's 199-feature design). A pattern row sees the whole 3 x 3 geometry at once, which
a per-cell first layer has to rebuild from data. B-128 (61,483 parameters)
fitted 0.4 points better and played 6 Elo worse at 20 ms, being about 15%
slower.

## 3. Training data

`tools/nnue_train_blend.py` and `tools/eval_data.py` supply the 128-byte
record format and loaders ([eval_data.md](eval_data.md) section 1).

| Set | What | Rows |
| --- | --- | ---: |
| eval2 | section 55's positions: 3.5M from crossfish self-play, 1.32M from uttt.ai, depth-14 labels | 4.82M |
| d8 | `datagen play OUT N d8 THREADS SEED datasets/eval2/uttt_openings.txt`: self-play at a fixed depth of 8 whose root scores are the labels ([eval_data.md](eval_data.md) section 2), about 2.5 hours on two machines | 176M |
| **d5M** (the training set) | eval2's training games + the labeled rows of the first 5M d8 records | 9.34M |
| V2 (holdout) | 10% of the eval2 games | 482,136 |
| D8H (holdout) | 1% of the d8 games, never streamed | 1.71M |

**Round twelve (r12_M2, improvement log section 57)** replaced the labels
with the NNUE's own:

| Set | What | Rows |
| --- | --- | ---: |
| eval2, relabelled | the same eval2 positions, labelled by a depth-14 search of the shipped NNUE engine (`datagen label`) | 4.82M |
| sp13 | NNUE self-play at a fixed depth of 13 (`datagen play ... d13`), same openings and randomisation as d8 | 13.44M |
| **r12_M2's training set** | relabelled eval2's training games + the first 5M non-SPH sp13 rows | 9.34M |
| SPH (holdout) | 3% of the sp13 games, never trained on | 402,405 |

The new labels predict game results better than the old ones (V2 log loss
0.4377 against 0.4593). Replacing the old engine's depth-8 rows with NNUE
self-play mattered as much as the relabelling: new labels alone gave about
-5% V2 loss, the self-play rows about another -4%, and +25 Elo more at 20 ms.

**Round thirteen (r13w_20, improvement log section 64)** generated far more
self-play and relabelled everything with the current engine (the section 62
search, net r12_M2):

| Set | What | Rows |
| --- | --- | ---: |
| sp13 (round thirteen) | depth-13 self-play of the current engine on three machines | 79.8M rows (77.4M train, 2.39M SPH13; 80.0M records) |
| e2b | eval2 relabelled at depth 14 by the current engine | 4.34M train |
| dumpb | the SPRT dumps' positions (30 files) labelled at depth 13 | 3.49M |
| SPH13 (holdout) | 3% of the sp13 games per shard | 2,388,278 |
| V2 (holdout) | round twelve's V2 rows, e2b's labels | 482,136 |
| DUMPH (holdout) | 3% of the dump games per file | 106,495 |
| LADH (holdout) | positions from 2,939 saved CodinGame games (ply 9+, deduplicated up to symmetry and colour, plus one-move children), depth 14 | 193,320 |

r13w_20 (and r13w_11 before it) trained on e2b (46% of the rows seen) and
sp13 (54%); the dumps and
round twelve's self-play were in the search space but not in the winner's
mix. No training row is a LADH position.

Mates are clamped to +/-20,000 in both label sources. The d8 labels come from
the process that played the moves, so the old free-move loader bug (improvement
log section 55) cannot reach them. Label with a `datagen` built from an unpatched Dev:
since the NNUE shipped, `datagen`'s `static_eval` column is the NNUE, and its
`hce` column is still the HCE.

## 4. Training recipe

`tools/experiments/nnue2/gen_nnue.py train --run B64_d5M_57ep,64,16x32,1e-2,epochs=57`:

- **Target and loss.** `sigmoid(search / 1600)` and the squared error of the
  predicted win probability `sigmoid(eval / 1600)`. K = 1600 is the fitted
  scale of the depth-14 labels (improvement log section 55).
- **Augmentation.** The 8 symmetries of the board (D4), applied to all nine
  miniboards and their positions together.
- **Optimizer.** AdamW, peak lr 1e-2 (no weight decay on the first layer,
  1e-5 on the head), 2% linear warmup, cosine decay to 1e-5, batches of
  16,384, seed 1.
- **Length.** 57 epochs of 570 steps (32,490 steps): 14 minutes on the
  desktop's GPU through DirectML.
- **Result.** V2 -51.41% and D8H -47.28% against the old static eval's loss
  on the same rows; correlation with the search label 0.788 on non-mate rows.

What the recipe search found (improvement log section 56):

- **Steps, not data.** Streaming all 176M d8 rows for the same number of
  steps was worse (-40 Elo for B-64); ten passes did not help.
- **57 epochs at lr 1e-2 is the best point.** 114 and 200 epochs were worse
  (-10, -21 Elo), and not from overfitting: the long high-lr phase leaves
  fewer lanes in the linear range and larger first-layer rows. At lr 5e-3,
  114 epochs tie it.
- **The d8 rows matter.** eval2 alone for the same steps memorizes eval2 and
  loses 19 Elo.
- **Held-out loss did not rank play reliably** here either; every decision
  was made on 20 ms round robins and confirmed at 90 ms.

**r12_M2** used the same model, loss, augmentation, optimizer and length
(32,490 steps at batch 16,384, lr 1e-2), on round twelve's training set
(section 3). Its trainer, `gen_r12.py`, imports all of that from
`gen_nnue.py` unchanged. It adds memory-mapped data mixes and a fixed step
budget, and it scores V2 with both label sets and SPH every epoch. It lives
with the round's data tools in `datasets/nnue2/r12/tools/` and is not yet in
the repository:

```bash
gen_r12.py train --run r12_M2,64,16x32,1e-2 --e2 new --sp 5000000
```

- **Loss.** V2 (new labels) -9.6% and SPH -8.5% against B64_d5M_57ep on the
  same rows.
- **Chaotic.** A control (M0) that retrained B64_d5M_57ep's exact data and
  recipe with the targets rounded one bit differently finished 1.7% worse on
  every holdout. So single-run gaps of 1-2% are weak evidence. Seed-2
  replicates of the round's candidates landed within 0.1-0.7%.
- **Twice the steps did not help.** The training loss kept falling past
  32,490 steps, but the held-out loss did not. At 20 ms, r12_M2 at 64,980
  steps tied r12_M2 (+1.6 +/- 9.1 head to head), and the same doubling cost
  the all-self-play mix about 11-16 Elo.
- **All 13M self-play rows (M3) were no better than 5M (M2).** They were
  better on SPH, worse on V2, and 2-4 Elo lower at 20 ms.

**r13w_20** is a warm start: r12_M2's weights (`--init r12_M2`, bit-exact)
fine-tuned by `datasets/nnue2/r13/tools/gen_r13.py`, a streaming trainer
over the packed sources, as trial 20 of the Optuna study r13w (a wide space:
rows seen, batch, lr and schedule, warmup, floor, weight decay, K, result
blend, source mix, PSQT weight, initialisation):

```bash
gen_r13.py train --name r13w_20 --init r12_M2 --lr 2e-3 --sched cosine --warmup 0.01 --lr-floor 1e-5 \
  --batch 16384 --steps 146484 --k 1600 --lam 0 --mix e2b=0.46,sp13=0.54 --v2-src e2b --dumph-src dumpb
```

- **Length.** 2.4G rows = 146,484 steps of batch 16,384 (about 254 passes
  over e2b, 17 over sp13), 73.5 minutes on the GPU; same loss, D4 augmentation
  and AdamW (a tensor-argument variant that does not leak under DirectML),
  seed 1.
- **Result.** Objective +22.9 Elo estimated against r12_M2 (SPH13 +23.0,
  LADH +24.5, V2 +24.0, DUMPH +18.1); held-out loss on SPH13 4.0% below the
  float r12_M2 net's. In games +16.6 +/- 5.3 at 20 ms, +20.2 +/- 12.4 at
  CodinGame compute, +16.0 +/- 10.9 as paste files (section 9). Trial 21,
  the same recipe at lr 1e-3, scored +22.94 against +22.92: the learning
  rate does not matter at this length.
- **What the search found.** Random-init retraining of the round-twelve
  recipe on the new data only matched r12_M2 (-2.3 offline, -13.8 +/- 5.5 in games);
  longer fine-tunes from r12_M2 kept improving (100M rows +11.5, 300M +14.5,
  600M +18.3, 1.2G +20.2, 2.4G +22.9; the 1.2G net, r13w_11, was ship-tested
  first and is the runner-up: SPH13 3.5% below the float r12_M2 net, +11.8
  +/- 11.0 as paste files); B-128 nets were +13 offline but tied r12_M2 in games (their
  tables cost 16-31% of the nodes); model soups of the warm starts were below
  the longest one alone. SPH13's labels are r12_M2's own search, so the warm
  starts were judged on V2 / LADH and in games.

## 5. Export and quantization

**Export.** `tools/experiments/fast_nnue/export_bgn.py export B64_d5M_57ep OUT.bin --perm`
writes the checkpoint as a BGN1 file: the baked float tables, the head and the
generator. `--perm` reorders the 64 lanes so that lanes which fire together
share an activation pair of the sparse head kernel (a maximum-weight matching
on co-activation), which cuts the nonzero pairs per evaluation from 37.7 to
28.6 without changing the function: the quantized evals are bit-identical.

**Integer scales.** Every table is quantized to a power of two:

| Scale | What | r13w_20 (and r13w_11) | r12_M2 | B64_d5M_57ep |
| --- | --- | ---: | ---: | ---: |
| `B64_QA` | the 64 accumulator lanes (int16) | 2^9 | 2^9 | 2^9 |
| `B64_QPS` | the PSQT lane (int16, separate arrays) | 2^12 | 2^12 | 2^13 |
| `B64_QB` | first dense layer weights (int16), sums int32 | 2^13 | 2^13 | 2^13 |
| `B64_Q2` | second dense layer | 2^13 | 2^13 | 2^13 |
| `B64_QO` | output weights | 2^10 | 2^11 | 2^10 |

`QA` is the largest value for which every row, every stored accumulator and
every accumulator plus constraint row plus forced-board row stays inside
int16 over every position the features can express. The bound is exact per
board (the extremes of each board's rows by the pattern's stone difference)
and is combined under the stone balance of a real game (the side to move has
as many stones as the other side or one fewer; search never passes). The
other scales are the largest that keep their layer's int32 sums bounded.
Hidden layers shift back with rounding to nearest; the output truncates
toward zero, as the float reference does. Nothing is clipped, so int16
arithmetic that wraps in between still ends exact.

**Error.** On 20,000 fixed positions the integer engine is 4.73 eval units
from the float net on average (max 164; the float evals' standard deviation
is 2,881), mostly at late plies. Its held-out loss equals the float net's to
within 3e-6.

**The payload.** The source carries the generator, not the tables: 35,243
parameters, rounded with GPTQ and a least-squares refit, one bf16 scale per
row, a bit width per group and Rice coding, in 30,923 CJK14 characters
([minification.md](minification.md) section 3.1).
`tools/nnue_emit_b64_header.py NET.bin --label NAME` writes
`cpp_impl/nnue_b64_net.hpp` from the lane-paired export and derives the five
scales itself (a port of the loader's bound). The payload's own rounding keeps
the bot at 4.72 / 163 from the float net, the same as the unrounded net.
r13w_20's payload is 53,834 bytes = 28,712 U15 characters (r13w_11: 53,927 =
28,762; r12_M2: 54,159 = 28,885); its rounding moves the float eval by 1.97
mean / 66.7 max units on the 20,000 parity positions (r13w_11: 1.69 / 41.8).

## 6. Runtime (`cpp_impl/nnue_b64.hpp`)

**Start-up.** `b64::load()` decodes the payload into the generator, runs the
encoder on the 11,093 live patterns, projects each through the nine location
projections and the forced-board one, and quantizes every row as it is made:
25.6 MB of static int16 tables in about 50 ms. The float operations run in a
fixed order without FMA, so every compiler and flag set bakes the same floats;
the unit tests pin the hashes of all 16 integer tables. `load()` uses a plain
ready flag, which the single-threaded bot needs; the engines call it through
one shared once-flag (`crossfish_nnue_load_once`) because the harnesses build
engines on many threads at once.

**Accumulators.** Both perspectives are kept per absolute player, so a move
never swaps them. A move in a live miniboard subtracts the board's old
pattern row and adds the new one, per perspective (6.6 ns); a move that
decides it swaps the pattern row for a decided row. The constraint and
forced-board rows are added at evaluation time and never stored.

**The lazy stack** (`b64::Stack`, one per engine):

- `on_make` in `make_move_fast(FastBoard&)`, just before `n_moves++`, only
  records the move (board, square, mover, decided state, the board's markers
  before, the parent's and the child's `tt_hash`) and prefetches the rows the
  update and the evaluation will read. Unmake does nothing.
- An evaluation walks back from the current ply while the recorded moves lead
  to the position it needs, and replays forward from the first entry that
  holds a parent. Siblings overwrite deeper entries, which is why entries are
  keyed by the position they hold: an entry is reused only for its own
  position.
- The empty board's `tt_hash` is 0, so an entry holding no position is marked
  `kNoKey`, not 0.
- A chain that reaches no valid entry refreshes from scratch, so a search
  entered without `refresh_root` is still exact, only slower. `getMove` and
  `search_fixed_depth` call `refresh_root` at the root.
- `evaluate_keyed` puts a 2^14-entry direct-mapped cache in front, keyed by
  `tt_hash` (which determines every feature). A third to a half of all
  evaluations hit it.

**Kernels.** AVX2 throughout: the accumulator add of the constraint and
forced rows, a clamp to [0, 2^QA], a 64-bit mask of the nonzero activation
pairs, and the first dense layer as `_mm256_madd_epi16` over those pairs
only; the 16 -> 32 -> 1 layers are dense. About 43 ns per uncached
evaluation. On the CodinGame build `eval_avx` and `Stack::sync` are real
calls (one per uncached evaluation); the per-move and per-evaluation helpers
are `always_inline`.

**Memory.** Each engine carries about 290 KB of stack and cache besides the
shared tables.

## 7. Search integration

The same wiring is in Dev, Prev and `codingame_nnue.cpp` (`make -C cpp_impl
port-check` proves the bot searches exactly like Dev):

- **Qsearch** stands pat on `NNUE + structural correction`. The HCE fail-high
  shortcut and the MiniNet/macro upper bound of Part II are gone.
- **Interior static eval** (reverse futility, futility) is the NNUE corrected
  by all three correction histories, which now learn the NNUE's errors. The
  depth-1 reverse-futility MiniNet check is gone. Pruning still stays away
  from mate-range bounds (improvement log section 51).
- **Make** keeps only the HCE's threat maps, which `has_immediate_global_win`
  and `has_forced_global_win_after_reply` read; the local and global HCE
  scores are no longer updated during search.
- **`evaluate(GlobalBoard&)`** (tests, `datagen label` depth 0) is
  `b64::evaluate_board`, computed from scratch; under `g_force_hce_eval` it
  still returns the HCE.
- **The macro net** stays as the macro correction history's prior
  (`corr_macro_entry`), and the HCE, MiniNet and macro code still compile in
  the engines for the tools that read them. The CodinGame bot drops the HCE
  evaluators and the MiniNet entirely and keeps three helpers of the MiniNet
  header in `d16_helpers.hpp`.

## 8. Tests

- `unit_tests` (`make test`):
  - `nnue_tables_match_verified_build`: the 16 table hashes and the five
    scales of the build checked with CodinGame's compiler.
  - `nnue_fixed_positions`: 16 positions (decided and drawn boards, forced
    boards, free moves) through `evaluate_board`, an independent scalar int64
    reference and Dev's `evaluate()`, against the verified bot's own evals;
    and `g_force_hce_eval` still gives the HCE.
  - `nnue_incremental_matches_scratch`: about 24,000 evaluations along a
    search-like walk from 300 random roots (siblings overwriting entries,
    evaluations before and after children, roots without `refresh_root`, a
    new stack at the empty board): keyed == incremental == from scratch ==
    scalar.
- `tools/test_nnue_emit_b64_header.py` (numpy only): the committed header
  decodes to the verified payload, its scales are the ones the net needs, the
  Python bake gives the pinned table hashes, and pack/unpack round-trips.
- `python tools/nnue_emit_b64_header.py --check [HEADER]` decodes any header,
  bakes it in float32 in `load()`'s order, derives the scales and prints the
  table hashes.
- `make -C cpp_impl port-check`: the bot against Dev at depths 5, 7, 9.

(`test_bots verify`'s "nnue incremental vs refresh" line checks the older
sparse `nnue.hpp` path, not this net.)

## 9. Strength and speed

**r13w_20** (improvement log section 64) against r12_M2, fixed-length games:
- the two CodinGame paste files through the protocol at 90 ms (`cg_match.py`, 5 referees, forfeit 1,000 ms): **N=1000, 321-404-275, +16.0 +/- 10.9 Elo**, pentanomial 4/80/296/106/14, 0 forfeits, one late reply (main's);
- the candidate harness at 20 ms on the ThinkPad: +16.6 +/- 5.3 (13,000 games per net);
- the Dell at its CodinGame compute (62 ms x 3 threads): +20.2 +/- 12.4 (1,000 games).

**r13w_11** (the 1.2G-row fine-tune, ship-tested first) the same three ways: paste files **N=1000,
317-400-283, +11.8 +/- 11.0**, pentanomial 9/75/301/103/12, 0 forfeits; 20 ms +11.0 (+11.4 +/- 5.5 in
the earlier 12,000-game fit); the Dell +13.6 +/- 12.6 and +9.4 +/- 12.8 in two runs.

Same architecture, so the same speed; `port-check` IDENTICAL at depths 5, 7 and 9. A
CodinGame-scaled SPRT net against net is not possible in the sprt harness (Dev and Prev
share the net header).

**r12_M2** (improvement log section 57) against B64_d5M_57ep:
- the official 90 ms SPRT: **N=504, 173-259-72, +70.6 +/- 19.7 Elo**, LLR +3.03 (H0=0, H1=+5), 0 timeouts;
- 3,000 fixed-length games of the two CodinGame paste files: +52.5 +/- 7.0;
- round twelve's 3,000-game match on the pre-NNUE search: +57.7 +/- 7.6.

Same architecture, so the same speed (Prev 14.63M, Dev 14.42M nodes/s in the SPRT header).

**B64_d5M_57ep**, the first NNUE: official 90 ms SPRT against the hybrid (the mate-window freeze), two pooled
shards: **N=420, 296-106-18, +276.6 +/- 30.4 Elo**, LLR +3.00 (H0=0, H1=+5),
0 timeouts. Against the hybrid it runs at 58-62% of the nodes per second
(`bench_ab nodes 400 9`), needs 87% of the nodes to reach depth 9, completes
about one ply less at 90 ms, and runs at 0.553 of the old paste file's
nodes/ms when both are built with CodinGame's flags. Improvement log section
56 has the full record, including the round robins that picked the net.

## 10. Shipping a new net

```bash
PY=toolchains/py312-dml/Scripts/python.exe    # torch for training and export; the emitter needs numpy only
$PY tools/experiments/nnue2/gen_nnue.py train --run NAME,64,16x32,1e-2,epochs=57
$PY tools/experiments/fast_nnue/export_bgn.py export NAME datasets/nnue2/fast/NAME_perm.bin --perm
$PY tools/experiments/fast_nnue/export_bgn.py verify datasets/nnue2/fast/NAME_perm.bin NAME
$PY tools/nnue_emit_b64_header.py datasets/nnue2/fast/NAME_perm.bin --label NAME
python tools/nnue_emit_b64_header.py --check  # new table hashes and scales for unit_tests.cpp
make -C cpp_impl cg-input && make test && make -C cpp_impl port-check
```

- **Update the pinned values in both test files.** In `cpp_impl/unit_tests.cpp`,
  the five `B64_Q*` checks and the 16 table hashes come from `--check`; take
  `want[16]` from the failing `nnue_fixed_positions` `CHECK_EQ` output after
  checking the new evals against the float net. In
  `tools/test_nnue_emit_b64_header.py`, update `PAYLOAD_SHA256`, the payload
  length, `SCALES` and `TABLE_HASHES` from the emitter and `--check` output.

- **Screen before shipping.** A new net is an eval change: screen it with a
  20 ms round robin against the current one and confirm it with the 90 ms
  SPRT. The candidate and pairing tools in `tools/experiments/fast_nnue/`
  patch the pre-NNUE engine (commit `c278cde`); for a net of the same shape,
  swapping `nnue_b64_net.hpp` in Dev's build is the simpler A/B, but Dev and
  Prev then need separately named copies of the runtime (the fast_nnue README
  explains why one shared header silently gives both engines one net).
- **Same shape only.** The runtime is specialised to B-64 with a 16 -> 32
  head; another shape needs `nnue_b64.hpp` changed with it, and the unit
  tests' expected values regenerated.
- **Calibration data.** The emitter's GPTQ and refit read
  `datasets/nnue2/d8_a.cfdg` (`--calib`), which is not in the repository.
  Its `--check` path is what CI runs.
- **Record** the checkpoint (`datasets/nnue2/probe/NAME.pt` and `.json`), the
  export's CRC, the payload sha256 (in the header's comment), the emitter
  command and the SPRT, as section 25 asks.

**Shipped nets** (the record section 25 asks for; the files are outside the
repository):

| Net | Since | Checkpoint | Export (`--perm`) | Emitter | Scales | Payload | Matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r13w_20 | 2026-10-01 (§64) | `datasets/nnue2/probe/r13w_20.pt` sha256 `eaf8c46b93bbbd03…`, `.json` (trial 20 of study r13w, `gen_r13.py`) | `datasets/nnue2/fast/r13w_20_perm.bin` CRC-32 `d18dccb2` | `tools/nnue_emit_b64_header.py datasets/nnue2/fast/r13w_20_perm.bin --label r13w_20` (defaults; GPTQ calibration `d8_a.cfdg`) | 9, 12, 13, 13, 10 | 53,834 bytes = 28,712 chars, sha256 `492a36eaf2f1c1011fdd5fb2533a4ee2416574dfb73346252511ef2accd11916` | paste files 90 ms N=1000 +16.0 +/- 10.9; 20 ms +16.6 +/- 5.3; Dell CodinGame compute +20.2 +/- 12.4 |
| r13w_11 | 2026-10-01 (§64; replaced by r13w_20 the same day, never submitted) | `datasets/nnue2/probe/r13w_11.pt` sha256 `8128258e168763b1…`, `.json` (trial 11 of study r13w, `gen_r13.py`) | `datasets/nnue2/fast/r13w_11_perm.bin` CRC-32 `75044cc4` | `tools/nnue_emit_b64_header.py datasets/nnue2/fast/r13w_11_perm.bin --label r13w_11` (defaults; GPTQ calibration `d8_a.cfdg`) | 9, 12, 13, 13, 10 | 53,927 bytes = 28,762 chars, sha256 `712039a021b912bc92503b7f66c70cd6079a798f7f1734e72be6108f45342047` | paste files 90 ms N=1000 +11.8 +/- 11.0; 20 ms +11.4 +/- 5.5; Dell CodinGame compute +13.6 +/- 12.6 |
| r12_M2 | 2026-09-28 (§57) | `datasets/nnue2/probe/r12_M2.pt` (`gen_r12.py`) | `datasets/nnue2/fast/r12_M2_perm.bin` CRC-32 `60f1f9ea` | `… r12_M2_perm.bin --label r12_M2` | 9, 12, 13, 13, 11 | 54,159 bytes = 28,885 chars, sha256 `4ba93b1a422c480c…` | 90 ms SPRT N=504 +70.6 +/- 19.7; paste files N=3000 +52.5 +/- 7.0 |
| B64_d5M_57ep | 2026-09-27 (§56) | `datasets/nnue2/probe/B64_d5M_57ep.pt` (`gen_nnue.py`) | `datasets/nnue2/fast/B64_d5M_57ep_perm.bin` | `… B64_d5M_57ep_perm.bin --label B64_d5M_57ep` | 9, 13, 13, 13, 10 | 30,923 CJK14 chars | 90 ms SPRT N=420 +276.6 +/- 30.4; N=3000 +267.4 +/- 11.9 |

# Part II. The hybrid evaluator it replaced (rounds seven to twelve)

This part describes the evaluator that shipped from round seven until the
NNUE (2026-09-13 to 2026-09-27), as it shipped: HCE, plus a D16/H8 MiniNet
residual, plus a macro residual. Its numbers (payload sizes, the
90,095-character paste file) are that ship's. The engines still contain its
code: the macro net is the macro correction history's prior, `datagen`
records the HCE, and the data formats and trainers below remain in `tools/`.

```text
position
  │
  ├─ incremental handcrafted evaluation (HCE)
  │
  ├─ local-pattern D16/H8 MiniNet residual
  │    └─ nine 3×3 boards, side-to-move relative
  │
  └─ compact macro-context residual
       └─ nine won/drawn/live classes + forced-board constraint

qsearch stand pat = HCE + local residual + macro residual
```

The hybrid was deliberately a sharp base with learned corrections: the HCE
stayed a fast, well-behaved baseline and the learned heads predicted
corrections to it. That was stronger and much easier to fit than asking a
small network to relearn all of the known Ultimate Tic-Tac-Toe structure,
until the pattern generator of Part I did so with far more data.

## 11. What “NNUE” means in this repository

Until the pattern-generator net of Part I, the name was historical: the
evaluator was not a Stockfish-style accumulator updated on every move.

Crossfish has experimented with several learned evaluators:

- a sparse 199-feature dual-accumulator network;
- the D8/H4 MiniNet that first shipped as an HCE residual;
- the D16/H8 MiniNet of this part;
- a separate macro-context residual head;
- a full Stockfish-style NNUE (199 features, 256-wide accumulator) that
  replaced HCE, MiniNet and macro together. It fitted holdout labels better
  and lost 193-296 Elo at equal depth; improvement log section 53 and
  `tools/experiments/full_nnue/` record it;
- the pattern-generator NNUE of Part I, which replaced all of the above.

The hybrid's local MiniNet was evaluated from the board at selected search nodes,
but its first layer is heavily preprojected, and the two pieces of its input
that change rarely (each miniboard's centroid code and the decided-miniboard
term) are kept up to date by make/unmake. The macro network is reduced to an
exact table lookup whose key is maintained incrementally. The HCE itself also
has an incremental local-board accumulator.

An experiment that maintained both local MiniNet perspectives on every
make/unmake was bit-exact but slower overall. Ultimate Tic-Tac-Toe evaluates
fewer nodes than it traverses, so paying update cost on every search edge was
more expensive than reconstructing the small preprojected network only when it
was needed.

## 12. The hybrid's score composition

Let:

- `H(p)` be the side-to-move-relative handcrafted evaluation;
- `R_local(p)` be the D16/H8 local MiniNet output;
- `R_macro(p)` be the macro-context output;
- `C(p)` be the search correction-history adjustment.

Three correction histories are learned online during search, each an exact
table of running static-eval errors: a structural one keyed by side to move,
forced miniboard and the decided-miniboard mask; one keyed by the shape of the
forced miniboard; and one keyed by the 18-bit macro state (improvement log
sections 14, 33 and 42). [hce_and_correction_history.md](hce_and_correction_history.md)
describes the HCE and these tables in detail.

The full public evaluator used by tests and probes is:

```text
E(p) = H(p) + R_local(p) + R_macro(p)
```

The qsearch stand-pat path additionally applies the structural correction
history only (the cheapest of the three):

```text
stand_pat(p) = H(p) + C(p) + R_local(p) + R_macro(p)
```

The code avoids running both learned heads when bounds already prove that their
exact value cannot matter:

1. If `H(p) + C(p) - 640 >= beta`, qsearch returns immediately.
2. If `H(p) + C(p) + MINI_MAX + MACRO_CLIP < alpha`, it uses that safe upper
   bound instead of executing the networks.
3. Otherwise it evaluates the local residual and performs the cached macro
   lookup.

`MINI_MAX` is 8000 and `MACRO_CLIP` is 2000 in the hybrid.

Interior reverse-futility pruning normally uses HCE corrected by all three
histories. At depth one, a selective second check evaluates the D16 local head
before pruning a borderline fail-high. The macro residual is not paid on that
interior pruning path. Reverse futility, futility and qsearch delta pruning do
not run against mate-range bounds at all, where a static eval says nothing
(improvement log section 51).

Relevant runtime files:

- `cpp_impl/crossfish_dev.hpp`
- `cpp_impl/crossfish_prev.hpp`
- `cpp_impl/codingame_nnue.cpp`
- `cpp_impl/mini_eval_d16.hpp`
- `cpp_impl/macro_eval.hpp`

The final two files are generated artifacts. Do not hand-edit their packed
weights.

## 13. Training-data format

### 13.1 `NNUEWDL1`

The local and macro trainers consume the repository’s `NNUEWDL1` binary format.
It starts with:

| Field | Size | Meaning |
| --- | ---: | --- |
| Magic | 8 bytes | ASCII `NNUEWDL1` |
| Record count | 8 bytes | little-endian `uint64_t` |

Each record is 101 bytes:

| Field | Size | Meaning after search labeling |
| --- | ---: | --- |
| UTTTAI state | 93 bytes | board, side to move, constraint, result |
| Static score | 4 bytes | little-endian `float32`, normally HCE |
| Teacher score | 4 bytes | little-endian `int32`, normally search score |

The 93-byte state is ASCII encoded:

| Byte range | Meaning |
| --- | --- |
| `0..80` | 81 local cells, nine consecutive cells per miniboard |
| `81..89` | nine super-board cells |
| `90` | side to move: `'1'` for player zero, `'2'` for player one |
| `91` | forced miniboard `'0'..'8'`, or `'9'` for free choice |
| `92` | game result field; unused by MiniNet feature extraction |

Local cell values are:

- `'0'`: empty;
- `'1'`: player zero;
- `'2'`: player one.

Super-board values are:

- `'0'`: live miniboard;
- `'1'`: won by player zero;
- `'2'`: won by player one;
- `'3'`: drawn.

The training loader rejects a dataset whose first score field still resembles
WDL probabilities (`mean(abs(y)) < 2`). This catches the most common pipeline
mistake, but it cannot detect every semantically bad label set.

### 13.2 Raw self-play data

Build the harness first:

```bash
make -C cpp_impl verify
```

Generate self-play positions:

```bash
cpp_impl/bin/test_bots dump nnue 8000 20 datasets/nnue_pos.bin
```

This starts every game with four to eight random moves, then lets the current
Dev engine play at the requested time. The ordinary `nnue` mode stores final
WDL in the float field and static HCE in the integer field. It is useful as a
position source, but it is not directly a search-score training set.

Generate random legal-game positions for broader coverage:

```bash
cpp_impl/bin/test_bots dump hce 2000000 datasets/nnue_hce_rand.bin
```

The fixed-depth search dumper expects files named `nnue_pos.bin` and
`nnue_hce_rand.bin` under `datasets/` unless equivalent paths can be found by
the harness.

### 13.3 Search-score labels

There are three supported labeling routes.

#### Mixed fixed-depth labels

```bash
cpp_impl/bin/test_bots dump search 8 800000 datasets/nnue_search_d8.bin
```

This deterministically samples half of the requested positions from random
legal play and half from engine self-play, shuffles them, and runs a full-window
fixed-depth search. The output already has:

```text
float32 = static HCE
int32   = fixed-depth search score
```

By default, the teacher search is forced to use HCE at its leaves. This is
useful when training a residual that learns what deeper search sees beyond the
static HCE without recursively teaching from the candidate network itself.

#### Relabel an existing position set

```bash
cpp_impl/bin/test_bots dump relabel 12 80000 \
  datasets/source.bin datasets/relabel_d12.bin hce
```

Use `current` instead of `hce` as the final argument to label with the current
full evaluator:

```bash
cpp_impl/bin/test_bots dump relabel 12 80000 \
  datasets/source.bin datasets/relabel_current_d12.bin current
```

Relabeling is useful for controlled distribution experiments because it keeps
the positions fixed while changing teacher depth or teacher evaluator.

#### Timed root-score labels

```bash
cpp_impl/bin/test_bots dump root 8000 20 datasets/nnue_root.bin
cpp_impl/bin/test_bots dump annotate \
  datasets/nnue_root.bin datasets/nnue_root_annotated.bin
```

`dump root` stores completed root depth in the float field and completed root
score in the integer field. `dump annotate` replaces the float field with
static HCE while preserving the integer search score.

Do not run `dump annotate` on an ordinary WDL dump and assume it has created
search labels. In an ordinary dump the integer field is already HCE, so the
result would effectively train HCE against itself.

### 13.4 Label clipping and mate handling

Search scores near the engine’s mate bounds dominate a regression loss while
providing little useful calibration for ordinary qsearch leaves.

`nnue_train_mininet.py` supports two policies:

- clip both HCE and teacher scores to `±mate_clip`;
- drop rows with `abs(search) >= mate_clip`.

The round-seven D16/H8 artifact used the drop-mates path with a threshold of
8000. The macro trainer independently removes `abs(search) >= 8000`, then
clips its remaining residual target by dropping rows outside
`±target_clip` (4000 by default).

### 13.5 Distribution quality

Offline loss is only meaningful when the position distribution resembles the
nodes where the engine actually calls the evaluator.

Useful coverage includes:

- early positions where no miniboard has been decided;
- forced-board and free-choice states;
- random legal play, which reaches shapes self-play may avoid;
- engine self-play, which emphasizes realistic tactical structures;
- positions near local and macro captures;
- multiple teacher depths;
- independent qsearch-leaf samples.

Round seven tested an additional 80,000 targeted depth-12 labels. They improved
MAE on their own targeted set by 1.77 points but worsened independent depth-12,
depth-8, and qsearch-leaf sets. The broader training distribution was retained.

The current trainer uses a deterministic random 90/10 position split. Adjacent
positions from the same game can therefore appear on both sides of the split.
Treat that validation loss as an optimizer/early-stopping signal, not as proof
of generalization. Use a separately generated dump for honest model
comparison, and use SPRT for the final decision.

## 14. Side-to-move-relative feature encoding

Both learned heads are side-to-move relative. One network handles both players.

For each local square:

```text
0 = empty
1 = occupied by the side to move
2 = occupied by the opponent
```

The nine digits of one miniboard form a ternary index:

```text
local_index = Σ digit[square] × 3^square
```

There are exactly:

```text
3^9 = 19,683
```

possible local patterns. Square zero is the least significant ternary digit,
matching `mini_index` in C++.

Each super-board cell is also converted to the side-to-move-relative class:

| Class | Meaning |
| ---: | --- |
| 0 | live |
| 1 | won by the side to move |
| 2 | won by the opponent |
| 3 | drawn |

The constraint is an integer from 0 through 9:

- `0..8`: the next move is forced into that miniboard;
- `9`: free choice, including the initial position and a send to a finished
  miniboard.

This canonicalization is important. Without it, the model would need separate
parameters for positions that are identical after swapping player colors.

## 15. Local D16/H8 MiniNet

### 15.1 Architecture

The active round-seven local network has:

- embedding width `D = 16`;
- hidden width `H = 8`;
- 19,683 local-pattern embeddings;
- four super-state embeddings;
- nine location embeddings;
- two active/inactive embeddings;
- ten constraint embeddings.

For miniboard `m`, define:

```text
v[m] =
    local_embedding[local_index[m]]
  + super_embedding[super_class[m]]
  + location_embedding[m]
  + active_embedding[constraint == m]
```

The nine vectors are concatenated with one global constraint embedding:

```text
x = concat(v[0], v[1], ..., v[8], constraint_embedding[constraint])
```

Since there are ten D-wide blocks:

```text
x has 10 × 16 = 160 elements
```

The residual is:

```text
hidden  = ReLU(W1 × x + b1)      # 8 values
R_local = W2 × hidden + b2       # one scalar in eval units
```

The active inference path contains 316,625 trainable scalar parameters:

```text
19,683×16 local embeddings
4×16      super embeddings
9×16      location embeddings
2×16      active embeddings
10×16     constraint embeddings
8×160     first-layer weights
8         first-layer biases
8         output weights
1         output bias
```

Most parameters are in the local-pattern table. The hidden mixer is
intentionally tiny because it runs at many qsearch leaves and must fit inside
the CodinGame source limit after packing.

### 15.2 Residual objective

The winning training mode is `--residual`. For each position:

```text
prediction = static_HCE + MiniNet(position)
target     = teacher_search_score
loss       = Huber(prediction, target)
```

Equivalently, the network learns the residual:

```text
teacher_search_score - static_HCE
```

The loss is expressed directly in engine evaluation units. The default Huber
delta is 1500. Huber loss behaves quadratically for ordinary errors and
linearly in the tails, reducing the influence of tactical and mate outliers.

The trainer also supports:

- direct replacement of HCE instead of residual learning;
- `asinh(score / S)` label compression;
- L2 regularization on the residual;
- residual upsampling by error magnitude;
- an output ReLU;
- a frozen baked-HCE additive head.

Those are experiment controls, not part of the shipped D16/H8 inference path.

### 15.3 Empty-board anchoring

HCE already supplies the empty-position tempo score. A residual network must
therefore contribute zero on the empty board.

The trainer can enforce this after every optimizer step with `--pin-empty`.
Regardless of that option, final MiniNet export shifts `b2` so the empty-board
residual is exactly zero in the float checkpoint.

The packing step checks the empty output again after embedding compression and
adjusts `b2` by:

```text
original_empty_output - packed_empty_output
```

This prevents a compression artifact from silently changing the opening
baseline.

### 15.4 Expanding an existing network

`--init-mini old.bin` can initialize a larger MiniNet from a smaller CFM2
checkpoint.

When D grows, each of the ten old input blocks is copied into the beginning of
the corresponding wider block. New columns start outside the inherited
function.

When H grows, old hidden rows and output weights are copied. New hidden
neurons keep randomized incoming weights but start with zero output weights.
The expanded model therefore reproduces the old model at initialization while
allowing additional capacity to learn.

The initializer rejects a request whose target D or H is smaller than the
source checkpoint.

### 15.5 Training loop

The MiniNet trainer:

1. loads the `NNUEWDL1` records;
2. verifies that the float field resembles an eval, not WDL;
3. drops or clips mate scores;
4. optionally upsamples large residuals;
5. creates or reuses a `.npz` feature cache;
6. splits records 90/10 with NumPy seed 42;
7. trains with Adam;
8. applies cosine learning-rate annealing;
9. early-stops on validation Huber loss;
10. restores the best validation checkpoint;
11. reports MAE, median absolute error, correlation, and HCE baseline metrics;
12. zero-centers the empty residual;
13. writes a CFM2 checkpoint.

Feature caching is keyed by data path, D, row count, mate policy, and upsampling
configuration. Delete the cache if the underlying dataset is replaced in
place; the filename alone cannot detect changed file contents.

### 15.6 Representative D16/H8 command

The round-seven artifact is a D16/H8 linear residual with mate rows dropped and
a learning rate of `2e-4`. A representative invocation is:

```bash
python3 tools/nnue_train_mininet.py \
  --data datasets/nnue_search.bin \
  --out artifacts/mininet_d16h8.bin \
  --arch mini \
  --residual \
  --mini-d 16 \
  --hidden 8 \
  --label linear \
  --mate-clip 8000 \
  --drop-mates \
  --upsample 0 \
  --huber 1500 \
  --lr 2e-4 \
  --epochs 40 \
  --batch 2048 \
  --patience 8
```

This is a documented template, not a byte-for-byte reconstruction command for
the checked-in header. The generated header does not contain the original
dataset path, dataset hash, or complete optimizer invocation.

## 16. CFM2 MiniNet checkpoint

`nnue_train_mininet.py` writes a little-endian float32 CFM2 file:

```text
4 bytes   magic "CFM2"
int32     D
int32     H
float32   local embeddings[19683][D]
float32   super embeddings[4][D]
float32   location embeddings[9][D]
float32   constraint embeddings[10][D]
float32   active embeddings[2][D]
float32   W1[H][10*D]
float32   b1[H]
float32   W2[H]
float32   b2
```

The trainer appends legacy scalar additive-head tables and a bake flag after
`b2`. They support older frozen-HCE experiments. The D16 header emitter reads
only the active MiniNet fields listed above.

CFM2 is an inference checkpoint, not a complete experiment record. It does not
store optimizer state, epoch, data provenance, command-line arguments, or
validation metrics.

## 17. Packing the local network for C++

The full local embedding table alone is:

```text
19,683 × 16 × 4 bytes = 1,259,712 bytes
```

Embedding it literally would exceed CodinGame’s 100,000-character source cap.
`tools/nnue_emit_mininet_header.py` compresses and preprojects it.

### 17.1 Projection-aware clustering

Ordinary k-means in embedding space treats every embedding dimension equally.
The engine only cares about errors after the first layer and output mixer.

The emitter reshapes the local portion of W1 into:

```text
[hidden=8][miniboard=9][D=16]
```

For every local embedding it computes all 72 first-layer projections, weighted
by the absolute output weight of each hidden neuron:

```text
projection_feature[index, hidden, miniboard] =
    abs(W2[hidden])
    × dot(local_embedding[index], W1[hidden, miniboard, :])
```

K-means runs in this flattened 72-dimensional projection space. This directs
capacity toward embedding differences that can actually affect the output.

There are 256 centroid codes:

- code 0 is reserved for the empty local board;
- the other 19,682 patterns are clustered into codes 1 through 255;
- each stored centroid is the mean of its members in original embedding space.

Keeping centroids in embedding space preserves the additive decomposition with
super-state, location, and active-board embeddings.

### 17.2 Packed payload

The generated payload contains:

- 19,683 one-byte centroid codes;
- 256 D16 float32 centroids;
- super, location, constraint, and active embeddings;
- W1, b1, W2, and b2.

It is CJK14 encoded into `cpp_impl/mini_eval_d16.hpp`: each 14 bits of
payload become one character in U+4E00..U+8DFF. CodinGame counts UTF-16 code
units and each of those characters is one unit, so the header stores 14
payload bits per counted character, against 6.4 for the ASCII85 encoding it
replaced. The macro payload uses the same decoder. See
[minification.md](minification.md) section 4 for the encoding details.

The raw C++ string uses `~` as its delimiter. Every payload character is
non-ASCII, so payload text cannot accidentally terminate the literal. At startup the header decodes the payload
into static storage and builds runtime tables.

### 17.3 Mask-to-code lookup

The board stores each miniboard as two 9-bit bitboards. The runtime creates:

```text
D16_MN_MASK_CODE[1 << 18]
```

indexed by:

```text
(mine_mask << 9) | opponent_mask
```

Valid disjoint masks map to the ternary index and then to the centroid code.
Overlapping masks map to code zero as a defensive fallback.

This removes ternary-index arithmetic from the hot inference loop.

### 17.4 First-layer factorization

The float model appears to require reconstruction of 160 features followed by
an `8 × 160` matrix multiply. Almost every term is constant for a small
categorical choice, so the emitter/runtime preprojects those terms once.

The shipped tables are:

```text
D16_MN_FACTOR_INIT[constraint][hidden]
D16_MN_FACTOR_CODE[miniboard][centroid][hidden]
D16_MN_FACTOR_SUPER[miniboard][super_class][hidden]
D16_MN_FACTOR_ACTIVE[miniboard][hidden]
```

`FACTOR_CODE` includes the common live/inactive local contribution.
`FACTOR_SUPER` stores a delta from the live class.
`FACTOR_ACTIVE` stores a delta from inactive.

Every projection is rounded to int32 with a fixed scale of 192. Runtime
inference is approximately:

```text
hidden = FACTOR_INIT[constraint] + super_acc[side_to_move]

for each miniboard:
    hidden += FACTOR_CODE[miniboard][mini_code[side_to_move][miniboard]]

if a miniboard is forced:
    hidden += FACTOR_ACTIVE[forced_miniboard]

output =
    b2
    + dot(W2, ReLU(float(hidden))) / 192
```

Two inputs are maintained by the search rather than looked up per leaf:

- `mini_code[perspective][miniboard]` caches each miniboard's centroid code
  (round nine). Only the miniboard a move is played in can change, so make
  updates two bytes and unmake restores them.
- `super_acc[perspective]` is the running sum of `FACTOR_SUPER` over all nine
  miniboards (round ten). It changes only when a miniboard becomes decided,
  exactly where the macro key changes. This is exact because the live-class
  rows are exactly zero (each is `lround(x - x)`).

This is deliberately not a full incremental accumulator: maintaining the whole
hidden vector on every make/unmake was measured slower (section 11).

All eight hidden lanes fit in one AVX2 vector. The main code table occupies
72 KiB:

```text
9 × 256 × 8 × 4 bytes = 73,728 bytes
```

The scalar path reconstructs the 160 float features and serves as a reference.
The unit test permits at most eight eval units of difference between the
scalar packed model and the int32 factored path across randomized games.

### 17.5 Header generation

```bash
python3 tools/nnue_emit_mininet_header.py \
  artifacts/mininet_d16h8.bin \
  -o cpp_impl/mini_eval_d16.hpp
```

The emitter currently requires exactly D16/H8 for the factored runtime.
`--no-reserve-empty` exists for experiments but was weaker than reserving the
empty pattern exactly.

## 18. Macro-context residual

The local MiniNet has detailed 3×3 pattern information, but its tiny mixer is
not an efficient way to learn every interaction among decided miniboards and
the forced-board constraint. The macro head targets the remaining error.

### 18.1 Target construction

`tools/nnue_train_macro_context.py` loads:

- the annotated search-label dataset;
- the trained local CFM2 checkpoint.

It computes:

```text
macro_target =
    teacher_search_score
  - static_HCE
  - float_local_MiniNet
```

Rows with `abs(search) >= 8000` are removed. Rows with
`abs(macro_target) > target_clip` are also removed.

The macro head therefore learns only what remains after both existing
evaluators.

### 18.2 Features and architecture

The macro model sees:

- nine side-to-move-relative super-board classes;
- the forced-board constraint.

It does not see local stones.

For miniboard `m`, the class embedding is selected from a location-specific
table entry `4*m + class[m]`. The nine embeddings and one constraint embedding
are summed:

```text
u =
    Σ macro_embedding[4*m + class[m]]
  + macro_constraint[constraint]
```

The network is:

```text
hidden  = ReLU(W_hidden × u + b_hidden)  # 16 values
raw     = W_out × hidden + b_out
R_macro = raw(position) - raw(empty_position)
```

The accepted model uses embedding width 8 and hidden width 16. Subtracting the
empty-position output in `forward()` makes zero anchoring structural rather
than an optimizer preference.

### 18.3 Training

The macro trainer uses:

- deterministic PyTorch and NumPy seeds;
- a deterministic 90/10 split;
- AdamW with weight decay `1e-4`;
- batch size 4096;
- Huber loss, delta 800 by default;
- early stopping, patience 30 by default;
- up to 200 epochs;
- 16 CPU threads by default.

A representative command is:

```bash
python3 tools/nnue_train_macro_context.py \
  --data datasets/nnue_search.bin \
  --net artifacts/mininet_d16h8.bin \
  --out artifacts/macro_d8h16.pt \
  --d 8 \
  --hidden 16 \
  --epochs 200 \
  --patience 30 \
  --lr 3e-3 \
  --huber 800 \
  --target-clip 4000 \
  --threads 16
```

The `.pt` checkpoint stores the model state, D, H, best validation MAE, base
net path, and data path.

### 18.4 Preprojection

The macro header emitter folds every categorical embedding through the hidden
weight matrix:

```text
projected_embedding = embedding × W_hiddenᵀ
projected_constraint = constraint × W_hiddenᵀ
```

The runtime payload therefore contains hidden-space contributions directly:

- hidden bias: 16 floats;
- ten projected constraints: `10 × 16` floats;
- nine-by-four projected super classes: `9 × 4 × 16` floats;
- output weights: 16 floats;
- output bias: one float.

The raw float payload is 3,076 bytes.

The accepted export multiplies output weights and the empty-adjusted output
bias by 1.25, then clips the final value to `[-2000, 2000]`.

```bash
python3 tools/nnue_emit_macro_header.py \
  artifacts/macro_d8h16.pt \
  -o cpp_impl/macro_eval.hpp \
  --scale 1.25 \
  --clip 2000
```

## 19. Exact macro lookup

Even a 16-hidden-unit AVX2 macro MLP was expensive at every qsearch leaf.
There are only:

```text
4^9 = 2^18 = 262,144
```

possible side-to-move-relative super-board states and ten possible
constraints. The generated header evaluates every combination once and stores:

```text
int16_t MACRO_SCORE[10][1 << 18]
```

The table occupies exactly 5 MiB:

```text
10 × 262,144 × 2 bytes = 5,242,880 bytes
```

Each macro key uses two bits per miniboard:

```text
key = Σ class[miniboard] << (2 × miniboard)
```

`FastBoard` carries one key for each player perspective:

```text
uint32_t macro_key[2]
```

For a miniboard won by player zero:

- player-zero key stores class 1;
- player-one key stores class 2.

For a draw, both keys store class 3. Live boards remain class 0.

The key changes only when a miniboard becomes decided or when that move is
unmade. Ordinary moves inside a live miniboard do not touch it. At evaluation,
the engine selects `macro_key[side_to_move]`, computes the current constraint,
and performs one indexed load.

The public/reference evaluator can still execute the original macro MLP from
board state. Unit tests compare that path against the lookup across randomized
games and require exact equality.

Table construction is a one-time initialization cost. It uses static storage,
not an engine-instance member, avoiding stack overflows in match workers and
the CodinGame process.

The lookup recovered roughly 6% start-position NPS within the D16 build.

## 20. Search integration

### 20.1 Initialization

At the start of a search:

1. `FastBoard` is copied from `GlobalBoard`;
2. HCE local accumulators are initialized;
3. both macro keys are reconstructed;
4. generated packed weights/tables initialize lazily on first evaluation.

### 20.2 Make/unmake

For an ordinary move in a still-live miniboard:

- local HCE state is updated;
- that miniboard's two cached centroid codes are refreshed;
- no macro key changes.

When a move wins or draws a miniboard:

- the super-board state changes;
- the cached out-of-play mask and terminal flag change;
- the relevant two-bit macro class is updated in both perspective keys, and
  the matching `FACTOR_SUPER` rows are added to both `super_acc` sums.

Make records everything it overwrites (hash, HCE scores and threat maps, the
miniboard's HCE entry, both centroid codes, active board, terminal flag and
decided state) in a 32-byte undo record, and unmake restores from it rather
than re-deriving each value (round ten).

### 20.3 Qsearch

The learned stack is primarily a leaf evaluator. Qsearch:

1. detects terminal positions;
2. computes incremental HCE plus structural correction;
3. applies the cheap HCE fail-high shortcut;
4. applies the safe upper-bound shortcut;
5. computes D16 MiniNet and cached macro residual when needed;
6. uses the result as stand pat;
7. searches local capture moves.

This placement matters. A network can have better full-position MAE yet lose
Elo if it is trained on states unlike actual qsearch leaves or costs enough
nodes to reduce search depth.

### 20.4 Interior pruning

The normal reverse-futility and futility checks remain HCE based. This avoids
paying the full learned evaluator at every interior node. A guarded depth-one
reverse-futility path adds the D16 local residual only when a cheap coarse
bound says it might confirm the cutoff.

That split is part of the accepted design; moving the networks to every static
evaluation call is a different search experiment and must be SPRT tested.

## 21. Generating a CodinGame submission

After generating both evaluator headers:

```bash
make -C cpp_impl test
make -C cpp_impl verify
make -C cpp_impl cg-input
python3 -c "s=open('cpp_impl/cg_input.cpp',encoding='utf-8').read(); print(len(s.encode('utf-16-le'))//2)"
```

`tools/cg_minify.py --inline-local` recursively expands the local generated
headers into `codingame_nnue.cpp`, then strips and renames the combined source.

The hybrid's last `cg_input.cpp` was 90,095 UTF-16 code units, leaving 9,905 below
the 100,000-unit limit; the evaluator payloads account for 26,247 of them and
the opening book for 13,456. `wc -c` reports bytes, which overstate the count
because each payload character is three UTF-8 bytes; the minifier prints the
unit count. The lossless CJK14 payload encoding keeps both evaluator payloads
compact; their decoded data remains byte-for-byte identical to the accepted
round-seven networks.

Always compile both the readable and minified sources. Packing bugs can preserve
Python validation metrics while producing a broken submission. CodinGame
compiles without `-O`, so a regenerated header must keep its hot helpers
(`d16_mini_hsum256`, `evaluate_macro_key`) `always_inline`; the emitters write
the attribute ([minification.md](minification.md) section 13).

## 22. Correctness and equivalence tests

Run:

```bash
make test
make -C cpp_impl verify
```

The relevant checks include:

- scalar packed MiniNet versus the fast D16 factor path;
- original macro MLP versus exact macro-key lookup;
- macro clipping bounds;
- HCE linear/LUT consistency;
- board make/unmake and legal move generation;
- NNUE incremental-versus-refresh checks for the older sparse path;
- readable and minified CodinGame compilation.

The D16 unit test currently allows at most eight eval units between scalar
centroid inference and the int32 factor path. Macro lookup must match the float
macro evaluator exactly because the table is generated from that evaluator at
startup.

Do not enable FMA casually. The reference and generated paths are designed
around the repository’s AVX2 mul-plus-add behavior, and changing floating-point
association can change the effective network.

## 23. Strength validation

Offline validation is diagnostic. It is not the ship criterion.

For an evaluation change:

1. freeze `crossfish_prev.hpp`;
2. change only Dev and intended generated headers;
3. run correctness tests;
4. run equal-depth testing to isolate leaf quality;
5. optionally run a cheap 20 ms screen;
6. run the authoritative 90 ms SPRT with the external 100 ms referee;
7. freeze and port only after a pass.

The timed referee measures wall-clock response time outside the engine. A move
returned after CodinGame's 100 ms limit is scored as an immediate loss and
included in the printed timeout totals. This catches internal timer
regressions that ordinary W/D/L testing would otherwise misclassify as extra
search strength. Timed tests reserve one physical core for scheduler and
referee headroom; fixed-depth tests remain exempt.

The timeout-hardened direct validation against merged round six used the same
90 ms allocation as the CodinGame bot:

```text
N: 2954 W: 1100 D: 937 L: 917
Elo diff: +21.55 +/- 10.37
LLR: +3.110 (H0=0, H1=+5) — PASS
Timeouts: Prev=0 Dev=0
Maximum response: Prev=97.99 ms Dev=90.19 ms
```

The accepted direct round-seven result against the merged round-six engine was:

```text
95 ms: N 5152 W 2012 D 1591 L 1549
Elo diff: +31.31 +/- 7.91
LLR: +3.063, H0=+20, H1=+25 — PASS
Prev NPS: 13,333,760
Dev NPS: 11,787,264
```

The gain is a mixture:

- algorithmic quality from the larger local residual;
- algorithmic quality from explicit learned macro context;
- implementation speed from projection-aware centroid packing;
- implementation speed from int32 first-layer factors;
- implementation speed from exact macro lookup;
- search integration that avoids paying learned inference where HCE already
  proves the bound.

## 24. Failure modes and lessons

### Better MAE can still lose Elo

The engine chooses moves, not regression examples. Calibration improvements
that do not alter useful move ordering may be neutral. A slower evaluator can
also erase its equal-depth gain at fixed time.

### More capacity is not automatically useful

Wider hidden layers can sharply reduce NPS. Earlier H128 experiments were much
too expensive. D16/H8 is a balance among representation, packed source size,
and leaf cost.

### Incremental updates are not automatically faster

Maintaining learned state on every make/unmake only wins if enough descendant
evaluations amortize that work. The rejected dual-perspective D16 accumulator
updated far more often than the evaluator was called.

### Targeted data can overfit its own slice

The rejected depth-12 fine-tune improved its targeted set and worsened three
independent sets. Preserve broad data and evaluate on datasets generated by a
different run.

### The macro head can duplicate HCE ideas

An explicit active macro-target HCE bonus added cost and was nearly neutral.
The learned macro head already represented much of that interaction.

### Packing is part of the model

Centroid assignment, empty-board reservation, projection rounding, output
scale, clipping, and arithmetic order all define the deployed evaluator. Test
the packed C++ network, not only the PyTorch checkpoint.

## 25. Reproducibility checklist for future nets

Large training dumps and the accepted round-seven CFM2/PyTorch checkpoints are
not currently tracked in Git. The generated headers are the versioned runtime
source of truth, but they are insufficient to reconstruct training.

For each future accepted net, record:

- base commit;
- exact data-generation command;
- teacher evaluator and search depth/time;
- dataset record count and SHA-256;
- feature-cache provenance;
- exact training command;
- Python, NumPy, and PyTorch versions;
- random seeds;
- checkpoint SHA-256;
- emitter command;
- generated-header SHA-256;
- scalar-versus-packed error statistics;
- minified source character count;
- equal-depth result;
- timed SPRT result.

A practical artifact layout is:

```text
artifacts/
  round-N/
    training-command.txt
    dataset.sha256
    mininet.cfm2
    mininet.cfm2.sha256
    macro.pt
    macro.pt.sha256
    metrics.txt
```

Whether those large files live in Git, release storage, or external artifact
storage is a repository policy decision. The hashes and commands should still
be committed with the generated headers.

## 26. Source map (both parts)

| File | Responsibility |
| --- | --- |
| `cpp_impl/nnue_b64.hpp` | the NNUE runtime: decode, bake, quantize, lazy stack, kernels (Part I) |
| `cpp_impl/nnue_b64_net.hpp` | the generated NNUE payload and its integer scales |
| `tools/nnue_emit_b64_header.py` | NNUE payload emitter, scale derivation and `--check` |
| `tools/experiments/nnue2/` | NNUE trainers (`gen_nnue.py`, `gen_nnue_stream.py`) and analyses |
| `tools/experiments/fast_nnue/` | BGN1 export and parity, the experiments' integer inference, candidate builds, checks, two-net matches |
| `cpp_impl/datagen.cpp` | self-play (timed or self-labelling fixed depth) and labeling |
| `cpp_impl/test_bots.cpp` | data generation, search labeling, probes, SPRT |
| `tools/nnue_train_mininet.py` | local MiniNet feature extraction and training |
| `tools/nnue_train_macro_context.py` | residual macro-head training |
| `tools/nnue_emit_mininet_header.py` | D16/H8 clustering, packing, and factor runtime generation |
| `tools/nnue_emit_macro_header.py` | macro preprojection, scaling, clipping, and lookup generation |
| `tools/nnue_emit_mininet_cg.py` | legacy D8/H4 packer and shared packing helpers |
| `cpp_impl/mini_eval_d16.hpp` | generated local evaluator |
| `cpp_impl/macro_eval.hpp` | generated macro evaluator and exact lookup |
| `cpp_impl/crossfish_dev.hpp` | local search integration |
| `cpp_impl/codingame_nnue.cpp` | readable CodinGame integration |
| `tools/cg_minify.py` | header inlining and final source minification |
| `cpp_impl/unit_tests.cpp` | packed-eval equivalence checks |

For the chronological experiment history and rejected alternatives, see
[the improvement log](improvement_log.md).
