# Eval training data and the win-probability trainer

This pipeline retrains the shipped evaluation (HCE + D16/H8 MiniNet + macro
head) on millions of positions from current Crossfish and uttt.ai play,
labeled by a deep Crossfish search. It replaces the `NNUEWDL1` dumps of
[nnue_training_and_implementation.md](nnue_training_and_implementation.md)
for new work; the runtime evaluator and its packing are unchanged.

```text
datagen play ──► cf_play.cfdg ─┐
uttt.ai self-play (fork npz) ──┴► import ──► datagen label (depth 14) ──► datagen diffs
                                                  │                        (HCE features)
                                   uttt.ai net4 value (GPU) ◄──┘
                                                  │
                         nnue_train_blend.py ──► eval_candidate.py emit / build / verify / match
```

## 1. Record format

`cpp_impl/datagen.cpp` and `tools/eval_data.py` share one fixed 128-byte
record with no file header, so files can be appended to, resumed and
memory-mapped:

| Field | Type | Meaning |
| --- | --- | --- |
| `s` | 93 bytes | uttt.ai state, ASCII digits (same layout as `NNUEWDL1`) |
| `source` | u8 | 0 light-random start, 1 uttt.ai opening, 2 heavy-random start, 3 uttt.ai self-play |
| `ply` | u8 | stones on the board |
| `result` | i8 | game result for the side to move: 1, 0, -1 (CodinGame rules) |
| `flags` | u8 | 1 result valid, 2 search labeled, 4 uttt.ai value, 8 uttt.ai root value |
| `game` | u32 | game id, unique within a file |
| `hce`, `static_eval`, `search` | i32 | static HCE, full static eval, fixed-depth search score (side to move) |
| `game_score` | i32 | Crossfish's root score when it played the move |
| `uttt_q`, `uttt_v` | f32 | uttt.ai MCTS root value and network value (side to move) |

`datagen diffs` writes a sidecar of ten `int8` per record: `eval_diffs`, the
HCE feature differences (player 0 minus player 1). The HCE is exactly linear
in them: `HCE = stm * sum(w_i * d_i) + 112 (tempo) + 300 (free move) - 800
(opponent latent capture)`.

## 2. Positions

**Crossfish self-play** (`datagen play OUT N MS THREADS SEED OPENINGS`),
3.5M positions from 69,626 games at 40 ms per move on 16 threads (about
90 minutes). Each game starts from one of:

- 0-8 uniform-random plies (45%);
- a position from uttt.ai's self-play at plies 4-24 (35%): uttt.ai samples
  its first 12 moves by visit count with Dirichlet noise, so these openings
  are varied but sensible;
- 9-20 uniform-random plies (20%), for coverage.

Before ply 40, 8% of Crossfish's moves are drawn uniformly from the moves a
depth-4 search puts within 300 of the best. These keep the games varied
without making their results meaningless. A result is valid for every
position after the last uniform-random move.

**uttt.ai self-play**: the 1.32M positions of the fork's generations 1-5
(25,000 games at 800 simulations, `tools/eval_data.py import-utttai`).

## 3. Labels

- `datagen label IN OUT 14 16`: a full-window depth-14 search with the
  current evaluator, fresh history per position, mates clamped to ±20,000
  (about 850 positions a second on 16 threads). Label with a `datagen`
  built from an unmodified Dev, or the candidate teaches itself.
- `tools/eval_data.py uttt-value FILE net4.onnx`: uttt.ai's network value,
  batched on the GPU through DirectML (about 37,000 positions a second).

`tools/eval_pipeline.py [--dir datasets/eval2] [--depth 14] [--threads 16]`
runs play, both label passes, diffs and uttt.ai values in order, resumably,
and writes `pipeline.json` in the data directory for the lab dashboard.
Before any stage runs, it checks the inputs it does not make itself and exits
with a list of what is missing: `cpp_impl/bin/datagen.exe`
(`make -C cpp_impl datagen`),
`uttt_openings.txt` and `uttt_sp.cfdg` in the data directory, and the uttt.ai
ONNX file (`UTTTAI_ONNX`) with the fork's `onnx_weights.py` in
`UTTTAI_TRAIN_DIR` or beside the ONNX file.

A stage is done when its outputs are complete, not when its log says so, so a
new `--depth` builds new files (D is the depth):

| Stage | Log | Done when |
| --- | --- | --- |
| play | `cf_play.log` | `cf_play.cfdg` holds 3.5M records |
| label_cf | `label_cf_dD.log` | `cf_play_dD.cfdg` is as long as `cf_play.cfdg` |
| label_uttt | `label_uttt_dD.log` | `uttt_sp_dD.cfdg` is as long as `uttt_sp.cfdg` |
| diffs | `diffs_dD.log` | each `.diffs` file has ten bytes per record of its labeled file |
| uttt_value | `uttt_value_dD.log` | every row of `cf_play_dD.cfdg` has flag 4 |

- A stage that runs first renames the outputs of the stages it feeds to
  `NAME.stale`, so they are rebuilt from its new output.
- While a stage's child runs, `LOGSTEM.lock` in the data directory holds its
  pid. A restarted driver waits for that process instead of starting a
  second one on the same output.
- When the driver stops a stage, it ends the stage's log with
  `pipeline: stage KEY done`, `failed` or `interrupted`, and `pipeline.json`
  shows the same status.

## 4. Trainer

`tools/nnue_train_blend.py` trains in win-probability space:

```text
E      = HCE + MiniNet + macro
pred   = sigmoid(E / K)
target = lam * sigmoid(search / K) + (1 - lam) * result      (lam = 1 for rows without a valid result)
         optionally (1 - mu) * target + mu * (uttt_v + 1) / 2
loss   = |pred - target| ** 2
```

K is fitted so that `sigmoid(search / K)` best predicts the results (1,600
for the depth-14 labels, 3,150 before the loader fix of improvement log
section 55; the shipped static eval alone would fit 2,350).

- The MiniNet and macro head start from the shipped headers, decoded
  exactly; the decoded model reproduces the C++ static eval to within one
  unit. They train jointly.
- Default `--mode centroid` trains the 256 centroids with the shipped
  pattern-to-centroid codes fixed, so the trained model is exactly the
  deployable one. `--mode full` trains all 19,683 pattern embeddings and
  re-clusters at export.
- `--hce-weights DIFFS...` also trains the HCE weights (the pawn weight
  stays fixed because every search margin is a multiple of it).
- 10% of games, not positions, are held out.
- `--device dml` trains on the GPU; ten epochs over 4.3M rows take about
  eight minutes.
- The HCE term is each row's `hce` column, so the net is fitted on top of
  the Dev that labeled the data. When that Dev carried source patches (the
  `*_hd` files: the drawn-miniboard HCE fix), pass the same patch list as
  `--dev-patches FILE` (the `dev_patches.json` format below). It is only
  recorded in `OUT.json`, from which `tools/eval_screen_plan.py` writes the
  candidate's `dev_patches.json`.

## 5. Candidates

`tools/eval_candidate.py`:

- `emit PREFIX DIR` writes `mini_eval_d16.hpp`, `macro_eval.hpp` and the HCE
  weights.
- `build DIR` compiles `test_bots`, `datagen` and `bench_ab` in DIR, with
  Dev on the candidate and Prev on the shipped headers under renamed symbols,
  so the two cannot share weights. HCE weights are patched into both
  `eval_weights` and the `LUT_W_*` constants the miniboard table is built
  from. `DIR/dev_patches.json`, if present, is a list of
  `{"old", "new", "count"}` exact source replacements applied to Dev.
  `--prev-dir OTHER` makes Prev candidate OTHER instead of the shipped eval
  (section 6).
- `verify DIR [SAMPLE] [--train-prefix PREFIX]` (at least one of the two)
  checks that the candidate's C++ static eval equals the trained model, and
  that its HCE is the one the model was trained on:
  - With `--train-prefix` (the run's `nnue_train_blend.py --out`), the
    candidate's `datagen` recomputes the statics (depth 0) of the first
    20,000 rows of the run's first `--data` file. Its HCE must equal that
    data's HCE plus exactly the trained weight change, so a missing or wrong
    Dev patch fails. `tools/eval_screen_plan.py` verifies this way.
  - With SAMPLE (labeled at depth 1), the HCE is compared with the shipped
    HCE instead. That comparison is skipped when SAMPLE is not
    shipped-labeled or DIR has `dev_patches.json`.

  An untrained export (`--epochs 0`) passes `bench_ab equiv` against Prev:
  identical scores and node counts.
- `match DIR NAME --depth D --games N` is an equal-depth screen;
  `match DIR NAME --ms 90` is the timed SPRT (six game threads: `test_bots`
  counts this machine's 16 logical CPUs as physical cores).

`tools/eval_train_plan.py` and `tools/eval_screen_plan.py` train and screen
a sweep of variants unattended.

## 6. Round robin

Screens against the shipped eval rank candidates only through one common
opponent. `tools/round_robin.py` plays every pair of candidates against each
other and fits one set of ratings to all the results, so each rating uses
every game its candidate played:

```text
round_robin.py plan NAME noop A_search F_full D_full --depth 8 --games-per-pair 1000
round_robin.py build NAME
round_robin.py run NAME [--local-threads 6] [--worker --worker-threads 4]
round_robin.py rate NAME [--anchor noop]
round_robin.py status NAME
```

- **plan** writes `datasets/eval2/rr/NAME/plan.json`: every unordered pair
  with its own opening range (`SPRT_GAME_OFFSET`, in opening pairs; N games
  use N/2 openings). A candidate is a name (`cpp_impl/bin/cand_NAME`), a
  directory, or `NAME=DIR`; `--ms 90` plays timed games instead of
  `--depth`. The book has 50,000 openings and book wrap is off, so a
  tournament plays at most 100,000 games in total: 10 candidates are 45
  pairs, at most 2,222 games each. Planning an existing NAME again with more
  candidates adds their pairs after the openings already used.
- **build** compiles `cpp_impl/bin/rr_A__B/test_bots.exe` for every pair with
  `eval_candidate.py build DIR --prev-dir B --no-tools`, where DIR holds A's
  emitted files. Dev is A; Prev is B's headers under the same renamed symbols
  as the shipped ones, with B's HCE weights and `dev_patches.json` applied to
  `crossfish_prev.hpp` exactly as A's are to Dev (a weight constant or patch
  that does not match fails the build). That is B as Dev would play it only
  while `crossfish_dev.hpp` and `crossfish_prev.hpp` are the same engine, so
  the build refuses when they differ beyond comments and the class name (a
  search experiment in progress in `crossfish_dev.hpp`). `rr_build.json`
  holds a digest of the candidates' files and of the engine sources the build
  compiles (`test_bots.cpp`, `datagen.cpp`, `bench_ab.cpp`, both engine
  headers and their includes), so up-to-date pairs are skipped. Changing one
  of those sources during a run stops it after the current pair, with a
  message to rebuild; the play book and the shipped eval headers are not
  inputs.
- **run** plays unfinished pairs one at a time locally and, with `--worker`,
  one at a time on the Linux worker (`sprt_worker.py setup` / `start` under
  the name `rr_NAME__A__B`, polled every minute; its remote directory is
  removed once the log is copied back). Each pair runs `test_bots` with
  `SPRT_MAX_GAMES=N` and `SPRT_LLR_BOUND=1000000000`, so it never stops
  early (the screens' bound of 100 is within reach of a pair a few hundred
  Elo apart), and logs to `datasets/eval2/rr/NAME/A__B.log` (Elo is A's). A
  pair whose log has N games is done. An interrupted pair resumes from its
  last result line (`SPRT_RESUME_*` and the next opening), which reproduces
  the uninterrupted run exactly at fixed depth; a new run reattaches to a
  pair still playing on the worker, and first collects the remote log of one
  that finished or died there while no run was attached. Progress with ETAs
  goes to the console and `run.log`, and `ratings.json` is refitted after
  every pair.
- **rate** takes each pair's last pentanomial line, its Elo and 95% interval
  exactly as `test_bots` and `sprt_merge.py` compute them (standard error =
  interval / 1.96), and fits ratings by weighted least squares with the
  anchor (`--anchor`; otherwise the previous one, `noop` at first) at 0. The
  intervals come from the fit's
  covariance and are relative to the anchor. `fit.chi2 / fit.dof` near 1
  means the pairs agree with one rating per engine. A pair with no variance
  (every opening the same result) is fitted with half a pair added to each
  outcome; an engine with no chain of games to the anchor gets no rating.
  `ratings.json` holds `anchor`, `mode`, `engines` (`name`, `rating`, `ci`,
  `games`), `pairs` (`a`, `b`, `games`, `elo`, `ci`, `penta`) and `updated`.
  On simulated tournaments with known ratings, 95% of the intervals cover
  the truth; on the depth-8 screens treated as pairings against `noop`,
  the ratings equal the screens' own Elo and intervals.

The lab dashboard's *Candidate ranking* card shows every tournament's
ratings (newest first) with interval bars and a head-to-head matrix, and
running pairs appear under *Running now*.

## 7. SPRT shards on a second machine

`tools/sprt_worker.py` runs part of an SPRT on a Linux machine reachable by
SSH (`CROSSFISH_WORKER=user@host`, key `CROSSFISH_WORKER_KEY`, default
`~/.ssh/crossfish_worker`):

- `setup NAME DIR` ships `cpp_impl` and the candidate DIR, builds `test_bots`
  and `datagen` with g++ and runs the self-tests there;
  `crosscheck NAME SAMPLE --local-dir DIR` then requires the remote build's
  HCE to equal the local build's exactly and its static evals to within one
  unit.
- `start NAME --ms 90 --offset 25000` plays the shard on its own opening range
  (both engines on the remote machine, so every pair is a fair comparison)
  with no LLR stop of its own; 4 threads (the default) on a 12-thread laptop
  CPU have played every timed shard so far without a timeout (the code notes
  that 8 caused them).
- `sync NAME LOCAL_LOG` copies the remote log every minute and appends the
  pooled result to `datasets/eval2/sprt/NAME_combined.log`
  (`tools/sprt_merge.py`, which reproduces test_bots's pentanomial LLR and
  Elo; `tools/test_sprt_merge.py` pins it to test_bots's own output). When
  the pooled LLR reaches +/-2.94 it stops the remote shard.

## 8. Screening policy

From the retraining round (improvement log section 55):

- Screen eval candidates with a **20 ms timed round robin** (for example
  `round_robin.py plan NAME noop A B ... --ms 20 --games-per-pair 2000`);
  confirm the winner with a 90 ms SPRT against the shipped eval. Equal-depth
  screens turn off futility-style pruning and correction history and ignore
  speed, and rated the four nets played both ways 6-18 Elo higher than the
  20 ms round robin did.
- Use at least 12,000-16,000 games on fresh openings (`--offset`) before
  believing an equal-depth screen; the first 4,000-game block read 9-12 Elo
  higher than the next 12,000 games for three of four candidates.
- Do not rank candidates by held-out loss: lower loss on the same target
  played worse three times.

## 9. Crossfish vs ultimattt

`tools/vs_ultimattt.py PAIRS CF_MS UT_MS WORKERS LOG [--plies 4] [--seed 1]`
plays the CodinGame bot (`crossfish_cg_meta.exe match`, built by the fork's
`cg/tools/make_crossfish_variants.py`) against nelhage/ultimattt's minimax
(`ultimattt worker`) in colour-swapped pairs from seeded random openings. The
uttt.ai fork's `utttpy` judges every move and scores the result under
CodinGame rules (miniboard-count tiebreak); an illegal move loses the game,
and a pair whose engine dies or answers garbage is reported, not scored. The
fork is expected at `../utttai` beside this repository (`UTTTAI_ROOT`), the
bot in its `cg/` (`CF_EXE`) and ultimattt at
`../ultimattt/target/release/ultimattt.exe` (`UT_EXE`). ultimattt checks
its time limit only between iterative-deepening depths, so it overshoots its
budget (137 ms average at 90 ms, against crossfish's 68 ms); the log's
move-time columns show by how much.
