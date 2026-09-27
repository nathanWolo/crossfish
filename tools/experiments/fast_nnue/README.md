# Fast integer NNUE: candidates, checks and two-net matches (nnue2 stages 1-5 and the play rounds)

The engine-side half of the nnue2 work: fast, incremental, integer AVX2 inference for the nets that
[`../nnue2`](../nnue2/README.md) trains, wired into the engine as `tools/eval_candidate.py` candidates;
the checks that prove it exact; builds in which two different nets play each other; and a Linux worker
for those matches. Every result that picked the shipped net **B64_d5M_57ep** was measured with these tools.
The NNUE that ships in `cpp_impl` derives from `fast_nnue_b.hpp`'s B-64 path (through the CodinGame port),
with the net's generator embedded; nothing in this directory is compiled into the engine. The round write-ups stay with the data in
`datasets/nnue2/fast/*.md` (gitignored); this file is their summary.

## Which engine these tools patch

`make_cand.py` and `make_cand_b.py` wire the NNUE into the engine by exact-text patches
(`dev_patches.json`) of the pre-NNUE `cpp_impl/crossfish_dev.hpp`: HCE + MiniNet + macro, as of commit
`c278cde` (the tip of PR #29; in general the parent of the commit that puts the NNUE into `cpp_impl`).
Once `cpp_impl` carries the NNUE itself, `eval_candidate.py build` refuses those patches, because their text
no longer matches. To rerun a recorded experiment, work in a checkout of that engine with these tools and
the local data linked in:

```bash
git worktree add ../crossfish-prenn c278cde && cd ../crossfish-prenn
git checkout <branch with this directory> -- tools/experiments/fast_nnue tools/experiments/nnue2
# link toolchains/ and datasets/ from the main checkout (both are gitignored):
#   PowerShell: New-Item -ItemType Junction -Path toolchains -Target ..\crossfish\toolchains   (same for datasets)
#   Linux:      ln -s ../crossfish/toolchains toolchains; ln -s ../crossfish/datasets datasets
```

The curation smoke test used such a tree: `c278cde`'s `cpp_impl` and `tools` with these directories
copied in (`datasets/nnue2/ship/curate.md`). The candidate directories under `cpp_impl/bin/` (gitignored)
are disposable build trees; the recorded ones hold the binaries of every result below.

Toolchain: the repository's llvm-mingw (`toolchains/llvm-mingw-20260616-ucrt-x86_64`, as
`eval_candidate.py` uses) and a Python with numpy (`PY`, default `toolchains/py312-dml`; the exporters
also need torch). The Linux worker builds with its own g++ (11.4 on the laptop, the same major version as
CodinGame's 11.2).

## Files

| file | role |
| --- | --- |
| **engine headers** | |
| `fast_nnue.hpp` | The per-cell FNN1 net (199 features, A=256, 2A->32->1): loader (file from `FASTNNUE_PATH`, else the compiled-in `FASTNNUE_NET_FILE`; one stderr line with path and CRC-32), quantizer with rigorous int16/int32 bounds, AVX2 kernels, scalar reference, the lazy accumulator `Stack`, eval cache, `-DFASTNNUE_CHECK` verification. Also the two-side scheme for two-net builds. |
| `fast_nnue_b.hpp` | The pattern-generator BGN1 nets (A = 64 or 128, 2A->16->32->1 or 2A->32->1): the same pieces, plus the C++ re-bake of the tables from the generator (`FASTNNUE_BAKE=1`) and the joint forced-board and stone-balance first-layer bound. |
| `fast_nnue_any.hpp` | Runtime dispatch on the net file's magic (`AnyStack`), or one kind fixed at compile time (`FASTNNUE_KIND`, 3-5% faster). |
| **candidates** | |
| `make_cand_b.py` | Writes a candidate directory: the headers and the 10 patches (the make hook, root refreshes, the NNUE at qsearch stand-pat and the interior static eval, the dead HCE and MiniNet updates dropped). `--kind any\|cell\|b64\|b128`, `--net NET` compiles the net's path in. |
| `make_cand.py` | Stage 1's per-cell-only candidate (its `PATCHES` are the base of `make_cand_b.py`) and the engine guard that makes a second copy of the fast NNUE in one build a compile error. |
| `build_cand.sh` | `make_cand_b.py` + `eval_candidate.py build` + the check builds, repros and `fast_bench_b` + an AVX2 instruction count. |
| **export and parity** | |
| `export_bgn.py` | A `gen_nnue.py` checkpoint to BGN1: the tables the engine adds, baked in float, plus the generator (`export`, with `--perm` lane pairing). `verify` recomputes evals from the file against PyTorch; `same` checks that a file is a lane permutation of another. |
| `export_net.sh` | The per-net pipeline of the full-data and long rounds: export, verify (float64 fallback), candidate `cpp_impl/bin/cand_full_NAME`, parity as loaded and re-baked, and the quantizer's exponents. |
| `compare_bgn.py`, `compare_float.py` | The quantized static eval of a candidate (`datagen label ... 0`) against PyTorch float (B nets) or the FNN1 float math (per-cell), on the 20,000 parity positions, overall and by phase. |
| `permute_fnn1.py`, `analyze_fnn1.py` | Stage 1: co-activation lane pairing of an FNN1 file (same function, fewer nonzero pairs), and its weight and activation statistics. |
| `bound_compare_b.py` | A float prototype of the three first-layer bounds (independent, joint forced board, + stone balance): which QA each allows. |
| **checks and repros** | |
| `fast_check.cpp` | The check driver: fixed-depth 6/8/10 searches, 3 ms self-play games and from-scratch evals on parity positions, many threads, timestamped progress. |
| `run_checks.sh` | Test (a) on a candidate: `fast_check`, `bench_ab_check`, `test_bots_check` at depth-prune 8 and 20 ms, `datagen_check label`, and the two repros. |
| `fallback_repro.cpp` | Searches entered without `refresh_root` (S1-S4); stage 3's position keys fixed them. |
| `stack_repro.cpp` | The same for the empty board (Z1-Z5); stage 5's `kNoKey` sentinel fixed them. |
| **speed** | |
| `bench_variants.sh` | `bench_ab` (Dev = candidate, Prev = shipped in one process) for a candidate and compile-time variants, interleaved rounds, median Dev/Prev nodes/s. |
| `fast_bench.cpp`, `fast_bench_b.cpp` | Per-call costs: refresh, update, eval, scalar reference, cold rows, load and bake time. `fast_bench.cpp` needs a `make_cand.py` (per-cell) candidate. |
| `search_mt.cpp` | Multi-threaded throughput (one table, one L3, up to 14 search threads). |
| `nps_probe.py` | Single-thread Dev/Prev speed of a pairing on the desktop or on the worker. |
| **two nets in one match** | |
| `fast_pair.py` | `pair A B`: a test_bots with Dev on A's net and Prev on B's, where Prev gets a renamed copy of the fast NNUE (namespace `fnnue_prev`, every `FASTNNUE_*` / `FNNUE_*` name but `FASTNNUE_CHECK` prefixed `PREV_`). Source and symbol checks run after every build. `rr-build` / `rr-run` run round robins, and `check-logs` / `check-log` check that each side loaded its planned net. |
| `rr_parallel.py` | Parallel `rr-build`, and `rr-run` with engines kept off the worker (`--worker-exclude`). |
| `run_rr.sh` | Build, play, check the net lines and rate a round robin. |
| `sprt_pair.sh` | An early-stopping SPRT between two candidates in a fresh pairing. |
| `split_match.py` | A fixed-length match split between the desktop and the worker, pooled into one log. |
| `pair_statics.cpp`, `pair_statics_check.py` | The isolation test: each side's statics and depth-N searches equal its own net's single-engine labels. |
| `rr_report.py`, `rr_machine_check.py` | Tables of a round robin, and the desktop-against-worker consistency test. |
| **the Linux worker** | |
| `fast_worker.py` | `rr-run --worker` goes through it. It ships the nets by CRC-32 (`~/crossfish_worker/nets/CRC/FILE`) and the pairing, builds with g++, runs `test_bots verify`, and checks both engines' statics against the candidates' own local builds on 1,000 positions before any game. It starts test_bots with the engine variables removed and checks the recorded environment. |
| `fw_negative.py`, `fw_env_test.py` | Its negative tests: swapped or shared nets must fail the crosscheck, wrong net lines must fail `check-logs`, and a polluted login environment must not reach the engines. |

## Workflow

```bash
PY=toolchains/py312-dml/Scripts/python.exe; F=tools/experiments/fast_nnue; N=datasets/nnue2/fast
# 1. export a trained net, build its candidate, check parity, record the quantizer's exponents
bash $F/export_net.sh B64_d5M_57ep b64                 # -> $N/B64_d5M_57ep_perm.bin, cpp_impl/bin/cand_full_B64_d5M_57ep
# 2. correctness (check builds; ~7 min for 20,000 positions and 200 games per match)
bash $F/build_cand.sh cpp_impl/bin/cand_chk_b64 --kind b64 --net $N/B64_d5M_57ep_perm.bin
CHECK_GAMES=200 bash $F/run_checks.sh cpp_impl/bin/cand_chk_b64 $N/B64_d5M_57ep_perm.bin b64
# 3. speed against shipped (fixed trees, then in-game at 20 ms), and variants
bash $F/bench_variants.sh cpp_impl/bin/cand_chk_b64 $N/B64_d5M_57ep_perm.bin --args "nodes 400 9"
bash $F/bench_variants.sh cpp_impl/bin/cand_chk_b64 $N/B64_d5M_57ep_perm.bin --args "walk 16 40 20" nopf=-DFASTNNUE_B_PREFETCH=0
# 4. games: one candidate against shipped (eval_candidate's log in datasets/eval2/sprt/)
$PY tools/eval_candidate.py match cpp_impl/bin/cand_full_B64_d5M_57ep B64_d5M_57ep_20ms --ms 20 --games 1000
#    two nets: SPRT in one pairing, a round robin, a fixed-length desktop + worker match
bash $F/sprt_pair.sh B64_d5M_57ep B64_lr1e2 90
mkdir -p cpp_impl/bin/cand_noop && cp cpp_impl/mini_eval_d16.hpp cpp_impl/macro_eval.hpp cpp_impl/bin/cand_noop/ \
    && $PY tools/eval_candidate.py build cpp_impl/bin/cand_noop       # the engine's own eval as a candidate
$PY tools/round_robin.py plan NAME noop=cpp_impl/bin/cand_noop A=cpp_impl/bin/cand_full_A B=... --ms 20 --games-per-pair 2000
CROSSFISH_WORKER=user@host bash $F/run_rr.sh NAME --local-threads 6 --worker --worker-threads 4 --worker-exclude '^B128_'
$PY $F/rr_report.py NAME --ref A && $PY $F/rr_machine_check.py NAME
CROSSFISH_WORKER=user@host $PY $F/split_match.py NEW --opp OPP --games 1200 --offset 48000
```

- **Other net files.** Any FNN1 or BGN1 file runs through `FASTNNUE_PATH` in a `--kind any` candidate.
- **Linux worker.** `CROSSFISH_WORKER=user@host` is required, with `CROSSFISH_WORKER_KEY` (default
  `~/.ssh/crossfish_worker`), as for `tools/sprt_worker.py`. Remote files live in `~/crossfish_worker/`.
- **Worker crosscheck positions.** `fast_worker.py` makes its 1,000 positions
  (`datasets/nnue2/fast/iso_pos1000.cfdg`, every 20th parity record) when they are missing.
- **Check builds of a pairing.** Compile `test_bots.cpp` in the pairing directory with the candidate
  flags plus `-DFASTNNUE_CHECK`. Each side keeps its own counters (`side=Dev` / `side=Prev`).

**Compile-time switches** (on the command line they reach Dev only; Prev takes the `FASTNNUE_PREV_` name):

| flag | effect |
| --- | --- |
| `FASTNNUE_CHECK` | verify every evaluation (shared by both sides of a pairing) |
| `FASTNNUE_KIND KIND_CELL/B64/B128` | fix the net kind (`make_cand_b.py --kind`) |
| `FASTNNUE_NET_FILE "path"` | compiled-in net (`--net`); `FASTNNUE_PATH` overrides it |
| `FASTNNUE_CACHE_BITS=N` | eval cache of 2^N entries (default 14; 0 = none) |
| `FASTNNUE_B_PREFETCH=0/1/2` | row prefetch at make (default 1) |
| `FASTNNUE_B_COMPACT` | tables for live patterns only (slower) |
| `FASTNNUE_EAGER`, `FASTNNUE_LIST_KERNEL`, `FASTNNUE_PREFETCH` | stage-1 variants (per-cell net) |
| `FASTNNUE_FLOOR_SHIFTS`, `FASTNNUE_INDEP_BOUND`, `FASTNNUE_JOINT_ONLY` | stage 2's numerics: floored hidden shifts; the weaker first-layer bounds |
| `FASTNNUE_NO_CACHE_PREFETCH`, `FASTNNUE_NO_SYSV` | undo two stage-3 speed changes |

The environment variables are `FASTNNUE_PATH` and `FASTNNUE_BAKE=1`, which re-bakes the tables from the
generator in C++. Prev's are `FASTNNUE_PREV_PATH` and `FASTNNUE_PREV_BAKE`.

## Design in brief

- **Accumulators.** They are kept per absolute player, so a move never swaps them. The eval reads
  `[acc[stm], acc[stm^1]]`.
  - A B net adds one pattern row per live miniboard (`T[m][pattern]`, 3^9 rows each) and a decided row
    per decided board.
  - The constraint rows (`con`) and the forced board's pattern row (`F`) are added at evaluation time,
    not stored.
  - An update is one row out and one row in per perspective: 6.6 ns for B64. The per-cell net needed
    61 ns for a deciding move.
- **Lazy stack.** `make` only records the move. `evaluate` walks back to the nearest entry that holds
  the needed position and replays from there. Unmake does nothing.
  - Entries are keyed by position (`tt_hash`), with the parent and child keys on every recorded move.
    Unset entries hold `kNoKey`, not 0: 0 is the empty board's hash.
  - A broken chain refreshes from scratch, so a search entered without `refresh_root` is still correct
    (`fallback_repro`, `stack_repro`).
  - A 2^14-entry eval cache keyed by `tt_hash` saves a third to a half of all evaluations.
- **Quantization.** All scales are powers of two, chosen at load so that no int16 accumulator and no
  int32 sum can overflow.
  - For B nets the first-layer bound is exact per board. It is tightened with the forced board's joint
    T+F rows and with the stone balance (no passes in search). This moved B64_lr1e2 from QA 2^9 to 2^10.
  - Hidden shifts round to nearest. The output truncates toward zero, as the float reference does.
  - The loader prints the exponents: B64_d5M_57ep gets QA 2^9, PSQT 2^13, QB 2^13, Q2 2^13, QO 2^10,
    which the CodinGame port takes as `--qexp 9,13,13,13,10`.
- **Lane pairing.** The dense layer skips activation pairs that are both zero. Pairing lanes that fire
  together cuts the nonzero pairs per eval (B64 37.7 -> 28.6 of 64) at no cost in function: the
  quantized evals are bit-identical.
- **Two nets in one build.** test_bots compiles Dev and Prev into one program, and `#pragma once` would
  hand both the first-included net.
  - `fast_pair.py` renames Prev's copy completely.
  - The engine guard makes any other two-engine build of two fast candidates a compile error.
  - Every log line names each side's net file and CRC-32, and `check-logs` compares them with the plan.

## Results

**Stages 1-5** (`datasets/nnue2/fast/stage*.md`)

| | per-cell r10_aug_d8x5M_lr6e3 | B64_lr1e2 | B128_lr1e2 |
| --- | --- | --- | --- |
| checked evals, mismatches (stage 3 check builds) | 356.9M, 0 | 432.4M, 0 | 363.4M, 0 |
| quantized vs float on parity_in: mean / max abs d | 3.62 / 45 | 1.73 / 38 (3.30 / 60 before stage 3's bound) | 1.50 / 37 |
| eval / update per call (stage 2) | 83 / 12 ns | 43 / 6.6 ns | 56 / 12 ns |
| nodes/s vs shipped, `walk 16 40 20` (stage 3) | 42.4% | 56.6% | 49.5% |
| depth-prune 8 vs shipped, 2,000 games | +274.2 ± 17.9 | +365.0 ± 21.7 | +392.6 ± 23.2 |
| 20 ms vs shipped, 1,000 games | +182.6 ± 19.1 | +276.1 ± 21.8 | +288.1 ± 22.9 |
| round robin `nnue_fast_20ms` (4,000 games per pair, shipped = 0) | +190.5 | +279.5 | **+289.2** (±7) |

- **Speed.** The float per-cell hook ran at 0.8% of shipped; the integer path at 49% (stage 1) is 64
  times faster.
- **Stage 4 isolation.** In 10 of 10 two-net runs, each side's statics and depth-7 searches equaled its
  own net's labels on 1,000 positions, while two different nets differ on 997-1,000 of those statics. B64 against
  itself scored +8.1 ± 18.0.
- **90 ms SPRT.** B128_lr1e2 against shipped passed at 420 games, +252.7 ± 30.6.
- **Stage 5.**
  - The complete Prev renames took review 3's leak repro from 317 statics and 800 searches wrong (of
    4,000) to 0.
  - The empty-board sentinel took Z1-Z4 from all wrong to 0.
  - Node counts were unchanged in 18 of 18 runs.
  - The six pairings built with g++ 11.4 with 0 warnings; their check runs covered 130.3M evaluations
    with 0 mismatches.

**Full-data and long rounds** (`full_data_play.md`, `long_play.md`; 20 ms, 2,000 games per pair, shipped = 0)

| round robin | 1st | 2nd | 3rd | B64_lr1e2 | games |
| --- | --- | --- | --- | --- | ---: |
| `nnue_full_20ms` (9 engines) | **B64_d5M_57ep +318.1 ± 6.7** | B128_d5M_57ep +309.3 | B64_all_e2x10 +303.0 | +287.3 | 72,000 |
| `nnue_long_20ms` (9 engines) | **B64_d5M_57ep +300.5 ± 6.5** | B64_d5M_114ep_lr5e3 +298.8 | B128_d5M_57ep +294.2 | +270.8 | 72,000 |

- **90 ms SPRT.** B64_d5M_57ep against B64_lr1e2 (the first CodinGame net) passed at 864 games:
  **+36.3 ± 14.3**, LLR 3.05, 0 timeouts. The round robins' head to head gave +25.9 ± 10.5 and
  +30.1 ± 10.3.
- **90 ms, 1,200 games.** B64_d5M_114ep_lr5e3 against B64_d5M_57ep scored +9.0 ± 12.0, split across both
  machines, so no replacement.
- **Worker consistency.**
  - The same 600 games on the desktop and the worker gave -51.9 ± 19.1 and -32.5 ± 20.4 (z -1.36).
  - Machine joint tests: p = 0.04 in `nnue_full_20ms` (mostly one B-128 net) and p = 0.33 in
    `nnue_long_20ms`.
  - B-128 loses relative speed on the laptop's 12 MB L3. B-64 runs at 131% of B-128's nodes/s there,
    against 116% on the desktop, so B-128 pairs stay on the desktop.
- **Timeouts.** There were 0 in the two round robins' 144,000 games.

## Notes and limitations

- **Windows first.** The local builds are Windows builds: `.exe` names, a 16 MB stack flag, and
  `sysv_abi` on the kernels under `_WIN64`. On Linux the kernels compile with g++ 11 (the worker builds
  every pairing that way), but the local build scripts assume the repository's llvm-mingw.
- **Tables in memory.** The BGN1 files hold the baked tables: 51 MB for B64 and 102 MB for B128 as float.
  In memory they take 25.6 and 50.8 MB as int16. A search touches few rows: about 1,100-1,200 distinct rows per
  fresh engine.
- **Stale pairings after an edit.** `fast_pair.py`'s build stamp includes its own hash. Any edit (this
  curation removed the worker default) marks existing round-robin pairings stale: `rr-build` rebuilds
  them before `rr-run` accepts them. Finished games and ratings are unaffected. `fast_worker.py` is not
  in the stamp.
- **Refusals.** `fast_pair.py` refuses candidates made before stage 4 (no `kSide` / `net_path`).
  `round_robin.py build` cannot build fast pairings; use `fast_pair.py rr-build`.
- **The removed scripts.** The stage drivers (`build_s3.sh`, `iso_s4.sh`, `s5_fixes.sh`, ...) became
  `build_cand.sh`, `export_net.sh` and `run_rr.sh`; `run_checks_b.sh` became `run_checks.sh`,
  `sprt90_full.sh` became `sprt_pair.sh`, and `long90.py`, `long_play_rr.py` and `long_play_report.py`
  (from `nnue2/`) became `split_match.py`, `rr_parallel.py` and `rr_report.py`. The full file-by-file
  map is kept locally (untracked) in `datasets/nnue2/ship/curate.md`. They depended on stage-specific candidate
  directories, and some of those can no longer be rebuilt by design: stage 4's shared-net control now
  stops at the guard.
