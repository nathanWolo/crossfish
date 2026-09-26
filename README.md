# crossfish

Entry for the UVicAI UTTT tournament (first place) and a CodinGame Ultimate Tic-Tac-Toe engine.

The **active engine is C++**. `cpp_impl/codingame_nnue.cpp` is the readable
CodinGame engine and includes the generated packed-eval headers;
`cpp_impl/cg_input.cpp` is the bundled/minified file to paste into CodinGame.
Local SPRT compares `cpp_impl/crossfish_dev.hpp` against the frozen previous
in `cpp_impl/crossfish_prev.hpp`. `cpp_impl/cg_legend_hce.cpp` is a snapshot
from the first Legend hit. The Python tree under `python_impl/` is legacy.

Detailed project documentation:

- [Engine improvement log](documentation/improvement_log.md): the chronological
  record of every accepted and rejected experiment, with its gate
- [Handcrafted evaluation and correction history](documentation/hce_and_correction_history.md)
- [NNUE training and runtime implementation](documentation/nnue_training_and_implementation.md)
- [Eval training data and testing tools](documentation/eval_data.md): the self-play/labeling
  pipeline, the win-probability trainer, isolated candidate builds, SPRT shards on a second
  machine and round-robin ratings
- [CodinGame submission and minifier](documentation/minification.md)
- [Gameplay opening book](documentation/play_book.md): the book the CodinGame bot plays from
- [SPRT opening book](documentation/opening_book.md): the frozen starting positions the SPRT harness uses

Hill-climbing Elo is a specific loop: freeze Prev, edit only Dev, prove correctness, then SPRT. Read **Improving the engine** before changing search or eval.

## Verify a change

For the Python oracle, create the optional repo-local environment once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

Run this after every engine/rules/eval change. It is fast and deterministic:

```bash
make test
```

That builds and runs the C++ unit tests, the Python rules oracle and the
`tools/test_*.py` suites, validates the SPRT opening book, and compiles the
CodinGame binaries: the readable `codingame_nnue.cpp`, the paste file
`cg_input.cpp` both at `-O3` and with CodinGame's own flags, the HCE-only
`crossfish.cpp`, and the Legend snapshot. CI runs it on every push, alongside
the CodinGame performance gate (see **Compiler and local builds**).

| Target | What it checks |
| --- | --- |
| `make test` | Correctness: perft, make/unmake, Zobrist, legal moves vs an independent oracle, eval internals, search legality / instant wins, plus CG compile |
| `make test-cpp` | C++ unit tests only (`cpp_impl/unit_tests.cpp`) |
| `make test-python` | Independent Python board perft / make-undo / winners (`python_impl/test_rules.py`), plus `tools/test_*.py` |
| `make verify` | The harness's own checks in `test_bots.cpp` (movegen, the 100 ms referee, the opening traversal fingerprint, the pentanomial SPRT maths against a Fishtest reference, eval consistency), then exit (no SPRT) |
| `make -C cpp_impl bench` | Proves a speed-only change is tree-identical: fresh engine per position, identical scores **and** node counts, plus node counts and wall time |
| `make -C cpp_impl port-check` | Proves `codingame_nnue.cpp` searches exactly like Dev (fixed-depth scores and node counts at depths 5, 7, 9), for tree-changing ports too |
| `make sprt` | Strength: Dev vs Prev self-play with SPRT at **90 ms** (slow, noisy) |
| `make book-inspect` | Validate and summarize the 50,000-position SPRT opening book |
| `make opening-book` | Regenerate the depth-16-qualified SPRT opening book (slow; normally do not run) |
| `make roundrobin` | 10k-game pairs at 20 ms/move between the HCE-only `crossfish.cpp` bot, the Legend snapshot and the Python bot (a legacy sanity check, not the shipped engine) |
| `make cg` | Compile the CodinGame bot, the paste file (at `-O3` and with CodinGame's flags) and the HCE bot |
| `make cg-input` | Rebuild `cpp_impl/cg_input.cpp` from `codingame_nnue.cpp` with the minifier |
| `make -C cpp_impl cg-flags` | Build the paste file with CodinGame's exact command line (`bin/cg_input_cgflags`) |
| `make -C cpp_impl cg-speed` | Same-tree search time of the CodinGame port at `-O3` and with CodinGame's flags; checksums must match |
| `make -C cpp_impl cg-gate` | The CI CodinGame performance gate, locally in a `gcc:11.2` container (needs Docker) |
| `make -C cpp_impl play-book-match` | Gameplay book vs no book from the start position (see `documentation/play_book.md`) |

SPRT answers "is this stronger?". Unit tests answer "did I break the rules, hashing, eval, or search?". Do not skip `make test` because SPRT passed.

Timed referees enforce CodinGame's external clock independently of the bot's
own timer. Any move returned after 100 ms is an immediate loss, and SPRT /
round-robin summaries report timeout counts. Fixed-depth eval tests are exempt
because they intentionally run without a move clock.

SPRT hypotheses can be overridden through the environment. For example, this
tests whether Dev clears +50 Elo instead of the default 0-vs-5 screen:

```bash
SPRT_ELO0=50 SPRT_ELO1=55 make sprt
```

The SPRT opening records are traversed through a deterministic permutation,
not their generation order. `SPRT_GAME_OFFSET=N` starts at opening pair `N` of
that permutation. Use a fresh offset when you want a reading independent of an
earlier run: every SPRT from offset 0 walks the same openings, so
near-identical engines share their early noise. The full list of controls is
under **SPRT contract** below.

## Improving the engine

This section is the process. The goal is Elo on CodinGame Ultimate Tic-Tac-Toe, not a prettier loss, a higher training correlation, or a faster NPS number that plays worse. Other agents will hill-climb from here. Follow the loop; do not invent a parallel scoring system.

CodinGame gives 1000 ms on the first execute per player and 100 ms on later
moves. The fixed center opening, later moves, and official SPRT use a 90 ms
search budget, leaving time for eager evaluator initialization, per-turn setup,
scheduler jitter, search unwinding, and output before the independently
enforced 100 ms deadline. A change is not shipped until it passes that gate
without timeout forfeits. 20 ms is an optional cheap screen, not a ship. The
submission cap is **100,000 characters**.

### The three copies of the engine, and the files around them

| File | Role | Who may edit it |
| --- | --- | --- |
| `cpp_impl/crossfish_prev.hpp` | Frozen baseline. `CrossfishPrev` in SPRT. | Only when **freezing** a landed win. Never during an experiment. |
| `cpp_impl/crossfish_dev.hpp` | The experiment. `CrossfishDev` in SPRT. | The only search/eval file you change while testing. |
| `cpp_impl/mini_eval_d16.hpp` / `macro_eval.hpp` | Generated packed evaluators shared by Dev, Prev and the CG bot. | Regenerate only for an accepted eval candidate, with the emitters in `tools/`. |
| `cpp_impl/mini_eval.hpp` | The retired D8/H4 MiniNet, kept for unit tests of the packed-net code paths. | Do not change. |
| `cpp_impl/global_board.hpp` | Board, movegen, make/unmake, Zobrist. Shared by everyone. | Rules/hash only. Perft is frozen. |
| `cpp_impl/codingame_nnue.cpp` | Readable CG bot; local eval headers are bundled for submission. | After a pass, when porting. Not the experiment. |
| `cpp_impl/cg_input.cpp` | Bundled and minified paste file. | Regenerated from `codingame_nnue.cpp`; never hand-edited. |
| `cpp_impl/play_book.hpp` / `play_book_data.hpp` | Gameplay opening book runtime and its generated payload. | Only through the procedure in `documentation/play_book.md`. |
| `cpp_impl/crossfish.cpp` | Self-contained HCE CG bot; packing source for a new MiniNet emit. | Only if you are emitting a new net or changing the HCE template. |
| `cpp_impl/cg_legend_hce.cpp` | Historical Legend snapshot. | Do not touch. |

`test_bots` plays **Dev vs Prev**. Default (no extra net flags) is the current Dev search/eval against the frozen Prev snapshot. That is the strength gate.

Dev and Prev must stay the same program except for the experiment. If you "clean up" Prev, speed it up, or change its search while testing Dev, the SPRT is no longer measuring the experiment.

### The loop

Do this in order. One hypothesis per loop.

1. **Confirm the freeze.** Prev must be the last *accepted* engine (last SPRT pass that was frozen, or the last shipped CG bot). If Dev already contains leftover failed experiments, revert Dev to Prev before starting. The experiment is the diff `crossfish_dev.hpp` (plus any intended shared-header change) versus `crossfish_prev.hpp`.
2. **Change only Dev** (and shared headers only if the change is truly shared and you understand both sides get it). Do not edit Prev. Do not port to `codingame_nnue.cpp` yet.
3. **One axis.** Eval rewrite *or* search change *or* speed-only rewrite of the same eval *or* a new net. Not two of those in the same SPRT. If you cannot say in one sentence what Prev does that Dev should do better, the experiment is not ready.
4. **`make test`.** Always. SPRT passing a broken engine is how you ship illegal moves or a wrong hash. Unit tests answer "did I break rules, hashing, eval, or search?". SPRT answers "is this stronger?".
5. **Classify the change, then gate** (next subsection). Run **one** SPRT at a time, reserving one physical core for the OS and referee. Do not start a second SPRT on the same machine.
6. **On FAIL or a clearly marginal run:** stop the SPRT, revert Dev to the freeze, write down why it failed (eval, speed, or mixed). Do not keep a failed experiment in Dev as the new baseline.
7. **On PASS:** freeze immediately (copy the accepted Dev into Prev), then port to the CG files, then record the result. Only then start the next experiment.

### Freezing Prev

Freezing is how the hill climb keeps its baseline honest.

- After a real pass, make `crossfish_prev.hpp` a snapshot of the accepted Dev: same search, same eval, same constants. Update the file comment with the date and what was frozen.
- The next experiment starts from that pair being identical except for the new diff.
- Do not leave accepted changes only in Dev. The next agent will treat Prev as the truth and your unfrozen win as noise.
- Do not "improve" Prev to make Dev's SPRT look better. If Prev is weaker than the last ship, a pass is fake.
- Dead parameters in Prev (for example a `can_null` argument that is never read) are not a reason to change Prev behavior. Leave them unless you are freezing a Dev that already removed them.

### SPRT contract

Harness: `cpp_impl/test_bots.cpp`, built as `cpp_impl/bin/test_bots`. Rebuild after every Dev/Prev/header change.

```bash
make -C cpp_impl test          # correctness first
make -C cpp_impl sprt          # official: 90ms, H0=0, H1=+5, one core reserved
make -C cpp_impl book-inspect  # validate the frozen opening artifact
# or, from cpp_impl/bin:
./test_bots                    # 90ms (official bar)
./test_bots 95                 # optional historical comparison
./test_bots 20                 # optional cheap 20ms screen
./test_bots depth 4            # equal depth 4, eval pruning off
```

Default hypotheses: **H0 = 0 Elo**, **H1 = +5 Elo**. The run stops at `|LLR| >= 3` (override with `SPRT_LLR_BOUND`).

- `SPRT PASS: H1 … favored over H0` — accept the change (for that time control).
- `SPRT FAIL: H0 … favored over H1` — reject. This means "not a +5 Elo gain", not "Dev is worse". A true +2 Elo at 90 ms will often fail H0-vs-+5. That is still not a ship.
- `SPRT INCONCLUSIVE` — hit `SPRT_MAX_GAMES` without a decision.

Environment overrides:

| Variable | Meaning |
| --- | --- |
| `SPRT_THINK_MS` | Search allocation (default 90). Ignored for play when `depth` is set. |
| `SPRT_ELO0` / `SPRT_ELO1` | Hypotheses. `ELO1` must be greater than `ELO0`. |
| `SPRT_LLR_BOUND` | Stop when `|LLR|` reaches this (default 3). |
| `SPRT_MAX_GAMES` | Optional cap. 0 = run until LLR decides. |
| `SPRT_THREADS` | Worker count. On Linux the default uses physical-core topology and reserves one core; it falls back to logical CPUs when topology is unavailable. |
| `SPRT_GAME_OFFSET` | Starting index in the deterministic opening permutation. One index produces two color-swapped games. |
| `SPRT_OPENING_BOOK` | Override the shipped book path. `none` or `0` disables the file book and uses the legacy opener selected by `SPRT_BOOK`. |
| `SPRT_BOOK` | Legacy fallback only: 1 selects deterministic seeded 4-8-ply openings and 0 selects unseeded random openings. The shipped file book is the default. |
| `SPRT_PAIR_MODEL` | 1 (default) scores colour-swapped opening pairs with the pentanomial model; 0 is the old per-game trinomial model, for diagnostics only. |
| `SPRT_RESUME_WINS` / `_DRAWS` / `_LOSSES` / `_PENTA` | Resume a stopped run from its last printed line. `SPRT_RESUME_PENTA=LL,LD,MID,DW,WW` is required in the pair model, and `SPRT_GAME_OFFSET` must be the original offset plus half the resumed games (it defaults to half the resumed games, which is right only for a run that started at offset 0). |
| `SPRT_ALLOW_BOOK_WRAP` | Diagnostics only: reuse openings after the book is exhausted. Without it an exhausted book ends the run as inconclusive. |
| `SPRT_CENTER_ENUM` | Diagnostics only: the center-first deterministic opener used for the section 41 opening experiments. |

Example: prove a huge speed win is more than +50 Elo at 90 ms:

```bash
SPRT_ELO0=50 SPRT_ELO1=55 make -C cpp_impl sprt
```

Use a raised H0 only when the first thousands of games already show a blowout. Do not use it to dress up a +8 Elo run.

Each printed line reads like

```text
N: 2748 W: 818 D: 1234 L: 696 Penta=76,287,543,375,93 Elo diff: 15.43 +/- 9.05 LLR: 3.012 timeouts Prev=13 Dev=5 max_ms Prev=158.3 Dev=127.1
```

where `Penta` counts opening pairs from Dev's side (two losses, loss+draw,
split or two draws, win+draw, two wins), `timeouts` counts external-referee
forfeits and `max_ms` is each engine's slowest reply. The header also prints
**Prev NPS** and **Dev NPS** from a 1-second startpos search. Treat NPS as a
speed signal, not a strength score: it varies by several percent between
hosts and between runs.

Normal SPRTs load `cpp_impl/opening_book.bin`. Its 50,000 positions were
selected by a frozen merged baseline, independently searched to depth 16, and
limited to an absolute baseline score of 300. Each position is played twice
with colors swapped and scored as one pentanomial pair, so the book covers
100,000 games before an opening would repeat; the harness never reuses one
silently. Missing the expected book is an error rather than a silent
fallback. See the [opening-book guide](documentation/opening_book.md) before
regenerating or replacing it.

When the original 10,000-position book replaced random openings, a controlled
2,000-game comparison measured the same candidate at **+24.01 +/- 11.59 Elo**
with 42.2% draws on the book, versus **+6.95 +/- 12.67 Elo** and 30.8% draws
on the legacy random opener. Balanced, reasonable openings leave more room for
the engine change to decide the game; full methodology and caveats are in the
guide.

Referee forfeits count in W/D/L exactly as they would on CodinGame. A candidate that gains nodes by overrunning the clock is weaker, not
faster.

`depth N` sets a fixed search depth and turns **eval pruning off on both sides** (`g_disable_eval_prune`: no RFP / futility / qsearch-delta). That is the equal-depth gate: same node budget in ply, so a loss means a worse leaf, not a slower one.

### Which gate, in which order

| Kind of change | First gate | Then | Ship only after |
| --- | --- | --- | --- |
| Different leaf / different net / different HCE weights | `depth 4` | optional 20 ms, then **90 ms** | Official 90 ms pass. Depth-only is not enough. |
| Same eval, faster implementation (LUTs, MiniNet projection, AVX) | `make -C cpp_impl bench` must print **IDENTICAL** | optional 20 ms, then **90 ms** | Official 90 ms pass. If `bench` reports any node-count difference, the "speedup" changed the eval and the SPRT would be measuring something else. |
| Search (LMR, TT, move order, pruning) | optional 20 ms | **90 ms** | Official 90 ms pass. Equal-depth can lie: more nodes at a fixed depth is not the CG game. |
| Bug fix (the old behaviour is wrong, not just weaker) | `make test` plus a test or trace that shows the bug | **90 ms** with `SPRT_ELO0=-5 SPRT_ELO1=0` | Non-regression pass. Say in the PR that it is a bug fix; a strength claim still needs H0=0, H1=+5. |
| CodinGame port only (no Dev change) | `make -C cpp_impl port-check` (tree-changing) or `cg_selfcheck` before/after (tree-identical) | the CodinGame performance gate in CI | Gate passes; for a speed claim, a new-vs-old match with both paste files built with CodinGame's flags. |
| Gameplay opening book (`play_book_data.hpp`) | `make -C cpp_impl play-book` (pack + check against the text book) | `play-book-match` from the start position, plus a paired run against a different engine, same seed for old and new book | The official SPRT starts from its own 50,000 openings and never reaches the book; see `documentation/play_book.md`. |

Do not skip `make test` because a gate passed.

**`test_bots depth N` is not an equivalence test.** Each worker reuses one
engine pair across the two games of an opening pair, so it carries
transposition, history and correction-history state into the second game. A Dev
that was *provably* tree-identical to Prev measured **-8.44 +/- 23.84 Elo**
there at N=700. Use `make -C cpp_impl bench`, which builds a fresh engine per
position and requires identical scores **and** identical node counts; that is
the only real proof that a speed-only rewrite left the eval alone. Its `walk`
mode replays a scripted game through one engine at a real move budget and
reports mean completed root depth, which is the quantity a speed win has to
convert into: calibration on the round-nine host is **+3.3% nodes = +0.167
ply**.

A tree-changing candidate cannot be screened by node count at all. Round nine
measured a 16x transposition table searching 3.9% *more* nodes than the
shipped one, because the pseudo-singular extension fires on any sufficiently
deep TT hit, so a higher hit rate buys extra extensions. Fewer nodes is not
better and more nodes is not worse; only depth-at-equal-time and Elo are.

Kill a run that is clearly not going to a +5 pass: Elo stuck around 0 to +3 after many thousands of games, LLR wandering near 0. Waiting for H0 at 20k games is a waste of the machine. Record the last line and revert.

### Correctness checklist

Run `make test` after any rules, hash, eval, or search edit. It builds C++ unit tests, the Python rules oracle, and the CG binaries.

If you change HCE, `eval_consistency` in `unit_tests.cpp` still has to match `eval_diffs` / `eval_parts`. A LUT rewrite that disagrees with the linear features is a bug even if it is faster.

If you change the MiniNet or its packing, the scalar reference `d16_evaluate_mini` and the fast factored path `d16_evaluate_mini_fast` must agree within 8 eval units on random games, and the macro lookup must equal `evaluate_macro_key` exactly (`d16_fast_matches_scalar` in `unit_tests.cpp`). The AVX paths are **mul+add, not FMA**, so they match the scalar net.

Large tables (`1<<18` scores, threat maps, MiniNet projections) must be `static inline` (BSS), not instance members. Instance arrays of that size overflow the 1 MB Windows stack and kill `match` workers. CodinGame's Linux stack may hide this. `codingame_nnue.cpp` already hit it.

Do not put a 1 MB table on `main`'s stack in the CG file.

Startpos perft is frozen in C++ and Python. If one suite's counts change, update the other in the same commit. Do not "fix" perft to match a movegen bug.

| Depth | Nodes |
| ---: | ---: |
| 1 | 81 |
| 2 | 720 |
| 3 | 6336 |
| 4 | 55080 |
| 5 | 473256 |

### What is not strength

These have already fooled people in this repo:

- **Holdout MAE / correlation.** A net can fit search scores better and still play the same moves (equal-depth ~0 Elo).
- **Beating static HCE on a dump.** MiniNet only runs at qsearch leaves when HCE is inside the window. Training on mates and fail-highs that search never asks the net about is the wrong target.
- **Scaling MiniNet width (D) on old HCE-only depth-6 labels.** Extra unused concat; mixer stays tiny. Measured ~0 Elo at equal depth vs the D=8 H=4 residual shipped at the time. (The later D16/H8 net won because it was trained on better labels and packed to stay cheap.)
- **A full Stockfish-style NNUE with a better holdout fit.** It replaced HCE, MiniNet and macro and lost 193-296 Elo at equal depth; parity looked like it would need on the order of 100M labeled positions (improvement log section 53).
- **A much wider mixer (H=128) at 20 ms.** NPS collapse, hundreds of Elo lost.
- **A slower better leaf.** A +10 Elo equal-depth eval with a 5% NPS tax can be ~0 at 20 ms.
- **SPRT against a Prev you just weakened or against old HCE when the ship is MiniNet.** Gate vs the frozen ship, not vs a convenient opponent.
- **Nodes per move in a CodinGame log, compared across games.** `N` depends on the position and on how much of the clock the search used; compare the same position, or use `make -C cpp_impl cg-speed` and the CI gate.
- **A local speed-up alone.** The SPRT builds at `-O3`; CodinGame builds without `-O`. A change that is fast locally can be slow on CodinGame, and the reverse (improvement log section 47 was worth +163 there and nothing locally).

Early-game positions are where HCE is weakest (minis not yet decided). If you train a net, bias data toward low ply with MiniNet-search labels, not only late self-play.

### Shipping to CodinGame

A Dev SPRT pass does **not** update the CG bot. `codingame_nnue.cpp` keeps a
standalone copy of the search but shares generated packed-eval headers. After
a freeze:

1. Port accepted search changes into `codingame_nnue.cpp`, following the
   CodinGame compiler rules below (`always_inline` helpers, no `std::`
   containers on the hot path). For eval changes, regenerate
   `mini_eval_d16.hpp` and/or `macro_eval.hpp` with the matching tools. Keep
   large LUTs in static storage.
2. Prove the port: `make -C cpp_impl port-check` must print IDENTICAL at every
   depth (the CG search against Dev). For a tree-identical change,
   `cg_selfcheck` checksums must also match their pre-port values.
3. Run `make -C cpp_impl cg-input`; the minifier recursively inlines local
   headers before shortening the source. The CI gate fails if the committed
   `cg_input.cpp` is not exactly this output.
4. Confirm the minifier's count is under 100,000 and that the CodinGame
   performance gate passes. Paste **that** file into CodinGame, not the
   readable source. Its first line must be `#pragma GCC optimize("O3")`; a
   paste without it searches about 130k nodes per move instead of 600k-900k,
   which the bot's `N` output shows at once.
5. Record the result: the SPRT line (N, W/D/L, pentanomial counts, Elo, LLR,
   hypotheses, timeouts, NPS) in a new improvement-log section, and a row in
   **Latest strength result**.

`nnue_emit_mininet_cg.py` minifies by default and has a coarse vs-HCE match (`Elo < -80` fails). That is a packing-smoke test, not the Dev-vs-Prev gate.

### Compiler and local builds

```text
-O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread
```

**CodinGame does not compile with those flags.** Its C++ command line is g++
11.2 with `-std=gnu++17 -Werror=return-type -g -pthread` and **no `-O`**.
At a global `-O0`, `#pragma GCC optimize("O3")` still optimizes each
function, but GCC inlines only `always_inline` functions, so every
`std::array::operator[]`, `std::min` and small helper becomes a real call.
Until round ten the shipped bot therefore searched about 150k nodes per move
on CodinGame against 700k in every local test (a 4.5x gap no SPRT could see).
The CG source now puts the optimize pragmas before the includes, marks its
helpers `always_inline`, and uses `cf_array` / `cf_min` / `cf_max` instead
of the std versions. Keep it that way:

- `make -C cpp_impl cg-flags` (part of `make test`) builds `cg_input.cpp`
  with CodinGame's exact command line (`g++-11` if installed).
- `make -C cpp_impl cg-speed` times the CodinGame port on an identical tree
  at `-O3` and with CodinGame's flags (`cg_selfcheck 30 13`); the checksum
  lines must match and the seconds should be within a few percent. A new
  helper that is not `always_inline` shows up here and nowhere else.
  `python tools/cg_speed_check.py <bot> [<bot> ...]` compares nodes per move
  through the real protocol, but its games diverge with timing, so it only
  resolves large gaps (identical trees have read 630k vs 716k).
- Never add `std::` containers or algorithms to the CG hot path. Large
  tables that must live on the heap use `cf_heap_array` (the transposition
  table was a `std::vector` until round ten's follow-up).
- Mark every new hot-path helper `always_inline`, including templated ones
  and ones whose signature wraps across lines. That includes helpers in the
  generated eval headers: `tools/nnue_emit_mininet_header.py` and
  `tools/nnue_emit_macro_header.py` emit `d16_mini_hsum256` and
  `evaluate_macro_key` with the attribute, so regenerate rather than
  hand-edit.
- To see what is still out of line, build the readable source the CodinGame
  way and list the `call` targets inside `CrossfishDev::search`,
  `CrossfishDev::qsearch` and `CrossfishDev::search_leaf`:
  `g++-11 -std=gnu++17 -Werror=return-type -g -pthread -Icpp_impl -o /tmp/cg cpp_impl/codingame_nnue.cpp && objdump -d -C --no-show-raw-insn /tmp/cg`.
  Only recursion, the `std::chrono` clock read (once per 128 nodes),
  `memcpy`/`memmove`, `__stack_chk_fail` and the never-taken lazy
  `macro_load_packed()` branch should remain.

**CodinGame performance gate.** Every pull request (and every push to a
branch other than `main`) runs `tools/cg_perf_gate.py` in a `gcc:11.2` Linux container
(`tools/cg_gate/Dockerfile`, workflow `cg-perf-gate`), which builds the paste
file exactly as CodinGame does and compares it with `main`:

| Check | Fails when |
| --- | --- |
| fresh | `cg_input.cpp` is not the minifier's current output of `codingame_nnue.cpp` |
| size | over 100,000 characters (UTF-16 units) |
| speed | nodes per millisecond over full-budget replies (80 ms or more), paired protocol games vs a random opponent: slower than base by more than 5% *and* the 95% interval below 1 |
| inlining | the CodinGame-flags build below 85% of the same source at `-O3` (a hot helper lost `always_inline`) |
| latency | first reply 1,000 ms or more, or the 99th percentile of later replies 95 ms or more |
| smoke | candidate vs base at 90 ms (200 games, random openings): timeouts over 1%, or a score significantly below base (Elo is informational; strength is the SPRT's job) |
| book | `tools/play_book_protocol_check.py` fails on the candidate build |

Locally: `make -C cpp_impl cg-gate` (needs Docker; `CG_GATE_BASE=<rev>` to
compare with another revision), or `python tools/cg_perf_gate.py` on Linux with
g++ installed. A non-GCC compiler is refused: clang ignores the source's
`#pragma GCC optimize/target`, so its build says nothing about CodinGame
(`--allow-non-gcc` runs a labelled dry run; its inlining check fails, as it
should).

CI (`.github/workflows/`) runs `make test` on Ubuntu (`tests`, on every push
and pull request) and the performance gate above (`cg-perf-gate`, on pull
requests and pushes to branches other than `main`). A local
Windows toolchain that matches the `-O3` flags is enough for SPRT. Do not
enable FMA in MiniNet; it will disagree with the scalar reference.

`make -C cpp_impl sprt` rebuilds `bin/test_bots` when headers change, then runs it. If you run a stale `test_bots` binary, you are SPRTing yesterday's Dev.

### Rules for an auto-research agent

- One hypothesis, one Dev diff, one SPRT sequence. Log the hypothesis in the commit or the SPRT header comment.
- Revert Dev to Prev after every failed or killed gate. Do not stack unproven diffs.
- Freeze Prev on every pass before the next idea. Hill-climbing without a freeze is measuring against a moving leftover.
- Prefer the next experiment that is *cheap to falsify*: a LUT that must match `eval_consistency`, a search constant, a qsearch skip. Do not start with a new architecture and a week of training.
- If depth 4 is ~0 and 90 ms is a big win, you shipped speed. If depth 4 wins and 90 ms dies, you shipped a slow eval; make it cheaper or drop it.
- If 90 ms Elo is +1 to +3 after ~8k games, kill it. It will not clear H1 = +5 in a useful amount of time. A 20 ms screen that is already stuck near 0 can save the machine.
- Never run two SPRTs at once. Never edit Prev "just for the test". Never ship `codingame_nnue.cpp` from an unfrozen Dev.
- Sequential gates: depth 4 (if eval) → optional 20 ms screen → official 90 ms. Stop at the first fail. Do not ship on 20 ms alone.
- Sequential passes do not add. Rounds ten and eleven passed at +9.2 and +11.0 against moving baselines and measured +15.4 as one bundle. When several accepted steps ship together, measure the bundle against `main` directly.
- A reviewer should be able to reproduce the claim: say which offset (`SPRT_GAME_OFFSET`) and seed a run used. An SPRT stops at its first LLR crossing, so the Elo it prints is biased upward; an independent rerun on fresh openings usually lands lower. The reviews recorded in improvement-log sections 47, 48 and 51 all did; round ten alone went from +9.2 to +3.1.
- The local SPRT cannot see anything CodinGame's compiler does. Every port must pass `port-check` and the CodinGame performance gate as well as the SPRT.

## CodinGame file and minifier

CodinGame's source cap is **100,000 characters**, counted as UTF-16 code
units. The network weights and the opening book are packed at 14 bits per
character using CJK ideographs (see `documentation/minification.md`), so the file is larger in
bytes than in counted characters; trust the minifier's count, not `wc -c`. Paste
**`cpp_impl/cg_input.cpp`** into the IDE; the readable source and generated
headers are intentionally kept separate for review.

`tools/cg_minify.py` is an ice4-style minifier: it can inline local quoted
headers, strips comments and indentation, renames identifiers, and packs
tokens. It does not change search or eval. Rebuild the paste file after
editing the readable source:

```bash
python3 tools/cg_minify.py cpp_impl/codingame_nnue.cpp \
  -o cpp_impl/cg_input.cpp --inline-local
```

See the [minification guide](documentation/minification.md) for the complete
source-to-submission pipeline, neural payload encoding, minifier
implementation, and validation procedure.

`--no-rename` is whitespace-only (no identifier shortening). When regenerating the submission from a net, `tools/nnue_emit_mininet_cg.py` minifies by default; pass `--no-minify` to keep the readable file.

## Opening book

The CodinGame bot plays its first moves from an opening book chosen by
uttt.ai, `cpp_impl/play_book_data.hpp`: a tree after the first player's
center-center (moving second, the book assumes the opponent opened there) that
covers the replies uttt.ai's policy rates at 0.03 or more, grown deepest along
the likeliest lines (to ply 18). Our moves are uttt.ai's after a
3,200-simulation search, with a crossfish veto. It is 34,066 positions in 13,456
characters, stored without keys as digits along a fixed walk of the book.

Against the previous full-coverage book (same engine, 90 ms): **+99.4 ± 11.2**
vs +18.7 ± 11.1 head-to-head against the plain engine over 3,000 games. The
transfer test against the round-six engine, which the book was not built
from, gave a paired book value of +50.1 vs +12.2 in the author's run and
**+72.0 vs +26.3** in an independent review on a new seed (500 openings each):
about 2.7 times the old book. uttt.ai's reasonable replies hold 98-99% of what
three unrelated engines play, which is why this selective book transfers where
earlier ones did not. Moving second against anything but a center-center
opening, the bot has no book. Design, measurements and the regeneration
procedure are in `documentation/play_book.md`.

## Latest strength result

The shipped engine is `main`'s `cpp_impl/cg_input.cpp`: the round-eleven
search with the mate-window pruning fix, the D16/H8 MiniNet and macro
residual, the CodinGame-compiler inlining work and the uttt.ai opening book.
It is **90,095 characters**, 9,905 under the cap (the minifier's count; `wc -c`
reports UTF-8 bytes). Replies take 90.1-90.6 ms against the 100 ms referee,
and the first turn about 140-230 ms of its 1,000 ms.

Each accepted step, newest first. Elo is against the step before it unless
stated, at 90 ms with the external referee, H0=0 / H1=+5, pentanomial pairs on
the 50,000-position book; the improvement-log section has the full record.

| Date | Step | Result | Log |
| --- | --- | --- | ---: |
| 2026-09-24 | uttt.ai opening book | paired book value vs the round-six engine +72.0 (full-coverage book +26.3), same 500 openings | §54 |
| 2026-09-24 | Inline the remaining CodinGame hot-path calls | +4.6% to +8.7% nodes/ms with CodinGame's flags, tree identical | §52 |
| 2026-09-24 | Mate-window pruning fix (bug fix) | N=13212, +0.26 ± 4.17, LLR +3.05 (H0=-5, H1=0) PASS | §51 |
| 2026-09-24 | Speed rounds ten and eleven, tree identical | N=2748, 818-1234-696, +15.43 ± 9.05, LLR +3.01 PASS, as one bundle vs round nine | §48-49 |
| 2026-09-24 | Build fast under CodinGame's own flags (no `-O`) | 4.9x nodes per move built CodinGame's way; new vs old paste file +163.0 ± 31.7 (N=400) and +160.5 ± 25.2 (N=600) | §47 |
| 2026-09-23 | CJK14 payload repack; first opening book | size only; book +18.8 ± 9.1 head-to-head | §45-46 |
| 2026-09-22 | Round nine: hot-path rewrite, tree identical | N=2366, 723-1048-595, +18.81 ± 10.18, LLR +3.01 PASS (+22% NPS) | §44 |
| 2026-09-14 | Round eight: exact macro-state correction history | N=4672, +29.59 ± 7.63, LLR +3.02 (H0=+20, H1=+25) PASS | §42 |
| 2026-09-13 | Round seven: D16/H8 MiniNet, macro residual, timeout hardening | N=2954, 1100-937-917, +21.55 ± 10.37, LLR +3.11 PASS vs round six | §37-39 |
| 2026-09-12 | Round six | about +25 (direct H0=+30 run stopped at +23.70 ± 10.51), 95 ms | §33-36 |
| 2026-09-10 | Round five | N=1056, +53.72 ± 17.23, LLR +3.00 PASS, 95 ms | §26-32 |

The Dev-vs-Prev SPRT compiles both engines at `-O3`, so it measures the search
and eval but not the CodinGame build; that is why the rows for sections 47 and
52 are CodinGame-built matches and speed measurements instead.

## Layout

- `cpp_impl/global_board.hpp` — board, movegen, make/unmake (shared by tests and SPRT)
- `cpp_impl/crossfish_dev.hpp` / `crossfish_prev.hpp` — search + eval
- `cpp_impl/codingame_nnue.cpp` — readable CodinGame search source
- `cpp_impl/mini_eval_d16.hpp` / `macro_eval.hpp` — generated packed eval
- `cpp_impl/cg_input.cpp` — bundled/minified paste file for the CodinGame IDE
- `cpp_impl/crossfish.cpp` — self-contained HCE CG bot (packing source for MiniNet)
- `cpp_impl/test_bots.cpp` — SPRT / Texel harness
- `cpp_impl/bench_ab.cpp` — deterministic Dev-vs-Prev equivalence and node-count screen (`make -C cpp_impl bench`)
- `cpp_impl/cg_selfcheck.cpp` — checksums the CodinGame port's fixed-depth scores and node counts (and, with `cg-speed`, times them)
- `cpp_impl/engine_selfcheck.cpp` — the same checksum over Dev or Prev, so `make -C cpp_impl port-check` can compare a port with Dev
- `cpp_impl/opening_book.bin` — frozen 50,000-position depth-16-qualified SPRT book
- `cpp_impl/play_book.hpp` / `play_book_data.hpp` — gameplay opening book runtime and payload; `play_book_*.cpp` are its packer, checker, generator and match tools
- `cpp_impl/mini_eval.hpp` — retired D8/H4 MiniNet, kept for unit tests
- `tools/cg_minify.py` — ice4-style minifier used to build `cg_input.cpp`
- `tools/cg_perf_gate.py`, `tools/cg_gate/Dockerfile` — the CodinGame performance gate CI runs on every pull request
- `tools/cg_speed_check.py`, `tools/speed_ab.py` — nodes-per-move through the real protocol, and repeated paired Dev-vs-Prev timing with a confidence interval
- `tools/nnue_*.py` — MiniNet and macro-head training and header emitters; `tools/experiments/full_nnue/` archives the rejected full-NNUE study (log §53)
- `cpp_impl/datagen.cpp`, `tools/eval_*.py`, `tools/nnue_train_blend.py`, `tools/round_robin.py`, `tools/sprt_merge.py`, `tools/sprt_worker.py` — eval data, training and testing tools (log §55, `documentation/eval_data.md`); `tools/experiments/capacity/` holds the architecture probes
- `python_impl/crossfish.py` — original tournament entry; `python_impl/bots.py` has older bots used for backtesting
- `documentation/improvement_log.md` — chronological accepted and rejected engine experiments
- `documentation/hce_and_correction_history.md` — the handcrafted evaluation's features, tables and incremental updates, and the three correction histories
- `documentation/nnue_training_and_implementation.md` — data, training, packing, and runtime details for the learned evaluator
- `documentation/eval_data.md` — the eval data pipeline, win-probability trainer, candidate A/B builds, pooled SPRTs and round robins
- `documentation/opening_book.md` — SPRT book selection, binary format, validation, and versioning policy
- `documentation/play_book.md` — the gameplay opening book: design, measurements, regeneration
- `documentation/minification.md` — from readable source to the 100,000-character paste file
