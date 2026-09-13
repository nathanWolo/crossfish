# crossfish

Entry for the UVicAI UTTT tournament (first place) and a CodinGame Ultimate Tic-Tac-Toe engine.

The **active engine is C++**. `cpp_impl/codingame_nnue.cpp` is the readable
CodinGame engine and includes the generated packed-eval headers;
`cpp_impl/cg_input.cpp` is the bundled/minified file to paste into CodinGame.
Local SPRT compares `cpp_impl/crossfish_dev.hpp` against the frozen previous
in `cpp_impl/crossfish_prev.hpp`. `cpp_impl/cg_legend_hce.cpp` is a snapshot
from the first Legend hit. The Python tree under `python_impl/` is legacy.

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

That builds and runs the C++ unit tests, the Python rules oracle, and compiles the CodinGame binaries (`codingame_nnue.cpp`, `crossfish.cpp`, and the Legend snapshot) so they still build.

| Target | What it checks |
| --- | --- |
| `make test` | Correctness: perft, make/unmake, Zobrist, legal moves vs an independent oracle, eval internals, search legality / instant wins, plus CG compile |
| `make test-cpp` | C++ unit tests only (`cpp_impl/unit_tests.cpp`) |
| `make test-python` | Independent Python board perft / make-undo / winners (`python_impl/test_rules.py`), plus `tools/test_*.py` |
| `make verify` | The older smoke checks inside `test_bots.cpp`, then exit (no SPRT) |
| `make sprt` | Strength: Dev vs Prev self-play with SPRT at **95 ms** (slow, noisy) |
| `make roundrobin` | 10k-game pairs of current C++ vs Legend vs Python at 20ms/move |
| `make cg` | Compile the CodinGame submission |
| `make cg-input` | Rebuild `cpp_impl/cg_input.cpp` from `codingame_nnue.cpp` with the minifier |

SPRT answers "is this stronger?". Unit tests answer "did I break the rules, hashing, eval, or search?". Do not skip `make test` because SPRT passed.

SPRT hypotheses can be overridden through the environment. For example, this
tests whether Dev clears +50 Elo instead of the default 0-vs-5 screen:

```bash
SPRT_ELO0=50 SPRT_ELO1=55 make sprt
```

Other controls are `SPRT_THINK_MS`, `SPRT_LLR_BOUND`, `SPRT_MAX_GAMES`, and
`SPRT_THREADS`.

## Improving the engine

This section is the process. The goal is Elo on CodinGame Ultimate Tic-Tac-Toe, not a prettier loss, a higher training correlation, or a faster NPS number that plays worse. Other agents will hill-climb from here. Follow the loop; do not invent a parallel scoring system.

CodinGame gives 1000 ms on the first execute per player and 100 ms on later moves. The engine searches about 800 ms / 95 ms. The official SPRT bar is **95 ms/move** (the CodinGame later-move budget). A change is not shipped until it passes that gate. 20 ms is an optional cheap screen, not a ship. The submission cap is **100,000 characters**.

### The three copies of the engine

| File | Role | Who may edit it |
| --- | --- | --- |
| `cpp_impl/crossfish_prev.hpp` | Frozen baseline. `CrossfishPrev` in SPRT. | Only when **freezing** a landed win. Never during an experiment. |
| `cpp_impl/crossfish_dev.hpp` | The experiment. `CrossfishDev` in SPRT. | The only search/eval file you change while testing. |
| `cpp_impl/mini_eval.hpp` | Frozen packed D8/H4 MiniNet used by Prev and regression tests. | Baseline eval; do not change during a Dev experiment. |
| `cpp_impl/mini_eval_d16.hpp` / `macro_eval.hpp` | Generated packed evaluators used by Dev and the CG bot. | Regenerate only for an accepted eval candidate. |
| `cpp_impl/global_board.hpp` | Board, movegen, make/unmake, Zobrist. Shared by everyone. | Rules/hash only. Perft is frozen. |
| `cpp_impl/codingame_nnue.cpp` | Readable CG bot; local eval headers are bundled for submission. | After a pass, when porting. Not the experiment. |
| `cpp_impl/cg_input.cpp` | Bundled and minified paste file. | Regenerated from `codingame_nnue.cpp`. |
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
5. **Classify the change, then gate** (next subsection). Run **one** SPRT at a time, using all cores. Do not start a second SPRT on the same machine.
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
make -C cpp_impl sprt          # official: 95ms, H0=0, H1=+5, all cores
# or, from cpp_impl/bin:
./test_bots                    # 95ms (official bar)
./test_bots 20                 # optional cheap 20ms screen
./test_bots depth 4            # equal depth 4, eval pruning off
```

Default hypotheses: **H0 = 0 Elo**, **H1 = +5 Elo**. The run stops at `|LLR| >= 3` (override with `SPRT_LLR_BOUND`).

- `SPRT PASS: H1 … favored over H0` — accept the change (for that time control).
- `SPRT FAIL: H0 … favored over H1` — reject. This means "not a +5 Elo gain", not "Dev is worse". A true +2 Elo at 95 ms will often fail H0-vs-+5. That is still not a ship.
- `SPRT INCONCLUSIVE` — hit `SPRT_MAX_GAMES` without a decision.

Environment overrides:

| Variable | Meaning |
| --- | --- |
| `SPRT_THINK_MS` | Move time (default 95). Ignored for play when `depth` is set. |
| `SPRT_ELO0` / `SPRT_ELO1` | Hypotheses. `ELO1` must be greater than `ELO0`. |
| `SPRT_LLR_BOUND` | Stop when `|LLR|` reaches this (default 3). |
| `SPRT_MAX_GAMES` | Optional cap. 0 = run until LLR decides. |
| `SPRT_THREADS` | Worker count. Default is `hardware_concurrency`. |

Example: prove a huge speed win is more than +50 Elo at 95 ms:

```bash
SPRT_ELO0=50 SPRT_ELO1=55 make -C cpp_impl sprt
```

Use a raised H0 only when the first thousands of games already show a blowout. Do not use it to dress up a +8 Elo run.

Each printed line is `N W D L Elo +/- CI LLR`. The header also prints **Prev NPS** and **Dev NPS** from a 1-second startpos search. Treat NPS as a speed signal, not a strength score.

`depth N` sets a fixed search depth and turns **eval pruning off on both sides** (`g_disable_eval_prune`: no RFP / futility / qsearch-delta). That is the equal-depth gate: same node budget in ply, so a loss means a worse leaf, not a slower one.

### Which gate, in which order

| Kind of change | First gate | Then | Ship only after |
| --- | --- | --- | --- |
| Different leaf / different net / different HCE weights | `depth 4` | optional 20 ms, then **95 ms** | Official 95 ms pass. Depth-only is not enough. |
| Same eval, faster implementation (LUTs, MiniNet projection, AVX) | optional 20 ms | **95 ms** | Official 95 ms pass. Depth 4 should be ~0 Elo if the rewrite is equivalent. If depth 4 fails, the "speedup" changed the eval. |
| Search (LMR, TT, move order, pruning) | optional 20 ms | **95 ms** | Official 95 ms pass. Equal-depth can lie: more nodes at a fixed depth is not the CG game. |

Do not skip `make test` because a gate passed.

Kill a run that is clearly not going to a +5 pass: Elo stuck around 0 to +3 after many thousands of games, LLR wandering near 0. Waiting for H0 at 20k games is a waste of the machine. Record the last line and revert.

### Correctness checklist

Run `make test` after any rules, hash, eval, or search edit. It builds C++ unit tests, the Python rules oracle, and the CG binaries.

If you change HCE, `eval_consistency` in `unit_tests.cpp` still has to match `eval_diffs` / `eval_parts`. A LUT rewrite that disagrees with the linear features is a bug even if it is faster.

If you change MiniNet, scalar `evaluate_mini`, AVX `evaluate_mini_avx`, and any fast projection path must match within a couple of points on random games. The AVX path is **mul+add, not FMA**, so it matches the scalar net.

Large tables (`1<<18` scores, threat maps, MiniNet projections) must be `static inline` (BSS), not instance members. Instance arrays of that size overflow the 1 MB Windows stack and kill `match` workers. CodinGame's Linux stack may hide this. `codingame_nnue.cpp` already hit it.

Do not put a 1 MB table on `main`'s stack in the CG file.

Startpos perft is frozen in C++ and Python. If one suite's counts change, update the other in the same commit. Do not "fix" perft to match a movegen bug.

### What is not strength

These have already fooled people in this repo:

- **Holdout MAE / correlation.** A net can fit search scores better and still play the same moves (equal-depth ~0 Elo).
- **Beating static HCE on a dump.** MiniNet only runs at qsearch leaves when HCE is inside the window. Training on mates and fail-highs that search never asks the net about is the wrong target.
- **Scaling MiniNet width (D) on old HCE-only depth-6 labels.** Extra unused concat; mixer stays tiny. Measured ~0 Elo at equal depth vs the shipped D=8 H=4 residual.
- **A much wider mixer (H=128) at 20 ms.** NPS collapse, hundreds of Elo lost.
- **A slower better leaf.** A +10 Elo equal-depth eval with a 5% NPS tax can be ~0 at 20 ms.
- **SPRT against a Prev you just weakened or against old HCE when the ship is MiniNet.** Gate vs the frozen ship, not vs a convenient opponent.

Early-game positions are where HCE is weakest (minis not yet decided). If you train a net, bias data toward low ply with MiniNet-search labels, not only late self-play.

### Shipping to CodinGame

A Dev SPRT pass does **not** update the CG bot. `codingame_nnue.cpp` keeps a
standalone copy of the search but shares generated packed-eval headers. After
a freeze:

1. Port accepted search changes into `codingame_nnue.cpp`. For eval changes,
   regenerate `mini_eval_d16.hpp` and/or `macro_eval.hpp` with the matching
   tools. Keep large LUTs in static storage.
2. Run `make -C cpp_impl cg-input`; the minifier recursively inlines local
   headers before shortening the source.
3. Confirm `cg_input.cpp` is under 100k characters. Paste **that** file into CodinGame, not the readable source unless you are debugging.
4. Optionally play minified vs readable at 20 ms (`roundrobin.run_pair` on the two `match` binaries). Expect ~0 Elo if minify only renamed tokens.
5. Write the SPRT line into **Latest strength result** (N, W/D/L, Elo, LLR, hypotheses, NPS).

`nnue_emit_mininet_cg.py` minifies by default and has a coarse vs-HCE match (`Elo < -80` fails). That is a packing-smoke test, not the Dev-vs-Prev gate.

### Compiler and local builds

```text
-O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread
```

CI is `make test` on Ubuntu. A local Windows toolchain that matches those flags is enough for SPRT. Do not enable FMA in MiniNet; it will disagree with the scalar reference.

`make -C cpp_impl sprt` rebuilds `bin/test_bots` when headers change, then runs it. If you run a stale `test_bots` binary, you are SPRTing yesterday's Dev.

### Rules for an auto-research agent

- One hypothesis, one Dev diff, one SPRT sequence. Log the hypothesis in the commit or the SPRT header comment.
- Revert Dev to Prev after every failed or killed gate. Do not stack unproven diffs.
- Freeze Prev on every pass before the next idea. Hill-climbing without a freeze is measuring against a moving leftover.
- Prefer the next experiment that is *cheap to falsify*: a LUT that must match `eval_consistency`, a search constant, a qsearch skip. Do not start with a new architecture and a week of training.
- If depth 4 is ~0 and 95 ms is a big win, you shipped speed. If depth 4 wins and 95 ms dies, you shipped a slow eval; make it cheaper or drop it.
- If 95 ms Elo is +1 to +3 after ~8k games, kill it. It will not clear H1 = +5 in a useful amount of time. A 20 ms screen that is already stuck near 0 can save the machine.
- Never run two SPRTs at once. Never edit Prev "just for the test". Never ship `codingame_nnue.cpp` from an unfrozen Dev.
- Sequential gates: depth 4 (if eval) → optional 20 ms screen → official 95 ms. Stop at the first fail. Do not ship on 20 ms alone.

## CodinGame file and minifier

CodinGame's source cap is **100,000 characters**. Paste
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

`--no-rename` is whitespace-only (no identifier shortening). When regenerating the submission from a net, `tools/nnue_emit_mininet_cg.py` minifies by default; pass `--no-minify` to keep the readable file.

## Latest strength result

On 2026-09-13, the round-seven eval stack passed a strict direct 95 ms SPRT
against the merged round-six engine (`db8bce60e8f88ad4201042484dccc578fb59ecd4`):

```text
95 ms: N 5152 W 2012 D 1591 L 1549
Elo diff: +31.31 +/- 7.91
LLR: +3.063 (H0=+20, H1=+25) — PASS
Prev NPS: 13,333,760  Dev NPS: 11,787,264
```

The optional 20 ms screen also passed at N=2208, 878-630-700,
**+28.07 +/- 12.28 Elo**, LLR +3.03 (H0=0, H1=+5).

The accepted evaluator combines a larger D16/H8 local-pattern MiniNet with a
compact learned macro-context residual. The macro head is precomputed into a
5 MiB static score table keyed by constraint and base-4 super-board state;
search maintains both perspective keys only when a miniboard becomes decided.
The table exactly matches the original macro MLP and recovered about 6% NPS
within the new eval. Paste `cpp_impl/cg_input.cpp` (96,672 characters).

The prior round-five direct bundle passed the official 95 ms gate at N=1056,
433-352-271, **+53.72 +/- 17.23 Elo**, LLR +3.00 (H0=0, H1=+5).
The next sequential winner on top of that exact merged baseline adds a
lower-weight correction history keyed by the STM-relative shape of the forced
miniboard. It passed the official 95 ms gate at N=6240, 2236-1968-2036,
**+11.14 +/- 7.14 Elo**, LLR +3.02 (H0=0, H1=+5). This is an accepted
hill-climb step, not yet a new direct +30 bundle proof.

The following exact hot-path bundle removes the unused full search-board
hash, compacts killer/counter storage, keeps counter moves packed, and reuses
correction-entry references. It matched the frozen engine on move, score, and
node count across 400 randomized depth-4 positions. The authoritative 95 ms
gate passed at N=14816, 5146-4783-4887, **+6.07 +/- 4.60 Elo**, LLR +3.11
(H0=0, H1=+5). This is the second accepted step toward the next direct proof.

The third accepted step caches the finished-miniboard mask and already-scaled
correction values, then adds a bounded HCE penalty when the opponent has a
latent local capture on a macro-winning target. The authoritative 95 ms gate
passed at N=7392, 2646-2308-2438, **+9.78 +/- 6.57 Elo**, LLR +3.02
(H0=0, H1=+5). A direct 95 ms H0=+30/H1=+35 run against `d5617e5` was
stopped at N=2848, 1061-920-867, **+23.70 +/- 10.51 Elo**, LLR -0.84.
Round six therefore ships as an approximately +25 Elo improvement, not as a
formal direct +30 proof.

Startpos perft is frozen in both C++ and Python. If one suite's counts change, update the other in the same commit:

| Depth | Nodes |
| ---: | ---: |
| 1 | 81 |
| 2 | 720 |
| 3 | 6336 |
| 4 | 55080 |
| 5 | 473256 |

## Layout

- `cpp_impl/global_board.hpp` — board, movegen, make/unmake (shared by tests and SPRT)
- `cpp_impl/crossfish_dev.hpp` / `crossfish_prev.hpp` — search + eval
- `cpp_impl/codingame_nnue.cpp` — readable CodinGame search source
- `cpp_impl/mini_eval_d16.hpp` / `macro_eval.hpp` — generated packed eval
- `cpp_impl/cg_input.cpp` — bundled/minified paste file for the CodinGame IDE
- `cpp_impl/crossfish.cpp` — self-contained HCE CG bot (packing source for MiniNet)
- `cpp_impl/test_bots.cpp` — SPRT / Texel harness
- `tools/cg_minify.py` — ice4-style minifier used to build `cg_input.cpp`
- `python_impl/crossfish.py` — original tournament entry; `python_impl/bots.py` has older bots used for backtesting
