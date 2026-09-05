# crossfish

Entry for the UVicAI UTTT tournament (first place) and a CodinGame Ultimate Tic-Tac-Toe engine.

The **active engine is C++**. `cpp_impl/codingame_nnue.cpp` is the CodinGame MiniNet submission. `cpp_impl/crossfish.cpp` is the self-contained HCE bot used as the packing source for that file. Local SPRT compares `cpp_impl/crossfish_dev.hpp` against the frozen previous in `cpp_impl/crossfish_prev.hpp`. `cpp_impl/cg_legend_hce.cpp` is a snapshot from the first Legend hit. The Python tree under `python_impl/` is legacy.

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
| `make test-python` | Independent Python board perft / make-undo / winners (`python_impl/test_rules.py`) |
| `make verify` | The older smoke checks inside `test_bots.cpp`, then exit (no SPRT) |
| `make sprt` | Strength: Dev vs Prev self-play with SPRT (slow, noisy) |
| `make roundrobin` | 10k-game pairs of current C++ vs Legend vs Python at 20ms/move |
| `make cg` | Compile the CodinGame submission |

SPRT answers "is this stronger?". Unit tests answer "did I break the rules, hashing, eval, or search?". Do not skip `make test` because SPRT passed.

SPRT hypotheses can be overridden through the environment. For example, this
tests whether Dev clears +50 Elo instead of the default 0-vs-5 screen:

```bash
SPRT_ELO0=50 SPRT_ELO1=55 make sprt
```

Other controls are `SPRT_THINK_MS`, `SPRT_LLR_BOUND`, `SPRT_MAX_GAMES`, and
`SPRT_THREADS`.

## Latest strength result

On 2026-09-05, projected MiniNet first-layer lookups plus precomputed HCE local
scores/global threats passed the strict 20ms SPRT against the frozen pre-change
engine:

```text
N: 1184 W: 611 D: 293 L: 280
Elo diff: +99.7854 +/- 17.6482
LLR: +3.06318 (H0=+50, H1=+55) — PASS
Prev NPS: 4,320,256  Dev NPS: 8,737,280
```

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
- `cpp_impl/codingame_nnue.cpp` — MiniNet CodinGame submission
- `cpp_impl/crossfish.cpp` — self-contained HCE CG bot (packing source for MiniNet)
- `cpp_impl/test_bots.cpp` — SPRT / Texel harness
- `python_impl/crossfish.py` — original tournament entry; `python_impl/bots.py` has older bots used for backtesting
