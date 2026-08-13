# crossfish

Entry for the UVicAI UTTT tournament (first place) and a CodinGame Ultimate Tic-Tac-Toe engine.

The **active engine is C++**. `cpp_impl/crossfish.cpp` is the CodinGame submission (single file). Local search experiments live in `cpp_impl/crossfish_dev.hpp` vs the frozen previous in `cpp_impl/crossfish_prev.hpp`. `cpp_impl/cg_legend_hce.cpp` is a snapshot from the first Legend hit. The Python tree under `python_impl/` is legacy.

## Verify a change

Run this after every engine/rules/eval change. It is fast and deterministic:

```bash
make test
```

That builds and runs the C++ unit tests, the Python rules oracle, and compiles the CodinGame binaries (`crossfish.cpp` and the Legend snapshot) so they still build.

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
- `cpp_impl/crossfish.cpp` — self-contained CG bot
- `cpp_impl/test_bots.cpp` — SPRT / Texel harness
- `python_impl/crossfish.py` — original tournament entry; `python_impl/bots.py` has older bots used for backtesting
