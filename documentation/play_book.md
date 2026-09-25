# Gameplay opening book

`cpp_impl/play_book_data.hpp` is the opening book the CodinGame bot plays
from. It is unrelated to `cpp_impl/opening_book.bin`, which is the frozen set of
SPRT *starting positions* described in [opening_book.md](opening_book.md).

## What it covers

The book is a tree that starts after the first player's **center-center**:

- moving first, the bot opens center-center (unchanged) and the book starts at
  the opponent's reply;
- moving second, the book **assumes** the opponent opened center-center and
  starts at our reply to it. Any other first move means no book in that game.

Inside the tree, the book covers the opponent replies a strong independent
engine considers **reasonable**: those uttt.ai's policy network (the
CodinGame-rules fork, net4) gives prior 0.03 or more. Other replies leave the
book, and the bot's normal search takes over. Lines are grown best-first by
their estimated chance of being reached, so likely lines run deep (to ply 18)
and unlikely ones stop early. Our move in each position is uttt.ai's after a
3,200-simulation search, unless crossfish (500 ms) prefers another move and
scores uttt.ai's more than 500 worse; it vetoed 3.3% of moves.

| | |
| --- | --- |
| Our positions | 34,066 (after merging symmetric and transposed positions) |
| Opponent positions expanded | 6,033 |
| Deepest line | ply 18 |
| Book moves per game (3,000 games vs the plain engine) | 6.0 moving first, 6.2 moving second |

In book, the bot still runs its normal 90 ms search before playing the book
move. That warms its transposition, history and correction tables for the
moves after the book ends, and it is exactly how the book was tested.

## Why uttt.ai's reasonable replies

The previous book covered **every** reply to a fixed depth (our first 5 moves
moving first, 4 moving second), because the selective books tried before it
did not transfer: a pilot that pruned replies with crossfish's own depth-12
ranking left the book after about three moves (crossfish's 90 ms reply was that
ranking's top choice only 29% of the time), and a book grown from crossfish's
self-play was worth +38.5 against crossfish but about 0 against a different
engine, because it had learned one opponent's replies.

uttt.ai's policy predicts what *other* engines play far better. Over 500 early
positions from mixed play by five engines, the share of each engine's reply
inside uttt.ai's reasonable set (`cg/analysis/book_coverage.py` in the
[uttt.ai fork](https://github.com/nathanWolo/utttai/tree/codingame-rules)):

| Prior threshold | Set size (of ~9) | crossfish | crossfish HCE bot | Legend bot | legacy Python bot |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 7.2 | 99.8% | 99.8% | 99.6% | 87% |
| **0.03** | **6.7** | **99.4%** | **98.8%** | **98.4%** | **77%** |
| 0.05 | 5.6 | 92% | 93% | 92% | 68% |

At 0.03 the book drops about a fifth of the replies at every opponent move
while missing 1-2% of what three unrelated competent engines play; replies
outside the set are mostly weak, and the bot's search handles them. The space
saved, and not spending the second-player book on 80 first moves other than
center-center, pay for lines far deeper than full coverage can reach. uttt.ai
also chooses the early moves better: in 300 opening positions its 90 ms move
beat crossfish's where the two engines' deep searches agreed on which was
better (75 to 24).

## Measured strength

Old (full coverage) and new book, identical engine, seeds and 90 ms, through
`play_book_match`:

| Match | Old book | **uttt.ai book** |
| --- | ---: | ---: |
| Mode 0: book vs plain engine, 3,000 games | +18.7 ± 11.1 | **+99.4 ± 11.2** |
| Mode 1: diverse opponent (off-center half the time), 1,000 paired openings: book value | +16.3 | **+62.8** |
| Mode 2: round-six engine, 1,000 paired openings: book value | +12.2 | **+50.1** |
| Book moves per game, first / second (mode 2) | 5.0 / 4.0 | 5.3 / 6.7 |

A paired book value is about ±27 at 1,000 openings. Mode 2 is the transfer
test the previous selective books failed: a different engine, and the new book
still gains far more.

An independent review reran mode 2 for both books on the same new seed (7),
500 paired openings each (`play_book_match_old 500 2 3 7`):

| Book | With book | Without | Book value | Book moves (first / second) |
| --- | ---: | ---: | ---: | ---: |
| Full coverage | +91.0 ± 28.2 | +64.7 ± 26.8 | +26.3 | 5.0 / 4.0 |
| **uttt.ai** | **+143.9 ± 29.1** | **+71.9 ± 27.2** | **+72.0** | **5.7 / 6.6** |

On identical openings the new book is worth about 2.7 times the old one
(+46), rather than the four times the table above suggests: the old book read
higher in this run than in the author's. Compare books on the same seed.

Mode 1 shows the cost of the center-center assumption: when the opponent opens
elsewhere there is no book, and the second player averaged 3.4 book moves, yet
the book value still quadrupled.

Against uttt.ai itself, from the empty board (the crossfish CodinGame bot with
each book vs uttt.ai net4, 90 ms each, one game at a time; crossfish opens
center-center when first, uttt.ai samples its first two moves by visit count
for variety and opened center-center in all 200 of its first-move games):

| | crossfish W / D / L | Elo | book moves (first / second) |
| --- | ---: | ---: | ---: |
| Old book | 257 / 44 / 99 | +145 ± 35 | 5.0 / 4.0 |
| **uttt.ai book** | **314 / 18 / 68** | **+249 ± 42** | **6.6 / 8.9** |

This is the setting most favourable to the new book: its moves are uttt.ai's own
deep choices and uttt.ai's replies are covered by construction. The games are
also less varied than 400 suggests (about 60 distinct lines through ply 8), so
the interval is optimistic; mode 2 above is the engine-independent evidence.

## Size and cost

Positions need no keys. `PbWalker` (in `play_book.hpp`) visits the book in one
fixed order and the payload holds only mixed-radix digits: at each of our
positions, the book move's index among the legal moves and a 0/1 "the book
continues" digit; at each opponent position the book continues from, one 0/1
"covered" digit per reply that does not end the game. The digits are packed in
56-bit chunks and carried as CJK14 text like the network weights
([minification.md](minification.md) section 4).

| | |
| --- | --- |
| Information content | 184,846 bits |
| Payload | 23,548 bytes, 13,456 characters (the full-coverage book: 4,656) |
| `cg_input.cpp` | 90,095 characters, 9,905 left |
| Decode at startup | about 9 ms, inside the 1,000 ms first turn |

The packer (`play_book_pack.cpp`) and the runtime drive the same `PbWalker`,
so their order cannot drift. At runtime the walk rebuilds a table from a
symmetry-canonical 64-bit position hash to the book move in canonical
orientation. `pb_lookup` maps the move back to the real orientation and only
returns it if it is legal in the actual position.

## Files

| File | Role |
| --- | --- |
| `cpp_impl/play_book.hpp` | Runtime: canonical hash, walk, decoder, lookup. Shipped. |
| `cpp_impl/play_book_data.hpp` | Generated payload. Shipped. Do not edit. |
| `cpp_impl/play_book_text.hpp` | Text book format and string-keyed symmetry helpers for the tools. |
| `cpp_impl/play_book_pack.cpp` | Packs a text book into `play_book_data.hpp`. |
| `cpp_impl/play_book_check.cpp` | Decodes the payload and checks every entry against the text book. |
| `cpp_impl/play_book_gen.cpp` | Generates a full-coverage book with crossfish's own search (the previous method). |
| `cpp_impl/play_book_match.cpp` | Book-vs-no-book matches on the shipped book. |
| `tools/play_book_protocol_check.py` | Plays the CodinGame binary through the real protocol. |
| `cg/book/uttt_book_gen.py` ([uttt.ai fork](https://github.com/nathanWolo/utttai/tree/codingame-rules)) | Generates the shipped book; `cg/book/uttt_book_v1.txt` is its text. |

## The text book

The text book lists only **our** positions. A reply is covered when it leads to
one of them, and the book continues after our move when any reply does; there
are no separate coverage records. Two line forms are accepted:

- `<key> <mb> <sq> <score> <ply> <prob>`: a canonical key and the move in
  canonical orientation (what `play_book_gen` writes);
- `S <seq> <move>`: the position reached from the empty board by `seq`
  (comma-separated cells, `mb * 9 + sq`) and our move there, in real
  orientation. The packer replays and keys it, so an external generator never
  has to reproduce the canonical key.

## Regenerating the book

The shipped book needs the uttt.ai fork (its GPU toolchain and net4):

```bash
# in the uttt.ai fork's cg/ directory
python book/uttt_book_gen.py book/uttt_book_v2.txt --cf crossfish_cg_debug.exe --chars 13000
# about 1.8 h on the reference machine (GPU search plus 7 crossfish threads)

cp <that text book> cpp_impl/bin/play_book.txt
make -C cpp_impl play-book          # pack + check against the text book
make -C cpp_impl test               # update the pinned table checksum in test_play_book first
make -C cpp_impl cg-input
make -C cpp_impl play-book-protocol # exact book use through the real protocol
```

`--chars` sets the payload budget (the packed result lands within a few percent
of it), `--cover` the prior threshold, `--sims` uttt.ai's search, and
`--cf-ms` / `--veto` crossfish's veto. `make -C cpp_impl play-book-gen` still
writes a full-coverage book with crossfish's own search, in the same format.
`cg_selfcheck` also prints the decoded table's entry count and checksum, which
must equal the local build's (`play_book_check` prints the same checksum).

## Testing a book change

The official Dev-vs-Prev SPRT (`make sprt`) does not exercise the book: it
starts every game from one of the 50,000 SPRT openings, where a book that
begins at the empty board rarely applies. Gate book changes with start-position
matches instead:

```bash
make -C cpp_impl play-book-match    # 3,000 games, book vs no book
```

For a paired test against a different engine, extract an older engine and
build `play_book_match` with it (round six shown; it uses the older
`mini_eval.hpp`, whose symbols do not clash with the current evaluator):

```bash
mkdir -p /tmp/opp
git show db8bce6:cpp_impl/mini_eval.hpp > /tmp/opp/mini_eval.hpp
git show db8bce6:cpp_impl/crossfish_dev.hpp | sed 's/\bCrossfishDev\b/CrossfishOld/g' > /tmp/opp/crossfish_old.hpp
cd cpp_impl
g++ -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread -I. -I/tmp/opp \
    -DPLAY_BOOK_OPPONENT_HEADER='"crossfish_old.hpp"' -o bin/play_book_match_old play_book_match.cpp
bin/play_book_match_old 1000 2
```

To compare two books fairly, build `play_book_match` from each (a git worktree
of the old revision works) and run the same modes with the same seed.
