# Gameplay opening book

`cpp_impl/play_book_data.hpp` is the opening book the CodinGame bot plays
from. It is unrelated to `cpp_impl/opening_book.bin`, which is the frozen set of
SPRT *starting positions* described in [opening_book.md](opening_book.md).

## What it covers

The book covers **every** opponent reply to a fixed depth:

| Bot moves | Book moves per game | Coverage |
| --- | ---: | --- |
| First | 5 | the bot opens center-center (unchanged), then our next 5 moves against any replies |
| Second | 4 | every opponent first move, then our first 4 moves against any replies |

That is 20,883 positions after merging those equivalent under the 8 board
symmetries and transpositions. Each book move is the engine's choice after a
2-second search (depth about 18), against about depth 11-12 in the 90 ms the
bot has per move.

In book, the bot still runs its normal 90 ms search before playing the book
move. That warms its transposition, history and correction tables for the
moves after the book ends, and it is exactly how the book was tested.

## Why full coverage, not likely lines

Chess books cover a few likely lines deeply because a chess position has 30+
legal moves and a handful cover almost all serious play. Ultimate Tic-Tac-Toe
opponents usually have 9 or fewer replies, and engine replies are hard to
predict: in early positions the engine's own 90 ms choice was the top-ranked
reply under a depth-12 search only 29% of the time, and among the top five
only 84%. A selective book therefore loses a large share of games at every
level, and the losses compound.

Measured history (all at 90 ms from the start position, CodinGame protocol):

| Book | vs | Book moves / game (first / second) | Book value |
| --- | --- | --- | ---: |
| Selective pilot (margin pruning, 2,506 positions) | current engine | 3.2 / 3.1 | +12 ± 17 |
| Grown on-policy from self-play (6,506 positions) | current engine | 6.8 / 8.0 | +38.5 ± 11.0 |
| same | diverse opponent (paired) | 6.2 / 4.0 | ~+21 |
| same | round-six engine (paired) | 2.5 / 3.0 | ~0 |
| **Full coverage (20,883 positions)** | **round-six engine (paired)** | **5.0 / 4.0** | **+21.3** |
| **Full coverage, second run (shipped packed book, new seed)** | **round-six engine (paired)** | **5.0 / 4.0** | **+19.5** |
| **Full coverage** | **current engine, 3,000 games** | **5.0 / 4.0** | **+19.6 ± 11.1** |
| **Full coverage, extension (shipped packed book)** | **current engine, 1,500 games** | **5.0 / 4.0** | **+17.2 ± 15.7** |

The two head-to-head runs pool to **+18.8 ± 9.1** over 4,500 games
(1915-913-1672), LLR 3.67 against H0=0 / H1=+5: a pass under the repo's SPRT
bounds. The two round-six runs pool to about +20 over 2,000 openings.

The on-policy book looked strong only against the engine it was grown from:
it had learned that opponent's replies. Against a different engine it fell
out of book after two or three moves and gained nothing. Full coverage is
opponent-independent by construction, so it is the version that transfers to
CodinGame, where opponents are other people's engines.

"Paired" runs play every opening twice, with and without the book, against
the same opponent; the book's value is the Elo difference. Each side of a
1,000-opening paired run has a 95% interval of about ±19, so a single paired
difference is about ±27.

## Size and cost

Full coverage needs no position keys. `PbWalker` (in `play_book.hpp`) visits
the book's positions in one fixed order: our positions in turn, every legal
opponent reply in move-generation order, symmetric and transposed duplicates
skipped. The payload stores only each book move's index among the legal moves
at that position, packed in mixed radix (a 9-way choice costs log2 9 bits) in
56-bit chunks and carried as CJK14 text like the network weights
([minification.md](minification.md) section 4).

| | |
| --- | --- |
| Positions | 20,883 |
| Information content | 63,053 bits |
| Payload | 8,148 bytes, 4,656 characters |
| Whole feature in `cg_input.cpp` | +8,312 characters (74,043 total, 25,957 left) |
| Decode at startup | about 5 ms, inside the 1,000 ms first turn |

Each extra move of full coverage multiplies the book by about 8: covering one
more move per side would take roughly 9 hours of generation per side and about
16,000-18,000 more characters.

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
| `cpp_impl/play_book_gen.cpp` | Generates the text book. |
| `cpp_impl/play_book_pack.cpp` | Packs the text book into `play_book_data.hpp`. |
| `cpp_impl/play_book_check.cpp` | Decodes the payload and checks every entry against the text book. |
| `cpp_impl/play_book_match.cpp` | Book-vs-no-book matches on the shipped book. |
| `tools/play_book_protocol_check.py` | Plays the CodinGame binary through the real protocol against a random opponent. |

## Regenerating the book

Regenerate when the engine changes enough that its deep choices would differ,
or to change the depths.

```bash
make -C cpp_impl play-book-gen      # ~1.7 h on 7 threads; writes cpp_impl/bin/play_book.txt
make -C cpp_impl play-book          # pack + check against the text book
make -C cpp_impl test               # update the pinned table checksum in test_play_book first
make -C cpp_impl cg-input
make -C cpp_impl play-book-protocol # exact book coverage through the real protocol
```

Change `PLAY_BOOK_FIRST` / `PLAY_BOOK_SECOND` / `PLAY_BOOK_SEARCH_MS` in the
Makefile to change the depths or the search time; the packer fails if the
depths do not match the book it is given. `cg_selfcheck` also prints the
decoded table's entry count and checksum, which must equal the local build's
(`play_book_check` prints the same checksum).

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
