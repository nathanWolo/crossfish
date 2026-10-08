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

Since 2026-10-08 the two halves come from two generators:

| | First player (we opened 4 4) | Second player (we answer 4 4) |
| --- | --- | --- |
| Source | uttt.ai's book of 2026-09-24 (below), unchanged | crossfish's own deep searches: snapshot s5 of the P2 book builder (see **The second player's book**) |
| Our positions | 8,828 | 1,646, of which 5 store two or three moves |
| Our moves per position | one | one, or up to three near-tied moves the bot picks between at random |
| Deepest stored move (half-move index) | 14 | 23, the 25-ply cap |

Together that is **10,474 positions** (symmetric and transposed positions
merged) with 2,336 opponent positions whose replies the book covers.

In book, the bot still runs its normal 90 ms search before playing the book
move. That warms its transposition, history and correction tables for the
moves after the book ends, and it is exactly how the book was tested.

## The first player's half: uttt.ai's reasonable replies

Inside the first player's tree, the book covers the opponent replies a strong
independent engine considers **reasonable**: those uttt.ai's policy network
(the CodinGame-rules fork, net4) gives prior 0.03 or more. Other replies leave
the book, and the bot's normal search takes over. Lines are grown best-first
by their estimated chance of being reached, so likely lines run deep and
unlikely ones stop early. Our move in each position is uttt.ai's after a
3,200-simulation search, unless crossfish (500 ms) prefers another move and
scores uttt.ai's more than 500 worse; it vetoed 3.3% of moves.

Until 2026-10-08 the whole book came from uttt.ai, the second player's half
included:

| uttt.ai book of 2026-09-24 | |
| --- | --- |
| Our positions | 34,066 (8,828 moving first, 25,238 moving second) |
| Opponent positions expanded | 6,033 |
| Deepest line | ply 18 |
| Book moves per game (3,000 games vs the plain engine) | 6.0 moving first, 6.2 moving second |

The rest of this section and **Measured strength** describe that whole book,
as it was measured in September 2026.

### Why uttt.ai's reasonable replies

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

### Measured strength (the whole uttt.ai book, September 2026)

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

## The second player's book (deep search, from 2026-10-08)

**Why it was replaced.** By October the uttt.ai book's second-player half had
become a liability. With the NNUE engine the whole book was worth about
nothing in self-play (+8.6 +/- 12.1 Elo against no book, 3,000 games with
r13w_20), and on the ladder the second player's games against the top seven
bots decide the rank (the ladder score fits 31.6 + 10.3 x that cell's
score). The top bots play the same moves every game, so a book line with a
flaw is lost every game. Against the top seven, as second player, the old
book scored 0.064 with
r13w_20 and 0.063 +/- 0.019 with the current engine (r14 + ProbCut, n 126);
r13w_20 without any book scored 0.172 +/- 0.023 (n 198).

**How it was built.** The second player's half is snapshot **s5** of
`build_p2book.py`, run1 (2026-10-01 to 10-03, on the desktop and both
laptops; the builder, its README and the run live under
`datasets/nnue2/cg/ladder/book/p2book/`, outside the repository). The rules:

- **The stored move is the deepest search's best move.** At each position a
  depth-32 search (64 MB table) gives the engine's move, and every legal move's
  child is searched to depth 28. When the two disagree by more than 300, a
  second depth-32 search with the bot's 4 MB table decides. Positions with a
  high chance of being reached were re-searched deeper (depth 34-38). No move
  is stored because a ladder bot happens to lose to it.
- **Covered replies come from the engine's own scores**, each reply covered
  when its probability under them is 0.03 or more (at most 6 per position).
  Two things only decide where the compute goes, never which move is stored:
  a population model of first players (most top bots follow the same
  "Teccles" heuristic), which sets the reach of each line, and the ladder.
  Positions on the lines that earlier snapshots lost on the ladder were
  searched deeper (depth 36 for our first three moves) or added where the
  book had ended, and the deeper search's own best move was stored.
- **The 25-ply cap.** No stored move is deeper than half-move index 23 (ply
  24), so the book never plays a whole game.
- **Near-tied moves are stored together.** At our first three decisions, the
  other moves whose every-move score is within 150 of the stored move's (at a
  reach of 0.1 or more) are stored too, at most 3 per position, and the bot
  picks one uniformly at random. A book line with a flaw is then not lost
  every game against a deterministic opponent. Five positions have them: the
  root (3 3 or 3 4 after 4 4), 44 33 00 (1 2, 0 1 or 2 2), 44 34 04 (2 3 or
  1 3), 44 34 04 13 40 (5 1, 4 2 or 3 1) and 44 34 04 23 80 (7 0 or 7 2). The
  text book lists each position's primary move first; it is the deepest
  search's move.

s5 is the run's state when it was stopped (2026-10-03 13:43) for the round-14
net: snapshot s4 plus five deeper (depth 34) corrections. Its moves were
chosen with net r13w_20, one net before the shipped one; run2 is re-searching
the book with the shipped engine (r14 + ProbCut).

**The ladder test** (2026-10-08, improvement log section 68). The live
engine, main adda324, was submitted with three books that differ only in the
second player's half. As second player against the seven top bots whose
agents did not change:

| Second player's book | Agents | Placements | Score vs the top 7 |
| --- | --- | --- | ---: |
| uttt.ai (control) | 6783120, 6783154 | #6 32.27, #5 33.30 | 0.063 +/- 0.019 (n 126) |
| none (only the root reply 5 5) | 6785179, 6785224 | #4 32.22, #4 32.73 | 0.231 +/- 0.043 (n 78) |
| **s5 (deep search)** | 6785305, 6785340 | #4 32.32, #3 32.69 | **0.207 +/- 0.040 (n 75)** |

Both new arms beat the control by more than 2 standard errors (s5 +0.143 +/-
0.044, no book +0.167 +/- 0.047), and they are level with each other (-0.024
+/- 0.059). By the pre-registered rule the deep-search book stays: its moves
are the engine's own deepest choices, and it is the build that was live.

## Size and cost: the coded walk (payload format 2)

Positions need no keys. `PbWalker` (in `play_book.hpp`) visits the book in one
fixed order and the payload is one arithmetic-coded stream of the walk's
decisions (the LZMA binary range coder with adaptive 12-bit probabilities):

- at each of our positions, the primary book move's **rank** among the legal
  moves ordered by the NNUE's static evaluation of the position after the move
  (a game-winning move first, ties in generation order), as truncated unary
  with a context per legal-move count and rank position;
- then up to two **"another move" bits** (one adaptive context each: after
  the first stored move, after the second). A 1 is followed by the next
  stored move's rank, coded exactly like the primary's; a 0, or a third
  stored move, ends the position's list;
- then, for each stored move in order (unless it ends the game), a "the book
  continues" bit with the ply as context, and if it is 1 the opponent
  position after that move;
- at each opponent position the book continues from, one "covered" bit per
  reply that does not end the game, in the same evaluation order, with the
  ply and the reply's rank as context.

The ordering does the work: in the uttt.ai book the book move ranked first
63% of the time, and a reply of rank 0 was covered 99% of the time against 7%
at rank 7 and beyond. The "another" bits are almost free in a book where few
positions have several moves: the adaptive model brings each down to about
0.01 bits, 4 bytes for a 578-position single-move book, and each stored
alternative costs about one byte plus whatever its own line covers. Children
are evaluated through a light `PbView` of the board (markers, miniboard
states, move count) rather than `GlobalBoard::makeMove`, whose move-history
stack is slow when CodinGame compiles without `-O`. The stream is carried as
U15 text like the network weights ([minification.md](minification.md)
section 4).

| Shipped book (2026-10-08) | |
| --- | --- |
| Positions | 10,474 ours (5 with several moves: 3 with two, 2 with three; 7 further moves), 2,336 opponent positions expanded |
| Decisions as plain digits | 60,834 bits (the "another" bits not counted) |
| Payload | 3,552 bytes, **1,895 characters** (r14_d5_final_s2_rs's ordering) |
| `cg_input.cpp` | 69,861 characters, 30,139 left |
| `cg_input_native.py` | 68,860 characters, 31,140 left |
| Decode at startup | 9.5 ms in the native build, about 12 ms built with CodinGame's flags (selfcheck `book_ms`, ThinkPad) |

The uttt.ai book it replaced was 34,066 positions: 184,846 bits as plain
digits, 10,832 bytes and 5,778 characters coded with r14_d5_final_s2_rs's
ordering (10,654 bytes and 5,683 characters with r13w_20's; 10,863 bytes and
5,794 characters with r12_M2's; the mixed-radix digits needed 13,456 and the
full-coverage book before it 4,656), and took about 50 ms to decode at -O3 and
90 ms with CodinGame's flags (27.6 ms in the native build).

The packer (`play_book_pack.cpp`) and the runtime drive the same `PbWalker`
with the same evaluator (`b64::evaluate_board`, passed to `pb_init`), so
neither the order nor the probabilities can drift. **A new net changes the
ordering, so the book must be re-packed whenever `nnue_b64_net.hpp`
changes** (`make -C cpp_impl play-book`, from `cpp_impl/play_book.txt`). A
stale payload is refused, not decoded: the header carries
`PLAY_BOOK_EVAL_FINGERPRINT`, a hash of the evaluator's values on 64 fixed
pseudo-positions, and `pb_init` compares it with the evaluator it is given
before reading a bit (0.1 ms). Should the fingerprints ever agree while the
ordering differs, the walk stops as soon as it passes the expected entry
count (19 ms at -O3, 132 ms at -O0 in a test with a perturbed evaluator;
without the bound a stale book walked a tree of nonsense for 6-25 s). Either
way `pb_init` returns false, the bot plays without a book, `cg_selfcheck`
prints `book=FAILED` and exits 1, `test_play_book` fails, and the CI gate's
exact protocol check (`--book cpp_impl/play_book.txt`) fails.

**The runtime table and the random choice.** The walk rebuilds a table from
a symmetry-canonical 64-bit position hash to the position's moves in
canonical orientation, one `uint32_t` per position: bits 8i..8i+7 hold move i
(`mb * 9 + sq`; move 0 is the primary) and bits 24..31 the move count. The
low byte is the primary, so a `uint8_t` copy of an entry is exactly what the
single-move table held. `pb_lookup` maps the move back to the real
orientation and only returns it if it is legal in the actual position. At a
position with k > 1 moves it returns move `pb_random() % k`: splitmix64 over a
per-process state, seeded on the first draw from
`std::chrono::high_resolution_clock` and a stack address. CodinGame starts a
new process for every game, so every game draws its own choices; with one
move nothing is drawn. Measured on the shipped book: 3,000 draws at each of
the five positions came out within about 5% of a uniform share
(`play_book_check`, which fails beyond 20%), and 300 fresh processes of the
native launcher answered 4 4 with 3 3 161 times and 3 4 139 times.

**Format 1 is not read any more.** The single-move stream of the books before
2026-10-08 had no "another" bits, so the format-2 runtime would decode it
wrongly. The generated header declares `PLAY_BOOK_FORMAT = 2` and `pb_init`
has `static_assert(PLAY_BOOK_FORMAT == 2)`, so a format-1 payload does not
compile. Nothing in the repository needs to read one: the only format-1
payload was main's own `play_book_data.hpp`, which this change replaced; any
book is re-packed from its text (`cpp_impl/play_book.txt` is committed, and
any single-move text book packs to format 2 with the same positions and
moves, for a few bytes more); and the CI performance gate builds its base from
the base revision's own self-contained paste file (`git show
<base>:cpp_impl/cg_input.cpp`). Keeping a second decoder would only cost
paste characters.

## Files

| File | Role |
| --- | --- |
| `cpp_impl/play_book.hpp` | Runtime: canonical hash, walk, decoder, lookup and the random choice. Shipped. |
| `cpp_impl/play_book_data.hpp` | Generated payload (format 2). Shipped. Do not edit. |
| `cpp_impl/play_book_text.hpp` | Text book format (several moves per position) and string-keyed symmetry helpers for the tools. |
| `cpp_impl/play_book.txt` | The shipped book as a text book (`S` lines), the packer's default input. |
| `cpp_impl/play_book_pack.cpp` | Packs a text book into `play_book_data.hpp` (needs the NNUE: it orders moves like the bot). |
| `cpp_impl/play_book_text_dump.cpp` | Writes the shipped payload back out as a text book (`make play-book-text`); it packs again to the same payload. |
| `cpp_impl/play_book_check.cpp` | Decodes the payload and checks every stored move against the text book, and the runtime's choice at positions with several moves. |
| `cpp_impl/play_book_gen.cpp` | Generates a full-coverage book with crossfish's own search (an older method). |
| `cpp_impl/play_book_match.cpp` | Book-vs-no-book matches on the shipped book. |
| `tools/play_book_protocol_check.py` | Plays the CodinGame binary through the real protocol, with the exact book check. |
| `cg/book/uttt_book_gen.py` ([uttt.ai fork](https://github.com/nathanWolo/utttai/tree/codingame-rules)) | Generated the uttt.ai book; its first-player half is the shipped first-player book. |
| `datasets/nnue2/cg/ladder/book/p2book/build_p2book.py` (local, not in the repository) | Builds the deep-search second-player book; its README documents the method, the runs and the snapshots. |

## The text book

The text book lists only **our** positions. A reply is covered when it leads to
one of them, and the book continues after our move when any reply does; there
are no separate coverage records. Two line forms are accepted:

- `<key> <mb> <sq> <score> <ply> <prob>`: a canonical key and the move in
  canonical orientation (what `play_book_gen` writes);
- `S <seq> <move>`: the position reached from the empty board by `seq`
  (comma-separated cells, `mb * 9 + sq`, or `-` for the empty board) and our
  move there, in real orientation. The packer replays and keys it, so an
  external generator never has to reproduce the canonical key.

**Several moves per position.** Several lines may give the same position (the
same canonical key, so a line for a symmetric twin of the position counts as
the same position). Their moves, in file order, are the position's book
moves; the first is the primary. For example, the shipped book's root:

```text
S 40 36        after 4 4, play 3 3 (the primary)
S 40 37        after 4 4, play 3 4 (the alternative)
```

- A move that repeats one of the position's moves is ignored: the same
  canonical move, or any move that leads to the same canonical position (at a
  symmetric position, a mirror image of a stored move).
- A position may have at most 3 distinct moves. With more, the packer prints
  the position's lines and fails.
- A line whose move is illegal in its position makes the packer fail.
- A book with one line per position is a single-move book. CRLF line ends are
  accepted.

## Regenerating the book

The two halves have separate generators, both outside this repository. The
first player's half needs the uttt.ai fork (its GPU toolchain and net4):

```bash
# in the uttt.ai fork's cg/ directory
python book/uttt_book_gen.py book/uttt_book_v2.txt --cf crossfish_cg_debug.exe --chars 13000
# about 1.8 h on the reference machine (GPU search plus 7 crossfish threads)
```

`--chars` sets the payload budget (the packed result lands within a few percent
of it), `--cover` the prior threshold, `--sims` uttt.ai's search, and
`--cf-ms` / `--veto` crossfish's veto. The second player's half comes from
the P2 book builder (`datasets/nnue2/cg/ladder/book/p2book/`, README sections
3-8e): its `book_p2.txt` is a text book of second-player `S` lines with the
alternatives after each primary, and it keeps the 25-ply cap itself
(`--max-idx 23`). The shipped `cpp_impl/play_book.txt` is the uttt.ai book's
first-player lines (8,828 lines, in their original order) followed by s5's
`book_p2.txt` (1,653 lines). Then:

```bash
cp <the text book> cpp_impl/play_book.txt
make -C cpp_impl play-book          # pack + check against the text book (every move, the random choice)
make -C cpp_impl test               # update the two pinned table checksums in test_play_book first
make -C cpp_impl cg-input
make -C cpp_impl play-book-protocol # exact book use through the real protocol
```

and the native submission (`make cg-native`, `make cg-native-check`;
[native_build.md](native_build.md)). Check the 25-ply cap before packing: the
deepest stored move must be at half-move index 24 or less moving first and 23
or less moving second (a line's half-move index is the number of moves in its
`seq`).

Re-packing the shipped book (after a net change, or a change to the coder)
needs no generator: `cpp_impl/play_book.txt` is that book, and
`make -C cpp_impl play-book-text` regenerates it from the current payload with
the current net (one line per stored move, a position's primary first).
`make -C cpp_impl play-book-gen` still writes a full-coverage book with
crossfish's own search, in the key form.

**Checks and their numbers.** `play_book_check` prints two checksums of the
decoded table: `table_checksum`, an FNV-1a over the hashes and the primary
moves, and `moves_checksum`, over the hashes and every stored move with the
move count. `cg_selfcheck` (and so the native launcher's `selfcheck` mode)
prints the first, which must equal the local build's; `test_play_book` pins
both, and `test_play_book_moves` walks the decoded table from both roots
(every entry reached, every stored move legal and leading to its own
position, the random choice uniform). For the shipped book:
`table_checksum=10147742875230593747`, `moves_checksum=7272843604574170237`,
10,474 entries.

## Testing a book change

The official Dev-vs-Prev SPRT (`make sprt`) does not exercise the book: it
starts every game from one of the 50,000 SPRT openings, where a book that
begins at the empty board rarely applies. Gate book changes with start-position
matches instead:

```bash
make -C cpp_impl play-book-match    # 3,000 games, book vs no book
```

At a position with several book moves the match's book side picks one from
a generator seeded by the game, so the threads do not share `pb_lookup`'s
per-process generator.

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

Self-play says little about the second player's book: the local engine
already finds moves as good as the book's, and the ladder's top bots, which
play one fixed line each, decide its value. The second player's book was
judged on the ladder (above): two submissions per book, pooled, as second
player against the top seven agents that did not change during the test.
