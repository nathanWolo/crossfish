# SPRT opening book

`cpp_impl/opening_book.bin` is a frozen set of 10,000 legal Ultimate
Tic-Tac-Toe positions used by the Dev-versus-Prev SPRT harness. It is a test
book, not a gameplay opening book: neither the CodinGame bot nor the local
engine follows book moves during a real game.

The artifact has three jobs:

1. give every candidate and baseline exactly the same starting positions;
2. avoid openings created by obvious random blunders; and
3. keep both sides close enough to equality that the engine change, rather
   than an already-decided opening, determines the result.

Every opening is paired. The harness starts one game with Dev as the first
player and one with Prev as the first player from the same position. A full
10,000-position traversal is therefore 20,000 games.

## Frozen production definition

The production book is generated against the frozen `CrossfishPrev` engine
from merged main. Candidate selection is deterministic for the recorded
engine, seed, compiler behavior, and generator settings.

The generator uses these rules:

| Property | Production value |
| --- | ---: |
| Positions | 10,000 |
| Opening length | 4 through 10 plies |
| Minimum coverage | 500 positions at every included ply |
| RNG seed | `0xC0FFEE42` (`3237998146`) |
| Reasonable-move margin | 250 evaluation units |
| Shallow candidate search | depth 4, `abs(score) <= 300` |
| Deep prefilter | depth 12, `abs(score) <= 525` |
| Authoritative fairness search | depth 16, `abs(score) <= 300` |

The tracked version was generated on 14 September 2026 against merged-main
baseline `5a509c4` (plus the tooling-only fixed-depth state-initialization
fix):

| Artifact property | Value |
| --- | --- |
| File size | 160,032 bytes |
| SHA-256 | `6bc7556e530ec0e1cd9c395dae16809596503480f2929c2c2ca3ba5f906c509d` |
| Mean signed score | -13.5385 |
| Mean absolute score | 161.444 |
| Absolute-score p50 / p90 / p99 / max | 169 / 275 / 298 / 300 |
| Positions by ply | 4: 784, 5: 1,854, 6: 1,004, 7: 2,087, 8: 1,190, 9: 1,891, 10: 1,190 |
| Distinct first moves | 81 of 81 |
| First-move frequency | minimum 45, center-center 189, maximum 267 |

At every opening ply, the generator evaluates every legal child with the
frozen baseline's complete static evaluator: handcrafted terms, the packed
D16 local evaluator, and the learned macro evaluator. A move is "reasonable"
when its mover-relative value is no more than 250 units below the best legal
move at that node. The deterministic RNG chooses among those reasonable
moves. Terminal lines are rejected.

First-move proposals are stratified before that reasonableness check. Every
one of the 81 legal first moves receives one proposal slot, each move in the
center miniboard receives three additional slots, and center-center receives
eight more. Every proposal slot is paired once with each target opening
length. This compensates for the center openings' unusually low acceptance
under the balance filters while preserving the exact same legality,
reasonableness, and final depth-16 score requirements.

After the line is built, a depth-4 search removes clearly unsuitable
candidates cheaply. A depth-12 score then avoids spending the more expensive
depth-16 search on obvious failures. The depth-12 result never certifies a
shipped position: every retained record is searched again at depth 16 and
must independently satisfy the final absolute-score limit of 300.

The depth-12 cutoff was calibrated on a 140-position sample from the original
book. Its score had 0.910 Pearson correlation with depth 16, and all 61
positions that passed the final depth-16 `abs(score) <= 300` condition also
passed the depth-12 `abs(score) <= 525` prefilter. This is an efficiency
choice, not a fairness claim.

Each fixed-depth score uses a fresh `CrossfishPrev` object. That prevents a
position's result from depending on transposition-table, history, or
counter-move state left by an earlier root. The fixed-depth helper explicitly
zeros history and counter-move ordering state, and the harness warms shared
evaluator tables before launching workers. Audits with 7 and 16 workers then
produced identical scores and node counts. Exact encoded board states are
deduplicated, including transpositions reached by different move sequences.

At least 500 positions are retained at each ply from 4 through 10. Remaining
slots are filled in deterministic candidate order by any position that passes
the same depth-16 condition. This keeps broad opening-length coverage without
artificially making the rarest bucket determine generation time.

## Why depth 16

The earlier prototype stored depth-4 scores. Those scores were useful as a
fast coarse filter but were not a convincing fairness certificate. In a
140-position comparison, only 61 of the depth-4-qualified positions remained
within 300 units at depth 16. The deeper search is therefore the authoritative
book score; depth 4 and depth 12 are only prefilters.

Depth 20 was also evaluated. On the same sample, depth-16 and depth-20 scores
had 0.944 Pearson correlation. Of the 61 positions inside the depth-16
`+/-300` band, 50 also remained inside the depth-20 band. Depth 20 took
several seconds per root and projected to hours for a 10,000-position
production build. Depth 16 was chosen as the practical frozen gate; depth-20
audits remain available for samples or future book versions.

After generation, the first 140 production records were independently
re-searched. At depth 16 all 140 reproduced their stored score exactly. At
depth 20, 105/140 remained within 300 units and 137/140 remained within 500;
mean absolute score was 208.30, with p50 193, p90 381, p99 511, and maximum
550. This is a deterministic audit sample, not a claim that every record
would satisfy the depth-20 gate.

Depth means the engine's normal selective fixed-depth search, including its
ordinary move ordering, extensions, reductions, pruning, qsearch, and
evaluation. It is analogous to an engine depth rather than an exhaustive
game-tree proof. The score is from the side-to-move perspective, so
fairness is tested using its absolute value.

This procedure is still engine-relative. It does not prove that a position is
objectively drawn, and a future much stronger evaluator may disagree. It does
provide a substantially deeper, reproducible baseline judgment than the
search used in normal 90 ms play.

## Commands

Validate the tracked artifact and print its metadata and distributions:

```bash
make book-inspect
```

Regenerate the production artifact:

```bash
make opening-book
```

Generation is deliberately not part of `make test` or CI. It performs tens
of thousands of depth-16 searches and is still much slower than the normal
test suite. By default the generator detects physical cores and reserves one;
an explicit worker count can be supplied directly:

```bash
./cpp_impl/bin/test_bots book generate \
  cpp_impl/opening_book.bin 10000 7
```

Audit an existing book at an arbitrary depth. A count of zero means every
position. The optional final argument writes per-position TSV data:

```bash
./cpp_impl/bin/test_bots book audit \
  cpp_impl/opening_book.bin 20 140 7 /tmp/opening-depth20.tsv
```

The audit creates a fresh baseline engine for every root and reports score
percentiles, counts inside several balance bands, nodes, and latency.

## SPRT behavior

Normal invocations automatically find the tracked book when run from the
repository root, `cpp_impl`, or `cpp_impl/bin`:

```bash
make sprt
```

If the expected artifact is missing or malformed, the harness exits with an
error. Silent fallback would make two nominally identical SPRTs use different
opening populations.

Use an explicit alternate artifact only when intentionally comparing book
versions:

```bash
SPRT_OPENING_BOOK=/path/to/alternate.bin make sprt
```

For diagnostics, `SPRT_OPENING_BOOK=none` disables the file. The legacy
`SPRT_BOOK=1` mode then uses deterministic seeded 4-8-ply random openings;
`SPRT_BOOK=0` uses unseeded random openings. Those modes are not the long-term
strength gate.

`SPRT_GAME_OFFSET` and the resume counters preserve deterministic traversal.
The records are not consumed in generation order. On load, the harness makes
a fixed Fisher-Yates permutation using the book seed XOR a versioned constant,
then selects traversal index `i` modulo the book size. The shuffle is written
out explicitly rather than delegated to `std::shuffle`, so the same artifact
has the same order across standard-library implementations. `book inspect`
prints the traversal seed, a fingerprint, and the first record indices.
For the tracked artifact, the traversal seed is `2916390839`, the FNV-1a
fingerprint is `8698397342672575767`, and the first records are
`9147,6031,9603,8274,9238,1815,2827,9220`. The verify suite pins that
fingerprint so an accidental order change cannot silently redefine the
benchmark.

This distinction matters for short SPRTs. Generation writes positions in
acceptance order, which may contain subtle local correlations even when broad
score, ply, and first-move distributions look uniform. Permuted prefixes are
therefore a more representative sample of the full benchmark. One opening
still produces two games with colors swapped, so resume totals must be even.

## Binary format

The file is intentionally compact and dependency-free. Version 2 is little
endian and starts with a 32-byte header:

| Bytes | Field |
| ---: | --- |
| 8 | Magic `CFBOOK2\0` |
| 4 | Position count |
| 4 | RNG seed |
| 2 | Final balance limit |
| 2 | Reasonable-move margin |
| 1 each | Guide depth, shallow depth, prefilter depth, final score depth |
| 1 each | Minimum ply, maximum ply, maximum encoded line length |
| 2 | Prefilter score limit |
| 3 | Reserved |

Each position is one 16-byte record:

| Bytes | Field |
| ---: | --- |
| 1 | Number of moves |
| 12 | Packed moves, `mini_board * 9 + square`; unused tail bytes are zero |
| 2 | Signed final baseline score |
| 1 | Reserved |

The loader replays every line through `GlobalBoard`, rejects illegal moves,
rejects terminal starting positions, validates bounds and version metadata,
and only then exposes the records to SPRT.

## Reproducibility and maintenance policy

Treat the book like a benchmark dataset:

- Do not regenerate it for each candidate or after every accepted engine
  change. That destroys longitudinal comparability.
- Never select positions using the Dev candidate being measured. Selection
  must use a frozen accepted baseline.
- Do not remove openings because a candidate performs poorly on them.
- Record the generator baseline commit, format version, settings, size,
  SHA-256, and summary statistics whenever a new book is intentionally
  adopted.
- Keep an old artifact or its commit reachable when introducing a new
  version, and run an A/A sanity test before using the replacement for Elo
  decisions.
- Version the book only when there is a concrete reason, such as a much
  stronger baseline, a known sampling defect, broader ply coverage, or a
  format change.

Because the set is selected for balance under one accepted engine, it can
carry some baseline-selection bias. Pairing removes first-player/color bias,
while the broad move-margin sampling and large position count reduce
line-specific overfitting. The right response to future concern is a
deliberate, documented book version—not silent regeneration.
