# SPRT opening book

`cpp_impl/opening_book.bin` is a frozen set of 50,000 legal Ultimate
Tic-Tac-Toe positions used by the Dev-versus-Prev SPRT harness. It is a test
book, not a gameplay opening book: the local engine never follows it. The
CodinGame bot's gameplay book is a separate artifact,
`cpp_impl/play_book_data.hpp`, described in [play_book.md](play_book.md).

The artifact has three jobs:

1. give every candidate and baseline exactly the same starting positions;
2. avoid openings created by obvious random blunders; and
3. keep both sides close enough to equality that the engine change, rather
   than an already-decided opening, determines the result.

Every opening is paired. The harness starts one game with Dev as the first
player and one with Prev as the first player from the same position. A full
50,000-position traversal is therefore 100,000 games.

The earlier 10,000-position book supported only 20,000 unique games. A
decision-length H0=0/H1=+5 SPRT can run longer than that, especially when the
true effect is near the indifference midpoint. Reusing the same roots would
not necessarily move the point estimate, but it would violate the
independence assumptions behind the reported confidence interval and LLR.
The harness therefore never wraps a book silently: it reports an inconclusive
result when all unique pairs are exhausted.

## Frozen production definition

The production book is generated against the frozen `CrossfishPrev` engine
from merged main. Candidate selection is deterministic for the recorded
engine, seed, compiler behavior, and generator settings.

The generator uses these rules:

| Property | Production value |
| --- | ---: |
| Positions | 50,000 |
| Opening length | 4 through 10 plies |
| Minimum coverage | 500 positions at every included ply |
| RNG seed | `0xC0FFEE42` (`3237998146`) |
| Reasonable-move margin | 250 evaluation units |
| Shallow candidate search | depth 4, `abs(score) <= 300` |
| Deep prefilter | depth 12, `abs(score) <= 525` |
| Authoritative fairness search | depth 16, `abs(score) <= 300` |

The tracked version was generated on 15 September 2026 against frozen
`CrossfishPrev` baseline `88c57b4`, which freezes the engine accepted on
merged main `a7a64e9`:

| Artifact property | Value |
| --- | --- |
| File size | 800,032 bytes |
| SHA-256 | `1863013821e446ae2aab3b132a86863af64bcb83d815c2516c84a366809dfae5` |
| Mean signed score | -12.0914 |
| Mean absolute score | 159.826 |
| Absolute-score p50 / p90 / p99 / max | 165 / 275 / 298 / 300 |
| Positions by ply | 4: 3,019, 5: 8,788, 6: 5,326, 7: 10,377, 8: 6,407, 9: 9,868, 10: 6,215 |
| Distinct first moves | 81 of 81 |
| First-move frequency | minimum 278, center-center 877, maximum 1,243 |

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
several seconds per root and projected to many hours for a 50,000-position
production build. Depth 16 was chosen as the practical frozen gate; depth-20
audits remain available for samples or future book versions.

After generation, the first 500 production records were independently
re-searched at depth 16. All 500 searches succeeded, all 500 remained within
the required band, and every score reproduced its stored value exactly. A
separate depth-20 audit of the first 140 records kept 106/140 inside 300 units
and 136/140 inside 500. Its mean absolute score was 207.61, with p50 197, p90
372, p99 537, and maximum 616. This is a deterministic audit sample, not a
claim that every record would satisfy the depth-20 gate.

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
  cpp_impl/opening_book.bin 50000 7
```

Generation writes atomic progress checkpoints after every normal status
report:

- `<book>.partial` stores the accepted prefix as a valid book artifact;
- `<book>.next` stores the requested size and next deterministic candidate
  identifier.

Running the same generation command again resumes from those files. The
loader reconstructs the deduplication set and ply counts from the partial
book. On successful completion the final artifact is written and both
sidecars are removed.

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

The default statistical model is a pentanomial SPRT over complete opening
pairs. From Dev's perspective its five bins are:

1. two losses;
2. one loss and one draw;
3. one win and one loss, or two draws;
4. one win and one draw;
5. two wins.

The generalized-logistic LLR and pair-aware Elo interval follow the same
method used by official Stockfish Fishtest. The repository verifier pins a
known Fishtest reference vector and expected LLR. `SPRT_PAIR_MODEL=0` exists
only for legacy diagnostics.

`SPRT_GAME_OFFSET` and the resume counters preserve deterministic traversal.
The records are not consumed in generation order. On load, the harness makes
a fixed Fisher-Yates permutation using the book seed XOR a versioned constant,
then selects traversal index `i`. The shuffle is written out explicitly
rather than delegated to `std::shuffle`, so the same artifact has the same
order across standard-library implementations. `book inspect` prints the
traversal seed, a fingerprint, and the first record indices. For the tracked
artifact, the traversal seed is `2916390839`, the FNV-1a fingerprint is
`5306913481027657611`, and the first records are
`36648,32816,26293,43975,24552,29073,34191,17490`. The verify suite pins both
this fingerprint and the historical 10,000-position fingerprint so an
accidental order change cannot silently redefine either benchmark.

This distinction matters for short SPRTs. Generation writes positions in
acceptance order, which may contain subtle local correlations even when broad
score, ply, and first-move distributions look uniform. Permuted prefixes are
therefore a more representative sample of the full benchmark. One opening
still produces two games with colors swapped, so resume totals must be even.

Resuming a paired run requires all aggregate state:

```bash
SPRT_RESUME_WINS=... \
SPRT_RESUME_DRAWS=... \
SPRT_RESUME_LOSSES=... \
SPRT_RESUME_PENTA=LL,LD,MID,DW,WW \
SPRT_GAME_OFFSET=<next-opening-pair> \
make sprt
```

The pentanomial bins must account for exactly half of resumed W/D/L games or
the harness exits. Without `SPRT_ALLOW_BOOK_WRAP=1`, scheduling is capped at
the remaining unique records and book exhaustion is reported as
inconclusive. Explicit wrapping is reserved for diagnostics and must not be
used for an official strength decision.

## Replacement-book A/A validation

Before adoption, the 50,000-position artifact completed a 1,000-game
Prev-versus-Prev smoke test over 500 unique pairs at the official 90 ms search
budget, seven physical workers, and the external 100 ms referee:

```text
W 281 / D 456 / L 263
Penta 31 / 122 / 180 / 132 / 35
+6.25 +/- 15.50 Elo
LLR +0.300 for H0=0 / H1=+5
Timeout losses: Prev 0 / Dev 0
Maximum response: Prev 90.18 ms / Dev 90.12 ms
```

The identical engines are statistically compatible with zero, the paired
model remained numerically stable, and the new book introduced no timeout or
replay failure.

## Superseded 10,000-position artifact

The original production book remains reachable in Git history. Its benchmark
identity is retained here so old reports can be interpreted:

| Artifact property | Historical value |
| --- | --- |
| Generation date | 14 September 2026 |
| Positions / paired games | 10,000 / 20,000 |
| File size | 160,032 bytes |
| SHA-256 | `6bc7556e530ec0e1cd9c395dae16809596503480f2929c2c2ca3ba5f906c509d` |
| Traversal fingerprint | `8698397342672575767` |
| First traversal records | `9147,6031,9603,8274,9238,1815,2827,9220` |

It was replaced because official +5 Elo SPRTs can require more than 20,000
games, not because its individual positions were invalid.

## Historical comparison with the legacy random opener

A controlled fixed-size comparison measured the final Round 8 candidate
against the same frozen Prev baseline under both opening systems. Both runs
used:

- 2,000 games at 90 ms;
- eight physical workers;
- H0=0, H1=+5, and an LLR bound of 100 to prevent early stopping;
- opening-pair offset 5,000; and
- two games per opening with engine colors swapped.

The book arm used the pinned Fisher-Yates traversal. The legacy arm used
`SPRT_OPENING_BOOK=none SPRT_BOOK=1`, which reproduces the old deterministic
seeded 4-8-ply random opener.

| Opening source | W / D / L | Draw rate | Elo | LLR | Timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shuffled balanced book | 647 / 844 / 509 | 42.2% | +24.01 +/- 11.59 | +2.587 | 0 / 0 |
| Legacy random 4-8 ply | 712 / 616 / 672 | 30.8% | +6.95 +/- 12.67 | +0.508 | 0 / 0 |

The book's point estimate was 17.06 Elo higher. Treating the two Elo estimates
as independent Gaussian measurements gives an approximate 95% interval of
`-0.11` to `+34.23` Elo for that difference (`p=0.0515`, two-sided). The
strength-estimate gap is therefore strongly suggestive but narrowly misses a
conventional two-sided 95% threshold in this single matched comparison.

For reporting the candidate's expected improvement in normal benchmark play,
the book result is the preferred estimate: **+24.01 +/- 11.59 Elo**. It has
better nominal precision and samples intentionally balanced, reasonable
openings where engine quality can determine the result. This remains a
distribution-conditional estimate, not a claim that the candidate is exactly
+24 Elo under every possible opening policy.

The balance difference is unambiguous. The book produced 11.4 percentage
points more draws, with an approximate 95% interval of +/-2.96 points. Its Elo
confidence interval was 8.6% narrower; on the reported per-game model, the
legacy opener would need about 20% more games to match that nominal precision.
The book also accumulated about 5.1 times as much H0=0/H1=+5 LLR in the same
number of games, although that ratio reflects both better precision and the
larger measured effect.

Color swapping makes a lopsided random position fair in expectation, but does
not make it informative. A position already strongly favoring one player can
produce a split pair in which each engine wins once when assigned the favored
color. That contributes two decisive games but almost no evidence about the
engine change. Near-even book positions leave more scope for the candidate's
evaluation improvement to determine the result.

The two opening populations also test somewhat different games. The book uses
reasonable 4-10-ply lines filtered for depth-16 balance; the legacy opener
uses arbitrary 4-8-ply legal moves. The Elo difference can therefore include
a real interaction between the macro-evaluation change and more realistic,
later opening positions, not only reduced sampling noise.

These numbers were produced by the historical per-game trinomial harness and
the 10,000-position book. They remain useful evidence about opener quality,
but should not be mixed directly with new pair-level pentanomial results. The
current harness fixes that limitation by treating each color-swapped opening
pair as one observation.

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
