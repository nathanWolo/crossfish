# Handcrafted evaluation and correction history

This document describes the two non-neural parts of Crossfish's evaluation:

- the **handcrafted evaluation (HCE)**, a linear function of a few board
  features that has been the base of every evaluator since 2024;
- **correction history**, three tables the search fills during play with the
  HCE's measured errors and adds back to it.

The learned parts that sit on top (the D16/H8 MiniNet and the macro residual)
are in [nnue_training_and_implementation.md](nnue_training_and_implementation.md).
The chronology of every change mentioned here, with its SPRT, is in the
[improvement log](improvement_log.md).

Everything below describes `cpp_impl/crossfish_dev.hpp`. `crossfish_prev.hpp`
and the CodinGame port in `codingame_nnue.cpp` compute the same values
(`make -C cpp_impl port-check` proves the port matches Dev);
`cpp_impl/crossfish.cpp` is an older HCE-only bot and
`cpp_impl/cg_legend_hce.cpp` the 2024 original.

## 1. Where the HCE sits

```text
static eval at an interior node   = HCE + three corrections          (RFP, futility)
qsearch stand pat                 = HCE + structural correction
                                    + MiniNet residual + macro residual
```

The HCE is the only part of the evaluation that runs at every node where the
search needs a static score. The networks run only at qsearch leaves, and only
when the HCE cannot already decide the bound (improvement log section 9), so
the HCE decides most pruning on its own. That is why it is kept cheap, and why
its error is worth correcting online.

## 2. Units and sign

- Scores are integers in "eval units". `PAWN = 10`: one pawn is one extra
  corner square on a live miniboard, the smallest feature, and pruning margins
  are written in pawns (`RFP_PAWNS = 50` is 500 units).
- The raw features are computed from player 0's point of view. `finish_hce`
  multiplies by `+1` or `-1` for the side to move, then adds the tempo and the
  side-to-move extras, so `evaluate_hce()` returns a side-to-move-relative
  score, as negamax needs.
- A won miniboard is worth 2,410 units. Mates are `±(99999 - ply)`, far outside
  any eval (section 8.2 relies on that gap).

## 3. The features

`eval_weights` holds the ten linear weights (tuned by Texel in section 6 of the
log, then frozen); `LUT_W_*` repeat the local ones for the table builder. All
"two-in-a-row" counts use the same definition: a pair of squares on one line
held by the player, with the line's third square **not held by the opponent**.

### 3.1 Global terms (the super-board)

These depend only on which miniboards are won by whom.

| Weight | Value | Feature (player 0 minus player 1) |
| --- | ---: | --- |
| `eval_weights[0]` | 2410 | miniboards won |
| `eval_weights[1]` | 836 | the center miniboard won |
| `eval_weights[2]` | 464 | corner miniboards won |
| `eval_weights[3]` | 1316 | global two-in-a-rows: two won miniboards on a line whose third is not the opponent's |
| `eval_weights[5]` | 424 | "lined-up" two-in-a-rows: as above, but a live miniboard where the player has a local two-in-a-row also counts as held |

The lined-up term is the one piece of geometry the HCE has between the two
levels: two local threats that point at the same global line are worth much
more than two isolated ones. It arrived in two 2024 steps worth +29 and +55
and has survived every later rewrite (log section 3).

A quirk worth knowing: a drawn miniboard is not "held by the opponent", so a
global two-in-a-row whose third cell is a drawn miniboard still counts, even
though that line can never be completed. The weights were tuned with this
behavior in place, and dead-board variants of the eval were tried and lost
(log sections 6 and 7).

### 3.2 Local terms (each live miniboard)

Decided (won or drawn) miniboards contribute nothing locally; only live ones
are scored.

| Weight | Value | Feature (player 0 minus player 1), per live miniboard |
| --- | ---: | --- |
| `eval_weights[4]`, `LUT_W_TIAR` | 534 | local two-in-a-rows |
| `eval_weights[6]`, `LUT_W_CENTER_SQ` | 33 | the center square |
| `eval_weights[7]`, `LUT_W_CORNER_SQ` | 10 (`PAWN`) | corner squares |
| `eval_weights[8]`, `LUT_W_SQUARES` | 33 | squares held |

A worked example: on a live miniboard, the side to move holding the center and
one corner scores 534 (one open two-in-a-row through the center) + 33 (center)
+ 10 (corner) + 2 × 33 (squares) = **643**. If the opponent takes the opposite
corner, the two-in-a-row is blocked and the opponent's corner square counts
against: 33 + 10 + 66 - 43 = **66**.

### 3.3 Side-to-move terms

| Name | Value | Applied when |
| --- | ---: | --- |
| `eval_weights[9]` (tempo) | +112 | always; the empty board scores exactly 112 |
| `FREE_MOVE_PAWNS * PAWN` | +300 | the side to move may play anywhere (it was sent to a decided miniboard), after the first move (log section 6, +16 Elo) |
| `OPP_LATENT_CAPTURE_BONUS` | −800 | the opponent has a local two-in-a-row on a live miniboard that would complete one of the opponent's global lines (log section 35, +9.8 Elo) |

The latent-capture term's sign was found empirically: a +800 version failed at
fixed depth, −400 passed fixed depth but not timed play, and −800 passed both.
Read it as "the opponent has a winning threat the static board count does not
see".

## 4. Making it cheap

A naive HCE loops over nine miniboards and dozens of line masks per call.
Three layers remove almost all of that work while computing exactly the same
number.

### 4.1 The 3×3 table

A miniboard has 3^9 = 19,683 states. `init_mini_lut()` enumerates them once and
stores, per state, the feature counts (`mini_lut`), the local score
(`mini_score`), and the squares where each side would win or make a
two-in-a-row (`mini_win_sq`, `mini_tiar_sq`, used by move ordering). This
replaced about 24 popcount pairs per miniboard and doubled NPS for the same
eval (log section 6, +55 Elo).

### 4.2 Packed-occupancy tables

Search stores a miniboard as two 9-bit masks, one per player. Converting those
to a ternary index costs arithmetic, so the tables are re-indexed by
`(p0_mask << 9) | p1_mask` (2^18 entries, most unused):

| Table | Size | Contents |
| --- | ---: | --- |
| `fast_local_score` | 512 KiB | the local score of section 3.2 |
| `fast_tiar_flags` | 256 KiB | bit 0 / bit 1: player 0 / player 1 has a local two-in-a-row |
| `fast_threat_count` | 256 KiB | global two-in-a-row count for `(ours << 9) \| theirs` over the nine super-board cells |
| `fast_win_moves` | 1 KiB | squares that complete a line for one player's 9-bit mask |

The same tables serve both levels: `fast_threat_count` indexed by the two
players' won-miniboard masks gives the global two-in-a-row counts, and indexed
with the local two-in-a-row map OR-ed in gives the lined-up count. All of them
are `static inline` (in BSS, not on the stack; README "Correctness
checklist"). This change came from PR #8 (log section 10).

### 4.3 Incremental accumulation

`evaluate_hce()` recomputes from the board; the search calls
`evaluate_hce_incremental()` instead, which reads state kept up to date by
make/unmake:

- `hce_local_score`, the sum of the live miniboards' local scores;
- `hce_mb_scores[9]` and `hce_mb_flags[9]`, each miniboard's contribution;
- `hce_tiar_maps[2]`, which miniboards hold a local two-in-a-row for each
  player;
- `hce_global_score`, the global terms of section 3.1 except the lined-up term.

A move changes only the miniboard it is played in, so make calls
`set_hce_mb(board, mb)` for that one miniboard (two table loads).
`hce_global_score` depends only on decided miniboards, so it is recomputed only
when a move decides one: about 168k times instead of 1.2M in a 2.1M-node
search (log section 44). Unmake restores the old values from the move's
32-byte undo record rather than recomputing (log section 48).

What remains per evaluated node is `finish_hce_with_global()`: two
`fast_threat_count` loads for the lined-up term, a sign, the tempo and
`eval_extra_from_maps()` for the side-to-move terms, which reads only masks
the search already keeps.

### 4.4 Reference paths and tests

Every fast path has a slow reference that must agree with it:

- `evaluate_hce()` (from the board) must equal `evaluate_hce_incremental()`;
  `lut_capture_block_tiar` in `unit_tests.cpp` checks the incremental state
  and move-scoring masks against it.
- `eval_diffs()` returns the ten raw feature differences and `eval_parts()` the
  frozen global part plus the live miniboard indices. They exist for Texel
  tuning (`test_bots tune hce`) and for `eval_consistency` in
  `unit_tests.cpp`, which replays 300 random games and requires both the
  linear formula over `eval_diffs` (plus `eval_extra`) and the `eval_parts`
  decomposition to reproduce `evaluate_hce()` exactly, and the empty board to
  score exactly the tempo.

A table rewrite that disagrees with the linear features is a bug even when it
is faster. That rule exists because a 2024 eval typo was worth 76 Elo (log
section 3).

## 5. How the search uses the HCE

- **Reverse futility.** A non-PV node returns beta when the corrected static
  eval minus `RFP_PAWNS * depth` pawns (500 units per ply) is still at least
  beta. At depth 1 a second, guarded check adds the MiniNet residual before
  pruning a borderline case.
- **Futility.** At a non-PV node whose corrected static eval plus
  `FP_PAWNS * depth` (800 units per ply) cannot reach alpha, quiet moves after
  the first are skipped.
- **Qsearch.** The stand pat starts from HCE plus the structural correction.
  If that is at least 640 above beta (`QHCE_FAIL_HIGH_MARGIN`), qsearch returns
  without running either network. If even the largest possible network output
  cannot reach alpha, it uses that bound instead of running them. Delta
  pruning stops when the stand pat plus 350 pawns is still below alpha.
- **Mate-range guard.** None of these prunes runs against a mate-range bound
  (`|beta|` or `|alpha|` at or above 90,000): a static eval says nothing about
  how soon anyone is mated. Without that guard, every aspiration iteration
  after a mate score failed low several times (log section 51).
- **Fixed-depth mode.** `test_bots depth N` sets `g_disable_eval_prune`, which
  turns all three prunes off. That is the equal-depth gate for eval changes.

The aspiration window (`ASP_PAWNS = 40`, 400 units) is also expressed in HCE
units.

## 6. Correction history

### 6.1 What it corrects

A static eval is systematically wrong in some kinds of positions: early,
locked into a bad miniboard, or facing a macro shape the linear terms misprice.
Correction history measures that error during the search itself, as the gap
between what a search returned and what the static eval predicted, and adds a
running estimate of it back to the static eval the next time a position of the
same kind appears. The HCE's weights never change; the correction learns what
they miss, per game, online.

### 6.2 The three tables

Each table is indexed by an **exact** key, so there is no hashing and no
collision:

| Table | Key | Entries | Grain | Max correction | Log |
| --- | --- | ---: | ---: | ---: | --- |
| `corr_hist` (structural) | side to move × forced miniboard (0-8, or 9 for a free move) × 9-bit mask of decided miniboards | 2 × 10 × 512 | 24 | ±682 | §14, §19 |
| `corr_local_hist` | the exact side-to-move-relative contents of the forced miniboard (one of 19,683 states, or free move) | 19,684 | 48 | ±341 | §33 |
| `corr_macro_hist` | the 18-bit macro key: each miniboard's class (live / mine / theirs / drawn) from the side to move's view | 262,144 | 12 | ±1365 | §42 |

Each entry is a 4-byte `CorrEntry`: a `raw` accumulator and the `applied`
value `raw / grain`, stored side by side so reading the correction is one load
(log section 35). The macro table is 1 MiB; all three live in the engine
object.

The structural key separates the two facts the fixed eval misprices most, game
phase and being locked into one miniboard. The local key adds the forced
miniboard's exact shape. The macro key adds which player owns each decided
miniboard, which the structural mask cannot see. Replacing the exact mask with
an owner/material count fitted held-out residuals better but lost about 9 Elo
in play (log section 19): the exact layout matters.

### 6.3 Applying it

```text
corrected = HCE + structural.applied + local.applied + macro.applied
            (clamped to ±(90000 - 8000 - 1))
```

`corrected_eval()` is the static eval used by reverse futility and futility.
Qsearch adds only the structural entry: it is the cheapest to look up, and
reading the 1 MiB macro table at every tactical node cost more than it gained
(log section 42). The clamp keeps a corrected eval out of the mate band with
room for the MiniNet residual on top.

### 6.4 Learning it

After a non-PV node finishes its search and stores its TT entry, it updates all
three entries it read, if the result actually says something about the static
eval:

- an **exact** score always;
- a **lower bound** (fail high) only if it is above the corrected static eval;
- an **upper bound** (fail low) only if it is below it;
- never for a mate-range score.

The update, per table:

```text
diff = clamp(best - corrected_static_eval, ±1024)
w    = min(depth, 8)
raw += diff * w  -  raw * |diff| * w / 16384
raw  = clamp(raw, ±16384)
applied = raw / grain
```

The first term moves the entry toward the error, weighted by search depth. The
second is a "gravity" term: it shrinks the entry toward zero in proportion to
the size of the new evidence, so an entry cannot run away, and old evidence
fades as new evidence arrives. With `|diff| * w ≤ 8192 ≤ 16384` each update is
a contraction. Because `diff` is measured against the already-corrected eval,
the entry stops moving once the corrected eval's errors in that bucket balance
out. In effect it tracks a depth-weighted, recency-weighted, slightly shrunk
average of the static eval's error, capped at `16384 / grain`.

The grains set how strongly each table speaks. A grain of 32 for the
structural table was the original; 24 (a larger maximum correction) passed at
+8.4 Elo, while 48 was flat (log section 19). The local table is deliberately
weaker (48), the macro table stronger (12).

### 6.5 The macro prior

An untouched macro entry is not started at zero. Once at least three
miniboards are decided, its first use initializes it from the shipped macro
network's free-choice output for that exact macro key (`evaluate_macro_key(9,
key)`), scaled into raw units. A raw value of 1 marks "initialized to zero" so
the prior is written only once. Starting the prior at every game phase, or
without the three-miniboard gate, was weaker (log section 42).

### 6.6 Lifetime

- The tables are members of the engine object and **persist across moves** of
  a game (log section 15). On CodinGame one engine plays the whole match.
  Halving them at the start of each turn cost about 7 Elo (section 17).
- `search_fixed_depth()` clears all three, so fixed-depth labels, opening-book
  scores and `bench_ab` runs are deterministic functions of the position.
- `test_bots` reuses one engine pair for both games of an opening pair, so the
  second game inherits the first game's tables. This is part of why
  `test_bots depth N` is not an equivalence test (README).

### 6.7 What it was worth

| Change | Result | Log |
| --- | --- | --- |
| Structural correction history | +22.5 ± 10.7 measured alone (N=3000) | §14 |
| Persist it across moves | part of the +40 bundle of PR #10 | §15 |
| Grain 32 → 24 | +8.44 ± 5.99 | §19 |
| Forced-miniboard shape table | +11.14 ± 7.14 at 95 ms | §33 |
| Macro-state table with the learned prior | +29.59 ± 7.63 (H0=+20) at 90 ms | §42 |

Rejected nearby: an owner/material key instead of the exact mask (−9), halving
each turn (−7), an exact local-threat key over both players' two-in-a-row maps
(−1.85 ± 14.32, section 11), and a late-game tempo term fitted to the measured
static-vs-search bias (−6.9; the correction histories already absorb it).

## 7. History in brief

| When | Change | Log |
| --- | --- | --- |
| 2024 | Board wins, two-in-a-rows and the lined-up term; eval typo fixes worth up to +76 | §1, §3 |
| Aug 2026 | Finished miniboards skipped correctly (an operator-precedence bug had skipped only miniboard 0) | §5 |
| Aug 2026 | 3×3 lookup table, same features, 2× NPS (+55); Texel-tuned weights (+25); free-move bonus (+16) | §6 |
| Sep 2026 | Packed `1 << 18` tables (with the MiniNet projection, ~2× NPS) | §10 |
| Sep 2026 | Incremental local accumulator | §15 |
| Sep 2026 | Correction history, three tables | §14, §19, §33, §42 |
| Sep 2026 | Opponent latent-capture term (−800) | §35 |
| Sep 2026 | Cached global term and undo records (tree-identical speed) | §44, §48 |
| Sep 2026 | Prunes kept away from mate-range bounds | §51 |

## 8. Changing the HCE

- **Features, not weights.** The weights sit at a local optimum: hand-moved
  weights lost 110 Elo, an unconstrained second Texel pass lost 11, and SPSA
  over the six global weights left every one within 3% (log sections 5, 6 and
  11). New features on top of the HCE mostly lost too (win-in-one, dead
  boards, forks, live-third threats; section 11), because qsearch already sees
  the tactics they encode. A feature has to describe something neither
  qsearch nor the networks can see.
- **Gate it as an eval change:** `make test` (including `eval_consistency`),
  then `test_bots depth 4`, then the official 90 ms SPRT. A better leaf that
  costs NPS can be worth nothing at 90 ms.
- **Keep the fast and reference paths identical.** Update the table builder,
  `evaluate_hce()`, `eval_diffs()` / `eval_parts()` and the incremental
  accumulator together.
- **Port it.** `codingame_nnue.cpp` has its own copy of the HCE. After a pass,
  port it and run `make -C cpp_impl port-check`; new hot-path helpers there
  must be `always_inline` (README "Compiler and local builds").
- **Check the correction histories still make sense.** They learn the new
  HCE's errors automatically, but a change that moves the static eval a long
  way in some phase changes what they have to absorb. Look at the equal-depth
  result before assuming the corrections will cover a regression.

`crossfish_dev.hpp` also carries compile-time hooks for experiments that are
off by default and have no recorded result: `CROSSFISH_ACTIVE_MINE_FORK_BONUS`
and `CROSSFISH_ONE_SAFE_SEND_BONUS` (extra side-to-move terms), and
`CROSSFISH_MOVE_CORRECTION_GRAIN` / `CROSSFISH_PREV_LOCAL_CORRECTION_GRAIN`
(correction tables keyed by the previous move and by the previous move's
miniboard). If you test one, record the result in the log.
