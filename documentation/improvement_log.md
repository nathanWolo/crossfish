# Crossfish improvement log

This is the oral history of the engine: what we tried, what landed, what it was worth, and what we already know does not work. It is written for the next person (or agent) who will hill-climb Elo. The process itself lives in the README under **Improving the engine**. This file is the memory.

Elo numbers here are almost always **self-play against the immediately previous accepted version**, not CodinGame ladder rating and not a running total. They do not add. A +400 jump in January 2024 and a +5 pass in 2026 are not the same kind of event: the first is "the search finally knows which move to try," the second is "this is still a real gain on a strong baseline." Time controls also change. Early Python numbers come from `faceoff` scripts. C++ SPRTs ran at 20 ms/move through round three, at 95 ms (the CodinGame later-move budget) from round four, and since round seven at the official 90 ms with an external 100 ms referee (sections 38-39); some gates are equal-depth 4 with eval pruning off. From round nine the harness scores colour-swapped opening pairs (pentanomial) on a 50,000-position book (section 43).

When a commit quotes `N / W / D / L / Elo / LLR`, that is the gate that shipped the change. Failures in this file are as important as passes. Several of the largest Elo numbers in the repo are bugs being fixed, not new ideas.

---

## 1. UVicAI: a chess search in a 9×9 house (January–February 2024)

Crossfish started as a Python entry for the UVicAI Ultimate Tic-Tac-Toe tournament. The first commits copy prior work from the official repo and then, over about two weeks, transplant a classical chess search onto bitboard minis.

Ultimate TTT is not chess. There are nine 3×3 boards; a move sends the opponent to the corresponding mini; a finished mini gives a free move. The tactics are local (complete or block a mini) and global (two-in-a-rows of minis). The first engine did not understand either. It searched, but it searched in generation order, and the evaluation was thin.

The first real search idea is a transposition table. Adding TT *cutoffs* was only about **+13 Elo**. Ordering the TT move first was **+422 ± 90**. That is the largest number in the history, and it is not mysterious: without a first move that is often best, every node wastes its budget on junk. After that, the engine is a searcher. Everything else is refinement.

What followed, in Python, was a compressed chess-engine education:

| Date | Change | Self-play Elo (vs previous) |
| --- | --- | ---: |
| 2024-01-25 | Better eval | +20 |
| 2024-01-25 | TT cutoffs | +13 |
| 2024-01-25 | TT move first | **+422 ± 90** |
| 2024-01-26 | Prefer sending the opponent to a finished mini | +22 ± 20 |
| 2024-01-26 | Order completes and blocks | +29 ± 16 |
| 2024-01-26 | Smaller TT entries | +23 ± 15 |
| 2024-01-26 | Refactor to negamax | **+87 ± 14** |
| 2024-01-27 | Put two-in-a-row back into the negamax eval | **+94 ± 22** |
| 2024-01-27 | Killer moves | +73 ± 16 |
| 2024-01-27 | History heuristic | +61 ± 17 |
| 2024-01-27 | Faster eval (more numpy) | **+122 ± 17** |
| 2024-01-27 | Faster still | **+107 ± 22** |

Two lessons from that week still apply.

**Negamax is not a style choice.** The +87 was the same algorithm written so every ply is one function. Alpha/beta bugs and eval-sign bugs go away. Later C++ work is all negamax.

**Two-in-a-row is the soul of the eval.** Dropping it during the refactor cost a fortune; putting it back was +94. A won mini is a "piece." Two won minis on a line is a threat to win the game. The handcrafted eval (HCE) that still sits under MiniNet is mostly: won minis, global two-in-a-rows, local two-in-a-rows, a little center/corner/square junk, and tempo.

Speed of the leaf mattered even in Python. Vectorizing the eval was worth more than another heuristic. Nodes are the currency; a prettier eval that you cannot call is a losing eval.

The tournament entry (`0f9ffda`, 2024-02-04) adds the rest of the chess toolkit: PVS, null-move pruning, reverse futility, futility, LMR. It won **first place**. There is no SPRT line on that commit. The strength is the tournament.

A few days later the eval was simplified to a minibox score (`crossfish_v17` in `python_impl/crossfish.py`). That is the opposite of "more features": on a huge sample it was **+30 ± 4 Elo** against the complicated HCE. In UTTT, extra local geometry is easy to invent and easy to make the search worse. That fight comes back in 2026.

---

## 2. CodinGame rules, then a rewrite in C++ (February 2024)

The hackathon and CodinGame are not the same game in the details that matter to an engine (input, time, some rule edges). `435c287` edits the Python to match CodinGame. Search features that had been tuned for the contest are re-added one at a time (FP, RFP, PVS, LMR, margin tweaks). This is not a new idea; it is "make the winner legal on the new server."

Python is too slow for CodinGame's 100 ms later-move budget if you want Legend. On 2024-02-09 the C++ rewrite starts (`ac2f2b6`); two days later it plays (`2a4389c`). The first C++ bot is a translation, not a new design: bitboards, negamax, a TT, the same HCE shape.

The important infrastructure commit is `7dee9c1` (2024-02-19): **SPRT for C++ bots**, and the repo splits into `cpp_impl/` and `python_impl/`. From here on, a change is real if Dev beats Prev under a likelihood-ratio test, not if it feels faster. That discipline is why this log can quote numbers at all.

---

## 3. The first C++ hill-climb (20–28 February 2024)

This is the original "freeze a baseline, edit the other copy, SPRT" era. The C++ bot is still weak relative to later Legend, so many changes print huge Elo. They are still real. The search is learning the same lessons Python already knew, plus things Python could not afford.

### Search and ordering

| Commit | Change | SPRT |
| --- | --- | --- |
| `9833a63` | Better move ordering | +88, N=672 |
| `fd04cb1` | More move ordering | +60, N=912 |
| `249b709` | Another ordering pass | +4.6, N=26k (small, needed a long run) |
| `2a457fc` | Aspiration windows | +9.9, N=9k |
| `b277022` | Futility + reverse futility | +9.2 |
| `a0281f9` | Quiescence search | +11, N=7848 |
| `d7f2e36` | Better qsearch | +13, N=6360 |
| `f342ade` | Capture-only movegen in qsearch | +21, N=3696 |
| `1a25041` | Win checks inside qsearch | +29, N=2496 |
| `8ade001` | PVS, LMR, misc | +17, N=4488 |
| `0eee45a` | Internal iterative deepening | +7.2, N=13.9k |
| `d5c38ef` | One-reply and singular extensions | +11, N=7320 |
| `9a59d3e` | Fuse killer + history | +11, N=7704 |
| `7f43d41` | Smaller aspiration window | +4.3, N=34.7k |

Qsearch is the first time the C++ bot stops calling static eval in the middle of a tactic. UTTT "captures" are completes and blocks. Without them, a stand-pat score lies. Adding qsearch is only +11; teaching it to see wins and generate only captures is another +50 stacked. Later MiniNet will live *only* in this qsearch leaf. That decision is already implied here: the interior of the tree can stay cheap if the leaf is honest.

### Evaluation

| Commit | Change | SPRT |
| --- | --- | --- |
| `f3ee9a6`, `e482122` | Better eval | +122, then +70 |
| `0168ba4` | Stop double-counting local two-in-a-rows | +35, N=2016 |
| `d7d339e` | Bonus for lining up two-in-a-rows | +29, N=2520 |
| `c34b7a7` | Better lineup term | **+55**, N=1296 |
| `268ecd0` | More positional features | +24, N=3096 |
| `bdc04d0` | Heavier global two-in-a-row | +23, N=3072 |

The lineup term is the first time eval thinks about *geometry of threats*, not just a count. Two local two-in-a-rows that point at the same global line are a lot more than two isolated threats. That idea keeps paying. Attempts in 2026 to replace it with "smarter" 3×3 features (dead boards, forks, win-in-one bonuses) mostly lose, because qsearch already sees the tactics those terms try to encode, and the extra bias fights the search.

### Speed, and a famous own-goal

| Commit | Change | SPRT |
| --- | --- | --- |
| `9c6434a` | Stop copying boards | +25, N=3096 |
| `2d36465` | Less branching in capture/block scoring | +35, N=2040 |
| `2150189` | AVX2 on hot functions | +13, N=6264 |
| `ac474f3` | Qsearch by reference | +23, N=3192 |
| `f6f315c` | `reserve()` on move lists | +15, N=4944 |
| `1f9575e` | **Remove TT cutoffs**, pass-by-reference | **+37**, N=1824 |

`1f9575e` is the commit that later 2026 work has to undo. The Zobrist keys were already sick (see below). Cutoffs on a broken table do not help, and removing them plus cheaper calling convention won +37. In 2024 that was correct *given the hash*. In 2026, with a working hash, putting cutoffs back is +121 of the revival patch. If you only read the 2024 message you would think TT cutoffs are bad. They were bad because the keys were zero.

### Bugs that played like features

| Commit | What was wrong | SPRT |
| --- | --- | --- |
| `da0c454` | Win detection missed wins that were not a new 3-in-a-row on the last ply | +16, N=5076 |
| `6c0d923` | Mini-board draw detection | no SPRT |
| `e186f1b` | Typo in eval | +21, N=3552 |
| `d60ad37` | Another typo in eval | **+76**, N=864 |

An eval typo at +76 is a reminder: the HCE is a handful of integers. One wrong coefficient or a swapped player is a different engine. That is why `eval_consistency` exists in 2026. A LUT that disagrees with the linear features is the same class of bug, just faster.

---

## 4. Legend, then two years of silence (March 2024 – August 2026)

`0be9452` / `95a898e` (2024-03-12) snapshot the bot that first hit **CodinGame Legend**. The file is `cpp_impl/cg_legend_hce.cpp`. Do not edit it. It is the fossil.

Then the repo sleeps until August 2026. The Legend HCE is the public identity of the project for two years: a strong, fully handcrafted, AVX2 C++ search with a 3×3-aware eval, qsearch, LMR, PVS, and a transposition table that, we later learned, was not hashing.

---

## 5. The 2026 revival: the Legend bot was broken (12 August 2026)

The first modern commit, `83fd137`, is not a new idea. It is an autopsy.

**Zobrist keys never reached the board.** The constructor declared *local* arrays with the same names as the members. Every position hashed to 0. The TT always wrote slot 0. Cutoffs had been commented out in 2024 for "speed." That comment was a confession.

**Eval skipped finished miniboards with the wrong operator precedence.** `out_of_play & (1 << miniboard != 0)` binds as `out_of_play & 1`. Only mini 0 was ever skipped. The other eight finished boards still contributed two-in-a-row and square terms. The leaf was noisy in exactly the positions where the game was already decided locally.

**TT bound flags did not match the cutoff.** Even after hashing worked, a leftover mismatch would have made the table lie.

The conservative patch — working keys, matching TT flags, skip every finished mini, history heuristic, `2^18` TT, less frequent time checks — passed 20 ms SPRT at **+121 Elo** (N=569, 349–61–159, LLR +3.00). A first kitchen-sink patch (null-move, extra eval terms, qsearch delta, all at once) was about **−50 Elo** and was thrown away. That is the first modern statement of the loop: one axis, or you do not know what failed.

`cg_legend_hce.cpp` stays the original submission. The living engine moves on.

### Cheap ideas that already failed on this baseline

Before anyone trains a net, the revival tried the obvious chess tweaks against the +121 engine:

- LMR fail-high re-search: ~0 Elo. More correct, not stronger.
- Penalize "send opponent to a threat" in move order: **−45 Elo**. LMR then under-searches tactics qsearch already sees.
- Reweight board-win 2000→1200 and global two-in-a-row 1500→2200: **−110 Elo**. The feature set was locally maxed. Texel on the same counters would only reweight noise.

The eval was not going to be saved by new coefficients on the same popcounts. It needed either a faster implementation of the *same* features, or a leaf that can see patterns the counters cannot.

---

## 6. Same features, less work (12–13 August 2026)

### Alloc-free move generation — `5840146`

Search and qsearch used `std::vector` for moves and scores. Every node paid two heap allocations. Stack `Move[81]` / `int[81]` plus `fillLegalMoves` / `fillCaptures` passed 20 ms SPRT at **+37 Elo** (N=1760, 805–338–617). Startpos NPS only rose about 7% (3.88M → 4.16M). The Elo is larger than the NPS because 20 ms games live in the middlegame, where allocation noise is worse than a 1 s opening bench. Prev kept vectors on purpose so the SPRT was a clean speed test.

### The 3×3 lookup table — `dbc0cdb`

Every live mini was scored with ~24 popcount pairs plus center/corner counts. There are only `3^9 = 19683` legal 3×3 states. A table built once at startup replaces the inner loop.

The first LUT experiment *changed the features*: win-in-one bonuses, skip dead boards (minis that can never be 3-in-a-row). That version was **2× NPS and −21 Elo**. Qsearch already sees win-in-one. The extra terms fought the search.

The version that shipped is a **drop-in for the same linear features**. Same score, ~2× NPS. SPRT at 95 ms: **+55 Elo** (N=1120, 506–283–331). Depth 4 should be ~0 if you ever rewrite this table; if it is not, you changed the eval.

### Texel, the hard way — `a68b437`

Naive Adam on 10 ms self-play exploded the small terms (squares, tempo) until they rivaled a won mini. First pass: **−59 Elo**. The fix is per-weight step sizes so a pawn-scale term cannot outrun a board win, plus a pull toward the original coefficients. That pass: **+25 Elo** at 20 ms (N=2688). A second unconstrained round failed again (~−11). The landed weights are still the HCE in `crossfish_dev.hpp` (board 2410, local two-in-a-row 534, tempo 112, …).

A later attempt to Texel a *per-state* 3×3 score table on 4.5M positions moved the empty board by 7 points and was not worth an SPRT. The 3×3 table does not see new shapes from random 12-move prefixes; almost every mini still has one mark.

### Free move — `f6d23fa`

When you send the opponent to a finished mini, they get a free move: they may play anywhere. HCE treated that as a normal constraint. Scoring the free-move right to move is **+16 Elo** at 20 ms (N=4640). It is a real term. Scaling it or making it fancier later failed.

### Infrastructure that is not Elo

`b4f1fa8` / PR #2 extracts `GlobalBoard`, freezes startpos perft against a Python oracle, and makes `make test` the correctness gate. `68f2afd` adds a NEW/APPLY/GO match protocol and a 10k-game round-robin. These commits are 0 Elo and they are why later nets did not ship illegal moves. Perft is frozen (81 / 720 / 6336 / 55080 / 473256). If those counts change, you changed the rules.

---

## 7. Search quieter lines less (14 August 2026) — `96e4feb`

After free-move, the HCE leaf is good enough that the 95 ms budget is the bottleneck. The engine was spending too much time on quiet junk and not enough on the tactics that decide Legend games.

This was a *sequential* hill-climb, not one kitchen-sink SPRT. Each pass froze into Prev. Failures reverted. The commit message quotes the bundle versus previous main at 95 ms: **+80 Elo** (N=736, 349–204–183). The pieces, in order:

**Landed**

- LMR on late non-captures (not only on negative scores)
- History gravity (`h += bonus - h * bonus / 10000`) so a hot square cannot saturate and go silent
- Countermove heuristic
- History malus on quiets that fail
- Qsearch delta (stand-pat far below alpha → stop)
- LMR reductions `i/3`
- Tighter RFP (margin 500)

**Failed against the then-current baseline** (do not retry without a new reason)

- Dead-board eval, NMP, TT replacement scheme
- Extra weight on the active mini, razoring, qsearch TT, TT `1<<20`, aspiration 200
- Exact two-slot killers
- IID at depth 2, scaled free-move
- Futility margin 600

Chess folklore is not a patch list. NMP and a bigger TT are "supposed" to win. Here they did not, on this branching factor, this eval scale, and this time control. The README bar (H0=0, H1=+5) exists because a +2 idea at 20 ms is not worth the NPS tax of a more complicated search.

---

## 8. MiniNet: a residual leaf, not a second engine (15 August 2026) — `93eac80`

This is the second architecture in the project's life. The first is HCE. The second is **HCE plus a tiny net at qsearch leaves**.

### Why a net at all

HCE is a linear function of a few 3×3 counters. It is weak in the early game (nothing is decided) and it cannot represent "this shape is a fork but that identical count is dead." Search-d6 labels already know things HCE does not. A net that *replaces* HCE has to relearn won minis, tempo, and global threats from scratch. A net that *adds* to HCE only has to learn the residual.

### What we burned before MiniNet

The Stockfish-shaped idea — 199 sparse features, dual accumulator, CReLU, quantized int8 — is in `tools/nnue_train_sparse199.py`. It was trained, packed, even injected into a CG file. At equal depth it was a worse leaf: sparse d6 replace was about **−277 Elo** at depth 4. Residual WDL trained on 20 ms games was also a worse leaf than HCE. The unused sparse blob later came out in `0c3637c`. The trainer remains as a museum and as a warning: **architecture from chess plus a convenient dump is not a teacher**.

Other graves from the same week:

- Train to 20 ms WDL: the teacher is weaker than HCE.
- Mixed random+play dumps (`minires2`): −32 at depth 4.
- CReLU(4) on the mini net: collapsed to a constant residual (~+560). Uncapped ReLU was required.
- Mates (±20000) dominating Huber: clip to ±8000, keep them compressed, do not upsample.

### What actually beat HCE

A **mini-index embedding** over every 3×3 (`3^9` rows), plus location / super-cell / constraint / "am I the active board," concatenated, then a tiny ReLU MLP, **added to static HCE**.

Teacher: depth-6 full-window HCE search on **self-play boards only**, labels clipped to ±8000, Huber 1500.

The first net that won equal-depth was fat: D=32, H=128, **+48 Elo** at depth 4. At 20 ms it was about **−288 Elo**. Twelve times too slow. A better leaf you cannot afford is a worse engine. That single fact is the MiniNet design constraint.

Shrinking to **D=8, H=4** barely hurt the fit (val corr vs search 0.924 vs HCE 0.908) and is what can run at qsearch. Gates versus HCE, net only at leaves, HCE still used for RFP/futility:

| Gate | Result |
| --- | --- |
| Depth 4, eval prune off | **+54 Elo**, N=1248 |
| 20 ms | **+11 Elo**, N=6720 |
| 95 ms | **+7 Elo**, N=11264 |

That is the shipped residual: `minires_d8h4`. Equal-depth says the leaf is better. Timed Elo is much smaller because every qsearch leaf got more expensive. The CG ladder moved **Legend rank 82 → 68**. Ladder is not SPRT, but it is why the net shipped.

HCE stays on interior RFP because MiniNet is clamped near ±2000. It cannot raise a fail-high that HCE already sees, and it must not be the value RFP uses to prune a branch.

### How the net is packed

`tools/nnue_train_mininet.py` writes a CFM2 blob. `tools/nnue_emit_mininet_cg.py` embeds it in the single-file CG bot. `cpp_impl/mini_eval.hpp` is the shared packed eval for Dev, Prev, and tests. AVX is **mul+add, not FMA**, so it matches the scalar reference. FMA is a different net.

---

## 9. Spend MiniNet only when the leaf needs it (16 August 2026)

Once the net is in the leaf, the next Elo is not a bigger net. It is *not calling the net*.

### Skip on HCE fail-high — `70d8ffd`

If stand-pat HCE is already `>= beta`, qsearch can return without MiniNet. RFP also stopped evaluating the net only to throw the score away. SPRT vs the previous `codingame_nnue`:

- 20 ms: **+31 Elo** (N=2048, 869–492–687)
- 95 ms: **+20 Elo** (N=3272, 1257–942–1073)

This is the same philosophy as the 3×3 LUT: the expensive thing must be a no-op when HCE already knows.

### AVX, fail-low skip, LUT tactics — `8010df1`

Three speed/accuracy pieces, one SPRT, then frozen into Prev:

- AVX2 MiniNet (mul+add)
- Skip MiniNet when HCE cannot reach alpha even with a +8000 residual
- Capture / block / two-in-a-row move scoring from the 3×3 LUT instead of per-move AVX

- 20 ms: **+29 Elo** (N=2152, 887–558–707)
- 95 ms: **+26 Elo** (N=2416, 939–715–762)

After this, the engine is: HCE everywhere cheap, MiniNet only in the qsearch band where HCE is uncertain.

### Cleanup — `0c3637c` / #7

`SEARCH_EXPERIMENT` / `EVAL_EXPERIMENT` if-constexpr stubs, the sparse-199 inject path, and Texel helpers that only existed for dead tests come out. Dev is the landed engine again, not a museum of `#if 0`.

---

## 10. The same eval, twice as fast (5 September 2026) — #8 / `97d7ecc`

PR #8 does not change what the position scores. It precomputes it.

**HCE.** Local mini scores and global threat counts become `1<<18` lookups from the packed occupancy of a mini. The inner eval loop is a table load.

**MiniNet.** The first layer is pre-projected: each mini index already knows its contribution to the mixer, so qsearch does not rebuild embeddings from scratch.

That is a speed-only rewrite of a frozen eval. The right gate is timed SPRT; depth 4 should be ~0. The published 20 ms run used a raised hypothesis because the first thousands of games were a blowout:

```text
N: 1184 W: 611 D: 293 L: 280
Elo: +99.8 ± 17.6
LLR: +3.06  (H0=+50, H1=+55)
Prev NPS: 4.32M   Dev NPS: 8.74M
```

An independent local check saw about +56 Elo at N=320 with NPS 10.4M → 19.2M. Eval still matches `eval_consistency`. This is the largest *clean speed* jump since the 3×3 LUT.

The first implementation stored those megabyte tables as **instance members**. Windows match workers use a 1 MB stack. They died (`errno 22`). CodinGame's Linux stack hid it. `ee8be26` moves the tables to `static inline` (BSS) in the CG files. Any new LUT of this size must be static. Do not put a 1 MB array on `main`'s stack either.

The same commit writes the hill-climb process into the README, adds `tools/cg_minify.py`, and ships `cpp_impl/cg_input.cpp` (~57k characters, ~43k under the 100k cap). Paste the minified file. The readable `codingame_nnue.cpp` is ~95k and is for humans.

---

## 11. Experiments that did not ship (keep this current)

This section is the other half of the history. Retrying these without a new hypothesis wastes the machine.

**Eval features on top of HCE**

- Win-in-one / dead-board LUT extras: −21 Elo, 2× NPS
- Forks: ~−55
- Live-third global threats, extra active-mini local: fail
- Hand-moved board-win / global-2-in-a-row: −110
- Per-state Texel of 19683 scores: no movement worth SPRT

**Search**

- Kitchen-sink first 2026 patch: −50
- NMP, TT-replace, qsearch TT, TT `1<<20`, aspiration 200, razoring
- Exact killers, IID depth 2, scaled free-move, futility 600
- Send-to-threat ordering: −45
- Parallel SPRTs on one machine: do not; they steal NPS and lie
- Ply-relative TT mate scores (`value_to_tt`/`value_from_tt`): **+1.9 ± 10.6** at N=3000 on
  the #10 baseline, i.e. below a null control run the same day. The pairing is correct
  Stockfish; at 95 ms this engine almost never resolves a true mate, so the re-anchoring
  does not fire. Do not retry without a deeper time control.
- qsearch TT, second attempt on the #10 baseline: **−0.9 ± 10.6** at N=3000. Independently
  reproduces the earlier verdict above. Two baselines, same answer.
- Capture-only ProbCut on the #14 (corrhist+LMR) baseline: added and reverted in PR #10.
  Do not retry without a new margin story.
- Skip duplicate full-depth PVS probes on the same baseline: added and reverted in PR #10.

**Round ten (24 September 2026, 90 ms, H0=0/H1=+5, vs round-nine Prev)**

- History-modulated LMR, `r -= (h - 1500) / 3000` ply in hundredths:
  walk -0.67 ply; N=960, -10.86 +/- 15.34. Killed.
- Continuation history `[stm][previous move][move]` in ordering, same
  bonus/malus as history: NPS -4%; N=2106, +1.15 +/- 10.63. Killed. (On the
  NNUE engine with IIR it passed: section 62.)
- SPSA over 16 constants (margins, LMR, ordering weights, free-move and
  latent-capture terms), 8,000 pairs at 25 ms: drift was small; N=1794,
  -0.58 +/- 11.13. The constants are at a local optimum.
- Exact local-threat correction history (2^18 key: both sides' two-in-a-row
  maps): NPS -5.4%; N=1128, -1.85 +/- 14.32. Killed.
- Root move chosen from a fail-low bound: instrumented, 0 of 616 self-play
  moves affected. Not tested.
- TT bound as the pruning eval (an EXACT entry, a LOWER bound above or an
  UPPER bound below the static eval replaces it for RFP and futility):
  peaked at +14.8 at N=1200, then N=3702, +2.16 +/- 7.93. Killed.
- Singular extension with a verification search (depth >= 6, margin
  30 * depth, multi-cut; the pseudo-singular extension kept below depth 6):
  N=1938, +2.87 +/- 10.81. Killed.
- Both of the above as one bundle: N=1206, -8.64 +/- 13.94. They do not add;
  the small positive readings were noise.
- "Improving" (RFP margin half a margin smaller when the corrected static
  eval rose since ply-2): N=1812, -2.49 +/- 11.88.
- Internal iterative reduction replacing IID (no TT hit, depth >= 4: search
  one ply shallower): N=1206, -5.47 +/- 13.57. (At non-PV nodes only, keeping
  IID at PV nodes, it passed on the NNUE engine: section 60.)
- Late-move pruning (non-PV, depth <= 3, quiet moves after 3 + 2*depth^2):
  N=1230, -5.08 +/- 13.68. Only 6.5% of nodes have more than 9 moves.
- Qsearch plays a forced single move instead of standing pat: N=1212,
  -6.88 +/- 13.46.
- Late-game tempo `min(600, 18 * (n_moves - 32))`, fitted to the measured
  static-vs-search bias (+75 at 32-39 stones up to +544 at 56-63): N=1212,
  -6.88 +/- 13.51. The correction histories already absorb that bias.
- MiniNet fine-tune from the shipped net (the packed D16/H8 expands to a
  CFM2 checkpoint bit-exactly) on 600k depth-12 HCE-leaf labels with target
  search - HCE - macro: holdout MAE 1057 -> 1025, but N=1812, -0.96 +/- 11.40.
  `--pin-empty` diverged (loss rose every epoch); `--relative-empty` works.
- SPSA over the six HCE global weights: after 3,300 pairs at 25 ms every
  weight was within 3% of its value. Not worth an SPRT.
- Full Stockfish-style NNUE replacing HCE + MiniNet + macro, including a
  data-scaling study and qsearch-leaf training data: see section 53. (A
  pattern-generator NNUE on far more data did replace them: section 56.)
- Every SPRT walks the book in the same order, so near-identical engines
  share early noise; three unrelated candidates read -11 to -13 near N=600.
  Use `SPRT_GAME_OFFSET` for independent early readings.

**Middlegame search round (30 September 2026, pooled SPRTs at CodinGame-scaled budgets, section 60)**

- Macro-threat extension: a capture that gives the mover a new live global
  two-in-a-row is searched one ply deeper (depth >= 2, never stacked, once per
  path, ply < 2 x root depth). Not selective: about half of middlegame captures
  at depth >= 2 qualify, -0.3 completed ply at 90 ms. Vs the section 60 freeze:
  N=25606, +2.47 +/- 2.44, LLR -0.10 at a 25,000-game cap. Inconclusive; a
  real gain of about +2 at most.
- False-draw re-search: an LMR-reduced move whose null-window search returns
  exactly 0 at alpha 0 is re-searched at full depth. It removes the false
  exact draws of replay 906589060 but costs 5.7x nodes in drawn positions.
  N=2918, -8.81 +/- 7.55, LLR -3.79 FAIL. The mechanism: after a real draw
  sets alpha to 0, a reduced search of the winning move stops short of the win
  and returns exactly 0, which no longer triggers the re-search; the false 0
  then spreads through TT bounds.
- Futility floor (the node's stored and returned value raised to the futility
  value it pruned against): fixes the false mate bounds of section 61 too, but
  also discards the correct ones and costs +4-6% nodes at fixed depth. Stopped
  at N=6722, +1.14 +/- 4.73, LLR -1.17, in favour of section 61's guard.

**Speed round ten (section 48; all tree-identical, none kept)**

- Ternary-indexed 77 KiB miniboard table replacing the sparse 512/256 KiB
  LUTs: fewer simulated L1 misses, -1.6% to -2.6% wall clock. Tried twice.
- Huge-page (`MADV_HUGEPAGE`) transposition table: granted, but no measurable
  gain (+1.3% +/- 2.4% sat; walk -4.2% to +1.2%).
- Caching the lined-up threat term in make: +19.6M instructions (flags change
  too often, and makes outnumber evaluations).
- Out-of-line qsearch capture loop; static Zobrist tables instead of
  references; fixed-length sort rank loop: all within +/-0.3%.
- Round eleven (section 49): prefetching the sparse per-miniboard tables
  (-0.9%) or the child's macro-table entry (-0.3%); branchless LMR (+0.5%,
  n.s.); the outlined qsearch capture loop again, now on the wall clock
  (+0.5%, n.s.).
- Round twelve (section 50): branch hints (-2.7%), skipping the repeated
  leaf null-window probe (3% fewer nodes, -0.5% time), prefetching the
  children's macro-correction line (-2.3%).

**Nets**

- Sparse-199 replace: −277 at depth 4
- Residual trained on 20 ms WDL: worse leaf than HCE
- Fat MiniNet (H=128) at 20 ms: hundreds of Elo lost (NPS collapse)
- Wider embedding D on old HCE-only depth-6 labels: ~0 Elo at equal depth (mixer stays tiny; extra concat unused)
- Holdout MAE / correlation without a move-level or Elo gate: a net can fit scores and play the same moves
- Early-game MiniNet teacher (D=8, H=8, extra low-ply data): about **+10 Elo at depth 4**, ~**+1.5 Elo at 20 ms**, killed as not a +5 timed win. Better leaf, ~5% NPS tax. Not a ship.
- Training on mates and fail-highs qsearch never asks the net about
- The NNUE round (section 56), against the shipped B64_d5M_57ep at 20 ms:
  the same net trained on all 176M depth-8 rows at equal steps (-44 head to
  head), with eval2 upweighted (-20), for 114 or 200 epochs at lr 1e-2 (-10,
  -21), at lr 5e-3 for 114 epochs (a tie, 11% slower per node; +9.0 +/- 12.0
  at 90 ms), on eval2 alone (-19), and the 128-lane B128_d5M_57ep (-6, about
  15% slower). The float per-cell NNUE hook: -255 at 20 ms at 0.8% of the
  shipped speed.

**Process failures (the kind that fake a pass)**

- SPRT against a Prev you just weakened
- SPRT against old HCE after MiniNet is the ship
- Two ideas in one SPRT
- Shipping `codingame_nnue.cpp` from an unfrozen Dev
- FMA in MiniNet (disagrees with scalar)
- Instance-sized `1<<18` tables on Windows

---

## 12. How the numbers sit together

An honest running story, not a sum:

1. Python learns to search (TT order, negamax, two-in-a-row, killers, history) and wins UVicAI.
2. C++ relearns the same search, adds qsearch and AVX, hits Legend, with a dead hash.
3. 2026 turns the hash on and the eval skip on: **+121** against the fossil.
4. Same HCE, less overhead (stack movegen, 3×3 LUT, Texel, free-move): roughly another **+30 to +50** per honest step, with dead ends in between.
5. Search becomes selective at 95 ms: **+80** as a bundle.
6. MiniNet is a *small* timed gain (**+7 to +11** vs HCE) and a *large* equal-depth gain (**+54**), then skip/AVX buy back the tax (**+20 to +31**, then **+26 to +29**).
7. Precomputed HCE + MiniNet projection: **~2× NPS**, **+50 to +100** timed depending on the hypothesis, same leaf.
8. Correction history + log LMR: **+33** at 20 ms. Then the #15 hot-path bundle: another **+40** at 20 ms, mostly speed (1.56× NPS) plus persist-corrhist and global-win ordering.
9. The #16 compact-state / canonical-TT bundle: **+59** at 20 ms vs #15.
10. The #17–#25 search/speed bundle vs #16: independent **+28** at 20 ms and
    **+38** at 95 ms. The ship bar moved to 95 ms here.
11. Round five (one-reply tactical proofs, two-entry TT buckets, a narrower
    aspiration window, a fine-tuned MiniNet, signed history malus): **+54** at
    95 ms. Round six (active-miniboard correction history, hot-path trims,
    latent macro captures): about **+25**.
12. Round seven, the D16/H8 MiniNet plus a learned macro residual: **+31** at
    95 ms against H0=+20; the external referee then moved the bar to **90 ms**
    with zero tolerated timeouts (**+21.6**). Round eight's exact macro-state
    correction history: **+29.6** against H0=+20.
13. Round nine, a bit-identical hot-path rewrite: **+18.8** (+22% NPS).
14. Section 47: the shipped bot had been compiled by CodinGame **without `-O`**
    and ran at about a fifth of its tested speed. Fixing that was **+163** in a
    CodinGame-built match and invisible to every local SPRT.
15. Rounds ten and eleven, bit-identical speed again: **+15.4** as a bundle in
    an independent review SPRT. The mate-window pruning fix (section 51) passed
    for non-regression only; section 52 closed the rest of the CodinGame
    inlining gap (+4% to +9% nodes per millisecond).
16. The uttt.ai opening book (section 54): about **+72** paired book value
    against a different engine, against +26 for the full-coverage book it
    replaced.
17. The NNUE round (section 56): one 35,243-parameter pattern-generator NNUE
    replaced HCE + MiniNet + macro. It costs about 40% of the nodes per
    second and a ply at 90 ms, and passed at **+277** against the
    mate-window freeze.

CodinGame rank is a different axis. Legend HCE got us into the league. MiniNet
moved 82 → 68. Absolute ladder Elo is noisy and not what SPRT measures. The
README's **Latest strength result** names the current ship.

---

## 13. Where the code stood after round four (7 September 2026)

This snapshot is historical; the README's **Layout** and **The three copies of
the engine** describe the current tree. Since then Prev and Dev moved to the
D16/H8 MiniNet in `mini_eval_d16.hpp` plus `macro_eval.hpp` (section 37), and
`mini_eval.hpp` is kept only for unit tests.

| Piece | Role |
| --- | --- |
| `crossfish_prev.hpp` | Frozen last accepted engine (#25 round-4 bundle) |
| `crossfish_dev.hpp` | Same program as Prev until the next experiment |
| `mini_eval.hpp` | Packed D=8 H=4 residual |
| `codingame_nnue.cpp` | Readable CG bot |
| `cg_input.cpp` | Minified paste file |
| `crossfish.cpp` | HCE-only CG bot / packing template |
| `cg_legend_hce.cpp` | March 2024 Legend fossil |
| `python_impl/` | UVicAI winner; not the active engine |

The next cheap experiment is whatever you can falsify in one SPRT: a constant, a skip, a LUT that must match `eval_consistency`. The next expensive experiment is a net that is better at equal depth *and* not slower at 20 ms. We already know H=128 and "more D on old labels" are not that net.

When you land something, freeze Prev, port the CG file, minify, and add a short section at the end of this log. When you fail, add a line to section 11. The log is only useful if the graves stay marked.

---

## 14. Correction history and logarithmic LMR (5 September 2026)

Two search patches on top of #10 (`c2aae17`), landed together: **+30.6 ± 6.5 Elo at
N=8016**, LLR 11.98 against H0=0 / H1=+5. Four times the ±3 decision bound.

**Correction history** (the bulk of it, +22.5 ± 10.7 measured alone at N=3000). A small
exact-indexed table `[stm][constrained miniboard, 9 = free][decided-miniboard mask]`
accumulates the signed gap between what search returned and what the static eval said,
then shifts the static eval by that running average at RFP, futility, and qsearch
stand-pat. No hashing and no collisions: the index is the two structural facts the fixed
eval misprices most — game phase, and being locked into one miniboard. Only a bound that
*contradicts* the static eval updates the entry, and the gravity term
`e += diff*w - e*|diff|*w/CORR_SCALE` makes the update a contraction, so entries cannot
run away. Table is per-instance but small (2·10·512 ints), reset per `getMove`.

**Logarithmic LMR** (+16.1 ± 10.6 alone at N=3000, so real but the weaker half).
`reduction = i / 3` becomes a precomputed `ln(depth)·ln(i+1)` table in hundredths of a
ply, one reduction shaved in PV nodes. Table is `static inline` — see the Windows note in
section 11; it is 20 KB, not `1<<18`, but the rule stands.

**Method notes, because two of them changed the answer.**

- A **null control** (Dev compiled identically to Prev) ran in both batches. At N=3000 it
  read **+7.07 ± 10.66** and I nearly subtracted that as a harness floor; at N=8016 it
  resolved to **−1.43 ± 6.53**. The floor was noise. Re-measure the null at the same N as
  the candidate before correcting anything by it.
- Runs were **fixed-sample** (`SPRT_LLR_BOUND=1e9`, `SPRT_MAX_GAMES=N`). Early-stopped Elo
  is not comparable between candidates, which is the whole reason the screen used one N.
- The full 4-patch stack measured **+27.5 ± 6.5** versus this 2-patch build's **+30.6 ±
  6.5** — statistically the same (difference error ≈ ±9.2), so the two extra patches were
  dropped on parsimony, not because they measured negative. 98 changed lines, not 161.
  Their individual graves are in section 11.

Shipped: `codingame_nnue.cpp` and `cg_input.cpp` carry this Dev. Independent 20 ms
SPRT on the merge host: N 1920 W 784 D 533 L 603, **+32.85 ± 13.25**, LLR +3.10.

---

## 15. Hot-path LUTs, compact TT, persist corrhist (6 September 2026)

GitHub PR #10. Five landed changes on top of #14 (`ab420b3`), measured as one
bundle. Two more search ideas were tried and reverted (graves in section 11).

Independent official 20 ms SPRT vs `ab420b3` (H0=0, H1=+5, bound=3, max games
uncapped). Tip Prev was already frozen to this winner, so `make sprt` on the
branch is a null test; this run compiled PR Dev against main's Dev renamed to
Prev.

```text
N: 1536 W: 642 D: 427 L: 467
Elo diff: +39.76 +/- 14.83
LLR: +3.04 (H0=0, H1=+5) — PASS
Prev NPS: 15,877,888  Dev NPS: 24,697,088
```

Author reported a longer H0=+50 / H1=+55 run at N=6624, **+57.91 ± 7.31**, LLR
+3.02. Think-ms was not written down. The official 20 ms gate is the ship
number; the author's larger point estimate is the same direction with a tighter
CI if it was also 20 ms.

**What actually changed**

- **Exact local-win LUTs** (`fast_win_moves` / `fast_has_win`, 512 entries).
  One player's 9-bit occupancy is enough: opponent stones are just squares that
  are not ours, so they block lines the same way `mini_win_sq[ternary][stm]`
  did. Used for capture/block, capture movegen, `check_winner_fast`, and
  `make_move_fast` win detection. Replaces AVX line tests in the hot path.
- **Fast make / unmake / movegen / winner.** Search no longer calls
  `GlobalBoard::makeMove` / `fillLegalMoves` / `checkWinner`. Legal-move order
  matches the slow path (ctz is low-square-first, same as 0..8). Snap and
  incremental-HCE oracles are in `lut_capture_block_tiar`.
- **Compact TT, still `1<<18` entries.** 16 bytes: hash, score, `i16` depth,
  flag, packed move (`mb*9+sq`, 255 = none). Same slot count, about half the
  old `TTEntry` footprint, so more of the table stays in cache. Depth as
  `int16` is safer than the CG `int8` it replaced.
- **Persist correction history across `getMove`.** History and TT already
  survived between moves; corrhist was the leftover per-move wipe. `search_fixed_depth`
  still zeros it, so Texel stays clean. `play_game` reuses the same bot objects
  for both paired games, so game 2 inherits Dev's table from game 1. That can
  plump the persist-corrhist slice; it is also how CG will play (one object,
  one match).
- **`+800` if a capture completes a global win.** TT stays at 1000, so a
  winning capture scores 900 and does not override the hash move.
- **Incremental HCE.** Per-mini local score and 2-in-a-row flags ride
  make/unmake; `evaluate_hce` remains the oracle. Global terms still read live
  mini-states. Same leaf as #14.

This is more than one axis. Sequential freezes were the research loop; the
gate is the bundle vs `ab420b3`. Most of the Elo is speed (1.56× startpos NPS).
Persist-corrhist and global-win order are real search changes, not just a
faster rewrite of the same tree.

Shipped: `codingame_nnue.cpp` and `cg_input.cpp` carry this Dev.

---

## 16. Compact search state and canonical TT (7 September 2026)

Five sequentially frozen changes landed on top of PR #10 (`f453086`):

- Reset the aspiration window to its base width after every successful
  iteration instead of carrying a widened failure window forward:
  **+12.26 +/- 7.58 Elo**, LLR +3.09.
- Precompute the 9-bit-mask contribution to each local ternary index:
  **+10.97 +/- 7.18 Elo**, LLR +3.00.
- Search on a compact board with a fixed move stack rather than copying and
  mutating the full `GlobalBoard`: **+8.03 +/- 5.78 Elo**, LLR +3.05.
- Give the TT a canonical key that removes stones from already-decided
  miniboards. Those stones can never affect legal play again, so positions
  reached through different local move orders now merge:
  N=3584 W=1439 D=900 L=1245, **+18.82 +/- 9.86 Elo**, LLR +3.06.
- Halve move-history values at the start of each turn so tactical ordering
  retains recent evidence without letting early-game cutoffs dominate the
  whole match:
  N=2016 W=837 D=521 L=658, **+30.93 +/- 13.10 Elo**, LLR +3.00.

Those sequential results are selection gates, not additive Elo. The shipping
proof was one direct 20 ms SPRT of the complete branch against the exact
current `origin/main` engine, with H0=+50 and H1=+55:

```text
N: 3328 W: 1573 D: 817 L: 938
Elo diff: +67.12 +/- 10.38
LLR: +3.05 — PASS
Prev NPS: 13,149,952  Dev NPS: 14,300,416
```

Independent 20 ms SPRT on the merge host vs `f453086` (H0=0, H1=+5, uncapped):

```text
N: 1056 W: 486 D: 262 L: 308
Elo diff: +59.13 +/- 18.36
LLR: +3.09 — PASS
Prev NPS: 22,007,808  Dev NPS: 20,476,928
```

Startpos NPS was slightly *lower* on Dev, so this is a search win, not a
faster rewrite of the same tree. MiniNet still encodes stones on decided
minis; the canonical TT key is therefore an eval approximation (legal play
is identical). `codingame_nnue.cpp` and `cg_input.cpp` carry this Dev.

---

## 17. Earlier late-quiet reduction (7 September 2026)

The first sequential winner after #16 starts logarithmic LMR on the third
ordered quiet move (`i >= 2`) instead of the fourth. Negative-score moves keep
their existing reduction rule; no reduction amount, extension, or move score
changed.

```text
N: 4992 W: 1945 D: 1304 L: 1743
Elo diff: +14.07 +/- 8.29
LLR: +3.06 (H0=0, H1=+5) — PASS
```

Rejected on the same baseline: requiring depth 3 for LMR was about 0 Elo at
N=2976; resetting counter moves each turn screened -17 Elo; quarter-retaining
move history screened -20 Elo; halving correction history each turn screened
-7 Elo; removing the pseudo-singular extension screened -13 Elo despite higher
NPS; clearing killers between completed iterations screened +3 Elo; delaying
late-quiet LMR to `i >= 4` was -14.7 Elo at N=1344 in its formal run.

---

## 18. Reuse exact TT scores below the root (7 September 2026)

Sufficient-depth exact transposition entries now return immediately at PV
nodes below the root. Upper and lower bounds remain restricted to non-PV
nodes. Root exact hits are deliberately excluded: returning before the root
loop would leave `root_best_move` at the first generated legal move.

```text
N: 6560 W: 2487 D: 1795 L: 2278
Elo diff: +11.07 +/- 7.17
LLR: +3.03 (H0=0, H1=+5) — PASS
```

The unrestricted first prototype demonstrated the root hazard clearly,
falling about 303 Elo before it was stopped at N=256. The accepted version
changes neither evaluation nor TT replacement; it only reuses already exact
work where the caller needs a score rather than a root move.

---

## 19. Apply correction history more strongly (7 September 2026)

Correction history keeps the same exact structural key, update rule, and
bounded storage, but converts stored units back to evaluation units with a
grain of 24 instead of 32. This raises the maximum applied correction from
512 to about 683 evaluation units without changing how evidence is learned.

```text
N: 9344 W: 3490 D: 2591 L: 3263
Elo diff: +8.44 +/- 5.99
LLR: +3.03 (H0=0, H1=+5) — PASS
```

A weaker grain of 48 previously resolved near parity. Replacing the exact
decided-miniboard mask with an owner/material-count key looked better on
held-out score residuals but lost about 9 Elo in online self-play, confirming
that the layout-specific history remains important.

---

## 20. Search the hash move before scoring the rest (7 September 2026)

When a legal TT move is available, search now moves it to the front without
first scoring and sorting every legal move. If the hash move does not cut,
the remaining moves are scored and stably sorted before the second move.

Besides avoiding wasted ordering work on immediate cutoffs, the delay lets
recursive cutoffs from the hash-move search refresh history, counter, and
killer data before the remaining moves are scored. A fixed-depth screen was
positive as a result; this is not merely a timed NPS change.

```text
N: 6560 W: 2466 D: 1840 L: 2254
Elo diff: +11.23 +/- 7.13
LLR: +3.11 (H0=0, H1=+5) — PASS
```

---

## 21. Score tactical move masks without per-move branches (7 September 2026)

The scorer now loads each miniboard's capture, block, two-in-a-row, and
global-win masks once, then applies their exact existing bonuses with bit
arithmetic. The hash-move comparison was also removed from this function:
section 20 now handles a legal hash move before calling the scorer, so every
remaining call passes no hash candidate and that comparison was dead work.

The ordering values and stable sort are unchanged. The gain is small enough
that the formal run needed a large sample, but it eventually crossed the
repository's normal acceptance boundary:

```text
N: 37696 W: 13651 D: 10810 L: 13235
Elo diff: +3.83 +/- 2.96
LLR: +3.09 (H0=0, H1=+5) — PASS
```

---

## 22. Precompute decided-miniboard TT hash subsets (7 September 2026)

Canonical TT keys omit stones inside decided miniboards. Previously, every
make and unmake that changed a miniboard's decided state scanned each stone
and XORed its individual Zobrist value. A thread-safe 72 KiB lookup now stores
the XOR for every 9-bit marker subset, reducing that work to one lookup per
player while preserving every search key and move choice exactly.

The fixed-depth gate was tree-identical to the frozen engine. The timed
screen also showed higher NPS, and the repository's formal SPRT passed:

```text
N: 6432 W: 2394 D: 1849 L: 2189
Elo diff: +11.08 +/- 7.17
LLR: +3.01 (H0=0, H1=+5) — PASS
```

---

## 23. Pack internal search moves into one byte (7 September 2026)

Search and qsearch now represent generated moves as a single byte, with the
miniboard in the high nibble and square in the low nibble. Public board
history and root interfaces still use `Move`, and TT entries retain their
existing 0–80 encoding. The compact form reduces move-generation writes,
hash-move comparisons, and stable-sort copies without changing move order.

The fixed-depth gate produced the exact same tree and game sequence as the
frozen engine. The timed screen showed a small NPS gain, and formal SPRT
confirmed the improvement:

```text
N: 13216 W: 4858 D: 3750 L: 4608
Elo diff: +6.57 +/- 5.01
LLR: +3.00 (H0=0, H1=+5) — PASS
```

---

## 24. Prefetch child transposition entries (7 September 2026)

After making each full-search move, the engine now prefetches the child
position's transposition-table slot before entering the recursive search.
The TT is a random-access 4 MiB table; winner detection and search setup
between the prefetch and probe provide useful memory-latency overlap without
changing the searched tree, replacement policy, or stored entries.

The fixed-depth gate remained neutral, while the timed screen showed a large
start-position NPS increase. Formal 20 ms SPRT confirmed the gain:

```text
N: 8032 W: 2990 D: 2268 L: 2774
Elo diff: +9.35 +/- 6.44
LLR: +3.01 (H0=0, H1=+5) — PASS
Prev NPS: 12.99M  Dev NPS: 14.97M
```

---

## 25. Round-four direct +50 proof (7 September 2026)

The complete sections 17–24 branch was tested at 20 ms against the exact
current `origin/main` engine (`1c5b3f7cab8deee036a12799ea36029680ce8d0d`),
not against the sequentially advanced Prev snapshot. The raised-hypothesis
SPRT verified that the bundle clears another 50 Elo:

```text
N: 5920 W: 2632 D: 1635 L: 1653
Elo diff: +57.99 +/- 7.60
LLR: +3.00 (H0=+50, H1=+55) — PASS
Prev NPS: 14.35M  Dev NPS: 15.87M
```

Independent merge-host gates vs the same `1c5b3f7` baseline, H0=0 / H1=+5.
The official bar is now **95 ms**. The author's +58 at H0=+50 was not
reproduced; these are the ship numbers:

```text
95 ms: N 1568 W 630 D 480 L 458
Elo diff: +38.27 +/- 14.38
LLR: +3.05 — PASS
Prev NPS: 19,946,496  Dev NPS: 20,539,392

20 ms: N 2272 W 936 D 585 L 751
Elo diff: +28.35 +/- 12.34
LLR: +3.08 — PASS
Prev NPS: 19,265,280  Dev NPS: 20,314,880
```

`codingame_nnue.cpp` and `cg_input.cpp` carry this Dev.

---

## 26. Prove immediate global losses without searching them (9 September 2026)

After a candidate move, the engine now checks whether the opponent can win
the global board immediately. The test intersects the opponent's global
winning targets with the legal destination miniboard and that miniboard's
local winning squares. When such a move exists, the child is an exact loss
and recursion is skipped.

```text
N: 12928 W: 4482 D: 4211 L: 4235
Elo diff: +6.64 +/- 4.92
LLR: +3.13 (H0=0, H1=+5) — PASS
```

---

## 27. Prove forced global wins one reply ahead (9 September 2026)

The symmetric tactical shortcut recognizes positions where every legal
opponent reply still leaves a global winning move. It accounts for captures,
drawn miniboards, a reply that wins the macro board, full-board count
termination, and replies that block the sole local winning square.

The original move-by-move proof passed SPRT. The shipped implementation is an
equivalent bit-parallel mask test; a randomized depth-4 oracle matched on
2,000 later-ply positions after catching and fixing an early missing-target
condition.

```text
N: 5696 W: 2017 D: 1858 L: 1821
Elo diff: +11.96 +/- 7.41
LLR: +3.04 (H0=0, H1=+5) — PASS
```

---

## 28. Keep two transposition entries per cache line (9 September 2026)

The 4 MiB TT is now 131,072 aligned 32-byte buckets containing two 16-byte
entries. Probes check both entries; stores preserve a matching or empty slot
and otherwise replace the shallower entry. Capacity doubles without changing
the table's memory footprint or requiring a second cache line.

```text
N: 11040 W: 3818 D: 3634 L: 3588
Elo diff: +7.24 +/- 5.31
LLR: +3.05 (H0=0, H1=+5) — PASS
```

---

## 29. Narrow the aspiration window (9 September 2026)

The initial iterative-deepening aspiration window moved from 50 to 40 HCE
pawns. On this stronger and more stable baseline, the extra cutoffs outweighed
the additional re-searches.

```text
N: 6496 W: 2326 D: 2050 L: 2120
Elo diff: +11.02 +/- 6.99
LLR: +3.10 (H0=0, H1=+5) — PASS
```

---

## 30. Fine-tune MiniNet on current depth-8 search labels (9 September 2026)

The packed D=8, H=4 residual MiniNet was fine-tuned from the shipped model on
106,283 positions labeled by the current engine at depth 8. Training pins the
empty-board residual to zero after every optimizer step, and the trainer can
now initialize from an existing CFM2 model.

The packed held-out MAE improved from 1123.66 to 1103.73, median absolute
error from 552.10 to 527.41, and correlation from 0.86425 to 0.86601. Online
play, not those offline metrics, was the acceptance gate:

```text
N: 13856 W: 4859 D: 4387 L: 4610
Elo diff: +6.24 +/- 4.78
LLR: +3.02 (H0=0, H1=+5) — PASS
```

---

## 31. Let failed quiet moves earn negative history (9 September 2026)

Quiet moves searched before a beta cutoff now receive a signed,
gravity-bounded malus instead of being clamped at zero. The final malus is
twice the cutoff bonus and remains bounded at -10000, matching the positive
history update's saturation behavior.

```text
Signed malus, 1x:
N: 13248 W: 4620 D: 4252 L: 4376
Elo diff: +6.40 +/- 4.88
LLR: +3.01 (H0=0, H1=+5) — PASS

Final 2x malus:
N: 4512 W: 1634 D: 1435 L: 1443
Elo diff: +14.72 +/- 8.38
LLR: +3.07 (H0=0, H1=+5) — PASS
```

---

## 32. Round-five direct +50 proof (10 September 2026)

The final stack also iterates live miniboard bits in free-choice qsearch,
uses the bit-parallel mate-in-three proof described above, and lowers the
qsearch delta margin from 400 to 350 pawns. These last hot-path changes were
kept as part of the complete stack rather than claimed as independent +5 Elo
winners.

The complete engine was then tested at the official 95 ms control against the
exact `origin/main` commit
`0c50c955327ff58767bfe3378a75aa2c8beb211f`, with separate MiniNet namespaces
so the comparator could not accidentally share the new weights:

```text
N: 7744 W: 3282 D: 2385 L: 2077
Elo diff: +54.51 +/- 6.48
LLR: +3.00274 (H0=+50, H1=+55) — PASS
```

Independent merge-host gate vs the same `0c50c95` baseline, H0=0 / H1=+5.
The author's +54 at H0=+50 reproduced; these are the ship numbers:

```text
95 ms: N 1056 W 433 D 352 L 271
Elo diff: +53.72 +/- 17.23
LLR: +3.00 — PASS
Prev NPS: 26,194,176  Dev NPS: 23,589,632
```

Rejected or inconclusive experiments on this baseline included a second
projected-network cycle, an extra mate-in-three guard, TT generations,
history-reward and divisor variants, killer/free/block/two-in-a-row weight
changes, aspiration recentering, packed capture scores, cached winner state,
a compact MiniNet lookup, searched-only maluses, a last-mover winner check,
and a dedicated qsearch scorer. The live-miniboard capture scan alone was
only +3.87 Elo at N=4128 and was not treated as an independent pass.

`crossfish_prev.hpp`, `codingame_nnue.cpp`, and `cg_input.cpp` carry the
verified round-five engine.

---

## 33. Correct active-miniboard shape online (10 September 2026)

The accepted correction history only keyed side to move, forced miniboard,
and the mask of finished miniboards. Cross-depth analysis of the round-five
depth-8 and depth-10 labels showed that the exact STM-relative shape of the
forced miniboard explained another 100-112 points of static-HCE MAE, slightly
more than the accepted structural key by itself.

A second 19,684-entry correction table now learns that residual online. The
first 19,683 entries are the ternary local-board states and the final entry is
free choice. The existing correction remains unchanged at grain 24; the new
table is deliberately weaker at grain 48. Both use the same bounded gravity
update, and fixed-depth data generation clears both tables.

The change passed the optional 20 ms screen:

```text
N: 11040 W: 4084 D: 3109 L: 3847
Elo diff: +7.46 +/- 5.49
LLR: +3.02 (H0=0, H1=+5) — PASS
Prev NPS: 13,369,856  Dev NPS: 13,222,144
```

It then passed the authoritative 95 ms gate against the exact merged
round-five baseline (`d5617e510925a3315215a2d14e6526839ca70933`):

```text
N: 6240 W: 2236 D: 1968 L: 2036
Elo diff: +11.14 +/- 7.14
LLR: +3.02 (H0=0, H1=+5) — PASS
Prev NPS: 14,519,808  Dev NPS: 13,890,304
```

The start-position NPS moved slightly backward, so this is an evaluation /
pruning-quality gain rather than an implementation-speed result.
`crossfish_prev.hpp`, `codingame_nnue.cpp`, and `cg_input.cpp` carry the
frozen winner.

---

## 34. Make the search hot path smaller without changing its tree (11 September 2026)

Four exact implementation changes were combined after each had been checked
against the current search:

- `FastBoard` no longer maintains an unused full Zobrist hash alongside its
  canonical transposition hash;
- killer moves use byte storage;
- counter moves stay packed in the engine's byte-sized internal format; and
- correction-history entry addresses are retained from static evaluation to
  the later update.

A randomized real-`getMove` oracle compared 400 legal positions through depth
4 and matched the frozen engine exactly on selected move, root score, and node
count. Depending on the clean benchmark run, start-position throughput rose
roughly 4-8%.

The bundle passed the optional 20 ms screen:

```text
N: 17504 W: 6413 D: 4957 L: 6134
Elo diff: +5.54 +/- 4.36
LLR: +3.01 (H0=0, H1=+5) — PASS
Prev NPS: 13,805,056  Dev NPS: 14,722,048
```

It then passed the authoritative 95 ms gate:

```text
N: 14816 W: 5146 D: 4783 L: 4887
Elo diff: +6.07 +/- 4.60
LLR: +3.11 (H0=0, H1=+5) — PASS
Prev NPS: 14,384,384  Dev NPS: 14,971,136
```

This is a pure implementation-speed gain: the search tree is unchanged at a
fixed node/depth budget. `crossfish_prev.hpp`, `codingame_nnue.cpp`, and the
70,056-character `cg_input.cpp` carry the frozen winner.

---

## 35. Cache exact state and price latent macro captures (12 September 2026)

Two exact hot-path values are now retained instead of recomputed: `FastBoard`
incrementally maintains the finished-miniboard mask, and each correction
history entry stores both its raw gravity value and its already-divided
applied value in the same four-byte footprint.

On top of those caches, HCE now applies a bounded `-800` side-to-move
correction when the opponent has a local two-in-a-row on any live miniboard
that would complete their macro line. The incremental evaluator already
maintains both local-threat maps, so the search path adds only bit operations.
The standalone reference evaluator reconstructs the maps for consistency
tests.

The sign was established empirically. A teacher-residual-inspired `+800`
version failed fixed depth four at `-26.95 +/- 13.55` Elo. Reversing it to
`-400` passed fixed depth but did not translate at 20 ms. The final `-800`
version passed fixed depth four:

```text
N: 6592 W: 2906 D: 1011 L: 2675
Elo diff: +12.18 +/- 7.72
LLR: +3.07 (H0=0, H1=+5) — PASS
```

Its complete 20 ms screen was positive but stopped just short of acceptance:

```text
N: 12000 W: 4461 D: 3301 L: 4238
Elo diff: +6.46 +/- 5.29
LLR: +2.63 (H0=0, H1=+5)
```

The authoritative 95 ms gate then passed:

```text
N: 7392 W: 2646 D: 2308 L: 2438
Elo diff: +9.78 +/- 6.57
LLR: +3.02 (H0=0, H1=+5) — PASS
Prev NPS: 14,095,360  Dev NPS: 14,220,288
```

The essentially flat NPS and strong equal-depth result indicate that the
measured gain is primarily evaluation/search quality; the exact caches make
the feature cheap and may contribute a smaller speed component. Full tests
passed, including the 23,511-position incremental evaluator verifier.
`crossfish_prev.hpp`, `codingame_nnue.cpp`, and the 70,638-character
`cg_input.cpp` carry the frozen winner.

---

## 36. Round-six direct measurement (12 September 2026)

The three accepted sequential 95 ms steps were measured directly as one stack
against the exact merged round-five baseline
`d5617e510925a3315215a2d14e6526839ca70933`. The run used the stricter
H0=+30/H1=+35 hypotheses:

```text
N: 2848 W: 1061 D: 920 L: 867
Elo diff: +23.70 +/- 10.51
LLR: -0.84 (H0=+30, H1=+35) — stopped
Prev NPS: 14,817,536  Dev NPS: 14,169,600
```

This does not establish the originally targeted direct +30 claim. It does
show that the sequential winners compose to roughly +25 Elo, and that result
was accepted as sufficient for the round-six handoff. The shipped stack
contains:

- active-miniboard-shape correction history;
- exact hot-path state/storage reductions;
- cached out-of-play and applied correction values; and
- the opponent latent macro-capture HCE correction.

---

## 37. Larger eval, better data, and exact macro lookup (13 September 2026)

This cycle deliberately tested all four eval directions: faster inference, a
larger NNUE, new training data, and new handcrafted features.

### Accepted evaluator

The shipped local net grows from D=8/H=4 to D=16/H=8. Its 19,683 local-board
embeddings are compressed to 256 centroids in first-layer projection space,
with the empty board reserved exactly. The hot path stores preprojected int32
contributions for each centroid, super-board class, and active-board flag.

A separate compact 16-hidden-unit residual models only the nine super-board
classes and the forced-board constraint. Offline it reduced MAE versus
independent deeper-search labels by roughly 40-53 points, while adding only a
3,076-byte float payload.

The final speed step precomputes that macro residual into
`MACRO_SCORE[10][1<<18]`, a 5 MiB static int16 table. The key is a base-4
encoding of the nine super-board cells. `FastBoard` maintains a key for each
player perspective, updating it only when a miniboard becomes won or drawn.
Random live-search verification compared every cached result with the
original macro MLP and found no mismatch. Start-position throughput rose from
about 11.05M to 11.71M NPS within the D16 build.

The optional 20 ms screen passed:

```text
N: 2208 W: 878 D: 630 L: 700
Elo diff: +28.07 +/- 12.28
LLR: +3.028 (H0=0, H1=+5) — PASS
Prev NPS: 14,469,120  Dev NPS: 11,710,720
```

The authoritative direct test used the stricter target hypotheses and passed:

```text
N: 5152 W: 2012 D: 1591 L: 1549
Elo diff: +31.31 +/- 7.91
LLR: +3.063 (H0=+20, H1=+25) — PASS
Prev NPS: 13,333,760  Dev NPS: 11,787,264
```

The bundled CodinGame source is 96,672 characters, 3,328 below the limit.

### Rejected experiments

- A targeted 80k-position depth-12 fine-tune improved its own targeted MAE by
  1.77, but worsened independent depth-12, depth-8, and qsearch-leaf sets.
  The broader labels remained the better training distribution.
- A tiny 8->16->1 proxy for the old H4 pruning net reached about 112 MAE on
  independent positions, but its 20 ms screen finished only
  `+13.29 +/- 10.60` at N=3008 and reduced start-position NPS to about 8.9M.
- Affine D16-to-H4 calibrations improved score MAE but weakened online play.
  The uncalibrated D16 depth-1 gate was stronger.
- An active macro-target HCE bonus was only `+3.23 +/- 10.58` in an exact
  3008-game A/B and cost about 6% NPS. The learned macro head already captures
  most of that context.
- Maintaining both D16 first-layer perspectives on every make/unmake was
  bit-exact, but updates at all nodes cost more than recomputation at eval
  nodes. The screen was `-5.87 +/- 17.17` at N=1184 and was stopped.
- Larger/smaller macro clipping, a standard-centroid pack, a qleaf-adapted
  macro head, and macro-aware HCE fail-high shortcuts all underperformed the
  accepted projected-centroid, scale-1.25, clip-2000 configuration.

This round is a mix of algorithmic evaluation quality and implementation
speed. Most of the measured gain comes from the larger local net plus learned
macro context; the exact macro lookup buys back part of their node cost.

---

## 38. Enforce the real CodinGame move deadline (13 September 2026)

The first round-seven CodinGame bundle searched for 800 ms before its fixed
center opening, even though that search result was discarded. Eager D16 and
macro-table initialization took about 80 ms locally, so process start to first
output was already roughly 879 ms. The slower CodinGame host crossed its
one-second first-turn limit. Replacing the discarded search with the normal
warm-up budget reduced local first output to roughly 174 ms.

Later turns exposed the same missing safety margin in smaller form. The engine
requested the full 95 ms, checked time only every 256 nodes, rounded elapsed
time down to whole milliseconds, and started its clock after per-move setup.
Across 250 varied positions, ordinary responses clustered near 96.1 ms and the
99th percentile approached 98 ms, leaving almost no room for scheduling,
recursive unwinding, or output.

The CodinGame-only timing path now:

- uses a 90 ms search budget;
- starts the deadline before per-move setup;
- compares the exact duration instead of rounded milliseconds;
- checks every 128 nodes.

Across 349 later-turn samples, the median was 90.08 ms, p99 was 91.73 ms, and
the maximum was 91.86 ms. SPRT still used 95 ms at this stage; external-referee
stress testing in the next section showed that this left too little wall-clock
headroom.

The C++ SPRT referee and process-based round-robin now independently enforce
CodinGame's external 100 ms limit. A late move is an immediate loss, timeout
counts are printed with match results, and a timed-out process is restarted so
its stale output cannot corrupt the next game. Fixed-depth evaluator tests are
explicitly exempt because they intentionally have no move clock.

---

## 39. Eliminate timeout forfeits in long SPRTs (13 September 2026)

The first real referee-enabled direct SPRT used the historical 95 ms search
allocation and all 16 logical CPUs of an m5.4xlarge. It passed for strength,
but 420 of 2,176 games ended by timeout:

```text
N: 2176 W: 893 D: 572 L: 711
Elo diff: +29.13 +/- 12.57
LLR: +3.048 — PASS
Timeouts: Prev=202 Dev=218
```

The machine has eight physical cores with two SMT threads per core. A 90 ms
probe using all 16 logical CPUs still produced 29 forfeits in only 160 games.
At eight workers, ordinary responses were stable, but rare VM scheduling
stalls still produced two 103.6 ms Dev responses by game 1,008. This was not
an engine-node-count problem; a non-real-time process can be descheduled after
its final internal clock check.

The final mitigation combines three layers:

- both SPRT engines use `steady_clock`, start the timer before per-move setup,
  compare exact durations, and check every 128 nodes;
- the official search allocation is 90 ms, matching the submitted bot and
  leaving ten milliseconds before the external deadline;
- Linux test harnesses detect physical-core topology and reserve one physical
  core by default instead of saturating every SMT context.

Result lines now include maximum observed response latency as well as timeout
counts. The process-based round-robin uses the same physical-core policy and
reports its latency maxima.

The exact merged-main versus PR-stack validation then completed 2,954 games
with no timeout losses:

```text
N: 2954 W: 1100 D: 937 L: 917
Elo diff: +21.55 +/- 10.37
LLR: +3.110 (H0=0, H1=+5) — PASS
Timeouts: Prev=0 Dev=0
Maximum response: Prev=97.99 ms Dev=90.19 ms
Prev NPS: 14,007,040  Dev NPS: 11,566,848
```

This does not make a hard real-time guarantee—general-purpose operating
systems can pause any process—but it turns timeout regression into a measured
failure and demonstrated zero forfeits across a multi-thousand-game SPRT.

---

## 40. Lossless CodinGame payload compaction (13 September 2026)

The D16 and macro evaluator payloads previously used Base64. Re-encoding the
same binary data as ASCII85 reduces expansion from four source characters per
three bytes to five per four bytes. Both generated headers now share the small
D16 ASCII85 decoder, and use a `~` raw-string delimiter that cannot occur in
the emitted `!`-through-`u` alphabet.

This is a representation-only change: no weights, evaluation arithmetic,
search logic, or time allocation changed. Decoding the old and new headers
produced identical payloads:

| Payload | Bytes | FNV-1a 64 |
| --- | ---: | --- |
| D16 local evaluator | 42,855 | `e35e987c17a453cf` |
| Macro residual | 3,076 | `626e29f3a8d65679` |

The C++ unit suite now verifies those lengths and hashes, in addition to a
known ASCII85 decoder vector. The Python tests cover every possible partial
tail length and randomized round trips.

The bundled submission shrank from 96,674 to 92,759 characters, saving 3,915
characters and increasing headroom under the 100,000-character cap from 3,326
to 7,241. In 24 alternating process-start probes, first-move median latency
was unchanged at 154.88 ms for the Base64 baseline and 154.91 ms for ASCII85.
A clock-free depth-four comparison over 96 generated positions also produced
identical scores and node counts; both complete outputs had SHA-256
`b5540483238832734d2c9798f8d432ce71c5ae21c6f81729b2aa9ec2c8c46d2e`.
Because the decoded bytes and all post-load engine code are identical, this
size reduction does not change playing strength.

An official-time smoke SPRT then exercised the frozen merged-main Prev search
against the PR Dev search at 90 ms, with the external 100 ms referee and the
default seven workers (one of eight physical cores reserved). Non-regression
hypotheses were H0=-5/H1=0, and the run was capped at 2,000 games; seven
workers produce 14 games per batch, so it finished at 2,002:

```text
N: 2002 W: 667 D: 638 L: 697
Elo diff: -5.21 +/- 12.57
LLR: -0.324 (H0=-5, H1=0) — INCONCLUSIVE
Timeouts: Prev=0 Dev=0
Maximum response: Prev=90.06 ms Dev=90.30 ms
```

The in-process harness shares the generated evaluator tables, so this smoke
primarily validates official-time search/referee stability after the source
representation change. The stronger equivalence evidence remains the exact
decoded payloads and clock-free identical search outputs above.

---

## 41. Controlled opening experiments and a durable SPRT book (14 September 2026)

The proposed "Teccles" opening rule prefers, during the first few turns, a
move whose square sends the opponent back to the miniboard just played. Early
screens using the old 4-8-ply random opener were not meaningful because that
opener consumed much of the heuristic's intended window. The harness gained a
diagnostic center-first deterministic opener so the test could begin at ply
four instead.

The corrected controlled test still rejected the heuristic. Applying a
60-point ordering bonus at all search nodes finished:

```text
N: 1092 W: 417 D: 234 L: 441
Elo diff: -7.64 +/- 18.29
```

Earlier random-opener variants were also unpromising: all-node bonuses of 60,
300, and tie-break-only strength finished at approximately -2.8, -12.1, and
-8.6 Elo respectively, while a root-only bonus was exactly flat at N=1134.
No Teccles ordering change was retained.

That methodology issue motivated replacing implicit random starts with a
versioned benchmark artifact. The new generator builds legal 4-10-ply lines
by choosing only moves within 250 static-eval units of the frozen baseline's
best move. It deduplicates exact states, applies cheap depth-4 and depth-12
prefilters, then gives every retained position an independent fresh-engine
depth-16 search. The final book admits only positions with
`abs(depth16_score) <= 300` and guarantees at least 500 positions at every
included ply.

To avoid technically covering but practically neglecting the standard center
opening, first-move proposals are stratified: all 81 moves receive one slot,
the nine center-miniboard moves receive three extra slots, and center-center
receives eight more. These are proposal weights only; every line still has to
pass the identical reasonable-move and depth-16 balance filters.

A 140-position calibration demonstrated why the deeper gate matters and
quantified the optional depth-20 check:

```text
Depth 16: mean |score| 356.55, p50 341, p90 661
Within +/-300: 61 / 140
Depth-12 vs depth-16 Pearson correlation: 0.910
Depth-12 abs(score) <= 525 retained all 61 depth-16 qualifiers
Depth-16 vs depth-20 Pearson correlation: 0.944
50 of 61 depth-16 qualifiers also remained within +/-300 at depth 20
```

The audit initially exposed an uninitialized counter-move array in the
fixed-depth helper. Normal timed search initialized that table, but the
offline scorer did not, so worker scheduling could change selective-search
ordering and scores. Both frozen and Dev fixed-depth paths now explicitly
clear history/counter state, and evaluator payloads are warmed before worker
launch. The same 28 roots then produced identical scores and node counts with
7 and 16 workers.

The tracked `cpp_impl/opening_book.bin` contains 10,000 positions. The SPRT
harness validates and loads it by default, plays every position twice with
colors swapped, reports its metadata in the test header, and fails loudly if
the expected artifact is absent. Generation, binary format, audit commands,
and benchmark-versioning policy are documented in
`documentation/opening_book.md`.

The production artifact is 160,032 bytes with SHA-256
`6bc7556e530ec0e1cd9c395dae16809596503480f2929c2c2ca3ba5f906c509d`.
Its mean absolute stored score is 161.44 (p50 169, p90 275, p99 298), all 81
first moves occur at least 45 times, center-center occurs 189 times, and the
4-10-ply buckets contain
784/1,854/1,004/2,087/1,190/1,891/1,190 positions. Re-searching the first 140
records at depth 16 reproduced every stored score exactly. A depth-20 audit
kept 105/140 inside +/-300 and 137/140 inside +/-500, quantifying the tradeoff
made when the production gate was relaxed from depth 20 to depth 16.

SPRT consumes the records through a deterministic Fisher-Yates permutation
rather than raw generation order. This preserves exact resume/replay behavior
while making short prefixes representative of the complete book instead of
coupling a test slice to candidate-acceptance order. The inspect command emits
the traversal seed, fingerprint, and initial record indices so benchmark order
is auditable as well as benchmark contents. The tracked traversal fingerprint
is `8698397342672575767`, and the verify suite pins it as part of the benchmark
contract.

A fixed 2,000-game comparison then measured the final Round 8 candidate under
the new book and the legacy deterministic 4-8-ply random opener. Both arms
used 90 ms, eight workers, opening-pair offset 5,000, color-swapped pairs, and
disabled early stopping:

| Opening source | W / D / L | Draw rate | Elo | LLR |
| --- | ---: | ---: | ---: | ---: |
| Shuffled depth-16 book | 647 / 844 / 509 | 42.2% | +24.01 +/- 11.59 | +2.587 |
| Legacy random opener | 712 / 616 / 672 | 30.8% | +6.95 +/- 12.67 | +0.508 |

Neither run recorded a timeout. The book estimate was 17.06 Elo higher, with
an approximate direct 95% interval of `-0.11` to `+34.23` Elo
(`p=0.0515`, two-sided). That single between-opener strength difference is
suggestive rather than conventionally conclusive, while the 11.4-point
draw-rate increase is clear. The book's reported Elo interval was 8.6%
narrower, meaning the legacy opener would require roughly 20% more games for
the same nominal per-game precision.

The depth-16 book's **+24.01 +/- 11.59 Elo** is therefore the preferred
strength estimate for the candidate under the repository's intended balanced,
reasonable-opening benchmark. It should still be described as conditional on
that opening population rather than as a universal Elo value.

The interpretation is methodological as well as engine-specific. Color
swapping removes expected side bias, but a lopsided random position often
produces a split pair that says little about relative engine strength.
Depth-16-balanced starts leave more room for the tested evaluation change to
decide the game. The book also reaches reasonable 4-10-ply positions rather
than arbitrary 4-8-ply lines, so part of the larger measured gain may be a real
interaction with later macro-evaluation decisions. The current trinomial SPRT
does not explicitly model color-pair correlation; a pentanomial model would be
a useful future refinement.

---

## 42. Exact macro-state correction history (14 September 2026)

The accepted evaluation change is primarily an algorithmic improvement, not
an NNUE inference-speed optimization. The engine already maintained two
online correction histories for static-eval errors:

- a coarse exact table keyed by side to move, forced miniboard, and the
  nine-bit decided-miniboard mask;
- an exact table keyed by the side-to-move-relative contents of the currently
  forced miniboard.

Those keys cannot distinguish *which player* owns each decided miniboard.
That omitted information becomes increasingly important late in the game,
when otherwise similar decided masks can imply very different macro threats.
The search board already maintains an exact two-bit classification for each
of the nine miniboards, packed into an 18-bit side-to-move-relative macro key.
Round 8 adds a third correction table indexed directly by that key:

```text
2^18 exact macro states * 4-byte CorrEntry = 1 MiB per engine
```

No hashing, collision handling, or key construction is added to the hot path.
The existing incremental macro key is the array index. Each entry stores the
same bounded raw/applied pair as the older histories, and receives the same
search-bound feedback whenever a stored exact/lower/upper result genuinely
contradicts the corrected static evaluation. A grain of 12 converts raw
history units back into evaluation units.

An untouched entry is lazily initialized from the shipped trained macro
evaluator's free-choice output. This reuses learned macro structure as a prior
without paying for a new network or adding another model payload. The prior is
enabled only after at least three miniboards are decided; earlier macro scores
were too noisy and weakened the test. A one-unit raw sentinel distinguishes a
real zero prior from untouched zero-filled storage.

The qsearch fast path deliberately does **not** touch the new table. It uses
only the original structural correction for its cheap HCE fail-high test, as
before. Calling the general correction accessor there would initialize macro
entries as a side effect and read the larger table at every tactical node,
despite qsearch not using the macro correction. Removing that accidental work
restored the lost NPS and improved strength.

Several nearby variants were screened and rejected:

| Variant | Result | Decision |
| --- | ---: | --- |
| Add the trained macro residual directly at every non-PV interior node | N=784, +4.43 +/- 19.71 Elo, about 4% slower | reject |
| Exact macro history, grain 24 | N=4,000, +10.51 +/- 8.56 | too weak |
| Hashed macro + forced-miniboard interaction history | N=2,000, +9.04 +/- 12.26 | no gain |
| Exact macro history, grain 12, no learned prior | N=2,000, +18.26 +/- 12.28 | promising but below target |
| Exact macro history, grain 8 | N=1,328, +13.35 +/- 15.01 | reject |
| Free-choice macro prior at every game phase | N=2,000, +20.00 +/- 12.31 | weaker than gating |
| Prior only after three decided boards | N=2,000, +24.88 +/- 12.29 | retain |

Mean-over-constraint priors, a late current-constraint delta, double-strength
updates, skipping depth-one updates, and scaling the prior by 1.25 were also
tested and reverted. None improved on the simple gated free-choice prior.

The early screens above used contiguous ranges of the newly generated book.
One later range produced a sharply different result despite similar broad
ply, score, and first-move statistics. That exposed acceptance-order coupling
in the benchmark rather than a useful engine parameter. Section 41's pinned
Fisher-Yates traversal was added before final evaluation, and all final
numbers below use that representative order.

The final candidate first completed a fixed 2,000-game calibration on
permutation indices 0-999:

```text
W 644 / D 863 / L 493
+26.28 +/- 11.49 Elo
Timeout losses: Prev 0 / Dev 0
```

The formal gate was then run from permutation offset 1,000, so it reused none
of those calibration games. At the official 90 ms search budget with the
external 100 ms referee, H0=+20 and H1=+25 finished:

```text
N 4672  W 1564 / D 1941 / L 1167
+29.59 +/- 7.63 Elo
LLR +3.02241: PASS H1=+25 over H0=+20
Timeout losses: Prev 0 / Dev 0
Maximum response: Prev 96.18 ms / Dev 91.30 ms
```

The one-second startup benchmark in that run measured 12.10M nodes/s for Prev
and 12.33M for Dev, so the strength gain did not require a throughput tradeoff.
`crossfish_dev.hpp`, the readable CodinGame source, and the generated
submission carry the same correction logic. The regenerated payload is 93,272
bytes, leaving 6,728 characters below the CodinGame cap, with SHA-256
`c5aef709a1d182def56d54d7633fd42ca244aeaccf6488e57789acbe670496ae`.

An external readable-versus-minified smoke initially appeared to show timeout
losses, but it exposed a referee bug rather than an engine regression. The
persistent match protocol sent `NEW`, opening moves, and `GO` without an
acknowledgement, so the first nominal 100 ms move could include process
startup, evaluator-table construction, and opening replay—work that receives
1000 ms on CodinGame. The protocol now acknowledges `NEW` and a post-opening
`SYNC`, and each referee/bot pair is pinned to a distinct physical core.

With setup excluded correctly, a 500-game external-process comparison at
90 ms completed with zero timeout losses for both source forms. Maximum
responses were 90.32 ms for readable `codingame_nnue.cpp` and 90.27 ms for
the generated `cg_input.cpp`.

---

## 43. Pair-level SPRT and a 50,000-position benchmark (15 September 2026)

Round 9 began with several small search changes that looked promising in
short screens. The first candidate taken to a real H0=0/H1=+5 decision was a
larger late-move-reduction base of 65. Its early 700-game screen estimated
about +14 Elo, but the uninterrupted official run reversed:

```text
N 9884  W 2782 / D 4230 / L 2872
-3.16 +/- 5.18 Elo
LLR -3.04331: FAIL H1=+5 against H0=0
Timeout losses: Prev 0 / Dev 0
```

This was a useful warning against freezing changes from fixed-size screens.
The engine is mature enough that a plausible +5 Elo effect can require many
thousands of games, and short estimates can point in the wrong direction.

The next long run tested aspiration width 20 against width 40. It reached the
end of the original opening corpus without an SPRT decision:

```text
N 19992  W 5730 / D 8759 / L 5503
+3.95 +/- 3.61 Elo
LLR +2.29267
Timeout losses: Prev 0 / Dev 0
```

Those are the last clean figures before a full traversal. The old book held
10,000 positions and therefore only 20,000 color-swapped games. The harness
then silently reused positions with modulo indexing. Repetition does not
necessarily bias the Elo point estimate, but treating repeated roots as
independent overstates confidence and invalidates the LLR. Post-20,000 data
from that run were discarded, and aspiration width 20 remains promising but
unresolved until rerun on the replacement corpus.

The replacement-corpus rerun used the official 90 ms search budget, the
100 ms referee deadline, pair-level pentanomial SPRT, and unique positions
from the 50,000-position book. At the user's requested stopping point it had
still not produced a clear result:

```text
N 23142  W 6676 / D 9973 / L 6493
Penta 809 / 2756 / 4299 / 2857 / 850
+2.75 +/- 3.26 Elo
LLR +0.448 for H0=0 / H1=+5
Timeout losses: Prev 0 / Dev 0
```

This is an inconclusive result, not a pass. Aspiration width 40 therefore
remains the accepted baseline; width 20 was not frozen into the engine.

The testing infrastructure was hardened before spending more CPU:

- the default SPRT now uses pair-level pentanomial outcomes rather than
  per-game trinomial outcomes;
- the generalized-logistic LLR and pair-aware Elo interval are pinned against
  a known Stockfish Fishtest reference vector;
- the book never wraps silently and reports an inconclusive result on
  exhaustion;
- paired resume requires W/D/L, all five pentanomial bins, and the next
  opening-pair offset;
- generation writes atomic book-prefix and cursor checkpoints and resumes
  deterministically;
- the verifier retains both historical and current traversal fingerprints.

The replacement artifact was generated against frozen baseline `88c57b4`
with the same reasonable-move sampling, depth-12 prefilter, and authoritative
depth-16 `abs(score) <= 300` gate:

```text
Positions: 50,000
Unique paired-game capacity: 100,000 games
File size: 800,032 bytes
SHA-256: 1863013821e446ae2aab3b132a86863af64bcb83d815c2516c84a366809dfae5
Mean absolute stored score: 159.826
Absolute-score p50 / p90 / p99 / max: 165 / 275 / 298 / 300
Traversal fingerprint: 5306913481027657611
```

A 500-position depth-16 replay reproduced every stored score exactly. A
140-position depth-20 sample kept 106 positions inside +/-300 and 136 inside
+/-500, with mean absolute score 207.61. This closely matches the original
book's deeper audit and preserves the intended fairness tradeoff.

Finally, a fixed 1,000-game Prev-versus-Prev smoke used the new book,
pentanomial model, official 90 ms search budget, seven physical workers, and
the external 100 ms deadline:

```text
W 281 / D 456 / L 263
Penta 31 / 122 / 180 / 132 / 35
+6.25 +/- 15.50 Elo
LLR +0.300 for H0=0 / H1=+5
Timeout losses: Prev 0 / Dev 0
Maximum response: Prev 90.18 ms / Dev 90.12 ms
```

The identical-engine result is compatible with zero and confirms that the
expanded book, paired model, and no-wrap scheduler are ready for the pending
decision runs. After the inconclusive width-20 rerun documented above,
simplified qsearch capture ordering was tested next. A passing change is
frozen before the next candidate is measured, so small accepted gains
accumulate against an honest incremental baseline.

The simplified qsearch ordering had looked positive in short screens, but
the official replacement-book run reversed and reached a formal decision:

```text
N 7812  W 2144 / D 3468 / L 2200
Penta 291 / 962 / 1421 / 976 / 256
-2.49 +/- 5.60 Elo
LLR -3.05837: FAIL H1=+5 against H0=0
Timeout losses: Prev 0 / Dev 0
Maximum response: Prev 91.15 ms / Dev 96.47 ms
```

The full qsearch capture ordering therefore remains the accepted default.
Incremental TT-hash restoration was the next isolated candidate. It avoids
reversing every Zobrist component during unmake by saving the complete key
before each move and restoring it from the move stack. Despite an initially
encouraging throughput probe, the official run found no meaningful strength
gain:

```text
N 16884  W 4770 / D 7355 / L 4759
Penta 607 / 2017 / 3160 / 2074 / 584
+0.23 +/- 3.80 Elo
LLR -3.03183: FAIL H1=+5 against H0=0
Timeout losses: Prev 1 / Dev 1
Maximum response: Prev 106.42 ms / Dev 105.90 ms
```

The timeout losses were symmetric host jitter. Hash recomputation on unmake
therefore remains the accepted default. A three-way transposition-table
bucket, packed into the same 32-byte / 4 MiB table footprint, is the next
isolated candidate.

That follow-up compressed each entry to ten bytes: a 46-bit hash signature,
18-bit signed score, seven-bit move, two-bit bound flag, and seven-bit depth.
This fits three entries plus padding in each existing 32-byte bucket, raising
associativity by 50% without increasing the 4 MiB table. Both the normal and
packed variants passed all 27 unit tests, including score extremes and packed
metadata round trips.

The extra capacity did not translate into stable strength. Fixed 700-game
book screens finished:

```text
20 ms: -2.98 +/- 19.81 Elo, timeouts Prev 0 / Dev 0
90 ms: +0.00 +/- 18.93 Elo, timeouts Prev 0 / Dev 0
```

The official-time startup probe was also about 3% slower for the packed
variant because two of the three ten-byte entries are not naturally aligned.
With two independent screens centered on zero and no throughput gain, the
candidate did not justify a long SPRT. The full-hash two-way 32-byte bucket
remains the accepted default.

A second TT policy experiment protected an existing same-key entry when a
new non-exact result was at least five plies shallower. Exact results still
replaced, matching the broad shape of depth-preferred policies used by larger
engines. It also failed both fixed-size screens:

```text
20 ms: -6.95 +/- 19.47 Elo, timeouts Prev 0 / Dev 0
90 ms: -2.48 +/- 19.35 Elo, timeouts Prev 0 / Dev 0
```

The unconditional same-key replacement policy therefore remains the
accepted default.

---

## 44. A tree-identical hot-path rewrite worth +22% NPS (22 September 2026)

Round nine had produced six candidates and no passes. Every one of them tried
to change the search tree, and the two that looked most promising on short
screens reversed on long runs. This round changed nothing about the tree and
only made it cheaper to compute.

The candidate is a bundle of speed-only changes, which the gate table treats
as one axis. Each was required to produce a **bit-identical** tree: identical
scores and identical node counts, not "close".

- Move ordering carried two parallel arrays (`int scores[81]` and
  `FastMove legal_moves[81]`) through a stable insertion sort whose
  compare-and-shift inner branch mispredicts on nearly every element. Packing
  each move into one 32-bit key, `((1<<20 - score) << 8) | packed_move`, makes
  ascending key order exactly descending score with ties going to the smaller
  packed move, which is the smaller original index — the same permutation.
  Keys inside a list are then unique, so a key's rank *is* its sorted index
  and the sort becomes a branchless AVX2 rank computation plus a scatter. 93%
  of lists fit in one 8-lane vector.
- `check_winner_fast` ran a full win/draw scan at the top of every search and
  qsearch node. A terminal state can only be created or destroyed by a move
  that decides a miniboard, so the answer is now a cached `FastBoard` field
  refreshed only inside the existing decided/was_decided branches.
- The move, destination-constraint and side-to-move Zobrist terms are
  pre-XORed into one 1,296-byte table, so make and unmake each do one XOR
  rather than three.
- Squares 0..7 of a miniboard now leave movegen in one table load, one
  popcount and one 8-byte store. The move buffer carries `FAST_MOVE_SLACK`
  bytes of tail room for the wide store.
- The MiniNet first layer looked up `D16_MN_MASK_CODE[(mine << 9) | opp]` for
  all nine miniboards on every eval, nine probes into a 256 KiB byte table. A
  miniboard's centroid code only changes when that miniboard changes, so it is
  cached per (perspective, miniboard) and the eval reads nine contiguous
  bytes. This is deliberately **not** the rejected idea of maintaining the
  full 32-byte hidden accumulator incrementally (-5.87 Elo at N=1,184): only
  the byte lookup moved, not the vector adds.
- Smaller items: the global-HCE term cached rather than recomputed, a single
  latent-capture mask instead of both sides', an inline integer `lround` for
  the MiniNet head, the macro key set directly from the state index
  `make_move_fast` already knows, and signed `% 2` replaced by `& 1` where
  `n_moves` is provably non-negative.

Official 90 ms SPRT against the round-eight freeze, pentanomial pairs, seven
physical workers, the 50,000-position book:

```text
N 2366  W 723 / D 1048 / L 595
Penta 69 / 257 / 434 / 323 / 100
+18.81 +/- 10.18 Elo
LLR +3.010 (H0=0, H1=+5) — PASS
Timeout losses: Prev 0 / Dev 0
Maximum response: Prev 98.37 ms / Dev 98.40 ms
Prev NPS 11,881,856  Dev NPS 14,548,864 (+22.4%)
```

### What measurement made possible, and what it killed

`test_bots depth N` is not an equivalence test: each worker reuses one engine
pair across the two games of an opening pair, so it carries transposition,
history and correction-history state into the second game. A Dev that was
*provably* tree-identical to Prev measured **-8.44 +/- 23.84 Elo** there at
N=700. `cpp_impl/bench_ab.cpp` was added for this round and builds a fresh
engine per position, which makes score and node count deterministic functions
of the engine alone. Its `walk` mode replays one scripted move sequence
through a single engine instance at a real millisecond budget, the way a game
does, and reports mean completed root depth. Calibration from `walk`: **+3.3%
nodes is +0.167 ply**, and the bundle as shipped is +22.9% nodes / +0.5 ply.

Two candidates were killed by measurement before they cost any SPRT time.

**Transposition-table capacity.** The table is genuinely saturated in play:
about 39% occupancy after the first 90 ms move, 94% by move 3, 100% by move 9,
at 690k-1.0M nodes per move. That is not the same as being the bottleneck.
Doubling the entry count moves the TT hit rate only from 9.42% to 9.47%, and a
**16x** table (27% occupancy, so no eviction pressure at all) searches 3.9%
*more* nodes, because the pseudo-singular extension fires on any `tt_hit` with
`entry.depth >= depth - 3`, so a higher hit rate buys more extensions than it
saves in re-search. Node count at fixed depth is therefore not even a valid
quality proxy for a capacity change. Measured at equal time, doubling the table
to 8 MiB is **-0.258 ply** and about -10% NPS. An 8-byte-entry / 4-way design
that would have bought the same capacity for free (and whose natural alignment
fixes the stated failure cause of section 43's 3-way 10-byte bucket) was
dropped for the same reason: the capacity is not worth having.

**Lazy move selection.** A stable selection sort that rotates the best
remaining move to the front of the unsearched suffix provably yields the same
permutation as the stable insertion sort at every prefix (brute-forced over
800,000 random arrays), and would let a beta cutoff at move 2 skip ordering
moves 3..n. It measured **10-12% slower**. Instrumentation showed why: the
move loop visits 60% of slots on average (4.49 of 7.47 moves, and still 50% at
40-55-move free-choice nodes), because futility-pruned moves `continue` rather
than break and the history-malus loop at a cutoff needs the whole ordered
prefix. Selection only wins below roughly 30% visitation. The branchless sort
that shipped came out of the instrumentation that disproved the premise.

### Porting to CodinGame

`codingame_nnue.cpp` keeps its own standalone copy of the board and the
search, and nothing in `make test` checked that a port preserved behaviour;
the only documented check was playing minified against readable, which cannot
tell a correct port from a subtly wrong one. `cpp_impl/cg_selfcheck.cpp` now
closes that: it renames the CG file's `main()` away, includes it whole, and
drives the CG engine's own `search_fixed_depth` over a deterministic position
set, printing a checksum over every search's score and node count. For a
tree-identical change the pre-port and post-port checksums must match exactly.

The port reproduced them at fifteen (positions, depth) configurations spanning
depths 5 through 16 and roughly 100M nodes, including a run against the
**minified bundle** itself (the minifier's rename map was recovered from
`tools/cg_minify.py` to drive `cg_input.cpp`'s renamed engine). Instrumented
builds additionally asserted, at every node, that each cached value equalled a
from-scratch recompute — terminal state, `out_of_play`, `active_board`,
centroid code, macro key, cached MiniNet output, `lround_bits`, and the cached
global HCE term — with zero failures over ~12M nodes per configuration.

Live 90 ms searches on the shipped binary: turn-one nodes rose from a mean of
1,104,597 to 1,363,328 (+23.4%) and turn-two from 1,076,096 to 1,321,856
(+22.8%), matching the +22.4% the SPRT header reported.

Submission size went from 93,272 to **96,887** characters of the 100,000 cap,
leaving 3,113. Static storage grew by 4,368 bytes (a 1,296-byte combo-hash
table, a 2 KiB empty-square table, a 1 KiB open-win table); the 4 MiB
transposition table and 5 MiB macro table are unchanged, and `FastBoard`
itself got smaller. The sort scratch is a member rather than a local so the
recursive search frame does not grow. First-turn initialisation uses about
155 ms of the 1,000 ms allowance, down from 181 ms.

Two things to know before the next change. Headroom is now thin: 3,113
characters. Dead public API the CG bot never calls (`fill_captures_lut`,
`is_capture_avx(Board&)`, `is_block_avx`, `creates_two_in_a_row`,
`get_move_scores`, `eval_extra`, `eval_diffs`, `eval_parts`) and the generic
no-op template overloads kept for reference parity are non-behavioural
reclaim candidates if a future change needs room. Separately, the shipped
`CODINGAME_MOVE_MS = 90` replies at 90.1-90.6 ms against a 100 ms referee, so
there is only about 10 ms of scheduling slack; that is an argument against
raising it, not for.

---

## 45. Pack network weights at 14 bits per character (23 September 2026)

Round nine left 3,113 characters of submission headroom, and 57,414 of the
96,887 characters were the two network payloads in ASCII85 (6.4 bits per
character). CodinGame counts the cap in UTF-16 code units, not bytes (a
forum user established this by testing), so any character from the Basic
Multilingual Plane outside the surrogate range costs one unit however many
UTF-8 bytes it takes.

The payloads now map 14-bit groups onto U+4E00..U+8DFF, the first 16,384 CJK
Unified Ideographs. That block has no combining marks, separators, invisible
or bidi characters, or normalization decompositions, which is why the alphabet
stops at 14 bits rather than reaching for 15. The decoder reads the UTF-8
bytes of an ordinary narrow literal.

| | ASCII85 | CJK14 |
| --- | ---: | ---: |
| D16 payload characters | 53,569 | 24,489 |
| Macro payload characters | 3,845 | 1,758 |
| `cg_input.cpp` (UTF-16 units) | 96,887 | 65,731 |
| Headroom | 3,113 | 34,269 |

This is a representation change only. Both payloads decode to the same bytes
as before (42,855 and 3,076 bytes, FNV-1a `e35e987c17a453cf` and
`626e29f3a8d65679`, pinned by the unit tests), `cg_selfcheck` reproduces the
round-nine checksums on both the readable source and a minified bundle at
depths 7, 9 and 11, and first-turn initialization is unchanged at about
167 ms. No SPRT is needed or meaningful.

Two caveats. The file is now 118,225 UTF-8 bytes, so if CodinGame ever
counted bytes the paste would be rejected at submit time; the counting rule
comes from a forum user's testing, not official documentation, and should
be confirmed by a real paste. And `wc -c` no longer reports the capped
quantity; the minifier's printed count does.

---

## 46. A full-coverage opening book (23 September 2026)

The CJK14 repack left 34,269 characters of headroom, enough for an opening
book. At 90 ms the engine reaches depth 11-12 in the early game; a 3-second
search reaches about 18, and the two chose different moves in 47% of early
positions (a 3 s vs 6 s control agreed 85% of the time, so the gap is real).
The question was how to turn that into Elo against opponents we cannot see.

**Selective books did not transfer.** A pilot that followed only likely
opponent replies (within a margin of the best, by a depth-12 ranking) left
book after about three moves: engine replies are hard to predict, and the
engine's own 90 ms choice was the depth-12 top reply only 29% of the time. A
book grown on-policy from self-play covered 6.8 and 8.0 moves per game and
measured +38.5 ± 11.0 against our own engine, but only because it had learned
that opponent's lines: against the round-six engine it fell out of book after
2-3 moves and was worth about 0 (paired, 1,000 openings).

**Full coverage did.** Covering every opponent reply through our 5th move
after the center opening (moving first) and our 4th move (moving second) is
20,883 positions after symmetry and transposition merging. Against the
round-six engine it gave exactly 5 and 4 book moves in every game and a paired
book value of +21.3. A 3,000-game head-to-head against the current engine,
which it was not fit to, measured +19.6 ± 11.1 (1281-607-1112); a 1,500-game
extension on the packed, shipped book added +17.2 ± 15.7. Pooled: +18.8 ± 9.1
over 4,500 games, LLR 3.67 (H0=0, H1=+5), a pass. A second paired round-six run
with a new seed gave +19.5.

The book stores no keys. The packer and the bot drive one shared walk over the
book, so the payload is only each move's index among the legal moves, in mixed
radix: 4,656 characters, 3% above the information content. The bot decodes it
in about 5 ms of the first turn, still runs its 90 ms search in book (warming
its tables, as tested), and only plays a book move that is legal. The whole
feature costs 8,312 characters; 25,957 remain.

The official SPRT cannot gate this: its games start from the 50,000 SPRT
openings, where a start-position book never applies. Book changes are gated
by start-position matches (`make -C cpp_impl play-book-match`) and a paired
run against a different engine. See `documentation/play_book.md`.

---

## 47. Make the submission fast under CodinGame's own compiler (24 September 2026)

Every measurement in this log was made with `-O3` on the command line.
CodinGame compiles C++ with g++ 11.2 and
`-std=gnu++17 -Werror=return-type -g -pthread`: no `-O` at all. The source's
`#pragma GCC optimize("O3")` then optimizes each function body, but a global
`-O0` leaves GCC's inliner off for everything that is not `always_inline`.
The hot path was full of such calls: `std::array::operator[]` (every
miniboard's markers), `std::min` / `std::max`, `FastMoveStack::top`, and the
engine's own make/unmake, hashing and eval helpers. Built exactly the way
CodinGame builds it, the shipped paste file searched about 147k nodes per
move against about 700k at `-O3`.

The fix is mechanical and tree-identical:

- the optimize pragmas move above the includes (the `target` pragma stays
  after them: GCC 13 rejects it in front of libstdc++);
- every engine member function except the recursive `search` / `qsearch`
  and the one-time setup paths is `__attribute__((always_inline))`, as are
  the move-stack methods and the move-generation lambdas;
- `std::array` becomes `cf_array` (same layout, always-inline indexing) and
  `std::min` / `std::max` become `cf_min` / `cf_max`;
- the minifier keeps `__attribute__` and `always_inline` (it had renamed the
  attribute to `q`, which silently disables it; a unit test pins this).

`__attribute__((flatten))` on `search` was tried first and does nothing at
`-O0`; moving only the pragmas recovered 1.6x; `always_inline` on the
helpers 1.9x; the std replacements the rest.

| Build (CodinGame flags, g++ 11) | Mean nodes per searched move |
| --- | ---: |
| Round-nine paste file | 147k |
| Pragmas before includes | 239k-323k |
| + `always_inline` helpers | 287k-328k |
| + `cf_array` / `cf_min` / `cf_max` (shipped) | 669k-722k |
| Round-nine paste file at `-O3` | 656k-762k |

`cg_selfcheck` checksums (scores and node counts) are identical to the
round-nine port at depths 7, 10 and 13, at `-O3` and at CodinGame's flags.
New versus old paste file, both compiled with CodinGame's flags, 90 ms,
process referee, random openings:

```text
N 400  W 238 / D 99 / L 63   Elo +163.0 +/- 31.7   timeouts 0 / 0
```

The protocol check with the CodinGame-flag build keeps exact book coverage
(30/30), first turn at most 187 ms, later moves median 90.2 ms and max
90.9 ms. The paste file grew 2,911 characters to 76,954.

`make test` now also builds the paste file with CodinGame's command line
(`make -C cpp_impl cg-flags`), and `tools/cg_speed_check.py` compares its
nodes per move against the `-O3` build. The Dev-vs-Prev SPRT cannot see any
of this: both sides are compiled at `-O3`. Any CodinGame ladder result from
before this change was obtained at roughly a fifth of the tested speed.

**Independent review.** The reviewer rebuilt both paste files with g++ 11.5
and CodinGame's flags: mean nodes per move went from 141k to 497k (3.5x) against
~583k at `-O3`, with identical `cg_selfcheck` checksums at depths 7 and 10. A
600-game new-versus-old match on openings disjoint from the ones above gave
348-163-89, **+160.5 +/- 25.2** Elo. On CodinGame itself the NPS jump was
visible in the bot's `N` output immediately.

**The symptom to recognise.** On 25 September a CodinGame self-play log showed
one side searching 85k-160k nodes per move and the other 400k-925k. The slow
side had been pasted from an old copy whose first line was not
`#pragma GCC optimize("O3")`. About 130k nodes per 90 ms move is this
section's signature; check the first line of the paste before suspecting the
engine.

---

## 48. Restore instead of re-derive: a tree-identical speed round (24 September 2026)

Speed round ten was speed only, on the round-nine freeze. (It is separate
from the round-ten search experiments listed in section 11.) Every accepted change
computes the bit-identical tree (`bench_ab equiv` IDENTICAL at depths 7, 8
and 11, 320 positions), so the gate was identity first, then timing, then the
official 90 ms SPRT.

### Measurement first

On this 4-core cloud VM a single `bench_ab sat` run swings by ±4% with Dev
and Prev being the same code, so one run cannot see a 2% change.
`tools/speed_ab.py` repeats the paired saturated-TT run, pins it to one core,
checks node counts are identical on every run and reports the mean time ratio
with a 95% CI. A pinned A/A control read +0.09% ± 2.1%.

Individual micro-changes are below that noise floor, so each was first
screened by a deterministic simulated cost: callgrind with cache and branch
simulation over a fixed Dev-only workload (two scripted games, depth 9,
persistent engine), estimated as Ir + 12·L1 misses + 150·LL misses + 16·branch
mispredicts. Only changes that cut that cost went into the bundle, and the
bundle was then timed on the wall clock. Two cautions from using it:
callgrind's branch predictor is indexed by code address, so mispredict counts
move with layout on lines a change never touched; and it cannot see the TLB.

The starting profile was flat. By source function (inlined code attributed
back): search 25%, make/unmake and the helpers they call 29%, move ordering
10%, HCE finish 7.5%, MiniNet 3.4%.

### Accepted (bundle)

- **Undo records.** Unmake used to re-derive everything make changed: XOR the
  hash back, re-look-up both MiniNet codes, re-accumulate the HCE score and
  threat maps, and probe the three decided-state masks to find which one the
  move set. Make already holds every value before it overwrites it, so it now
  writes a 32-byte record (hash, HCE local/global score, both threat maps, the
  miniboard's score and flags, both code bytes, active board, terminal, the
  decided state) and unmake restores it. -55M simulated instructions (-2.5%),
  but the largest wall-clock gain of the round: the bundle with it measured
  +10.7% [+8.6, +13.0] against +2.8% [+1.0, +4.6] without it. Removing the
  dependency chains mattered more than the instruction count.
- **Vector move scoring.** Every ordering term except the counter move depends
  only on (miniboard, square), and moves arrive grouped by miniboard, so each
  group scores all nine squares at once in 16 int16 lanes. Killers are
  mirrored as a per-ply bitmask and history as a `history / 20` shadow table
  written wherever history changes, so no per-move division remains. -109M
  simulated instructions.
- **Running decided-miniboard term.** The MiniNet's per-miniboard super-class
  vectors depend only on which miniboards are decided, so their sum is kept per
  perspective and updated where the macro key changes (rare), replacing nine
  class-dependent loads and branches per leaf. The class-0 rows are exactly
  zero (`lround(x - x)`), which is what makes this exact. This is not the
  rejected full hidden-accumulator (section 44): only the rarely-changing term
  is incremental. -35M.
- **constexpr `eval_weights`** (never written) lets every pruning margin fold
  to an immediate. -10M.
- **Exact fast `lround`.** Truncate, take the fractional part (exact for
  |x| < 2^31) and step away from zero at one half. Checked equal to
  `std::lround` on all 2,650,800,128 finite floats below 2^31; larger inputs
  keep the old conversion. -15M.

Bundle timing against the round-nine freeze: `speed_ab` +8.0% [+6.3, +9.8]
(n=40, node counts identical every run); `walk 10 40 90` +5.5% and +8.6% nodes
at a fixed move budget; the SPRT header's 1-second startpos search
17,689,984 → 19,563,392 NPS (+10.6%).

### Rejected (measured, not kept)

- **Ternary-indexed miniboard table** (one 77 KiB `{score, flags, code}` table
  indexed by an incrementally kept base-3 key, replacing reads from the
  512 KiB and 256 KiB sparse tables). Cut simulated L1 misses 21% but added
  instructions; wall clock -2.6% ± 1.5% alone and -1.6% [-3.6, +0.5] on top of
  the undo records. Tried twice; the sparse tables are L2-resident and their
  misses overlap.
- **Huge-page transposition table** (`aligned_alloc` + `madvise(MADV_HUGEPAGE)`;
  the kernel granted 4 MiB of huge pages). +1.3% ± 2.4% on `sat`, and
  +1.2%, -4.2%, +0.7% nodes in three `walk` runs. No evidence, and CodinGame's
  THP setting is unknown.
- **Static Zobrist tables** instead of `FastBoard` reference members: -0.1%
  instructions. Not worth the churn.
- **Cached lined-up threat term** in make: the local two-in-a-row flags change
  on too many moves, and there are fewer evaluations than makes. +19.6M
  instructions.
- **Out-of-line qsearch capture loop** to shrink the stand-pat prologue: -0.9M;
  GCC keeps the heavy prologue for the inlined eval.
- **Fixed-length rank loop** in the ≤8-key sort: +7M instructions, neutral.

### Result

Official 90 ms SPRT against the round-nine freeze (`35dd664` vs `bcc0e30`),
pentanomial pairs, three workers with one of four cores reserved, the
50,000-position book:

```text
N 5478  W 1580 / D 2463 / L 1435
Penta 167 / 612 / 1064 / 701 / 195
+9.20 +/- 6.53 Elo
LLR +3.015 (H0=0, H1=+5) — PASS
Timeout losses: Prev 13 / Dev 10
Maximum response: Prev 140.23 ms / Dev 163.83 ms
Prev NPS 17,689,984  Dev NPS 19,563,392 (+10.6%)
```

An independent review SPRT of the same candidate against the round-nine
engine, on openings disjoint from the ones above, was stopped undecided at
N=8184, **+3.06 +/- 5.44** Elo (LLR +0.36), and the reviewer's startpos NPS
gain on a different 4-core host was +5% rather than +10.6%. Round ten alone is
therefore a smaller gain than its first SPRT suggested; section 49 has the
direct measurement of rounds ten and eleven together.

The run started badly (-13 +/- 18 at N=606) and that was checked rather than
waited out: a persistent-engine test drove both engines through scripted games
via `getMove` at a fixed depth, carrying TT, history, killers, counters and
correction history across moves as a real game does, and found identical
moves, node counts and root scores in all 310 searches. Only speed could
separate them. The timeouts are host stalls on this shared VM (both engines,
up to 164 ms), not engine overruns: normal replies stayed at 90-93 ms.

Porting: the patch applied to `codingame_nnue.cpp` except where its layout
differs (the rounding and MiniNet block, and the move-scoring loop, which the
CG copy keeps without a disabled `#ifdef`), which were ported by hand. The
port was then rebased onto section 47's CodinGame-compiler fix, and its new
hot-path code follows those rules: `cf_array` for the undo records and the
killer and history shadows, `always_inline` on make/unmake, the super-term
helpers, `sync_history_div` and the lane lambda, and `__builtin_fabsf`
rather than `std::fabs` in the rounding. `cg_selfcheck` reproduces the
pre-port checksums at depths 5, 7, 9 and 11 at `-O3` and at depths 5, 7 and
9 with CodinGame's flags (g++ 11, no `-O`), and on a `--no-rename` bundle.
Built with CodinGame's flags, the paste file searches 764k nodes per move
against 721k for the section-47 build (`tools/cg_speed_check.py`, six games
against a random opponent). Submission: 79,587 characters, 20,413 below the
cap.

Rebuilding the paste file also exposed a minifier bug: it avoided only names
already present in the source, so it could hand out `j0`, `j1`, `jn`, `y0`,
`y1` or `yn`, which glibc's `<math.h>` declares globally as Bessel functions.
Harmless on a variable, fatal on a type (`PbReader` became `jn`, and the
function hid it). Those six names are now reserved, and a unit test fails
without the fix.

---

## 49. Hide the table latency, skip the heavy frame (24 September 2026)

Round eleven started from the round-ten freeze with the same rules: speed
only, bit-identical tree, a wall-clock screen with `tools/speed_ab.py`, then
the official SPRT. Round ten's lesson was that removing dependency chains
paid far more than instruction counts predicted, so this round ranked the
profile by simulated L1 misses and mispredicts too. The largest single miss
source was the transposition-table probe at the top of every search node.
The only prefetch was issued after `make`, just before recursing, so little
of the latency was hidden.

### Accepted (bundle)

Each increment was timed against the previous stage, n=40 paired runs unless
stated.

- **Prefetch the next sibling's TT line** before searching the current move.
  Its key is this node's key minus the old destination term plus one combo
  term, and the whole of the current move's subtree hides the latency. A move
  that also decides a miniboard hashes differently, and its prefetch is
  simply wasted. **+5.7% [+3.5, +8.0]**; `walk` at 90 ms +4.8%, +1.7%, +6.3%
  nodes.
- **Prefetch the hash move's child line** right after the probe, before
  pruning, move generation and ordering run. **+2.0% [+0.7, +3.4]**.
- **Prefetch the first ordered child's line before it is made** (move 0
  without a hash move, move 1 after deferred ordering). **+0.84% [+0.04,
  +1.66]** at n=100.
- **`search_leaf` for depth <= 0 children.** A third of all `search()` calls
  only run the entry checks and TT cutoffs before handing off to qsearch, but
  paid for search's full register-save prologue and move-loop frame. The
  move loop now sends those children to a small function with the same front
  code. **+3.8% [+2.4, +5.3]**.

Bundle against the round-ten freeze: `speed_ab` **+7.9% [+6.1, +9.8]**;
`walk` +5.7%, +6.2%, +7.9% nodes; SPRT header NPS 19,385,984 -> 20,566,400.
Identity: `bench_ab equiv` IDENTICAL at depths 8 and 11, and persistent
`getMove` identical over 531 searches at depths 7 and 9.

### Rejected (measured, not kept)

- Prefetching the sparse per-miniboard tables (`fast_local_score`,
  `fast_tiar_flags`, `D16_MN_CODE`) for the next move: -0.9% [-2.2, +0.3].
  Those misses are L2 hits that out-of-order execution already hides; only
  the table and macro-sized lines are worth fetching early, and of those
  only the TT paid.
- Prefetching the child's 5 MiB macro-table entry: -0.3% [-1.9, +1.5].
- Branchless LMR reduction: +0.5% [-0.6, +1.7].
- Outlining the qsearch capture loop, re-timed on the wall clock this round:
  +0.5% [-0.8, +1.9]. Its instruction-count rejection in section 48 stands.

### Result

Official 90 ms SPRT against the round-ten freeze (`109e2b7`), pentanomial
pairs, three workers with one of four cores reserved:

```text
N 4236  W 1214 / D 1942 / L 1080
Penta 132 / 438 / 848 / 564 / 136
+10.99 +/- 7.31 Elo
LLR +3.032 (H0=0, H1=+5) — PASS
Timeout losses: Prev 9 / Dev 8 (host stalls, up to 318 ms)
Prev NPS 19,385,984  Dev NPS 20,566,400
```

Porting: the patch applied except for the new `search_leaf`, whose copied
front called `tt_score_from_store`, which the CG file does not have (its
search reads `entry.score`, identical while `CROSSFISH_NORMALIZE_TT_MATES` is
off). The CG leaf was rebuilt from the CG file's own search front. Like
`search` and `qsearch`, `search_leaf` stays a real function under
section 47's rules (its point is to be a cheap call); the `search_child`
dispatcher is `always_inline`. `cg_selfcheck` reproduced the pre-port
checksums at depths 5, 7, 9 and 11 at `-O3` and at depths 5, 7 and 9 with
CodinGame's flags. Built with CodinGame's flags (g++ 11, no `-O`), 16 games
against a random opponent through `tools/cg_speed_check.py`:

| Paste file | Mean nodes per move | Median |
| --- | ---: | ---: |
| Section 47 (round nine) | 649k | 728k |
| + speed round ten | 758k | 869k |
| + round eleven | 784k | 924k |

Submission: 80,576 characters, 19,424 below the cap.

The two rounds merged as one PR, so the reviewer measured the bundle directly:
PR Dev (rounds ten and eleven) against `main`'s round-nine Prev, 90 ms,
openings from offset 30,000:

```text
N 2748  W 818 / D 1234 / L 696   Penta 76 / 287 / 543 / 375 / 93
+15.43 +/- 9.05 Elo
LLR +3.012 (H0=0, H1=+5) — PASS
Timeout losses: Prev 13 / Dev 5 (host stalls, up to 158 ms)
Prev NPS 13,394,176  Dev NPS 14,935,936 (+11.5%)
```

Sequential SPRTs do not add: +9.2 and +11.0 against moving baselines became
+15.4 measured in one step.

---

## 50. Round twelve: no candidate survived (24 September 2026)

Screened against the round-eleven freeze with `tools/speed_ab.py` (n=40
paired runs each; a same-day A/A control read +0.46% [-1.16, +2.14]). None
was kept, and Dev stays identical to Prev.

- **Branch hints** (`__builtin_expect` on `stopped`, terminal returns and the
  clock check in `time_up`): tree-identical but **-2.67% [-4.15, -1.15]**,
  measurably slower; GCC's layout and inlining changed for the worse.
- **Skip the repeated leaf probe.** With no reduction, a null-window probe
  that beats alpha is repeated with the identical depth and window. For a
  leaf child (depth <= 0) the repeat is provably redundant (search_leaf and
  qsearch only read engine state), so skipping it kept every score (320
  fixed-depth positions) and every persistent `getMove` move and root score
  (656 searches) while searching 1.8-3.8% fewer nodes. Wall clock:
  **-0.52% [-1.91, +0.91]**. The skipped probes run on lines that are
  already hot, so they were nearly free; node count is not cost.
- **Prefetch the children's macro-correction line** once per node:
  **-2.31% [-3.90, -0.66]**.

What is left of the profile is search control flow, make/unmake and move
ordering, with the TT latency now hidden. The next speed gain probably needs
a structural change (for example a specialised depth-1 node, or a smaller
per-node state) rather than another local rewrite.

---

## 51. Keep eval pruning away from mate-range bounds (24 September 2026)

Found from outside the SPRT: in a 1,000-game match against a CodinGame-rules
uttt.ai network, 155 of crossfish's 21,780 searched moves (in 146 games)
reported a root score between 20,000 and 90,000. That is neither an eval nor a
mate score (mates are `±(99999 - ply)`). The values clustered at 51,566-51,570,
which is `99969 - 48400`: a mate score minus the initial 400-unit aspiration
half-width and four widenings (400 + 1,200 + 3,600 + 10,800 + 32,400).

### What was wrong

A per-iteration trace of one such position (a forced win in 30) showed every
depth failing low five times before a wide re-search found the same mate:

```text
depth 29  window [99569, 100369] -> 99569 FAIL-LOW
depth 29  window [98369, 100369] -> 98369 FAIL-LOW
...
depth 29  window [-45631, 100369] -> 99969            (inside the first window)
```

74 of 197 iterations in a 3-second search were such mate-window fail-lows,
and when the 90 ms clock ran out inside a cascade the root reported the
widened bound (51,569) as its score. With eval pruning disabled the same
search had none.

The cause is three prunes that compare the static eval with the window:

- **Reverse futility** returns beta when `static_eval - margin >= beta`.
  With beta at -99,569 any ordinary eval passes, so the node claims to escape
  the mate without searching.
- **Futility pruning** skips quiet moves when `static_eval + margin <= alpha`.
  With alpha at +99,569 it skips every quiet move, including the one that
  mates sooner.
- **qsearch delta pruning** does the same for captures.

A static eval says nothing about how soon anyone is mated, so the three prunes
now run only against bounds outside mate range: reverse futility needs
`|beta| < CORR_MATE_BOUND`, futility `|alpha| < CORR_MATE_BOUND`, delta
pruning `alpha < CORR_MATE_BOUND`. Stand-pat is unchanged.

### Measurement

The change alters the tree wherever a mate score appears in it: `bench_ab
equiv 80 8` differs in 65 of 80 positions, at +1.1% nodes and -0.6% time
(`bench_ab nodes 120 8`). The symptom, on crossfish turns from recorded games
searched fresh at 90 ms (`cg/tools/mate_window_probe.py` in the uttt.ai fork):

| | before | after |
| --- | ---: | ---: |
| traced position: mate-window fail-lows in 3 s | 74 | 0 |
| after a mate score (150 positions): searches with any | 20 | 1 |
| middlegame (150 positions): searches with any | 5 | 0 |
| bound reported as the score | 2 | 0 |
| median depth, middlegame | 18 | 18 |

Middlegame move choices differed in 9 of 150 positions, inside the 12 of 150
that the unchanged build differs from itself at 90 ms; after a mate score
they differed in 17 of 150 against 1 of 150 for the unchanged build, as the
search now finds shorter mates (mate in 20 rather than 30 on the traced
position).

This is a bug fix, so it was gated for **non-regression** (H0 = -5, H1 = 0)
rather than the +5 bar a search change normally needs. SPRT at 90 ms against
the round-eleven freeze, six workers on eight physical cores:

```text
N 2880  W 801 / D 1354 / L 725
Penta 63 / 345 / 569 / 379 / 84
+9.17 +/- 8.56 Elo
LLR +3.055 (H0=-5, H1=0) — PASS
Timeouts: Prev 0 / Dev 0   Maximum response: 90.14 ms / 90.15 ms
Prev NPS 30,783,360  Dev NPS 31,464,192
```

A run at the usual bar (H0 = 0, H1 = +5) on fresh openings (`SPRT_GAME_OFFSET=1440`)
was stopped undecided:

```text
N 10488  W 2773 / D 5048 / L 2667
Penta 263 / 1274 / 2094 / 1320 / 293
+3.51 +/- 4.51 Elo
LLR +0.957 (H0=0, H1=+5) — stopped, undecided
Timeouts: Prev 0 / Dev 0
```

Over both runs (13,368 games) the fix measures about +4.7 Elo: a small gain,
too small to clear +5 in reasonable time, and shipped as a correctness fix.

An independent review ran the same non-regression test against `main`'s
round-eleven Prev on openings from offset 40,000. It needed many more games,
and one container restart (resumed from its W/D/L and pentanomial counters),
but passed:

```text
N 13212  W 3635 / D 5952 / L 3625   Penta 407 / 1621 / 2557 / 1597 / 424
+0.26 +/- 4.17 Elo
LLR +3.053 (H0=-5, H1=0) — PASS
Timeout losses: Prev 29 / Dev 36 (host stalls)
```

Pooled over all three runs the fix is roughly +2 Elo in self-play: it costs
nothing and removes the wrong scores. Its value is in positions a self-play
SPRT rarely reaches, converting won games faster and reporting real scores.
The non-regression gate (H0=-5, H1=0) is the repository's rule for bug fixes;
search and eval experiments still need H0=0, H1=+5.

The CodinGame performance gate (section 47's compiler, `tools/cg_perf_gate.py`)
passed against `main` with g++ 11.2: nodes per millisecond at full budget
0.968 [0.936, 1.000] (the candidate now uses its full budget on 93% of replies
against 71%, adding won positions the old build left early, so the two sets of
replies differ); inlining 96% of `-O3`; later replies p99 90.1 ms with none at
or over 100 ms; smoke match 58-77-65 with no timeouts; 40/40 book games.

### Port

The same three guards went into `codingame_nnue.cpp`. A tree-changing port
cannot be checked by `cg_selfcheck` against itself, so `engine_selfcheck.cpp`
runs `cg_selfcheck`'s positions and checksum over `CrossfishDev` or
`CrossfishPrev`: the unported CG file matched Prev at depths 7 and 9, and the
ported file matches Dev at depths 5, 7, 9 and 11 (and the no-rename minified
bundle at 7 and 9). `make -C cpp_impl port-check` repeats that comparison.
`cg_input.cpp`: 80,617 characters (19,383 left). The protocol check kept 40/40
exact book games, first turn at most 187 ms, later replies median 90.1 ms and
max 90.2 ms.

### Not fixed

A rarer fault remains: occasional wrong proofs with state carried between
moves. In one 1-second game crossfish played a move it scored as a forced win
(+99,976) that a fresh search proves lost; in a 90 ms game it reported a
forced loss (-99,977) in a position a fresh search proves won, and then
played the losing move. Neither reproduces from a fresh engine, and replaying
the game's history reproduced one of them in 1 of 15 tries, so it depends on
table state and timing. This change does not show whether it helps; the
probes to chase it (a traced debug build with THINK, PV and switches for each
pruning and shortcut) are in the uttt.ai fork's `cg/tools/`.

---

## 52. Close the rest of the CodinGame-flags gap (24 September 2026)

Section 47's transform only matched function signatures that fit on one
line, so every signature wrapped across lines was skipped, and the
`std::array` sweep did not cover the transposition table, a `std::vector`.
Built the CodinGame way and disassembled, `CrossfishDev::search` still made
26 out-of-line calls and `qsearch` 13:

| Call target | `search` | `qsearch` |
| --- | ---: | ---: |
| `std::vector<CompactTTBucket>::operator[]` (probe, IID re-probe, 2 prefetches, store) | 5 | 0 |
| `get_fast_move_scores<FastBoard>` | 5 | 3 |
| `finish_hce<FastBoard>`, `finish_hce_with_global<FastBoard>` | 2 | 2 |
| `cached_mini_key<FastBoard>` | 1 | 0 |
| `d16_mini_hsum256` (`mini_eval_d16.hpp`) | 1 | 1 |
| `evaluate_macro_key` (`macro_eval.hpp`) | 1 | 1 |
| recursion | 4 | 1 |
| `std::chrono` clock read and compare (once per 128 nodes) | 3 | 3 |
| `memcpy`, `memmove`, `__stack_chk_fail` | 4 | 2 |

The fix, again tree-identical:

- the table becomes a `cf_heap_array<CompactTTBucket>`: the same
  value-initialized heap allocation (`new T[n]()`, 32-byte aligned through
  C++17 aligned new), with deep copy, move, and an always-inline
  `operator[]`. `run_match` resets the engine by move assignment, which the
  type supports;
- `always_inline` on the multi-line helpers: `cached_mini_key`,
  `get_fast_move_scores`, `finish_hce_with_global`, `finish_hce`, and
  `eval_extra_from_maps` (called only from `finish_hce_with_global`, so it
  becomes the next call once that one inlines);
- `always_inline` (and `inline`, which silences GCC's "might not be
  inlinable" warning on a plain `static`) on `d16_mini_hsum256` and
  `evaluate_macro_key` in the **shared generated headers**. Dev, Prev and
  `test_bots` include them too; the attribute changes no code at `-O3`, and
  `bench_ab nodes 60 9` gives the same 1,681,665 nodes before and after.
  `tools/nnue_emit_mininet_header.py` and `tools/nnue_emit_macro_header.py`
  now emit the attribute, so a regenerated header keeps it.

Afterwards `search` makes 12 calls (4 recursion, 3 clock, `memcpy` x2,
`memmove`, `__stack_chk_fail`, and the never-taken lazy
`macro_load_packed()` fallback inside `evaluate_macro_key`) and `qsearch` 7
of the same kinds. `get_fast_move_scores` inlined whole, with nothing out of
line left inside it.

Search time for the identical tree (`cg_selfcheck 30 13`: 30 positions,
9,617,942 nodes, same checksum everywhere), g++-11 for every build, mean of
five interleaved runs with start-up subtracted:

| Build | `main` | This change |
| --- | ---: | ---: |
| CodinGame flags | 0.691 s | 0.628 s |
| `-O3` | 0.633 s | 0.601 s |
| CodinGame / `-O3` speed | 91.6% | 95.7% |

The CodinGame build is 10% faster and now matches the previous `-O3` build.
What remains (about 4%) is code generation that per-function `optimize`
attributes cannot reach under a global `-O0`, not calls.

`tools/cg_speed_check.py` cannot resolve a difference this size: its games
diverge with timing, so mean nodes per move mixes speed with game phase (two
builds of the same tree read 630k and 716k on the same seeds). `cg_selfcheck`
now also prints `seconds=` and `nps=` for its searches (after a depth-1
warm-up, so table initialization is outside the timed region), and
`make -C cpp_impl cg-speed` builds it at `-O3` and with CodinGame's flags and
runs both. The checksum lines must match; the seconds ratio is the gap.

The paste file is 77,647 characters (22,353 left); all 101
`__attribute__((always_inline))` in the readable source and inlined headers
survive minification.

New versus `main`, both paste files compiled with CodinGame's flags, 90 ms,
process referee, 1,000 games:

```text
N 1000  W 335 / D 345 / L 320   Elo +5.2 +/- 17.4
```

A 10% speed-up is worth single-digit Elo, so this is a no-regression check,
not a measurement. The run logged 3 timeouts for the new build and 6 for the
old one while other jobs shared the machine. On an idle machine the protocol
check gives the new build a first turn of at most 182.8 ms and later moves
of median 90.2 ms, max 91.6 ms, with exact book coverage (30/30).

The branch landed after sections 48-51, so its gain was re-measured against
that `main` in review. Same-tree search time with CodinGame's flags fell from
0.533 s to 0.512 s (-3.9%; `-O3` 0.507 s to 0.494 s), the CodinGame gate read
+4.6% [+0.4%, +9.0%] nodes per millisecond locally and **+8.7% [+6.6%,
+10.8%]** in CI with g++ 11.2, and out-of-line calls fell from 32 to 15 in
`search`, 13 to 7 in `qsearch` and 5 to 4 in `search_leaf`. After the merges
the paste file was 81,317 characters with 108 `always_inline` attributes.

---

## 53. A full Stockfish-style NNUE, and whether data fixes it (24 September 2026)

The CJK14 repack left room for a larger network, so this round asked whether
a Stockfish-style NNUE can replace the whole evaluation (HCE, MiniNet and the
macro head) instead of correcting it. Nothing here shipped; the code is in
`tools/experiments/full_nnue/` with a README.

**Network.** 199 sparse features per perspective: each stone of a live
miniboard as mine or theirs (162), each decided miniboard as mine, theirs or
drawn (27), and the move constraint (10, shared). A shared-weight
accumulator of width 256 per perspective, clipped ReLU, concatenated 512 ->
32 -> 1. The corrected static eval in interior pruning and the qsearch
stand-pat both became NNUE + correction history; the HCE's incremental
threat maps stayed for move ordering and the global-win tactics. A float
reference implementation matched PyTorch to within one eval unit. The only
earlier attempt (section 8) had a 32-wide accumulator and depth-6 HCE labels
and lost about 277 Elo at depth 4.

**Data.** 1.59M self-play positions (30,000 games at 10 ms) and 300k random
positions, all labeled by the full current engine searching to depth 12
(about 1.7 ms per position on four threads; 20% of labels are mate scores).
Every row also carries the current static eval, so the trainer reports the
baseline on the same 50,000-position holdout.

**First result: better fit, worse play.**

| Evaluator | Holdout WDL-MSE | MAE | Corr | Equal depth 4 vs Prev |
| --- | ---: | ---: | ---: | ---: |
| Current static eval (HCE + MiniNet + macro) | 0.02865 | 998 | 0.621 | - |
| NNUE, sigmoid-MSE (K = 2000), 1.89M rows | **0.02435** | 933 | 0.635 | **-192.7 +/- 34.0** (N=510) |
| NNUE, Huber, mates dropped | 0.0447 | **792** | **0.689** | **-295.6 +/- 44.0** (N=444) |

No sign or perspective bug: the NNUE agrees in sign with the old eval on 88%
of positions where the old eval exceeds +/-3000. The better the net
predicts a deep search's verdict, the worse it orders leaves: its outputs
are compressed (standard deviation 1,759-1,983 against 2,738) and correlate
only 0.61-0.65 with the evaluator the search was tuned around.

**Data scaling.** The same net on nested subsets, one holdout, 400 games at
depth 4 per point (+/- ~40 Elo each):

| Training rows | Holdout WDL-MSE | MAE | Equal depth 4 |
| ---: | ---: | ---: | ---: |
| 125,000 | 0.0378 | 1032 | -303 +/- 49 |
| 250,000 | 0.0339 | 1030 | -344 +/- 49 |
| 500,000 | 0.0311 | 1006 | -214 +/- 42 |
| 1,000,000 | 0.0280 | 981 | -172 +/- 37 |
| 1,837,147 | 0.0243 | 931 | -205 +/- 39 |

Offline error falls steadily with data and has not plateaued. Play improves
roughly 30-50 Elo per doubling up to 1M rows and then stops within the
noise. Even at the optimistic slope, reaching parity would take five or six
more doublings (on the order of 100M labeled positions, about two days of
labeling per doubling at this machine's rate for the last ones).

**Better-matched data made it worse.** Root self-play positions are not what
the evaluator is asked about, so 1.52M positions were sampled at qsearch
entry during 3,000 self-play games at 20 ms (`leaf_dump`, 1 in 4,096 nodes)
and labeled the same way. Leaf positions are sharp: the mean depth-12 score
for the side to move is +4,771 (roots: +96) and 26% are mates. There the
current evaluator is better even offline, and the net trained only on them
is far worse in play:

| 1M training rows | Holdout WDL-MSE (leaf holdout) | Corr | Equal depth 4 |
| --- | ---: | ---: | ---: |
| Current static eval | 0.0504 | 0.638 | - |
| NNUE on leaf positions (lr 3e-4; lr 1e-3 collapsed to a constant) | 0.0541 | 0.536 | **-474 +/- 75** |

**Reading.** The handcrafted terms and the MiniNet's exact per-miniboard
pattern tables encode two-in-a-row geometry and threat structure directly.
A per-cell linear first layer has to rebuild that from 1-2M positions and
cannot yet, least of all in the tactical positions where qsearch evaluates.
The architecture that works here is a sharp base with learned corrections
on top, which is what shipped. A future attempt should start from that:
an NNUE residual on top of HCE (as the MiniNet is), pattern-level input
features rather than cells, or a WDL-blended target over far more games.
Anything that relabels positions must use a `test_bots` built from the
unpatched Dev, or the experimental net becomes its own teacher.

## 54. An opening book chosen by uttt.ai (24 September 2026)

Section 46's full-coverage book spent its characters on every reply, including
replies no competent engine plays, and on all 81 first moves when moving
second. This replaces it with a tree grown by uttt.ai, the AlphaZero-style
engine retrained for CodinGame rules in the
[uttt.ai fork](https://github.com/nathanWolo/utttai/tree/codingame-rules):

- both roots follow the first player's center-center (the bot plays it; moving
  second, the book assumes the opponent did);
- opponent replies are covered when uttt.ai's network prior is at least 0.03;
- lines are expanded best-first by estimated reach until a 13,000-character
  budget is spent;
- our moves are uttt.ai's after 3,200 simulations, with a crossfish veto (it
  replaced 3.3% of them).

Two measurements made the design. uttt.ai's policy predicts other engines'
replies far better than crossfish's own ranking did for the rejected selective
pilot: at prior 0.03 its reasonable set (6.7 of about 9 moves) held 99.4%,
98.8% and 98.4% of the replies of crossfish, its HCE bot and the Legend bot,
and 77% of the weak legacy Python bot's. And in 300 opening positions uttt.ai's
90 ms move beat crossfish's 75 to 24 where deep searches of both engines agreed
on which was better.

The format changes with it: the payload now also carries a "continues" digit
per book position and a "covered" digit per non-terminal reply at each
expanded opponent position (`documentation/play_book.md`). The book holds
34,066 of our positions (was 20,883) in 13,456 payload characters (was 4,656);
`cg_input.cpp` is 90,095 characters after merging #22-#26. `play_book_check` found 0 mismatches and
the protocol check, now exact against the text book, passed 40/40.

Old and new book, same engine, seeds and 90 ms (`play_book_match`):

| Match | Old book | uttt.ai book |
| --- | ---: | ---: |
| vs plain engine, 3,000 games | +18.7 ± 11.1 | +99.4 ± 11.2 |
| diverse opponent, paired book value (1,000 openings) | +16.3 | +62.8 |
| round-six engine, paired book value (1,000 openings) | +12.2 | +50.1 |

The round-six run is the transfer test the earlier selective books failed; the
new book stayed in book 5.3 / 6.7 moves per game (first / second) against it,
against 5 / 4 for the old one. The cost of the center-center assumption shows in
the diverse run, whose opponent opens elsewhere half the time: the second
player then has no book, averaging 3.4 book moves, and the book value still
quadrupled.

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

The reviewer reran mode 2 for both books on the same new seed (7), 500 paired
openings each against the round-six engine:

| Book | With book | Without | Book value |
| --- | ---: | ---: | ---: |
| Full coverage | +91.0 ± 28.2 | +64.7 ± 26.8 | +26.3 |
| **uttt.ai** | **+143.9 ± 29.1** | **+71.9 ± 27.2** | **+72.0** |

On identical openings the new book is worth about 2.7 times the old one, a
+46 difference; the "four times" above rests on the author's lower reading
for the old book. The review also confirmed that the committed payload is
reproduced byte for byte by `play_book_pack` from the uttt.ai fork's
`cg/book/uttt_book_v1.txt`, and that the merged paste file is 90,095
characters (9,905 left).

## 55. Retraining the evaluation on 4.8M positions: better fit, no Elo (25-26 September 2026)

The engine had gained a lot since the MiniNet was last trained, so this round
retrained the evaluation (MiniNet + macro head on top of the fixed HCE) on
fresh, much larger data, Stockfish style: a win-probability loss on searched
evals and game results. None of the resulting nets beats the shipped one at
real time controls. The round did produce a bug fix, a reusable dataset and a
test toolchain, and it settled several questions about how to test an eval.
Nothing in the engine or the CodinGame bot changed. The pipeline is in
[eval_data.md](eval_data.md).

**Data.** 4.8M positions, each with a depth-14 search score from the current
engine and uttt.ai net4's value, and (for the 93% played after the last
random move) its game's result under CodinGame rules:

- 3.5M from 69,626 crossfish self-play games at 40 ms (`datagen play`), 45%
  starting after 0-8 random plies, 35% from uttt.ai self-play openings at
  plies 4-24, 20% after 9-20 random plies; before ply 40, 8% of moves are drawn
  from the moves within 300 of the best at depth 4.
- 1.32M from the uttt.ai fork's 25,000 self-play games (generations 1-5).

**A loader bug that has corrupted labels for a long time.** `test_bots`'s
`load_utttai_state` read a free-move position with stones on the board (a
player sent to a decided miniboard) as a pass: it set `prev_move_was_pass`,
which only `pass()`/`unpass()` ever clear, so every move in the tree searched
from that position was also a free move. About 11% of rows (530,000 here) got
such labels, and so did every older dataset relabeled through this loader
(the `NNUEWDL1` `dump search` / `relabel` / `rank` paths). The fix records the
free move as a last move into a decided miniboard; a round-trip self-test in
`test_bots verify` (encode, reload, compare legal moves at the position and
after every reply) fails on the old code at the first such position. After
relabeling, 22% of the affected labels changed (the rest scored a mate or
exactly 0 both times, or had one live miniboard left, where the rule does not
matter), and the depth-14 labels predict results much better: the fitted
sigmoid scale K fell from 3,150 to 1,600 and the result log loss from 0.506
to 0.459. 61 self-play games that had started from a free-move opening, and
so were played entirely under the bug, were dropped.

**Candidates.** Win-probability MSE, 10 epochs, the shipped MiniNet and macro
decoded exactly as the starting point, 10% of games held out. Equal-depth
screens are depth 8 against the shipped eval; the first 4,000 games of each use
the same 2,000 openings.

| Net | Labels | What changed | Held-out loss vs shipped | Depth 8 |
| --- | --- | --- | ---: | ---: |
| A_search | buggy | search target only | -1.9% | +12.8 ± 9.2 (4k), +2.6 ± 5.2 (next 12k) |
| A_blend | buggy | 30% game result | -3.6%* | +2.0 ± 9.1 |
| A_blend_uttt | buggy | + 25% uttt.ai value | -5.4%* | +3.5 ± 9.2 |
| B_blend_hce | buggy | + HCE weights trained | -4.8%* | **-25.0 ± 9.1** |
| F_search | fixed | K 1,600 (fitted) | -4.4% | -1.6 ± 9.2, +0.3 ± 5.2 (12k) |
| F_full | fixed | all 19,683 pattern embeddings, re-clustered | -7.0% | **+17.8 ± 9.1, +8.3 ± 5.2 (12k)** |
| F_hce | fixed | HCE weights trained | -9.0% | -10.0 ± 9.2 |
| F_k3150 / F_k2400 | fixed | K fixed | -2.1% / -2.7% | -6.7 / +3.6 (± 9) |
| D_full | fixed | F_full on an HCE that treats drawn miniboards as blocking lines | -7.0% | +9.3 ± 4.5 (16k) |
| a2 | fixed | F_full with the MiniNet's duplicate hidden units re-initialised | -11.5% | +5.9 ± 4.6 (16k) |

\* against that run's own target, which includes results or uttt.ai values. Loss
percentages are only comparable between runs with the same labels and K
(3,150 for the buggy labels, 1,600 after the fix unless stated); a2's is the
float model before re-clustering (-10.9% packed).

Timed results (pentanomial SPRT, H0 0 / H1 5):

| Net | 90 ms | Games |
| --- | ---: | ---: |
| F_full | +1.9 ± 2.9, stopped as not a gainer (LLR -1.39) | 26,436 (desktop + Linux worker) |
| a2 | -3.0 ± 5.9, H0 accepted | 6,640 |
| A_search | +4.7 ± 6.6 when paused for the loader fix | 5,220 |

F_full also ran 4.8% (desktop) and 4.0% (laptop) fewer nodes per second at
90 ms, which eats into its equal-depth gain.

Round robins (`tools/round_robin.py`, ratings with shipped anchored at 0):

| Net | Depth 8 (42,000 games) | 20 ms (20,000 games) |
| --- | ---: | ---: |
| F_full | +9.7 ± 6.9 | -4.0 ± 7.2 |
| D_full | +8.8 ± 6.8 | -9.7 ± 7.2 |
| a2 | +6.9 ± 6.9 | -9.5 ± 7.3 |
| A_search | +6.4 ± 6.8 | +0.8 ± 7.2 |
| hcedraw (the drawn-miniboard HCE fix alone) | +2.9 ± 6.9 | - |
| F_search | +1.4 ± 6.9 | - |

**What the round showed about testing an eval.**

- *Depth-8 screens are the wrong proxy for eval changes.* Four of the five
  retrained nets in the depth-8 round robin gained 6-10 there (F_search
  +1.4), and none gained at 20 or 90 ms. Depth mode turns off futility-style
  pruning and correction-history updates and ignores speed; the 20 ms timed
  round robin ordered the nets like the 90 ms SPRTs did. Use a 20 ms round
  robin as the screen and 90 ms to confirm.
- *4,000-game screens on one opening block are optimistic.* The first block
  read 9-12 Elo higher than the next 12,000 games, on fresh openings, for
  three of the four candidates re-run.
- *Offline loss does not rank play.* Three times the run with the lower loss on
  the same target played worse: B_blend_hce against A_blend, F_hce against
  F_search, and a2 against F_full.
- *Search scores are the better teacher.* Blending in 40 ms self-play results
  or uttt.ai's value lowered the equal-depth result.
- *Training the HCE weights jointly hurts* (-25 and -10 at equal depth), as
  round ten's SPSA suggested: the shipped HCE weights are close to optimal for
  play even though a better fit is available.
- *The pruning margins are not what holds the retrained nets back.* The
  retrained nets' learned correction is larger (standard deviation
  1,240-1,390 for the round-robin nets against 1,076 for the shipped one on
  20,000 sample positions), so the qsearch fail-high shortcut
  (`QHCE_FAIL_HIGH_MARGIN`, 640) is wrong more often for them. Widening it to
  1,344 did not help (20 ms, 9,000 games per engine): F_full +0.4, shipped
  with 1,344 -1.1, F_full with 1,344 -6.3 (all ± 6.6).

**Architecture study** (offline probes on the same data and split, with rough
cost and character-budget estimates):

- A typical NNUE (width 128-256) or a transformer at the leaves does not fit
  the 9,905 free characters (the probes' pattern NNUE has 22.7M new
  parameters, the transformer 0.64M); a rough speed estimate, not measured in
  play, put their cost at 100-230 Elo.
- With a 10x higher learning rate (3e-3), the same D16/H8 net reaches -17.4%
  held-out loss after packing (not yet played); more MiniNet capacity then
  gains at most one more point (D32/H32: -19.1% against -18.1%, both float)
  and starts to overfit within 3-4 epochs.
- The shipped MiniNet's eight hidden units are four near-identical copies
  each of two units; un-collapsing them (a2) did not help in play.
- Storing the weights as fp16 would free roughly 7,500 characters (an
  estimate from the payload sizes; not built), the most promising use of
  which is a bigger gameplay book.

**Other.** nelhage/ultimattt's minimax player lost 18 / 25 / 957 to the
crossfish CodinGame bot at 90 ms (1,000 games, +601 ± 56 for crossfish under
CodinGame rules) while using about 50% more time than its budget.

**Tools added** (all described in [eval_data.md](eval_data.md)):
`cpp_impl/datagen.cpp` (self-play, labeling, HCE features);
`tools/eval_data.py`, `tools/eval_pipeline.py` (resumable data pipeline);
`tools/nnue_train_blend.py` (win-probability trainer);
`tools/eval_candidate.py` (headers, isolated A/B builds, exactness checks,
matches); `tools/sprt_merge.py` and `tools/sprt_worker.py` (SPRT shards on a
second, Linux machine, pooled with test_bots's own LLR);
`tools/round_robin.py` (round-robin ratings); `tools/vs_ultimattt.py`;
`tools/experiments/capacity/` (the probes).

## 56. A pattern-generator NNUE replaces the whole evaluation (26-27 September 2026)

Sections 53 and 55 left two answers: a Stockfish-style per-cell NNUE lost
193-296 Elo at equal depth, and retraining the MiniNet on 4.8M positions fitted
better without playing better. This round went back to a net that replaces
HCE + MiniNet + macro, with three changes: far more self-labelled data, a
pattern-level first layer, and integer incremental inference. The shipped net,
**B64_d5M_57ep** (35,243 parameters), passed the official 90 ms SPRT at
**+276.6 +/- 30.4** against the mate-window freeze and is now the evaluation
of Dev, Prev and the CodinGame bot. The trainers are in
[`tools/experiments/nnue2/`](../tools/experiments/nnue2/README.md) and the
engine-side tools (the experiments' candidate builds, checks and two-net
matches) in [`tools/experiments/fast_nnue/`](../tools/experiments/fast_nnue/README.md);
[nnue_training_and_implementation.md](nnue_training_and_implementation.md)
describes the net, its training and its runtime.

### Data and recipe

- **Self-labelled depth-8 data.** `datagen play OUT N d8 ...` (a new depth mode,
  [eval_data.md](eval_data.md) section 2) plays every move with a fixed
  depth-8 search and records its root score as the label: 176M positions in
  about 2.5 hours on the desktop and the laptop together (`d8_*.cfdg`).
- **Training set "d5M".** Section 55's eval2 training rows (depth-14 labels,
  4.34M) plus the first 5M records of one depth-8 file: 9.34M rows.
- **Holdouts.** V2, 10% of the eval2 games (482,136 rows), and D8H, 1% of the
  depth-8 games (1.71M rows, never trained on). Losses below are the change
  against the old static eval (HCE + MiniNet + macro) on the same rows.
- **Recipe.** Win-probability MSE against sigmoid(search / 1600), the 8
  board symmetries as augmentation, AdamW at 1e-2 with a 2% warmup and a
  cosine decay to 1e-5, batches of 16,384.

The recipe alone turned section 53's per-cell design (199 features, 256-wide
accumulator) from a loser into a winner at fixed depth: -140 at depth 8 with
the old recipe, -12 with section 55's fixed labels, symmetries and a longer
schedule, +61 with the depth-8 rows, and +231 (+287 with pruning on) at
learning rate 6e-3, at -39.6% held-out loss. At 20 ms the float
implementation then lost **-255 +/- 38** (444 games): it recomputed the whole
net per evaluation and searched 0.8% of the old engine's nodes per second.

### Making it fast (stages 1-3)

- **Integer, incremental, lazy.** int16 accumulators per absolute player, so a
  move never swaps them; `make` only records the move and an evaluation
  replays the recorded moves from the nearest ancestor entry; unmake does
  nothing. The first dense layer runs over nonzero activation pairs with
  `_mm256_madd_epi16`; a 2^14-entry eval cache keyed by the transposition key
  saves a third to a half of all evaluations. Every scale is a power of two
  chosen so that no int16 accumulator and no int32 sum can overflow.
- **Result on the per-cell net:** 49% of the old engine's nodes per second (64
  times the float hook), 0 mismatches in 506.9M checked evaluations, and
  **+182.6 +/- 19.1** at 20 ms (1,000 games).
- **The pattern generator (B nets).** Each live miniboard contributes one row
  per perspective, `T[m][pattern]` over its 3^9 patterns, and those rows are
  generated by a small shared encoder (one-hot 27 -> 64 -> 64 -> 32) and one
  projection per location to 64 lanes plus a PSQT lane. Decided boards,
  the move constraint and the forced board's pattern add their own rows; the
  head is 128 -> 16 -> 32 -> 1. The engine bakes the tables, so a move costs
  one row out and one row in per perspective (6.6 ns) and an evaluation 43 ns.
  B-64 reached -47.4% on V2 against -39.6% for the per-cell net with half its
  parameters; direct pattern tables (22.7M parameters) stopped at -28%.

| Net, against the old eval | Nodes/s vs old | Depth-prune 8 (2,000 games) | 20 ms (1,000 games) |
| --- | ---: | ---: | ---: |
| per-cell r10_aug_d8x5M_lr6e3 | 42% | +274.2 +/- 17.9 | +182.6 +/- 19.1 |
| B64_lr1e2 (10 epochs) | 57% | +365.0 +/- 21.7 | +276.1 +/- 21.8 |
| B128_lr1e2 | 50% | +392.6 +/- 23.2 | +288.1 +/- 22.9 |

Nodes per second are the in-game `walk 16 40 20` ratios after stage 3; the
match columns come from stages 1-3. A 24,000-game 20 ms round robin rated them
+190.5, +279.5 and +289.2 against the old eval (+/- 7), and B128_lr1e2 passed
a 90 ms SPRT against the old engine at 420 games, **+252.7 +/- 30.6** (a
review rerun: +301 +/- 36).

Three reviews found latent defects, none of them in a game that was played:

- **Stale accumulators.** A search entered without `refresh_root` could be
  evaluated from another position's entry (200 of 200 in a repro). Every
  stack entry is now keyed by the position it holds, and a broken chain
  refreshes from scratch.
- **The empty board's key is 0.** "No position" needed a sentinel (`kNoKey`):
  with 0, a fresh stack took its unset first entry for the empty board's
  accumulator (every evaluation wrong in four repro scenarios, 0 after).
- **Two nets in one build.** `#pragma once` gave both engines of a two-net
  `test_bots` the first-included net, silently. Pairings now rename Prev's
  copy completely, a guard makes any other two-candidate build a compile
  error, and every log line names each side's net file and CRC-32. Each
  side's statics and depth-7 searches equalled its own net's single-engine
  labels on 1,000 positions in 10 of 10 pairings.

### More data did not help; more steps did

| Net | Data, schedule | V2 | D8H | 20 ms vs B64_d5M_57ep |
| --- | --- | ---: | ---: | ---: |
| B64_lr1e2 (first CodinGame net) | d5M, 10 epochs | -47.43% | -42.85% | -29.7 +/- 5.0 |
| **B64_d5M_57ep (shipped)** | d5M, 57 epochs | **-51.41%** | **-47.28%** | 0 |
| B128_d5M_57ep | d5M, 57 epochs | -51.77% | -47.44% | -6.3 +/- 5.0 |
| B64_all | all 176M depth-8 rows, 3 passes | -47.83% | -46.75% | -44.0 +/- 11.1 (head to head) |
| B64_all_e2x10 | all rows, eval2 x10 | -49.66% | -47.29% | -19.8 +/- 10.5 (head to head) |
| B64_d5M_114ep / 200ep | d5M, 114 / 200 epochs | -50.22% / -48.76% | -45.94% / -44.56% | -9.9 / -20.9 (+/- 5.0) |
| B64_d5M_114ep_lr5e3 | d5M, 114 epochs, lr 5e-3 | -51.33% | -46.97% | -1.7 +/- 5.0 |
| B64_e2only_114st | eval2 only, equal steps | -49.16% | -40.66% | -18.8 +/- 5.1 |

Ratings come from two 20 ms round robins of nine engines each
(`nnue_full_20ms` and `nnue_long_20ms`, 72,000 games apiece, played on the
desktop and on the Linux laptop through `fast_worker.py`, 0 timeouts), in
which B64_d5M_57ep was first both times (+318.1 and +300.5 +/- 7 against the
old eval).

- **Steps, not data, limited the 10-epoch nets.** At equal steps, streaming
  all 176M depth-8 rows lost to training 57 epochs on the 9.34M rows, for
  B-64 by 40 Elo; ten passes changed nothing.
- **57 epochs at lr 1e-2 is the best point.** Longer high-lr training damages
  the net (fewer lanes in the linear range, larger first-layer rows, worse
  loss on its own training rows), which is not overfitting. At lr 5e-3 a
  114-epoch run ties it (+9.0 +/- 12.0 at 90 ms over 1,200 games) but is 11%
  slower per node.
- **The depth-8 rows matter.** With the same steps on eval2 alone the net
  memorizes eval2 and loses 5 points of D8H and 19 Elo.
- **B-128 does not pay at these time controls:** +0.4 points of V2, about 15%
  slower.
- **Quantization costs nothing measurable:** the quantized engine eval's
  holdout loss equals the float net's within 3e-6 for every net.

B64_d5M_57ep then passed a 90 ms SPRT against B64_lr1e2 at 864 games:
**+36.3 +/- 14.3**, 0 timeouts.

### CodinGame

- **The payload is the generator, not the tables.** The baked tables are 25.6
  MB of int16; the generator is 35,243 parameters. Each matrix row gets a bf16
  scale and its own bit width (the PSQT lane separately: an error there moves
  the eval 500 times as far), GPTQ rounding over the live patterns weighted
  by frequency, a least-squares refit of the projections and the dense head,
  and Rice coding: 54,114 bytes, **30,923 CJK14 characters**
  ([minification.md](minification.md) section 3.1).
- **What the rounding may cost.** The integer engine is itself 4.73 mean / 164
  max eval units from the float net (this net quantizes at 2^9). The chosen
  bits keep the bot at 4.72 / 163, and the payload's own float error at 0.91.
  Refitting the encoder hurt: it moved rare patterns' embeddings (max error
  116 against 11 at 16 bits everywhere), so the encoder is only GPTQ-rounded.
- **The bot bakes the tables at start-up**, about 50 ms: the 11,093 patterns a
  live board can hold, in a fixed float operation order and without FMA, so
  every build bakes the same floats. All 16 integer tables equal the local
  loader's, with clang and with g++ 11 at -O3 and with CodinGame's flags.
- **A trial first.** The same port with B64_lr1e2 (90,515 characters) was
  uploaded to CodinGame before this round finished. Built with CodinGame's
  flags, it beat the shipped MiniNet bot **+212 +/- 49** at 90 ms (200 games,
  0 timeouts). The B64_d5M_57ep build scored **+30.3 +/- 14.9** against that
  trial net at 20 ms (1,000 games; a review rerun +37.8 +/- 17.8 over 600).

### Integration

- **One runtime, three users.** `cpp_impl/nnue_b64.hpp` (the runtime) and
  `cpp_impl/nnue_b64_net.hpp` (the generated payload) are shared by Dev, Prev
  and `codingame_nnue.cpp`, so the repository needs no net file to build or
  test. `tools/nnue_emit_b64_header.py` regenerates the payload byte for byte
  from the lane-paired export; `--check` decodes any header, bakes it in
  numpy exactly as `load()` does and prints the table hashes.
- **Search wiring.** Qsearch stands pat on NNUE + structural correction (the
  HCE fail-high shortcut is gone); the interior static eval is the NNUE with
  all three corrections; the depth-1 reverse-futility MiniNet prefilter is
  gone. `make` keeps only the HCE's threat maps, which the global-win checks
  read; the macro net stays as the macro correction history's prior. The HCE,
  MiniNet and macro code still compiles for datagen's HCE labels and
  test_bots' tuning tools; the CodinGame bot drops them.
- **Tests.** Three unit tests pin the 16 table hashes and the scales, 16 fixed
  positions' evals (from the verified bot's own runtime, through three
  paths), and incremental == from scratch == a scalar reference over about
  24,000 search-like evaluations including the empty-board case. Setting the
  sentinel to 0, skipping the decided-board update or corrupting one
  perspective each fails them.
- **The CodinGame gate.** The speed check would fail an eval that is slower
  per node by design, so the pull request declares the change in
  `tools/cg_gate/eval_change.json`: the reason, the base paste file's hash
  and the expected nodes/ms range, [0.45, 0.70]. It applies only against that
  base and fails outside the range at either end. On the laptop (g++ 11.4)
  the gate passed at 0.553 [0.539, 0.568], with inlining 98%, p99 reply 90.4
  ms, first reply at most 191 ms and the smoke match +244 +/- 50. Delete the
  file once this has merged.

### Result

Official 90 ms SPRT against the mate-window freeze (Prev as of `c278cde`),
pentanomial pairs, two shards pooled with `tools/sprt_merge.py` (desktop,
clang, 6 threads, openings from 0; laptop, g++ 11.4, 4 threads, openings from
25000), stopped when the pooled LLR crossed:

```text
N 420  W 296 / D 106 / L 18
Penta 0 / 1 / 29 / 81 / 99
+276.63 +/- 30.44 Elo
LLR +3.000 (H0=0, H1=+5) — PASS
Timeouts: Prev 0 / Dev 0; slowest reply Prev 90.07 ms / Dev 90.10 ms
Prev NPS 31,197,568  Dev NPS 15,953,280 (desktop header)
```

The shards read +287.1 +/- 40.8 (desktop, 252 games) and +261.6 +/- 46.2
(laptop, 168 games). With no double losses the pentanomial LLR grows by at
most 0.0143 per pair here, so 420 games is the fewest any pass could take;
the Elo carries the information, not N.

An early-stopped SPRT overstates the effect, so the same binaries then played
a 3,000-game fixed-length match (no early stop) on openings disjoint from the
SPRT's (desktop 1,800 games from opening 30000, laptop 1,200 from 40000):

```text
N 3000  W 2087 / D 766 / L 147
Penta 3 / 31 / 195 / 565 / 706
+267.37 +/- 11.86 Elo
Timeouts: Prev 0 / Dev 0; slowest reply Prev 90.48 ms / Dev 91.14 ms
```

The shards agree: +274.1 +/- 15.3 (desktop) and +257.6 +/- 18.8 (laptop).

| Speed (`bench_ab`, same binaries) | Desktop, clang | Laptop, g++ 11.4 |
| --- | ---: | ---: |
| `nodes 400 9`: Dev nodes/s vs Prev | 61.5% | 58.1% |
| `nodes 400 9`: nodes to depth 9 vs Prev | 86.9% | 87.4% |
| `walk 10 40 90`: mean completed depth | 17.37 -> 16.37 | 17.54 -> 16.39 |
| CodinGame gate: nodes/ms, CodinGame's flags | | 0.553 |

The net costs about a ply at 90 ms and is worth nearly 300 Elo. The paste file
is **94,897 characters, 5,103 under the cap** (the MiniNet payload's 24,489
characters gave way to the generator's 30,923, and the HCE and MiniNet code
left the bot); built with CodinGame's flags it runs at about 100% of its -O3
speed.

**Freeze.** `crossfish_prev.hpp` is this Dev. One change came with it: Dev
baked the tables under a once-flag that belonged to its class, so a frozen
copy would have had a second flag, and the first Dev and Prev engines built
at the same moment on two threads would have run the bake twice (the pattern
index table came out doubled in 50 of 50 runs of a repro). The flag is now one
inline function shared by both engines; nothing else changed.
`bench_ab equiv` is IDENTICAL at depths 8 and 11 (the SPRT's Dev binary
searches the same trees), and a depth-7 `test_bots` screen of the pair read
-4.9 +/- 16.7 over 1,000 games. `test_bots`'s SPRT header now says
`eval=NNUE-B64`.

**Not settled here.** The file has not been run on CodinGame itself; the
laptop checks stand in for it (the B64_lr1e2 trial is the one that ran
there). The MiniNet candidate tools of section 55 (`eval_candidate.py`
`emit` / `verify`) no longer change or describe the search's eval, and the
fast-NNUE candidate builds patch the pre-NNUE engine (`c278cde`; the
`fast_nnue` README says how to get it).

## 57. Round twelve: the NNUE labels its own data (27-28 September 2026)

Section 56's net learned from labels that the old evaluator's search wrote:
eval2's depth-14 labels and the d8 self-play of the pre-NNUE engine. Round
twelve closed the loop: the shipped NNUE engine relabelled eval2 and played
its own games, and a B-64 trained on those data, **r12_M2**, is the new net.
Same architecture, same recipe, same runtime; only `nnue_b64_net.hpp`'s
payload and its five integer scales change. It passed the official 90 ms SPRT against B64_d5M_57ep at **+70.6 +/- 19.7** (N=504), and **+52.5 +/- 7.0** over 3,000 games of the two CodinGame paste files against each other. On
CodinGame the paste file reached **rank 1** of the Ultimate Tic-Tac-Toe
ladder (score 34.27 after placement, ahead of Daporan's 33.88).

### Data

- **eval2, relabelled.** All 4,824,319 eval2 positions, labelled by a
  depth-14 search of the shipped engine (`datagen label`, desktop 3.90M rows
  and laptop 0.92M before it dropped off the network). Every other byte of
  every record is unchanged, V2 is the same 482,136 rows, and the labels
  predict game results better: V2 log loss 0.4377 against 0.4593 for the old
  labels. About half of the new-vs-old disagreement is the search's own
  run-to-run noise (`datagen label` keeps each thread's table across
  positions). In the 20 largest disagreements a depth-18 search sided with
  the new label 14 times and the old 5 times.
- **sp13, NNUE self-play.** 13,442,027 rows from 266,302 games at a fixed
  depth of 13 (`datagen play ... d13`), desktop only: depth 13 is the deepest
  that still cleared 10M rows once the laptop was gone. Same openings file
  and randomisation as d8. A depth-12 / 13 / 14 comparison on 6,000 positions
  found no side-to-move bias from the odd depth (-3.9 +/- 8.1).
- **SPH**, 3% of the sp13 games (402,405 rows), is a second holdout no run
  trains on.

### Training

Five mixes, each B-64 with section 56's 57-epoch recipe (32,490 steps). The
trainer, `gen_r12.py`, imports gen_nnue.py's model, loss, augmentation and
optimizer unchanged; it and the round's data tools are in
`datasets/nnue2/r12/tools/` (not in the repository). Loss is against
B64_d5M_57ep on the same rows:

| Net | Data | V2 new labels | SPH | V2 old labels |
| --- | --- | ---: | ---: | ---: |
| M0 (control) | B64_d5M_57ep's own data and recipe | +1.7% | +1.7% | +1.8% |
| M1 | relabelled eval2 + 5M old d8 rows | -5.7% | -4.1% | +1.2% |
| **M2** | **relabelled eval2 + 5M sp13 rows** | **-9.6%** | **-8.5%** | +3.4% |
| M3 | relabelled eval2 + all 13.04M sp13 rows (31 epochs) | -8.1% | -8.9% | +4.5% |
| M4 | eval2 twice + all sp13 (25 epochs) | -8.9% | -8.4% | +3.9% |

- **The control is the lesson.** M0 changed nothing but the rounding of
  the float32 targets (one bit), and finished 1.7% worse on every holdout: the
  recipe is chaotic, and B64_d5M_57ep was probably a good draw. Seed-2
  replicates of M2 and M3 moved by 0.1-0.7%.
- **The old-label cost** sits almost entirely in the opening (plies 0-19),
  where the two engines' labels disagree systematically.
- **Twice the steps** (64,980) did not help: the training loss kept falling,
  the held-out loss did not (M2 -9.44% vs -9.57%, M3 -7.40% vs -8.10%).

### Play

20 ms round robin `r12_20ms` (fast_pair two-net pairings on the pre-NNUE
search, 36 pairs x 2,000 games, desktop; fit chi2 19.1 on 28 dof), Elo against
B64_d5M_57ep:

| Net | Elo | | Net | Elo |
| --- | ---: | --- | --- | ---: |
| **r12_M2** | **+65.4** | | r12_M4 | +61.0 |
| r12_M3_s2 | +62.8 | | r12_M1 | +39.9 |
| r12_M2_s2 | +62.5 | | r12_M0 | -6.0 |
| r12_M3 | +61.8 | | B64_lr1e2 | -32.0 |

All +/-4.8. A follow-up round robin (`r12x_20ms`, 20,000 games) put r12_M2
at 64,980 steps level with r12_M2 (+1.6 +/- 9.1 head to head) and the
doubled M3 11-16 Elo below M3.

3,000-game fixed-length matches at 90 ms against B64_d5M_57ep (the same
pairings): **r12_M2 +57.7 +/- 7.6** (W 925 / D 1644 / L 431, 0 timeouts),
r12_M3_s2 +51.7 +/- 7.9.

The round-robin candidates are unquantized exports in the pre-NNUE search,
not exactly what ships, so the shipped code itself was then measured:

- **Official 90 ms SPRT**, `test_bots` on the ship branch: Dev = r12_M2, Prev =
  B64_d5M_57ep. Dev and Prev share `nnue_b64.hpp`'s namespace and payload
  globals, so Prev ran on scratch copies with renamed symbols (namespace
  `b64prev`, `B64P_*`, its own load-once). Before the SPRT, `engine_selfcheck`
  confirmed each side: Prev's search matched main's engine, and Dev's the ship
  branch's, IDENTICAL at depths 5, 7 and 9. The run used the desktop, 7
  threads, openings from 20000:

```text
90 ms: N 504  W 173 / D 259 / L 72
Penta 2 / 37 / 95 / 94 / 24
Elo diff: +70.58 +/- 19.66
LLR: +3.026 (H0=0, H1=+5) - PASS
Timeouts: Prev 0 / Dev 0; slowest reply Prev 97.36 ms / Dev 97.15 ms
Prev NPS 14,626,304  Dev NPS 14,418,304
```

- **The two CodinGame paste files**, r12_M2's against PR #30's: 3,000 games
  (1,500 opening pairs, random 4-8 ply openings, each opening with both
  colours), 90 ms per move, CodinGame's protocol and 100 ms forfeit, roundrobin
  match mode, clang -O3 builds, desktop, 6 workers:

```text
N 3000  W 1203 / D 1044 / L 753
Penta 18 / 155 / 768 / 477 / 82
+52.5 +/- 7.0 Elo
Forfeits (replies over 100 ms): r12_M2 8, PR #30 15
```

  Without the 19 opening pairs that had a forfeit, the result is +51.5 +/- 6.9.
  Round twelve's review had measured +57.1 +/- 11.6 over 1,000 games of the
  same files.

### The CodinGame file and the review

- **The file.** `cg_input.cpp` is PR #30's with one line changed: the
  payload and the scales 9, 12, 13, 13, 11. It is **94,922 characters**, 25
  more than before, with 5,078 left.
  - It was built with the repository's pipeline:
    `nnue_emit_b64_header.py datasets/nnue2/fast/r12_M2_perm.bin --label r12_M2`
    (export CRC-32 `60f1f9ea`, payload sha256 `4ba93b1a...`), then
    `make -C cpp_impl cg-input`.
  - Port check: IDENTICAL at depths 5, 7 and 9.
  - The dequantized generator is 1.08 mean / 45 max eval units from the
    float net.
- **Review.** An adversarial review re-derived every number above from the
  raw files:
  - data integrity, and no leakage between training data and holdouts;
  - 200 labels recomputed;
  - holdout losses from the checkpoints;
  - the match pooling;
  - 200 CodinGame-protocol games against PR #30's file, with 0 faults.
  
  It found no defect that changes a result. Two of its notes:
  - about 9% of V2's positions also occur in M2's self-play rows, so the V2
    gains are slightly flattering;
  - `datasets/nnue2/cg/build_cg.sh`, the experiments' CodinGame builder,
    patches the pre-NNUE engine and must not be used for this engine.
- **On CodinGame.** Submitted on 2026-09-27, the file finished placement at
  rank 1 (34.27). Over its first 334 ladder games it went 203 / 19 / 112:
  - 160 / 5 / 2 as first player;
  - 43 / 14 / 110 as second. Every top bot shares that asymmetry: among the
    top six, the first player scores 83%.

**Freeze.** Dev and Prev share `nnue_b64_net.hpp`, so both now carry r12_M2;
nothing else changed. An early-stopped SPRT overstates the effect, which is why
the paste files also played 3,000 fixed-length games: they put the gain at
about +52, in line with round twelve's +57.7. `unit_tests.cpp` and `tools/test_nnue_emit_b64_header.py`
pin the new payload, scales and table hashes, and the new fixed-position
evals. `tools/cg_gate/eval_change.json` (section 56's declaration against the
pre-NNUE paste file) is removed as its base no longer applies.

## 58. Speed round thirteen, and margins the NNUE made too loose (29 September 2026)

Speed hill-climb on the r12_M2 freeze (section 57), on a 4-core cloud VM:
SPRTs at 3 threads with the fourth core idle, screens against the speed bundle
alone. Speed candidates had to compute the bit-identical tree (`bench_ab equiv`
IDENTICAL at depths 7, 8 and 11) and were timed with `tools/speed_ab.py`
(paired, pinned, 95% CI). Search candidates were screened for 2,000 games at
90 ms, then stacked for the official SPRT.

### Accepted: three tree-identical speedups (+4.5%)

- **Drop the dead HCE and MiniNet upkeep.** Since the NNUE replaced HCE +
  MiniNet, nothing on the search path reads the HCE scores, MiniNet codes,
  `hce_mb_flags` or the MiniNet super-accumulator, yet make still updated them
  and unmake restored them. The undo record now holds only the hash, the threat
  maps, the active board, the terminal state and the decided state.
  `speed_ab` +3.07% [+1.67, +4.51]. (The CodinGame file never had this upkeep.)
- **Forward pass** (`nnue_b64.hpp` `eval_avx`). All 64 activation pairs land
  in one mask word (A = 64), so layer 1 is one ctz loop over the nonzero pairs
  (about 28 of 64), split over two accumulator chains. Layer 2 broadcasts its
  input pairs straight from the packed register with `permutevar8x32` instead
  of a store and reload. Bit-identical on 200,000 captured search evaluations;
  an interleaved kernel bench put it at -13% per evaluation; +1.78% on top of
  the first.
- **8-byte eval cache.** 32-bit tag plus eval instead of a 64-bit key, so the
  same 256 KiB holds 2^15 entries instead of 2^14 (a false hit needs 47 key
  bits to agree). +1.03% on `sat`, about +2% on `walk`.

The three together: `speed_ab` +4.54% [+2.66, +6.49] (n=40), `walk 10 40 90`
+3.5 / +9.5 / +5.6% nodes. Built with CodinGame's flags, the kernel and
cache alone (same tree, same checksum) run a depth-13 `cg_selfcheck` in 0.387 s
against 0.434 s.

An official SPRT of the speed bundle alone was cut off at N=732 by a container
restart (-6.2 +/- 14.1, LLR -0.84: undecided). A 4% speed gain is worth about
the H1 bar, too close to pass cheaply, so the round looked for Elo as well.

### Rejected speed candidates

- **Four-way eval cache** (2^13 buckets of four 8-byte entries): +0.87%
  [-1.11, +2.93].
- **Static eval stored in the TT entry** (32-bit check key, the freed bytes
  hold the raw eval; search and `search_leaf` reuse it on a hit): +16M
  simulated instructions and `speed_ab` -1.43% [-2.73, -0.11]. The eval cache
  already catches almost every repeat, so there are no evaluations left to save.
- **Lazy static eval:** every static eval the search computes is read (RFP,
  futility, or the correction-history update), so there is none to skip.

### The margins: measuring what they cost

RFP (50 pawns x depth), futility (80) and the qsearch delta (350) date from
the HCE era; nobody retuned them when the NNUE (+276) replaced it. An
instrumented copy disabled RFP and futility and searched 30 self-play games
(random first 6 plies, then the engine's own depth-9 moves with 1-in-8 random),
logging every non-PV node (6.5M) and every quiet move that futility would
consider (13.4M).

Pooled, the error barely moves with the margin: at depth 1 an RFP margin of 20
pawns fires on 57% of nodes with 2.2% wrong (the full search fails low), 50
pawns on 41% with 1.7% wrong. Split by game phase, the error falls steeply
with the margin once the phase is fixed; the pooled curve is flat only
because wider margins shift the mix toward late nodes:

| Nodes | RFP 20: fire / wrong | RFP 50 | futility 20: pruned / raise alpha | futility 80 |
| --- | --- | --- | --- | --- |
| forced, moves 16-28, depth 1 | 61% / 1.2% | 43% / 0.2% | 63% / 0.18% | 25% / 0.02% |
| forced, moves 40+, depth 1 | 59% / 6.7% | 55% / 5.2% | 77% / 1.94% | 66% / 1.32% |
| forced, moves 40+, depth 3 | 48% / 10.8% | 38% / 8.2% | 67% / 1.90% | 39% / 1.30% |

Before move 28 the static eval almost never misjudges a node by a margin's
worth; from move 40 on it misjudges 3-10% of pruned nodes at any margin.
Free-move nodes need wider RFP margins than forced ones (at a 25-50 pawn gap,
27% wrong against 3%).

| Candidate (screen vs the speed bundle, 90 ms) | Result |
| --- | --- |
| RFP 25 everywhere | N=2000, +2.4 +/- 8.7 |
| Futility 30 everywhere | stopped at N=144, -19 +/- 27 (the phase data argues against it) |
| **RFP 25 / futility 20 before move 28, 50 / 80 after** | N=2000, +4.5 +/- 8.6 |

### Late-game HCE (queued idea, offline only)

Switching from the NNUE to the from-scratch HCE (cheap with few live squares)
once a node has at most K empty squares on undecided miniboards. At 90 ms the
current search already proves the result of every position with at most 30
such squares (136 of 136, depth 46). It proves 59% at 30-40 and 13% at 40-50,
and K = 20, 30 or 40 does not raise either figure despite about 30% more
nodes. On 141 positions (30-45 squares) that a 3-second search solves,
neither the bundle nor K = 30 played a single move that loses the proven
result. Not game-tested; any gain would have to come from the unsolved
positions.

### Result

Official 90 ms SPRT, Dev = the speed bundle + early margins, Prev = the r12_M2
freeze, 3 threads on the cloud VM:

```text
N 3684  W 768 / D 2243 / L 673
Penta 41 / 377 / 907 / 480 / 37
+8.96 +/- 6.31 Elo
LLR +3.09 (H0=0, H1=+5) — PASS
Timeouts: Prev 35 / Dev 33 (host noise: max replies 146 / 218 ms on both sides)
```

**Freeze.** The forward pass and the 8-byte cache moved from a Dev-local copy
into `nnue_b64.hpp` (Dev against the SPRT-passed commit: IDENTICAL at depths
7, 8 and 11), so Dev, Prev and the CodinGame file share them. Prev is Dev
renamed. `codingame_nnue.cpp` carries the early margins; `make port-check`
IDENTICAL at depths 5, 7 and 9; `make cg-speed` checksums match at -O3 and
with CodinGame's flags; `cg_input.cpp` is 95,285 characters (4,715 left).

## 59. Futility margins from what the node looks like (29-30 September 2026)

Section 58 set the pruning margins by move number. This round asked which
node features actually predict a wrong prune, and set margins from the answer.
It ran on the same 4-core cloud VM (3 SPRT threads, one core idle), with Prev
the round-thirteen freeze.

### The survey

An instrumented Dev disabled RFP and futility and searched 60 self-play games
(random first 6 plies, then its own depth-9 moves with 1-in-8 random). It
logged every non-PV node (12.8M) and every quiet move futility would consider
(26.3M), each with its features:

- move number;
- empty squares on undecided miniboards ("open squares");
- live miniboards;
- free move;
- each side's global threats: two held miniboards of a line whose third is
  still live.

For RFP (fired at 30 pawns, depth <= 4, 2.4% wrong overall), two features
separate the wrong prunes far better than the move number:

| Feature | Wrong prunes |
| --- | --- |
| Open squares 70+ / 50-59 / 30-39 / 10-19 | 0.04% / 0.60% / 2.55% / 9.3% |
| Opponent global threats 0 / 1 / 2 / 3 | 1.3% / 8.3% / 14.9% / 18.7% |
| Own global threats 0 / 1+ | 2.2% / 3.6-4.3% |

Within a (threat, open squares) cell the error falls cleanly with the margin.
In section 58 it looked flat only because pooling mixed the cells. For RFP
with no opponent threat:

| Open squares | 15 | 25 | 40 | 60 | 100 |
| --- | --- | --- | --- | --- | --- |
| 60+ | 0.58% | 0.21% | 0.06% | 0.01% | 0.00% |
| 45-59 | 1.98% | 1.01% | 0.41% | 0.14% | 0.02% |
| 30-44 | 3.11% | 2.18% | 1.36% | 0.75% | 0.30% |
| <30 | 5.07% | 4.34% | 3.52% | 2.73% | 1.77% |

With an opponent threat the rates run about 4x higher, up to 11-12% with fewer
than 30 open squares.

For futility the risk factor is the side to move's own threat: a quiet move
can set up the miniboard that wins the game. With an own threat, 2.2-3.1% of
quiet moves pruned at a 10-pawn margin raise alpha, and still 0.3-2.2% at 80.
Without one:

| Open squares | 10 | 20 | 40 | 80 |
| --- | --- | --- | --- | --- |
| 60+ | 0.12% | 0.03% | 0.00% | 0.00% |
| 45-59 | 0.29% | 0.13% | 0.04% | 0.00% |
| 30-44 | 0.58% | 0.42% | 0.23% | 0.08% |
| <30 | 0.88% | 0.76% | 0.57% | 0.36% |

### Candidates

- **R4: RFP margin by class**, set so that about 0.5% of prunes are wrong. The
  margin is 100 with an opponent threat; otherwise 20 / 35 / 60 / 100 at 60+ /
  45-59 / 30-44 / <30 open squares. This tightens early and widens late and
  around threats. Stopped at N=918: -6.8 +/- 12.0, LLR -1.25. Widening costs
  nodes on a third of the tree, the late game is already solved at 90 ms
  (section 58), and the search tolerates the occasional wrong prune. Equal
  error rates are not the Elo optimum.
- R5, the futility analogue of R4 (it also widened), was cancelled unrun for
  the same reason.
- **R6: futility margins only tighten.** The side-to-move-threat case keeps
  the section-58 rule (20 before move 28, 80 after). Otherwise the margin is
  15 / 20 / 50 at 60+ / 45-59 / 30-44 open squares, and never above that rule.
  The effect is mostly on nodes from move 28 on, where futility still used 80.

### Result

R6's 2,000-game screen (openings from 40000) read +9.9 +/- 8.3 with LLR +2.09.
The same binary continued the sequential test from those counts
(`SPRT_RESUME_*`, openings from 41000) until it crossed the bound:

```text
N 3062  W 610 / D 1930 / L 522
Penta 21 / 317 / 790 / 359 / 44
+9.99 +/- 6.79 Elo
LLR +3.13 (H0=0, H1=+5) — PASS
Timeouts: Prev 17 / Dev 15 (screen 13 / 9, continuation 4 / 6)
```

**Freeze.** The shipped check uses an eight-line test (two held, third not
blocked) instead of HCE's 256 KiB `fast_threat_count` table, which the
CodinGame file does not carry. Against the SPRT binary's Dev it is IDENTICAL
at depths 7, 8 and 11, since a line with all three held means the game is
already over. Prev is Dev renamed. `codingame_nnue.cpp` carries `fp_pawns`:

- `make port-check` IDENTICAL at depths 5, 7 and 9;
- `make cg-speed` checksums match at -O3 and with CodinGame's flags;
- `cg_input.cpp` is 95,874 characters (4,126 left).

## 60. Internal iterative reduction at non-PV nodes (30 September 2026)

The first experiment of a middlegame search round on the section 59 freeze.
Round ten killed IIR early when it *replaced* IID everywhere (section 11,
-5.47 +/- 13.57 at N=1206, on the HCE engine). This version keeps IID exactly
where it runs (PV nodes without a TT hit, depth > 2) and only adds the
reduction elsewhere.

### The change

`IIR_MIN_DEPTH = 4`. In `search()`, after RFP, futility and the IID block:
`if (!pv_node && !tt_hit && depth >= IIR_MIN_DEPTH) depth--;`. RFP and futility
still decide on the full depth; the move loop, the TT store, the history bonus
and the correction weight see the depth actually searched, so a later probe at
the full depth does not cut off on the shallower result. When the parent
re-searches the move at full depth because the reduced result beat its alpha,
the node finds its own entry and is not reduced again.

Instrumentation (scratch copy, not shipped):

- It fires at 2.8% of non-PV interior nodes on the fresh-TT depth-8 bench, 5.2%
  with a warm TT, and 19.7% in timed 90 ms games (a TT miss is common at the
  shallow non-PV nodes: 60% at depth 4, 12% at depth 16).
- Reduced nodes fail high 67-80% of the time and never return exact scores.
  Re-searched at full depth, 3.3% of the reduced fail-highs fail low, and 10.2%
  of the reduced fail-lows fail high; the second kind is mostly re-searched by
  the parent anyway.
- Nodes at fixed depth: 96.0% on the depth-8 bench, 88.1% and 77.7% at depths
  10 and 12 with a warm TT. The timed walk completes +0.85 ply deeper on
  average (16.71 -> 17.56). Threshold 5 fires about half as often and gains
  +0.73 ply; 4 matches round ten's constant.

### Result

The SPRT ran at time budgets scaled so that each search gets about the nodes
of a 90 ms CodinGame move. A probe replayed 755 positions from our CodinGame
games and compared the engine's node counts with the `N` the bot printed on
CodinGame, with as many copies running as the machine's SPRT threads. A flat
90 ms gives 1.69x CodinGame's nodes on the desktop, 1.36x on the ThinkPad and
1.30x on the Dell. The budgets are 49 ms on the desktop (7 threads), 63 ms on
the ThinkPad (7 threads pinned to its E-cores) and 67 ms on the Dell (3
threads), 1.03-1.09x CodinGame's nodes when checked. Each machine played its
own opening range, and the pentanomial counts were pooled with
`tools/sprt_merge.py`:

```text
N 9582  W 1926 / D 5875 / L 1781
Penta 115 / 1010 / 2401 / 1145 / 120
+5.26 +/- 3.95 Elo
LLR +3.39 (H0=0, H1=+5, bound 2.94) — PASS
Timeouts: Prev 0 / Dev 0
Shards: desktop N 4494 +7.50 +/- 5.93, ThinkPad N 3612 +1.83 +/- 6.28,
        Dell N 1476 +6.83 +/- 9.81
```

**Freeze.** Prev is Dev renamed. `codingame_nnue.cpp` carries the reduction:

- `make port-check` IDENTICAL at depths 5, 7 and 9;
- `make cg-speed` checksums match at -O3 and with CodinGame's flags (g++ 11 on
  the ThinkPad: 30 positions at depth 13, 2,757,452 nodes, checksum
  4620691443947525582, no speed lost);
- `cg_input.cpp` is 95,927 characters (4,073 left).

---

## 61. No futility pruning while every searched move loses (30 September 2026)

A bug fix found while investigating the false exact draw in CodinGame replay
906589060 (section 11, this round). A node that futility-prunes quiet moves
keeps a fail-soft best value from the moves it searched. If all of those lose
by force, it stores and returns a "mated" upper bound although a pruned quiet
move may hold or win, and the false bound propagates through the TT.

### The audit

Five ladder games replayed through one persistent engine (our moves, depth 22),
every mate-range TT store at 45+ stones checked against an exact WDL solver:

| Engine | Mate claims checked | False | At futility-pruning nodes |
| --- | --- | --- | --- |
| Section 60 freeze | 1,709,772 | 48,401 (2.83%) | 25,614 of 133,169 (19.2%) |
| Futility floor (not kept) | 1,435,571 | 0 | 0 |
| **This change** | 1,498,862 | **0** | 0 of 1,795 |

The false claims away from pruning nodes (1.45% of those) were propagated ones;
they vanish with the source. False exact 0 stores fall from 3.20% to 2.18%.

### The change

At the futility skip in `search()`:
`if (can_futility_prune && i > 0 && !capture && best_val >= -CORR_MATE_BOUND)`.
While every move searched so far loses by force, the quiet moves futility would
skip are searched instead (Stockfish's `!is_loss(bestValue)` guard), so a node
claims to be mated only after searching them. Otherwise pruning is unchanged.
Nodes at fixed depth: +2.5% at depth 8, +1.8% at depth 11; the 49 ms timed walk
completes the same depth (+0.03 ply). Deep persistent searches pay more (+17.6%
nodes to depth 22, mostly from opening and middlegame roots).

### Result

The usual 0 / +5 SPRT (openings from 15000) did not show a gain: N=5848,
-1.37 +/- 4.99, LLR -2.98, H0 accepted. As a verified bug fix it was then gated
for non-regression (H0=-5, H1=0, as section 51's mate-window fix) on fresh
openings (from 32000), pooled at CodinGame-scaled budgets (desktop 49 ms x7,
ThinkPad 63 ms x7, Dell 62 ms x3):

```text
N 3436  W 697 / D 2093 / L 646
Penta 51 / 361 / 830 / 438 / 38
+5.16 +/- 6.73 Elo
LLR +3.21 (H0=-5, H1=0, bound 2.94) — PASS
Timeouts: Prev 0 / Dev 0
```

**Freeze.** Prev is Dev renamed. `codingame_nnue.cpp` carries the guard:

- `make port-check` IDENTICAL at depths 5, 7 and 9;
- `make cg-speed` checksums match at -O3 and with CodinGame's flags (g++ 11:
  30 positions at depth 13, 3,134,381 nodes, checksum 6637135766784572329);
- `cg_input.cpp` is 95,936 characters (4,064 left).

## 62. Continuation history (30 September 2026)

Round ten killed continuation history on the HCE engine (section 11: NPS -4%,
+1.15 +/- 10.63 at N=2106). It was retried in this round.

### The change

`cont_hist[stm][previous move's cell, or none][mb][sq]` (int16, 47 KB),
updated where the butterfly history is: the cutoff move gets the same
`depth * depth` bonus with gravity, and the earlier quiet moves get the same
malus. Aging is the same (halved every search, zeroed for fixed-depth search).
It adds `cont * 3277 >> 16` (about 1/20, history's weight) to the move score;
the vector scorer loads one miniboard's nine squares per row. The countermove
bonus is unchanged.

### Measurements

On the section 59 freeze the instrumentation predicted a loss. At fixed depth
14 the tree grew 8% (more nodes on 32 of 40 games), and the first move cut off
about 1 point less often. NPS fell 1.5-3%, and 90 ms walks completed 0.1 ply
less. Rebased onto IIR (sections 60-61), the same table gives 90.5% of Prev's
nodes on the depth-8 bench, with NPS about 4% lower. IIR reduces non-PV nodes
that have no hash move, and those nodes are ordered by the history terms alone,
which is the likely reason the sharper reply-conditioned ordering pays off
there.

### Result

All tests were pooled at CodinGame-scaled budgets (desktop 49 ms x7, ThinkPad
63 ms x7, Dell 62 ms x3) against the section 61 freeze. A 3,000-game screen
read +7.93 +/- 6.66 (LLR +2.34) at N=3464. The sequential test continued on new
openings (the screen's shard logs pooled with the new ones) and passed at
N=4012: +9.18 +/- 6.16, LLR +3.37. Most of that decision came from the screen,
and the ThinkPad shard read only +1.8, so an independent SPRT on fresh openings
was run before freezing:

```text
N 10622  W 2126 / D 6516 / L 1980
Penta 137 / 1100 / 2708 / 1212 / 154
+4.78 +/- 3.78 Elo
LLR +3.06 (H0=0, H1=+5, bound 2.94) — PASS
Timeouts: Prev 0 / Dev 0
Shards: desktop +3.57 +/- 5.44, ThinkPad +6.85 +/- 6.35, Dell +3.58 +/- 9.31
```

**Freeze.** Prev is Dev renamed. `codingame_nnue.cpp` carries the table, its
updates, aging and both scorers:

- `make port-check` IDENTICAL at depths 5, 7 and 9;
- `make cg-speed` checksums match at -O3 and with CodinGame's flags (g++ 11:
  30 positions at depth 13, 3,520,782 nodes, checksum 6810861239900586370);
- `cg_input.cpp` is 96,714 characters (3,286 left). The table costs about 780
  characters of code.

---

## 63. Submission size: 95,927 to 72,105 characters (30 September 2026)

Not an Elo round: the same program, 23,822 characters smaller, to make room
for a bigger net or more book. Where the paste file's characters went before:
code 49,765, NNUE payload 30,948, opening book 13,456, macro net 1,758.

- **The NNUE payload is incompressible losslessly.** The quantized weights use
  12-13 significant bits each (median |q| 300-1,600 on a 14-bit grid, ~1,500
  distinct values per 1,800), and per-row Rice, Laplacian, Gaussian and
  adaptive bit-length-class models all land within 1% of the shipped Rice
  code. Only fewer bits would shrink it (about 1,350 characters for 0.1 more
  mean error, minification.md 3.1).
- **`#define` pass in the minifier (-15,044).** Keywords and attributes
  survive renaming and repeat thousands of times; a greedy pass turns the most
  valuable runs of 1-8 tokens into one- or two-letter object-like macros,
  bracket-balanced (the AVX intrinsics are function-like macros under
  CodinGame's -O0, and an unbalanced expansion inside their argument lists
  broke them), with every `#include` hoisted ahead of the defines and one
  joint ranking of macros and renamed identifiers by use count. Half the
  bundled file had been out of the pass's reach until the hoist, because
  inlined headers repeat `#include` lines mid-file. `make cg-min-check`
  minifies `cg_selfcheck.cpp` itself and pins the fixed-depth checksums.
- **Opening book arithmetic-coded (13,456 -> 6,208 at 14 bits).** The digits
  were at their full uniform information content (184,846 bits). Modelled:
  our move as its rank under the NNUE's static evaluation of the children
  (rank 0 63% of the time; 101,931 -> 54,073 bits; a depth-1 search would
  give 48,236 for 8 s of decode, so no), "continues" by ply (34,066 ->
  14,108), "covered" by ply and the reply's eval rank (48,849 -> 22,350; the
  uttt.ai prior threshold the book was grown with tracks the NNUE's ranking
  closely: rank 0 covered 99%, rank 7+ 7%). LZMA's binary range coder, 88
  models. The decode evaluates 300k children: 50 ms at -O3, 90 ms with
  CodinGame's flags once the children were built as a light view instead of
  through `GlobalBoard::makeMove` (its `std::stack` move history costs 190
  ms under -O0). The book table is unchanged (pinned checksum). The book
  now depends on the net: `make play-book` after every net change, from the
  text book now committed as `cpp_impl/play_book.txt`. The payload carries
  a fingerprint of the evaluator (its values on 64 fixed pseudo-positions)
  and `pb_init` refuses a mismatch in 0.1 ms; a walk that outgrows the
  expected entry count stops too (review finding: without it a stale book
  decoded nonsense for 6 s at -O3 and 25 s with CodinGame's flags, a
  first-turn timeout). `cg_selfcheck` exits 1 on a failed book and the CI
  gate's protocol check runs with the text book, so a stale book cannot
  ship quietly.
- **U15 alphabet: 15 payload bits per character (-2,536).** No single block
  of 2^15 plain characters exists, but U+3400..U+9FFF (Extension A, the
  Yijing symbols, the unified ideographs) plus 5,120 private-use characters
  is one, with no normalization decompositions anywhere; a Python test walks
  it. The decoders change one line. (A top CodinGame bot goes to 15.875 bits
  with every non-surrogate code point and a bignum decode; that is another
  4% of the payloads if ever needed.)
- **`evaluate_macro_fast` out of the shipped header (-212).**

Everything is tree-identical: `port-check` and `cg-min-check` (at -O3 and
with CodinGame's flags) IDENTICAL at depths 5, 7 and 9, 31/31 unit tests, 96
Python tests, the same node rate through the CodinGame protocol with
CodinGame's flags (680k vs 679k nodes per searched move). The first turn
with CodinGame's flags takes about 300 ms of its 1,000 ms (was 250) for the
book's evaluations. Merged after sections 61 and 62, with the fingerprint,
the file is 73,088 characters (26,912 left).

Not pursued: shipping a compiled binary inside a wrapper, as that top bot
does (UPX-packed C, ~200 KB in 95k characters). Our dynamic g++-11 binary is
260 KB and 151 KB after UPX, which would fit at 15.875 bits per character
with room, and it would give the bot a real -O3 (and PGO) instead of the
pragma-optimized -O0 build. But CodinGame's compile costs only 5% of nodes
today (`cg-speed`: 9.14M against 9.59M nps), the binary must be built
against the arena's libstdc++ and glibc, and it can only be tested by
submitting; the source route above already leaves 27,895 characters free.

---

## 64. Round 13 net: warm start from r12_M2 on 80M depth-13 self-play (30 September - 1 October 2026)

Round twelve trained on 13M rows of the NNUE engine's self-play. Round
thirteen generated 80M, relabelled every other source with the current
engine, added a holdout of real ladder positions, and searched the training
recipe instead of fixing it. The net that ships, **r13w_20**, is not a fresh
training run: it is r12_M2's own weights fine-tuned on the new data for 2.4G
rows. Same architecture, runtime and speed; only `nnue_b64_net.hpp`'s
payload, its scales and the re-packed opening book change. In play it is
about **+16 to +20 Elo** over r12_M2 (three independent matches); the shipped
engine's paste file with it beat main's over 1,000 games at 90 ms by
**+16.0 +/- 10.9**. r13w_11, the 1.2G-row fine-tune that was ship-tested
first (+11.8 +/- 11.0 the same way), is the runner-up.

### Data

- **sp13: 80.0M records of depth-13 self-play** (79.8M rows: 77.4M train + 2.39M holdout) by the current engine (the
  section 62 freeze's search, net r12_M2; `datagen play ... d13`, round
  twelve's openings and randomisation) on three machines (desktop, ThinkPad,
  Dell). Holdout **SPH13**: 3% of the games per shard, 2,388,278 rows.
- **e2b: eval2 relabelled** by the same engine at depth 14 (4,339,877
  training rows). **V2** is round twelve's 482,136 rows with these labels.
- **dumpb: the SPRT dumps' positions relabelled at depth 13** (the 30
  `datasets/eval2/sprt_dump` files, 3.49M positions; the dumps' game scores
  are no longer labels). Holdout **DUMPH**: 3% of the games per file,
  106,495 rows.
- **LADH, a ladder holdout:** 193,320 positions from 2,939 saved CodinGame
  games (the board before each move from ply 9, not decided, deduplicated up
  to the 8 symmetries and colour, plus two one-move children per position;
  74,722 played positions, half of them from our games against the top
  seven), labelled at depth 14. No training row is a LADH position.

Every training label is the current engine's search. The data tools
(`r13_prep.py`, `ladh.py`) and the trainer live with the data in
`datasets/nnue2/r13/` (not in the repository; its README has the full
record).

### Training

`datasets/nnue2/r13/tools/gen_r13.py` streams a weighted mix of the packed
sources from disk (about 750k rows/s on the GPU through DirectML) for a fixed
step budget, and reports on the four holdouts every 1/20 of it. An Optuna
study, **r13w**, searched a wide space: rows seen (2e8..1.6e9) and batch, lr
and its schedule (cosine / linear / exponential, warmup, floor), weight
decay, K, the result blend, the source mix, the PSQT weight and the
initialisation (random, or r12_M2's weights). Its objective is an Elo
estimate against r12_M2 from the holdouts (SPH13 35%, LADH 35%, V2 15%,
DUMPH 15%), calibrated on round twelve's nets (measured 20 ms Elo = 3.38 +
0.940 x objective, r 0.998, residual 2.4 Elo).

| Run | Recipe | Objective (Elo est. vs r12_M2) |
| --- | --- | ---: |
| r13w_0 | round twelve's recipe (random init, 32,490 steps) on the new data | -2.3 |
| r13_pre_b128, _s2 | the same with 128 lanes, two seeds (not shippable: the runtime is B-64) | +12.7, +13.7 |
| r13w_1 | **warm start** from r12_M2, lr 2e-3, 100M rows | +11.5 |
| r13w_5 | warm start, 300M rows | +14.5 |
| r13w_10 | warm start, 600M rows | +18.3 |
| r13w_11 | warm start, 1.2G rows | +20.2 (SPH13 +20.7, LADH +20.8, V2 +20.5, DUMPH +17.6) |
| **r13w_20** | **warm start, 2.4G rows** | **+22.9** (SPH13 +23.0, LADH +24.5, V2 +24.0, DUMPH +18.1) |
| r13w_21 | the same at lr 1e-3 | +22.9 |

- **The winner, trial 20:** r12_M2's weights fine-tuned for 2.4G rows =
  146,484 steps of batch 16,384 (73 minutes on the GPU), lr 2e-3 cosine (1%
  warmup, floor 1e-5), K 1600, no result blend, mix e2b 46% / sp13 54%
  (about 254 passes over e2b and 17 over sp13), D4 augmentation, seed 1.
  Held-out loss on SPH13 4.0% below the float r12_M2 net's. Trial 21, the
  same recipe at lr 1e-3, scored +22.94 against +22.92: the learning rate
  does not matter at this length.
- **Length is the lever.** The fine-tune keeps improving with the rows seen:
  100M +11.5, 300M +14.5, 600M +18.3, 1.2G +20.2, 2.4G +22.9 offline, and the
  order holds in games (below). Trial 11 (1.2G rows, 73,242 steps, 28
  minutes; SPH13 3.6% below the float r12_M2 net, 3.8% below the shipped
  quantized one) was the first net tested in the shipped engine and is the
  runner-up.
- **Random-init retraining does not beat r12_M2.** Round twelve's recipe on
  the new data (trial 0) is -2.3 offline and -13.8 +/- 5.5 in games (20 ms,
  12,000-game fit): the relabel alone
  does not lift it, and three times the rows at equal steps are worth about
  one seed's difference. The data loop pays through the warm start.
- **Capacity pays offline, not in play.** B-128 nets were +13 offline but
  tied r12_M2 at 20 ms on the ThinkPad (-1.7 and +2.5 +/- 8.3) and at the
  Dell's CodinGame compute (-0.3 +/- 13.4): their tables are twice B-64's and
  the search loses 16-31% of its nodes. A generic-width runtime exists on a
  side branch for a later round.
- Model soups of the warm starts (+12.4, +13.2) were below the longest
  fine-tune alone. SPH13's labels are r12_M2's own search, so a warm start
  that stays close to it is flattered there; the warm starts were judged on
  V2 / LADH and in games.

### Games

Candidate engines as in round twelve (the fast-NNUE harness on the pre-NNUE
search, commit `c278cde`), fixed-length round robins, 1,000 games per pair:

- **20 ms on the ThinkPad** (13,000 games per net, ratings fitted with
  r12_M2 = 0): **r13w_20 +16.6 +/- 5.3**, r13w_10 +11.9, r13w_11 +11.0
  (+11.4 +/- 5.5 in the earlier 12,000-game fit), r13w_5 +8.0.
- **Dell at its CodinGame compute** (62 ms x 3 threads, 1,000 games each vs
  r12_M2): **r13w_20 +20.2 +/- 12.4**; r13w_11 +13.6 +/- 12.6 and +9.4 +/-
  12.8 in two runs, r13w_5 +12.9 +/- 12.2, r13w_3 +10.4 +/- 12.2, r13w_10
  +9.4 +/- 12.7 and +16.3 +/- 12.8 in two runs, r13w_1 +0.3, r13_pre_b128_s2 -0.3;
  head to head at the same compute r13w_20 beat r13w_11 by +8.7 +/- 12.2 (W214 D597 L189).
- **The shipped engine:** this branch's `cg_input.cpp` against main's
  (r12_M2), the CodinGame paste files through the protocol at 90 ms
  (`cg_match.py`, 5 referees, forfeit at 1,000 ms, late replies counted:
  0 forfeits, one late reply, main's):

```text
90 ms: N 1000  W 321 / D 404 / L 275
Penta 4 / 80 / 296 / 106 / 14
+16.0 +/- 10.9 Elo
Forfeits: 0
```

  r13w_11 measured the same way (the desktop then loaded by the trainer, a
  relabel and ten bots, so late replies were frequent and symmetric): N 1000,
  317-400-283, pentanomial 9/75/301/103/12, **+11.8 +/- 11.0**, 0 forfeits.

Offline +14..+23 became +9..+20 at CodinGame compute: the objective
overstates the fine-tunes by about 1.2-1.5x, with the ranking intact.

### The build

- `tools/nnue_emit_b64_header.py datasets/nnue2/fast/r13w_20_perm.bin --label r13w_20`
  (every other option at its default; export CRC-32 `d18dccb2`, checkpoint
  `datasets/nnue2/probe/r13w_20.pt` sha256 `eaf8c46b...`): scales **9, 12,
  13, 13, 10** (r13w_11's; QO one bit below r12_M2's), payload **53,834
  bytes = 28,712 U15 characters** (r13w_11: 28,762; r12_M2: 28,885), sha256
  `492a36ea...`. The payload's rounding moves the float eval by 1.97 mean /
  66.7 max units on the 20,000 parity positions (r13w_11: 1.69 / 41.8); on
  the 16 pinned test positions the integer engine is 6.1 mean / 30 max from
  the PyTorch net (r13w_11: 8.9 / 31).
- The opening book re-packed (`make -C cpp_impl play-book`): its moves are
  unchanged (table checksum 17441813851168678777), the coding differs, 10,654
  bytes = 5,683 characters (r13w_11's packing 5,619, r12_M2's 5,794);
  `cg_selfcheck` prints book=ok.
- `cg_input.cpp` is **72,803 characters** (27,197 left); `make cg-input`
  reproduces it byte for byte.
- `port-check` IDENTICAL at depths 5, 7 and 9 (nodes 536445 / 831874 /
  1819208, checksums 8838790865174526946 / 11722602351321275973 /
  14786683063888215671); 31/31 unit tests and 120 Python tests with the new
  pins (the 16 table hashes, the 16 fixed-position evals, the payload sha256
  and length, the book's net fingerprint; the five scales are r13w_11's).

**Freeze.** Dev and Prev share `nnue_b64_net.hpp`, so both now carry r13w_20;
nothing else in the engine changed. A same-shape net is not an eval-architecture
change, so the CodinGame gate needs no declaration.

**How r13w_20 replaced r13w_11.** r13w_11 (1.2G rows) was chosen and
ship-tested first; trial 20 finished while that was written, scored +22.9
offline, and then won every match it entered (+16.6 +/- 5.3 at 20 ms, +20.2
+/- 12.4 at the Dell's CodinGame compute, +16.0 +/- 10.9 as paste files), so
the branch was re-pinned to it the same morning. A 4.8G-row fine-tune (twice
r13w_20's length) is training and ungamed.

**Not yet done.** The 4.8G-row fine-tune's games (and a paste-file match
against r13w_20). The sprt harness cannot run a CodinGame-scaled SPRT net
against net, because Dev and Prev share the net header; the paste-file match
above is the ship test. The g++ 11 half of the gate ran on the ThinkPad at
06:12: `cg_input.cpp` compiles with CodinGame's exact command line and zero
diagnostics, `make cg-min-check CG_CXX=g++-11` is IDENTICAL on both legs at
depths 5, 7 and 9, `cg_selfcheck_cgflags` loads the book (book=ok, the same
depth-5 checksum as the clang build) at 15.0M nps, and the exact book
protocol check passes 40/40 (first turn at most 223 ms, later replies 90.1 ms
median, 90.4 ms max). The CodinGame IDE paste and copy-back test and CI's
`cg-perf-gate` remain to be run.

## 65. Round 14 net: a WDL filter and a power loss on round 13's data (3-4 October 2026)

Round fourteen was a pre-registered ablation round: one fixed base recipe,
one-factor changes judged by games against r13w_20 with confidence
intervals, and the decision rules written down before the games
(`datasets/nnue2/r14/DESIGN.md` with its dated log, and `REPORT.md`; not in
the repository). The net that ships, **r14_d5_final_s2_rs**, is r13w_20's
own weights fine-tuned for 600M rows on round thirteen's data and labels,
with the two trainer changes that survived: a WDL contradiction filter on
the self-play rows and a power loss. Same architecture, runtime, speed and
scales; only `nnue_b64_net.hpp`'s payload and the re-packed opening book
change. Against r13w_20, with the shipped engine's booked paste builds at
90 ms, it is **+9.1 +/- 5.9 Elo** over 4,000 fresh-opening games. On the
CodinGame ladder the difference is below resolution: no measurable change,
and no regression.

### Recipe

The base, **R-old**, is round thirteen's continuation: r13w_20's weights,
lr 1e-3 cosine (1% warmup, floor 1e-5), batch 16,384 for 36,621 steps (600M
rows), K 1600, no result blend, e2b 46.4% / sp13 53.6% of the rows with
round thirteen's labels. The shipped run adds the filter and the loss
(`datasets/nnue2/r14/tools/gen_r14.py`, frozen as `r14/frozen_x`; holdout
options left out):

```bash
gen_r14.py train --name r14_d5_final_s2 --init r13w_20 --lr 1e-3 --sched cosine --warmup 0.01 \
  --lr-floor 1e-5 --batch 16384 --steps 36621 --seed 2 --mix e2b=0.4644,sp13=0.5356 \
  --wdl-filter --wdl-src sp13 --wdl-only sp13 --pow-exp 2.5 --qp-asym 0.2 --psqt-w 0.06878
```

- **WDL filter.** A win/draw/loss model of sp13's labels and game results,
  P(win) = sigmoid((s - d) / s0) and P(loss) = sigmoid((-s - d) / s0),
  fitted by maximum likelihood (d 1,108, s0 608), skips each sp13 row with
  probability 1 - P(observed result | label): a row whose search score
  contradicts how its game ended is likely dropped. About 70% of sp13's rows
  are kept; eval2 is kept whole.
- **Power loss.** |p_net - p_target|^2.5, weighted 1.2x where the net is
  above the target (Stockfish's trainer), with the PSQT term's weight scaled
  by the new loss's ratio to the squared one (0.1 x 0.688 = 0.06878).
- **Eval scale.** The filter widens the eval: the raw evals' spread is 1.047
  times r13w_20's (standard deviation over 200,000 SPH13 positions with
  |eval| <= 2,000). The search margins are tuned to r13w_20's scale, so by a
  rule fixed before any games the net is multiplied by 1/1.0473 = 0.9548
  before export (`r14/tools/scale_b.py --apply`, which writes
  `r14_d5_final_s2_rs`): the PSQT lane of every first-layer table and the
  last dense layer, which scales the eval exactly and leaves the lanes'
  sparsity alone. Every WDL-filter net needed it; the power loss alone did
  not.
- 14.6 minutes on the GPU (730k rows/s), seed 2.

### What the round found

- **The labels stay old.** Nets fine-tuned on the same positions relabelled
  by r13w_20's search (`datagen label`) lost to their old-label twins:
  -10.6 +/- 2.5 Elo over three seed pairs of 10,000 games each at 20 ms
  (-11.3 over the first two, which triggered the pre-registered pivot back to
  the old labels), although every offline metric preferred them. The loss
  came from relabelling the self-play, whose old labels are the in-game
  searches (-10.4 +/- 5.5); relabelling eval2 was neutral (+0.9 +/- 3.4).
  About half of it remains at fixed depth, and the relabelled nets grow 1.09
  times larger trees.
- **Length bought nothing.** Continuing r13w_20 on data it has already fit
  gained +0.2 +/- 2.5 Elo per doubling from 600M to 2.4G rows at 20 ms while
  the offline objective rose, so 600M it is.
- **Of the trainer ideas, two passed their screens**, and a 2x2 factorial
  at 600M rows (four fresh seeds per cell, 4,000 games per net against
  r13w_20 at 20 ms) measured them:

| Cell | Seeds 11-14 | Mean |
| --- | --- | ---: |
| R-old | +1.4, +5.9, +4.6, +3.3 | +3.80 |
| + WDL filter | +4.1, +6.9, +11.6, +3.2 | +6.45 |
| + power loss | +8.2, +3.6, +5.2, +6.0 | +5.75 |
| + both (R\*) | +10.9, +11.0, +6.8, +14.4 | **+10.78** |

  Main effects: WDL filter **+3.84** (one-sided p 0.024), power loss
  **+3.14** (p 0.054), interaction +1.2 (not significant); the seeds vary no
  more than the games' noise. Three more fresh seeds of R\* scored +7.6,
  +8.3 and +2.2 (+/- 4.9 each, 8,000 games at 20 ms on the ThinkPad): +6.1 on
  average, and +6.8 over replays of R-old on the same openings (the factorial
  predicted +7.0).
- The activation-sparsity penalty, the lambda schedule, a low-lr finish and
  1-10% ladder positions in the mix failed their screens.
- **Offline metrics did not predict play.** Both kept ideas lower the
  label-based objectives: the shipped net is -8.8 on round thirteen's
  objective against r13w_20, and across round fourteen's nets that objective
  correlated with games at r = -0.43. Every decision was made on games.

### The ship tests

The candidate came from a 2,000-game screen at the Dell's CodinGame compute
(62 ms x 3 threads) of the three fresh R\* seeds and the factorial's best R\*
net: s2 +10.6, s4 +9.6, s3 +8.0, s14 +5.7, all within one standard error of
each other, so the pre-registered tie-break (the lowest fresh seed) picked
s2. Those games are selection data only. Then three pre-registered tests
against r13w_20, each engine the shipped one with its own net and re-packed
book (paste builds in their match mode, CodinGame's rules,
`r14/eval/gauntlet.py`):

- **90 ms GSPRT [0, 6]** (desktop, 7 pinned workers; pentanomial LLR, alpha
  = beta = 0.05): **accepts H1** at 4,200 games, LLR +3.323. Its Elo is
  sequentially stopped, so it is a decision, not an estimate (nElo +14.1 +/-
  10.5).
- **4,000 fresh-opening games at 90 ms**, the estimate:

```text
90 ms: N 4000 W 777 D 2551 L 672
Penta: 29 / 417 / 1028 / 472 / 54
Elo diff: +9.1 +/- 5.9 (nElo +16.5 +/- 10.8, LOS 99.9%)
Forfeits: 0, late replies: 0
```

- **CodinGame-compute veto** (Dell, 62 ms x 3 threads, 4,000 games):
  **+12.9 +/- 6.4**; the veto (an upper bound below zero) does not apply.

In both 90 ms runs neither engine forfeited or replied late (over 120 ms);
the longest moves were 90.7-90.9 ms. Against r12_M2 the net is +23.8 +/- 7.2
(4,000 games at 20 ms).

### On the ladder

The user submitted this paste twice on 2026-10-04 (agents 6780732 and
6780755): **#4 at 33.19 and #3 at 33.30**, against #3 to #7 (32.38 to 33.27)
for r13w_20's eleven placements with various books. Against r13w_20's
submission with the same book (agent 6776785):

| | r14_d5_final_s2_rs (2 agents) | r13w_20, same book |
| --- | ---: | ---: |
| second player vs the top 7 | 0.074 +/- 0.018 (n 148) | 0.064 +/- 0.028 (n 78) |
| first player vs the top 7 | 0.929 +/- 0.017 | 0.942 +/- 0.020 |
| both seats vs the 7 top agents unchanged since 10-01 | 0.518 +/- 0.029 (n 274) | 0.518 +/- 0.041 (n 142) |

**No measurable change, and no regression.** The local gain (+9 to +13 Elo,
about +0.013 to +0.019 per game) is below the ladder's resolution of about
+/-0.03 per cell, as round thirteen's +16 was. The higher rank is not a net
effect: no game cell explains it, and the field moved during the
placements. The second player's games against the top seven are still
decided by the book's lines. The first placement had 4 timeouts of ours in
276 games (and 8 by opponents), the second none in 260; the burst hit both
sides during one placement only, which points at CodinGame's judges rather
than the build. The analysis is
`datasets/nnue2/cg/ladder/prep/r14/ladder_r14_d5_final_s2_rs.md`.

### The build

- Checkpoint `datasets/nnue2/probe/r14_d5_final_s2_rs.pt` (sha256
  `68ad5e53...`, the rescaled copy of `r14_d5_final_s2.pt`); export
  `export_bgn.py export r14_d5_final_s2_rs datasets/nnue2/fast/r14_d5_final_s2_rs_perm.bin --perm`
  (CRC-32 `65bdbb1a`, sha256 `993ec0f5...`; it rebuilds from the checkpoint
  byte for byte).
- `tools/nnue_emit_b64_header.py datasets/nnue2/fast/r14_d5_final_s2_rs_perm.bin --label r14_d5_final_s2_rs`
  (every other option at its default): scales **9, 12, 13, 13, 10**
  (r13w_20's), payload **53,865 bytes = 28,728 U15 characters** (r13w_20:
  28,712), sha256 `cde8c610...`. The header is byte for byte round
  fourteen's verified build (sha256 `aa2819fe...`), from which the tested
  and submitted pastes were made. The payload's rounding moves the float eval
  by 1.73 mean / 74.4 max units on the 20,000 parity positions (r13w_20:
  1.97 / 66.7); on the 16 pinned test positions the integer engine is 8.8
  mean / 48 max from the PyTorch net (r13w_20: 6.1 / 30; the 48 is on an
  eval of 11,193).
- The opening book re-packed (`make -C cpp_impl play-book`): its moves are
  unchanged (table checksum 17441813851168678777), the coding differs, 10,832
  bytes = 5,778 characters (r13w_20's packing 5,683); `cg_selfcheck` prints
  book=ok.
- `cg_input.cpp` is **72,914 characters** (27,086 left); `make cg-input`
  reproduces it byte for byte, and it is byte for byte the paste submitted on
  2026-10-04 (sha256 `b32979a6...`).
- `port-check` IDENTICAL at depths 5, 7 and 9 (nodes 568480 / 897652 /
  1900326, checksums 14701287764179133873 / 253444004976430199 /
  5993005870751148654); 31/31 unit tests and 120 Python tests with the new
  pins (the 16 table hashes, the 16 fixed-position evals, the payload sha256
  and length, the book's net fingerprint; the five scales are r13w_20's).

**Speed.** Same architecture, so the same nodes per second: alternating with
r13w_20's build on one ThinkPad core, the median ratio is 0.997 over 15
pairs (the gate, below). The net
does search more nodes to a given depth: 4.5-7.9% more at depths 5, 7 and 9
over `port-check`'s 120 positions and 14.5% more at depth 13 over 30 (round
fourteen's plain replicate fine-tunes of r13w_20 also took about 5% longer
to depth 14), and on the ladder it reached depth 19.2 at plies 21-40 against
r13w_20's 19.9, on the same median of 316k nodes. The games above include
that cost.

**Freeze.** Dev and Prev share `nnue_b64_net.hpp`, so both now carry
r14_d5_final_s2_rs; nothing else in the engine changed. A same-shape net is
not an eval-architecture change, so the CodinGame gate needs no declaration.

**The g++ 11 gate** ran on the ThinkPad at 12:24 on CPUs 0-1, while
self-play datagen held CPUs 2-11 (the cores' clock moved between 2.0 and
4.0 GHz): `cg_input.cpp` compiles with CodinGame's exact command line and
zero diagnostics, `make cg-min-check CG_CXX=g++-11` is IDENTICAL on both
legs at depths 5, 7 and 9, `cg_selfcheck_cgflags` loads the book (book=ok)
and gives the clang build's checksums at depths 5, 7 and 9, and the exact
book protocol check passes 40/40 twice (first turn at most 538 and 527 ms,
later replies 90.1 ms median, 91.6 and 92.2 ms max). The load, not the net,
sets those times: main's r13w_20 bot in the same conditions took up to 653
ms on its first turn (90.1 / 91.6 ms later). The gate's single speed run
read 5.0M nps at 2.1 GHz (round thirteen's gate: 15.0M); alternating this
build's `cg_selfcheck_cgflags` with main's on one core, 15 pairs, the median
ratio is 0.997 (single runs 4.7M to 10.3M as the clock moved), so the speed
is unchanged.

**Not yet done.** The paste has run on the ladder itself, so the IDE
copy-back test was not repeated; CI's `cg-perf-gate` runs on the pull
request. Round fourteen's exploratory architecture track (wider encoder and
head, macro-board contexts) is still training.

## 66. Native submission: the clang build in a Python 3 launcher (5 October 2026)

The live CodinGame file is now **`cpp_impl/cg_input_native.py`** (ladder
submissions 41456153 and 41456253). It holds the unchanged r14 bot,
`codingame_nnue.cpp` with net r14_d5_final_s2_rs and the book, as a Linux
x86-64 executable. clang 23.1.2 builds it with `-O3 -march=haswell
-ffp-contract=off` and thin LTO, links it dynamically against libstdc++, then
xz-compresses and U15-encodes it into a stdlib-only launcher. The launcher
writes the binary to a memfd and execs it. The file is 72,056 characters, which
is 27,944 under the cap and smaller than the C++ paste. `cg_input.cpp` stays as
the reference and the fallback. [native_build.md](native_build.md) has the
toolchain, the commands and the checks.

**Same tree.** The launcher's `selfcheck` reproduces the C++ build's
fingerprint at depths 5, 7 and 9 (568,480 / 897,652 / 1,900,326 nodes), with
book=ok. `-ffp-contract=off` is required. A contracted NNUE bake fails at every
depth. The exact book protocol check passes 40/40 games through the launcher.

**Speed.** Measured in user-mode cycles at d12, the clang build is **+7.5% ±
2.0** against CodinGame's own build of the paste (40 paired rounds). g++
`-O3 -march=haswell` gains +0.1% ± 1.7, because the paste's pragmas already
give it -O3 code. PGO (FE, IR and CS-IR) added nothing. UPX was 4.7 KB larger
than xz and 9 ms slower to start.

**Strength.** Against CodinGame's build of the paste at 62 ms, GSPRT [0, 5]
(α = β = 0.05) accepted **H1 at 5,400 games**: W 1,685 / D 2,160 / L 1,555,
+8.4 ± 5.9 Elo, LLR +3.10. The first 2,000 games were a fixed block (+12.0 ±
11.8). The continuation alone gave +6.2 ± 6.3, so read the gain as about +6 to
+8 Elo. Neither side timed out.

**CodinGame.** The runtime is Python 3.11.5 with glibc 2.36 and GLIBCXX_3.4.30.
memfd works. The CPU is a Haswell without TSX. In its first 258 ladder games
the file timed out once, the same rate as the C++ build. On the rules,
TomAlard's write-up calls the method fine for bot-programming leaderboards and
banned in contests. So in a contest, submit `cg_input.cpp`.

**Reproducible.** `make cg-native` was run twice from a clean export of this
change, with LLVM 23.1.2 and g++ 11.4.0 headers on Pop!_OS 22.04. Both times it
rebuilt the live binary bit for bit (sha256 `68381d0c...cba61e58`) and the live
file byte for byte (`aaf96195...0848260`). `tools/cg_native/manifest.json`
records both hashes and the hashes of the sources. `tools/test_cg_native.py`
fails in CI if the bot changes without a native rebuild.

---

## 67. ProbCut (6-7 October 2026)

### The change

`search()`, after move ordering and before the move loop, at a null-window
node with depth >= `PC_MIN_DEPTH` (5) and beta outside mate range:
`pc_beta = beta + PC_PAWNS * eval_weights[PAWN_IDX]` with `PC_PAWNS = 60`.
Unless the table already holds a result of depth >= depth - 3 below `pc_beta`,
the first `PC_MOVES` (3) ordered moves are each tried (the deferred ordering
is completed first): the usual global-win shortcuts, then a qsearch at
`pc_beta`, and if it holds, a `depth - PC_REDUCTION` (4) null-window search at
`pc_beta`. The first move that holds stores a lower bound at depth - 3 (keeping
a deeper entry for the same position) and the node returns the shallow score
less the margin. Stockfish's ProbCut has the same shape; it restricts the
tried moves to captures with a good exchange, which UTTT has no analogue for,
so the ordering's first three moves stand in.

### Measurements (instrumented copy, not shipped)

Fixed depth, 120 positions, against the r14 freeze (d11 4.575M nodes, d13
11.004M):

| Margin (pawns) | d11 nodes | d13 nodes | Wrong cuts (d12) |
| ---: | ---: | ---: | ---: |
| 150 | 4.583M | | fires 602 times |
| 60 | 4.381M | 10.745M | 0.48% |
| 30 | 4.006M | 9.604M | 1.33% |
| 15 | 4.160M | | 2.17% |

A cut is wrong when a full-depth null-window search at beta, with ProbCut off
in its subtree, fails low. The rate is flat in depth. About a third of
eligible nodes try at all; the table check skips the rest. Depth 4 /
reduction 3 and depth 6 / reduction 4 were no better at 30 pawns.

Search depth at 90 ms along 214 replayed self-play games (7,174 positions,
persistent engines as in a game, both engines on the same position):

| Open squares | Freeze | ProbCut | Gain |
| --- | ---: | ---: | ---: |
| 70-81 | 13.37 | 13.36 | -0.01 +/- 0.04 |
| 60-69 | 15.24 | 15.59 | +0.35 +/- 0.06 |
| 50-59 | 16.41 | 17.24 | +0.83 +/- 0.09 |
| 40-49 | 17.88 | 19.01 | +1.13 +/- 0.12 |
| 30-39 | 23.84 | 24.50 | +0.66 +/- 0.27 |
| all | 17.60 | 18.02 | +0.42 +/- 0.04 |

Below 30 open squares both engines solve and reach the depth cap. The node
rate is unchanged (ratio 1.02).

### Result

90 ms, 3 threads on the 4-core cloud VM, openings from 31000, against the
r14 freeze:

```text
90 ms: N 1584 W 328 D 1001 L 255
Penta: 13 / 138 / 421 / 203 / 17
Elo diff: +16.0 +/- 9.2
LLR: +3.03 (H0=0, H1=+5) - PASS
Timeouts: Prev=7 Dev=2
```

The 30-pawn margin also passed (openings from 31000, N 1788, 386-1101-301,
+16.5 +/- 9.4, LLR +3.02). Head to head, 30 against 60 (openings from 41000)
stood at 0.0 +/- 13.0 after 744 games (134-476-134); 60 ships, with a third
of the wrong cuts. Multi-ProbCut on top of the 60-pawn version (first a
depth - 6 check against +90 pawns at depth >= 9, and a fail-low cut when the
static eval is below alpha and a depth - 4 search stays below alpha - 120
pawns; -9% nodes at d13, both cut kinds under 0.7% wrong) failed: N 1944,
321-1248-375, -9.7 +/- 8.7, LLR -3.05. The flat 90 ms budget gives this VM
about 1.05-1.2x CodinGame's nodes (section 60).

### Confirmation at CodinGame compute

A second SPRT, at CodinGame-equivalent compute, against main's
`crossfish_prev.hpp` (the r14 freeze), pooled over the desktop and the ThinkPad
with `tools/sprt_cluster.py` (6 October, openings from 10000 and 35000):

```text
CG compute: N 2534 W 563 D 1527 L 444
Penta: 26 / 255 / 614 / 318 / 54
Elo diff: +16.3 +/- 8.0
LLR: +4.15 (H0=0, H1=+5) - PASS
Timeouts: Prev=0 Dev=0
```

The budgets are the section 60 calibration (desktop 49 ms, ThinkPad 63 ms,
7 threads each) scaled by 1.075 for the native build's cycle gain (section
66): desktop 53 ms, ThinkPad 68 ms. By machine: desktop N 1414, +13.5 +/-
10.7; ThinkPad N 1120, +19.9 +/- 12.0. Both runs stopped at their first
crossing, so both estimates lean high, but they agree: two independent
openings ranges, two hardware setups, about +16 each.

One interaction worth knowing: the ProbCut entry is a lower bound at depth - 3,
which is exactly the pseudo-singular threshold (`entry.depth >= depth - 3`,
lower or exact). When a node ProbCut cut is revisited at the same depth and the
cut does not repeat, the ProbCut move gets the singular extension. Both SPRTs
include this behaviour.

### Freeze

Prev = Dev renamed. `codingame_nnue.cpp` carries the same block; it stores raw
table scores, which equal Dev's `tt_score_to_store` without
`CROSSFISH_NORMALIZE_TT_MATES`. `port-check` and `cg-min-check` (both legs)
IDENTICAL at depths 5, 7 and 9; 31/31 unit tests. The book is unchanged (same
net). `cg_input.cpp` is 73,510 characters. The native submission was rebuilt
in an `ubuntu:22.04` container (glibc 2.35, g++ 11.4.0 headers, LLVM 23.1.2;
`documentation/native_build.md`): binary 288,224 B, sha256
`d3850f74a4e821bf3312f08607df39684aa4f6dd52b96cb866c5c37e441cbd82`, needs
GLIBC_2.34; `cg_input_native.py` 73,189 units, sha256
`3f3267f6e56f49d833f7eeb378fd91c53b7abeefbe421f9530eb86da26cba860`;
`cg-native-check` OK (IDENTICAL at 5, 7, 9; 40/40 protocol games with the
exact book check).

