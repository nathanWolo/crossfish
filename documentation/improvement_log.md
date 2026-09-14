# Crossfish improvement log

This is the oral history of the engine: what we tried, what landed, what it was worth, and what we already know does not work. It is written for the next person (or agent) who will hill-climb Elo. The process itself lives in the README under **Improving the engine**. This file is the memory.

Elo numbers here are almost always **self-play against the immediately previous accepted version**, not CodinGame ladder rating and not a running total. They do not add. A +400 jump in January 2024 and a +5 pass in 2026 are not the same kind of event: the first is "the search finally knows which move to try," the second is "this is still a real gain on a strong baseline." Time controls also change. Early Python numbers come from `faceoff` scripts. C++ SPRT is usually 20 ms/move, sometimes 95 ms (the CodinGame later-move budget), sometimes equal-depth 4 with eval pruning off.

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

**Nets**

- Sparse-199 replace: −277 at depth 4
- Residual trained on 20 ms WDL: worse leaf than HCE
- Fat MiniNet (H=128) at 20 ms: hundreds of Elo lost (NPS collapse)
- Wider embedding D on old HCE-only depth-6 labels: ~0 Elo at equal depth (mixer stays tiny; extra concat unused)
- Holdout MAE / correlation without a move-level or Elo gate: a net can fit scores and play the same moves
- Early-game MiniNet teacher (D=8, H=8, extra low-ply data): about **+10 Elo at depth 4**, ~**+1.5 Elo at 20 ms**, killed as not a +5 timed win. Better leaf, ~5% NPS tax. Not a ship.
- Training on mates and fail-highs qsearch never asks the net about

**Process failures (the kind that fake a pass)**

- SPRT against a Prev you just weakened
- SPRT against old HCE after MiniNet is the ship
- Two ideas in one SPRT
- Shipping `codingame_nnue.cpp` from an unfrozen Dev
- FMA in MiniNet (disagrees with scalar)
- Instance-sized `1<<18` tables on Windows

---

## 12. How the numbers sit together

A honest running story, not a sum:

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
    **+38** at 95 ms. Official ship bar is now 95 ms.

CodinGame rank is a different axis. Legend HCE got us into the league. MiniNet
moved 82 → 68. The round-4 bundle is the current ship. Absolute ladder Elo is
noisy and not what SPRT measures.

---

## 13. Where the code is now

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

When you land something, freeze Prev, port the CG file, minify, and add a short section here. When you fail, add a line to section 11. The log is only useful if the graves stay marked.

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
