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

CodinGame rank is a different axis. Legend HCE got us into the league. MiniNet moved 82 → 68. The September speed patch is the current ship. Absolute ladder Elo is noisy and not what SPRT measures.

---

## 13. Where the code is now

| Piece | Role |
| --- | --- |
| `crossfish_prev.hpp` | Frozen last accepted engine (the #10 speed engine, `c2aae17`) |
| `crossfish_dev.hpp` | Current landed Dev (adds #14 correction history + log LMR) |
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
