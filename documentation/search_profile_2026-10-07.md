# Search profile, 7 October 2026

Baseline: the section 67 ProbCut freeze. Both engines in the worktree
were identical before profiling. The event counters were added to a
temporary copy of Dev and did not change the repository's engine code.

## Workloads and method

- **Random playouts:** `bench_ab walk 10 40 90`, ten legal random scripts,
  400 searches per engine. Counters cover only Dev.
- **Engine play:** ten scripts with six random opening moves followed by
  Prev's 20 ms choices. Each position was then searched by Dev and Prev
  at 90 ms with persistent engine state (`walk 10 50 90`); 498 searches
  per engine. Counters cover only Dev. This is the more relevant column
  for choosing an experiment.
- **Runtime samples:** `g++ -O3 -g -pg` with the normal AVX2 and BMI flags
  on a five-game 90 ms walk, read with `gprof`. The profiler instruments
  function entry, so its percentages are an approximate ranking of
  costs, not precise production wall-time shares. A second run on
  engine-play scripts confirmed the broad ranking.

The counter build adds increments to the hot path and therefore reaches
slightly less depth than uninstrumented Prev. The engine-play run
completed 498 searches and counted 192.1 million nodes. Its counts
are descriptive rates for that workload, not an Elo test or proof that
removing a prune is safe.

## Where time goes

| Named function or group | Sampled CPU share, engine-play walk |
| --- | ---: |
| NNUE forward pass (`b64::eval_avx`) | 25.6% |
| `search` body, including inlined work | 30.2% |
| Make and unmake | 16.8% |
| NNUE accumulator synchronization | 7.9% |
| Move scoring | 5.6% |
| `qsearch` body, excluding callees | 5.3% |
| Move-key sorting | 3.0% |

The 20 ms engine used to generate these scripts was also included in
the sample. A random-playout run independently put NNUE at 25.7% and
the search bodies at 27.5%, preserving the broad ranking.

## Pruning and search rates

Phase is the *position's* move count, including nodes below the root.
`TT cut` includes exact, lower-bound, and upper-bound returns from
`search`, but not leaf TT returns. `RFP` is a reverse-futility return as
a share of all `search` calls. `ProbCut` is a successful cut as a share
of nodes passing the depth/window eligibility check. These are
different denominators.

| Ply | `search` calls | TT cut | RFP | ProbCut cut | First-move beta cut | LMR retry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–11 | 1.85M | 4.5% | 26.5% | 2.9% | 91.6% | 1.8% |
| 12–27 | 17.15M | 3.9% | 38.2% | 12.0% | 79.9% | 3.0% |
| 28–44 | 32.69M | 5.8% | 36.0% | 25.9% | 77.4% | 3.5% |
| 45+ | 29.08M | 16.6% | 20.8% | 26.6% | 89.5% | 3.6% |

`First-move beta cut` is the first searched move's share of ordinary
move-loop beta cutoffs; the mean is 82.8% across 24.1 million cutoffs.
`LMR retry` is the share of reduced moves that triggered a full-depth
null-window search; 45.4 million moves were reduced and 1.50 million
retried. Of 156.8 million ordered move slots visited, futility skipped
22.8 million (14.6%). IIR ran 6.48 million times.

ProbCut ran at 5.66 million of 11.72 million eligible nodes and
made 11.61 million move probes, producing 2.75 million cuts. Its
success rate rises sharply after the opening:

| Ply | Runs / eligible | Cuts / eligible | Cuts / probe | Cut by first / second / third probe |
| --- | ---: | ---: | ---: | ---: |
| 0–11 | 15.9% | 2.9% | 6.8% | 7,282 / 519 / 202 |
| 12–27 | 42.1% | 12.0% | 11.5% | 185,564 / 25,328 / 9,814 |
| 28–44 | 50.4% | 25.9% | 25.1% | 1,139,256 / 121,294 / 49,663 |
| 45+ | 50.5% | 26.6% | 27.8% | 1,090,421 / 80,630 / 36,160 |

The TT-related skip suppresses most early ProbCut runs; 76% of those
that do run have a legal TT move. The low early yield also appeared
in the independent random playout scripts (2.7% cuts per eligible
node before ply 12). By contrast, late random playouts have only
8.5% TT cuts and 34.8% RFP returns, versus 16.6% and 20.8% in
engine play. Late random positions alone would mislead tuning.

Quiescence visited 64.5 million nodes. Stand-pat failed high at
52.6 million (81.4%). Delta pruning returned at 419,000 (0.65%);
it did not fire at all before ply 12, and reached 2.08% of quiescence
nodes at ply 45+. Of 11.27 million capture-list generations,
7.84 million captures were searched.

The shipped 256 KiB direct-mapped NNUE eval cache hit 32% of evaluation
requests in a separate five-game engine-play walk. A shadow-cache
probe estimated how much capacity could help:

| Cache | Hit rate in shadow probe |
| --- | ---: |
| 128 KiB | 27.7% |
| 256 KiB | 31.3% |
| 512 KiB | 35.4% |
| 1 MiB | 39.3% |
| 2 MiB | 42.6% |

The shadow lookups slowed the instrumented search by about 19%, so
their hit rates describe a shallower search tree. They support a
timed capacity experiment; they do not themselves show a speedup.

Late tactical shortcuts are substantial. The ordinary move loop
returned a proven immediate global loss without recursing on 13.37
million moves, plus 1.12 million proven forced global wins. Immediate
global loss cases were 9.7% of searched moves at plies 28–44 and
18.6% at ply 45+. The current code makes and unmakes each of these
moves before recognizing the loss.

## Experiments suggested by the profile

1. Skip opening ProbCut before ply 12. The opening has roughly
   7,800 cuts across 1.85 million `search` calls while still spending
   time on probes. Measure timed depth and Elo; the low yield alone
   does not prove the probes are expendable.
2. Recognize a subset of inevitable immediate global-loss moves
   from the parent position before make/unmake. This could preserve
   the exact tree while avoiding work on millions of late moves.
   Prove score and node identity before timing it.
3. Optimize NNUE evaluation or accumulator synchronization if a
   concrete kernel change measures faster. The profile gives these
   more headroom than move sorting. Changes to a shared NNUE header
   must still leave Prev at the accepted behavior during an SPRT.

The first suggestion was screened: Dev skipped ProbCut before ply 12.
It passed `make test` and used 1.5% fewer nodes on 120 fixed-depth-10
positions, but lost 0.053 ply in a 20 ms persistent-engine walk over
400 searches. The timed search did not support a match trial, so Dev
was restored to the freeze.

The second suggestion was also screened. Dev detected one sufficient
condition for an immediate global loss before make: the move sent the
opponent to a *different*, live, globally winning target miniboard,
without first winning the game. A validation build made those moves
and checked that the original post-make test agreed; it found no
disagreement on a 500-search engine-play fixed-depth walk. `make test`
passed, and scores and node counts were identical to Prev over 80
fresh depth-8 positions and 498 engine-play depth-10 searches. But
the fixed-depth walk took about 2.5% longer and a five-game 90 ms
engine-play walk lost 0.344 ply over 250 searches. The added
per-move check outweighed the saved makes. Dev was restored.

A Dev-only two-level cache (the shipped 256 KiB cache followed by a
1 MiB cache on misses) passed `make test` and searched the same tree
on 80 depth-8 positions. On a persistent depth-10 walk it took 5.9%
longer. The extra lookup cost more than the recovered NNUE evaluations,
so it was discarded.

A Dev-only direct 1 MiB cache avoided the second lookup. It passed
`make test` and the 80-position tree-equivalence check, but took
3–4% longer on random and engine-play fixed-depth-10 walks.
Two 90 ms engine-play walks with both the old and new cache lines
prefetched gained 0.380 ply over 250 searches and 0.234 ply over
500 searches on fresh scripts. Prefetching only the 1 MiB cache cut
the fixed-depth slowdown to 1.7%, but the corresponding fresh
500-search 90 ms walk lost 0.146 ply despite 2.2% higher NPS.
The timed-depth evidence is inconsistent and no SPRT was run.
Dev was restored to the freeze.

## Best remaining direction

The NNUE forward pass is the clearest speed target left by this
profile. It occupies about a quarter of instrumented CPU time; a
bit-identical 20% reduction in its cost would save roughly 5% of
that sampled work before other effects. A kernel candidate should
first prove identical evaluations and search trees, then demonstrate
a repeated paired speed gain before an SPRT.

For search changes, measure *wrong-cut rates* for RFP and ProbCut on
engine-play positions by phase before retuning margins. The large
late-game differences between random and engine-play TT and RFP rates
show why frequency counts on random positions are a poor tuning
target. Move-score tweaks have less apparent headroom: the first
searched move already causes 82.8% of ordinary beta cutoffs.
