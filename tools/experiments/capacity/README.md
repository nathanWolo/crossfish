# Eval capacity probes (improvement log section 55)

Offline probes behind the architecture study of section 55: how much held-out
loss more MiniNet capacity, a pattern-feature NNUE residual or a 9-token
transformer residual buy on the corrected 4.8M-position data, with the same
split, target and K (1,600) as the `F_full` run. Nothing here is used by the
engine. Offline loss did not rank play this round (section 55), so these
numbers show how well an architecture fits the data, not what it would gain
in play; play decides.

| File | Role |
| --- | --- |
| `capacity_probe.py` | Trains one arm (`--arm`: a, a2, a2s, b, c, d, e, f, fns, g, fj or gj, see its docstring; `a2s` is a2 with a side-to-move input, `fns` is f without one) on top of `tools/nnue_train_blend.py`; writes `datasets/eval2/capacity/<arm>.{pt,json}`. |
| `quant_eval.py` | Held-out loss after the shipped 256-centroid re-clustering; `--export ARM` writes a checkpoint the header emitter accepts. |
| `summarize.py` | Table of every probe's result. |
| `run_all.sh`, `run_extra*.sh` | The queues that were run (DirectML GPU, two CPU threads). |

Main results (held-out win-probability loss against the shipped eval, float
unless stated):

| Arm | Change | vs shipped |
| --- | --- | ---: |
| a | reproduces F_full (D16/H8, lr 3e-4) | -7.0% |
| a2 | duplicate hidden units re-initialised | -11.5% (played: H0 at 90 ms) |
| c | D16/H16 | -11.7% |
| f | 9-miniboard-token transformer residual | -20.0% |
| g | pattern NNUE residual, width 128 | -13.3% |
| a_lr3e3 | the shipped D16/H8 shape at lr 3e-3 | -17.4% after packing |
| d_lr3e3 | D32/H32 at lr 3e-3 | -19.1% |

At lr 3e-3 the larger nets start overfitting within 3-4 epochs. The
transformer (0.64M new parameters) and the NNUE (22.7M) do not fit the
CodinGame character budget; a rough speed estimate, not measured in play, put
their cost at 100-230 Elo.
