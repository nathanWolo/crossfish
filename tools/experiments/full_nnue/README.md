# Full-evaluation NNUE experiment (round ten, not shipped)

Code for improvement log section 52: a Stockfish-style NNUE replacing
HCE + MiniNet + macro. It lost at equal depth in every configuration, so
nothing here is used by the engine. It is kept so the direction can be
resumed without rebuilding the pipeline.

| File | Role |
| --- | --- |
| `train_full_nnue.py` | Trainer (PyTorch). 199 sparse features per perspective, dual accumulator, `--A`, `--L1`, `--loss wdl|huber`, `--subsample N`. Reports the current static eval as a baseline on the same holdout and writes `<out>.bin` for C++. |
| `full_nnue_float.hpp` | Float reference inference; reads `FULLNNUE_PATH`. Matches PyTorch to 1 eval unit. |
| `wire_into_dev.py` | Patches `crossfish_dev.hpp` so qsearch stand-pat and the interior static eval use the NNUE (plus correction history). Copy `full_nnue_float.hpp` to `cpp_impl/full_nnue.hpp` first. Never commit the patched Dev. |
| `full_annotate.cpp` | Writes the current full static eval into a dump's float field (the baseline column). Build next to `test_bots.cpp` with `-Icpp_impl`. |
| `make_leaf_engine.py`, `leaf_dump.cpp` | Sample positions where qsearch is called during self-play (`leaf_dump GAMES MS OUT [threshold/4096]`). |
| `scale.sh` | The data-scaling study. |

Pipeline used:

```bash
cpp_impl/bin/test_bots dump nnue 30000 10 datasets/nnue_pos2.bin        # 1.59M self-play positions
cpp_impl/bin/test_bots dump relabel 12 1587147 datasets/nnue_pos2.bin \
    datasets/full_play_d12.bin current                                  # full-engine teacher, depth 12
full_annotate datasets/full_play_d12.bin datasets/full_play_d12_base.bin
python3 tools/experiments/full_nnue/train_full_nnue.py \
    --data datasets/full_play_d12_base.bin --out /tmp/net.pt --A 256 --epochs 20
```

Relabel with the unpatched Dev: a `test_bots` built from the wired Dev would
label with the experimental net itself (it exits with "FULLNNUE_PATH not set"
when the variable is missing, which is how this was caught).
