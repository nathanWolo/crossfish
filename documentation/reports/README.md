# Research reports

Full write-ups of the NNUE training studies, each with its figures in `fig/` beside it.

- [Round 14: new labels, cheap training ideas and fine-tune length](round14/REPORT.md)
  (3-4 October 2026; report final 4 October, addendum 8 October). A six-stage pre-registered
  funnel tested four levers on the 35,243-parameter net r13w_20: labels from its own search,
  CodinGame ladder positions, fine-tune length and seven trainer ideas from Stockfish, bullet and
  Viridithas (48 nets, about 450,000 games). The shipped net `r14_d5_final_s2_rs`, from a WDL
  contradiction filter plus a power-2.5 loss, beat r13w_20 by +9.1 ± 5.9 Elo over 4,000 games at
  90 ms. New labels lost −10.6 ± 2.5 Elo although every aggregate offline metric preferred them, longer
  fine-tunes gave nothing measurable, and offline metrics could not certify the changes that
  mattered. The addendum reports the offline-only architecture track (wider encoder, wider head,
  macro-board contexts).
- [Scaling study: data, fine-tune length and encoder width (generation 15)](scaling_study/REPORT.md)
  (4-7 October 2026; report final 8 October). Warm fine-tunes of r13w_20 over nested data levels
  and lengths from 150M to 4.8G rows, a function-preserving widening of the encoder from 64 to 128
  and a scratch width ladder (118 nets, 933,000 games), with offline loss laws fitted to plan later
  generations. Along length and data, offline gains did not reach play; only encoder capacity moved
  it (widened enc128 over enc64 +7.3 Elo, Holm p 0.009). The generation-15 operating point lost to
  the shipped round-14 net by −7.7 ± 4.5 Elo, so nothing from the study shipped, and no law was
  validated for extrapolation along any axis.

Both studies were pre-registered: the design, hypotheses and decision rules were frozen before
the results they judge, each report lists its deviations, and "±" is a 95% half-width. The
records, the numbers scripts that compute the reports' tables and figures, and the training and
game logs live under `datasets/nnue2/` (`r14/` and `scaling/`), which is not in git; paths inside
the reports are relative to those directories.
