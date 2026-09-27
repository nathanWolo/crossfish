#!/usr/bin/env python3
"""Do the laptop worker's pairs of a round robin agree with the desktop's? (fast_pair.py rr-run --worker)
Written for the full-data play stage's nnue_full_20ms; the long-training play stage used the same
per-machine fits for nnue_long_20ms.

  rr_machine_check.py NAME [--anchor noop] [--json FILE] [--root DIR]

Fits the ratings from the desktop's pairs alone (round_robin.wls, anchor ANCHOR), then compares every
laptop pair's Elo with the difference those ratings predict: z = (observed - predicted) / sqrt(se_obs^2 +
se_pred^2), where se_pred comes from the desktop fit's covariance. Also the chi-square of the laptop pairs,
and the fitted scale k in observed = k * predicted over the laptop pairs (1 = same Elo scale) with and
without the pairs against the anchor. And the reverse (laptop-only fit predicting desktop pairs) when the
laptop's pairs connect every engine to the anchor. Then the joint test (one set of ratings for both
machines against one set per machine) and each machine's own ratings. Needs numpy.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import round_robin as rr  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("name")
ap.add_argument("--anchor", default="noop")
ap.add_argument("--json", help="write the numbers here")
ap.add_argument("--root", help="tournaments directory (round_robin.py --root; default datasets/eval2/rr)")
args = ap.parse_args()
if args.root:
    rr.RR_ROOT = Path(args.root).resolve()
ANCHOR = args.anchor


def save(out):
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


Z = rr.Z95
plan = rr.load_plan(args.name)
d = rr.tour_dir(args.name)
names = list(plan["candidates"])
obs = {"desktop": [], "laptop": []}
for p in plan["pairs"]:
    r = rr.last_result(d / f"{rr.pair_key(p)}.log")
    if not r:
        continue
    elo, ci, se, reg = rr.pair_estimate(r[2])
    obs["laptop" if p.get("where") == "worker" else "desktop"].append((p["a"], p["b"], elo, se))


def fit_cov(rows):
    """wls ratings and their covariance (anchor ANCHOR)."""
    free = [n for n in names if n != ANCHOR]
    idx = {n: i for i, n in enumerate(free)}
    X = np.zeros((len(rows), len(free)))
    y = np.array([r[2] for r in rows]); w = np.array([1 / r[3] ** 2 for r in rows])
    for k, (a, b, _, _) in enumerate(rows):
        if a in idx: X[k, idx[a]] = 1
        if b in idx: X[k, idx[b]] = -1
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ (X.T @ (w * y))
    return idx, beta, cov


if not obs["desktop"] or not obs["laptop"]:
    raise SystemExit(f"{args.name}: {len(obs['desktop'])} desktop and {len(obs['laptop'])} worker pairs with results;"
                     " a machine comparison needs both")
out = {}
for fit_on, test_on in (("desktop", "laptop"), ("laptop", "desktop")):
    try:
        idx, beta, cov = fit_cov(obs[fit_on])
    except np.linalg.LinAlgError:
        print(f"{fit_on}-only fit: the {fit_on}'s pairs do not connect every engine; skipped")
        continue
    zs, pred, got, wts = [], [], [], []
    print(f"{fit_on}-only fit ({len(obs[fit_on])} pairs) predicting the {len(obs[test_on])} {test_on} pairs:")
    for a, b, y, se in obs[test_on]:
        g = np.zeros(len(idx))
        if a in idx: g[idx[a]] = 1
        if b in idx: g[idx[b]] = -1
        mu, sp = float(g @ beta), math.sqrt(float(g @ cov @ g))
        z = (y - mu) / math.hypot(se, sp)
        zs.append(z); pred.append(mu); got.append(y); wts.append(1 / (se ** 2 + sp ** 2))
        print(f"  {a:>14} vs {b:<14} observed {y:+7.1f}  predicted {mu:+7.1f} +/- {Z * sp:4.1f}  z {z:+5.2f}")
    zs, pred, got, wts = map(np.array, (zs, pred, got, wts))
    chi2 = float((zs ** 2).sum())
    k_all = float((wts * pred * got).sum() / (wts * pred ** 2).sum())
    nn = np.array([ANCHOR not in (a, b) for a, b, _, _ in obs[test_on]], dtype=bool)  # without the anchor's pairs
    k_nn = float((wts[nn] * pred[nn] * got[nn]).sum() / (wts[nn] * pred[nn] ** 2).sum()) if nn.any() else float("nan")
    se_k = float(1 / math.sqrt((wts[nn] * pred[nn] ** 2).sum())) if nn.any() else float("nan")
    print(f"  chi2 {chi2:.1f} on {len(zs)} pairs; max |z| {abs(zs).max():.2f}; mean z {zs.mean():+.2f};"
          f" scale k {k_all:.3f} (all), {k_nn:.2f} +/- {Z * se_k:.2f} (without the {ANCHOR} pairs)")
    out[f"{fit_on}_predicts_{test_on}"] = dict(pairs=len(zs), chi2=round(chi2, 2), max_abs_z=round(float(abs(zs).max()), 2),
                                               mean_z=round(float(zs.mean()), 3), k_all=round(k_all, 4),
                                               k_without_noop=round(k_nn, 3), k_without_noop_ci95=round(Z * se_k, 3))
save(out)


# The proper joint test: one set of ratings for both machines (the tournament's fit) against separate
# ratings per machine. Under "the machines agree", chi2(shared) - chi2(desktop) - chi2(laptop) is chi-square
# with (number of free ratings) degrees of freedom.
def chi2_of(rows):
    _, c2, dof = rr.wls(names, ANCHOR, rows)
    return c2, dof


c_all, d_all = chi2_of(obs["desktop"] + obs["laptop"])
c_d, d_d = chi2_of(obs["desktop"])
c_l, d_l = chi2_of(obs["laptop"])
lr, ldof = c_all - c_d - c_l, d_all - d_d - d_l
try:
    from scipy.stats import chi2 as _chi2
    pval = float(_chi2.sf(lr, ldof))
except ImportError:  # Wilson-Hilferty
    zz = ((lr / ldof) ** (1 / 3) - (1 - 2 / (9 * ldof))) / math.sqrt(2 / (9 * ldof))
    pval = 0.5 * math.erfc(zz / math.sqrt(2))
print(f"joint test: shared fit chi2 {c_all:.1f} on {d_all} dof; desktop-only {c_d:.1f} on {d_d}; laptop-only"
      f" {c_l:.1f} on {d_l}; machine difference chi2 {lr:.1f} on {ldof} dof, p = {pval:.2f}")
out["joint"] = dict(shared_chi2=round(c_all, 2), shared_dof=d_all, desktop_chi2=round(c_d, 2), desktop_dof=d_d,
                    laptop_chi2=round(c_l, 2), laptop_dof=d_l, machine_chi2=round(lr, 2), machine_dof=ldof,
                    p=round(pval, 3))
save(out)

# Each machine's own ratings next to the tournament's.
fits = {k: rr.wls(names, ANCHOR, rows)[0] for k, rows in
        (("both", obs["desktop"] + obs["laptop"]), ("desktop", obs["desktop"]), ("laptop", obs["laptop"]))}
print(f"\n{'engine':<14} {'both':>15} {'desktop only':>15} {'laptop only':>15} {'laptop - desktop':>18}")
out["ratings"] = {}
for n in sorted(names, key=lambda n: -(fits["both"][n][0] or 0)):
    if any(fits[k][n][0] is None for k in fits):  # an engine one machine never played
        print(f"{n:<14} {fits['both'][n][0] if fits['both'][n][0] is not None else float('nan'):+7.1f}"
              "  (not rated by both machines)")
        continue
    cells = [f"{fits[k][n][0]:+7.1f} +/-{Z * (fits[k][n][1] or 0):4.1f}" for k in ("both", "desktop", "laptop")]
    dl = fits["laptop"][n][0] - fits["desktop"][n][0]
    sd = math.hypot(fits["laptop"][n][1] or 0, fits["desktop"][n][1] or 0)
    print(f"{n:<14} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15} {dl:+8.1f} (z {dl / sd if sd else 0:+.2f})")
    out["ratings"][n] = {k: [round(fits[k][n][0], 2), round(Z * (fits[k][n][1] or 0), 2)] for k in fits}
save(out)
