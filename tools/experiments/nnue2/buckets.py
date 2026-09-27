#!/usr/bin/env python3
"""Bucketed V2 holdout metrics (the nnue2 critique's leafproxy.py, as a reusable tool).

For every model: loss vs the shipped static eval and the correlation with the depth-14 label on
non-mate rows (|label| < 8000), per bucket: all / ply<20 / ply 20-40 / ply>=40 / side to move can
capture (win-in-one on its forced board, or on any live board when free) / |label| 4000-8000, plus
the critique's other buckets and its proposed gate rows (ply >= 20, not a random-prefix row).

  buckets.py NAME [NAME ...] [--device cpu|dml] [--threads 2]
NAME is shipped, hce, or a run in datasets/nnue2/probe/<NAME>.{json,pt} (probe.py arms r10/pat/patact
or gen_nnue.py's gen arm). Predictions are cached in datasets/nnue2/probe/v2pred/<NAME>.npy.
Writes datasets/nnue2/probe/buckets.json (merged) and a "buckets" entry in each gen run's json.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_nnue  # noqa: E402
import probe  # noqa: E402

import eval_data  # noqa: E402
import nnue_train_blend as ntb  # noqa: E402

K = 1600.0
CACHE = probe.OUT / "v2pred"
MAIN = ["all", "ply<20", "ply20-40", "ply>=40", "stm_capture_avail", "4000-8000"]


def holdout():
    rec, _, hold = ntb.load_all([str(p) for p in probe.DATA], None)
    r = rec[np.flatnonzero(hold)]
    del rec
    return r


def bucket_masks(r, x):
    s = r["search"].astype(np.float64)
    d = np.array([(p // probe.POW3) % 3 for p in range(probe.N_PAT)])

    def win1(owner):
        w = np.zeros(probe.N_PAT, bool)
        for line in eval_data.WIN_LINES:
            c = d[:, list(line)]
            w |= ((c == owner).sum(1) == 2) & ((c == 0).sum(1) == 1)
        return w
    w1m, w1t = win1(1), win1(2)
    cells = x[:, :81].astype(np.int64).reshape(-1, 9, 9)
    sup = x[:, 81:90].astype(np.int64)
    con = x[:, 90].astype(np.int64)
    pat = (cells * probe.POW3).sum(2)
    live = sup == 0
    mine_w1 = w1m[pat] & live
    opp_w1 = w1t[pat] & live
    forced = con < 9
    stm_cap = np.where(forced, mine_w1[np.arange(len(con)), np.clip(con, 0, 8)], mine_w1.any(1))
    comp_opp = np.zeros_like(live)
    for line in eval_data.WIN_LINES:
        line = list(line)
        for i, m in enumerate(line):
            o = [line[j] for j in range(3) if j != i]
            comp_opp[:, m] |= (sup[:, o[0]] == 2) & (sup[:, o[1]] == 2)
    latent = (opp_w1 & comp_opp).any(1)
    randpre = (r["flags"] & eval_data.F_RESULT) == 0
    ply = r["ply"].astype(np.int64)
    a = np.abs(s)
    return {
        "all": np.ones(len(s), bool),
        "ply<20": ply < 20, "ply20-40": (ply >= 20) & (ply < 40), "ply>=40": ply >= 40,
        "stm_capture_avail": stm_cap, "4000-8000": (a >= 4000) & (a < 8000),
        "no_stm_capture": ~stm_cap, "opp_latent_capture": latent, "free_move": ~forced, "forced": forced,
        "random_prefix": randpre, "played": ~randpre,
        "|s|<1000": a < 1000, "1000-4000": (a >= 1000) & (a < 4000), "mate": a >= 20000,
        "gate:ply>=20,played": (ply >= 20) & ~randpre,
        "gate:ply20-40,played": (ply >= 20) & (ply < 40) & ~randpre,
        "gate:ply>=40,played": (ply >= 40) & ~randpre,
    }


def load_model(name, fx):
    meta = json.loads((probe.OUT / f"{name}.json").read_text())
    a = meta["args"]
    if a["arm"] == "gen":
        m = gen_nnue.build(fx, a)
    elif a["arm"] == "r10":
        m = probe.R10(fx, a["A"], a["L1"])
    else:
        m = probe.Pat(fx, a["width"], a["L1"], active=a["arm"] == "patact")
    m.load_state_dict(torch.load(probe.OUT / f"{name}.pt", map_location="cpu"))
    return m.to(fx.dev).eval(), meta


def predict(name, x, dev, r, sym=0):
    if name == "shipped":
        return r["static_eval"].astype(np.float64)
    if name == "hce":
        return r["hce"].astype(np.float64)
    CACHE.mkdir(parents=True, exist_ok=True)
    pt = probe.OUT / f"{name}.pt"
    c = CACHE / (f"{name}.npy" if not sym else f"{name}.g{sym}.npy")
    if c.exists() and c.stat().st_mtime > pt.stat().st_mtime:
        return np.load(c).astype(np.float64)
    fx = probe.Feats(dev)
    m, meta = load_model(name, fx)
    if sym:  # the holdout under symmetry `sym` (gen_nnue.augment_np's maps, checked against transform_states)
        xs = x[:, gen_nnue.COLPERM[sym]]
        xs[:, 90] = probe.CONMAP[sym][x[:, 90]]
        x = xs
    X = torch.from_numpy(np.ascontiguousarray(x))
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32768):
            xb = X[i:i + 32768].to(dev)
            e = m.fast(xb) if meta["args"]["arm"] == "gen" else m(*fx.split(xb))
            out.append(e.cpu().numpy())
    e = np.concatenate(out).astype(np.float32)
    np.save(c, e)
    return e.astype(np.float64)


def st_sym(name, x, dev, r, g):
    return predict(name, x, dev, r, sym=g)


def metrics(e, s, masks, base):
    y = 1 / (1 + np.exp(-s / K))
    nm = np.abs(s) < 8000
    out = {}
    for bn, b in masks.items():
        p = 1 / (1 + np.exp(-e[b] / K))
        L = float(((p - y[b]) ** 2).mean())
        bb = b & nm
        c = float(np.corrcoef(e[bb], s[bb])[0, 1]) if bb.sum() > 100 else float("nan")
        out[bn] = dict(frac=float(b.mean()), loss=L, vs_shipped_pct=100 * (L / base[bn] - 1) if base else 0.0, corr=c)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--symgap", action="store_true", help="also score the 7 transformed holdouts (symmetry gap)")
    args = ap.parse_args()
    probe.below_normal_priority()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    r = holdout()
    x = probe.compact(r)
    s = r["search"].astype(np.float64)
    masks = bucket_masks(r, x)
    print(f"V2 holdout: {len(r):,} rows ({time.time() - t0:.0f}s)", flush=True)
    dev = gen_nnue.get_dev(args.device)
    names = ["shipped"] + [n for n in args.names if n != "shipped"]
    res = {}
    base = None
    for n in names:
        e = predict(n, x, dev, r)
        res[n] = metrics(e, s, masks, base)
        if n == "shipped":
            base = {bn: v["loss"] for bn, v in res[n].items()}
            res[n] = metrics(e, s, masks, base)
        nm = np.abs(s) < 8000
        st = r["static_eval"].astype(np.float64)
        ex = dict(std=float(e.std()), scale_to_shipped=float(st.std() / e.std()),
                  scale_to_shipped_nonmate=float(st[nm].std() / e[nm].std()),
                  calib_slope_nonmate=float(np.polyfit(e[nm], s[nm], 1)[0]))
        if args.symgap and n not in ("hce",):
            gaps = []
            for g in range(1, 8):
                eg = st_sym(n, x, dev, r, g) if n != "shipped" else None
                if eg is not None:
                    gaps.append(float(np.abs(eg - e).mean()))
            if gaps:
                ex["symgap_mean_abs"] = float(np.mean(gaps))
        res[n]["extra"] = ex
        print(f"{n} scored ({time.time() - t0:.0f}s)", flush=True)
    sh = res["shipped"]
    for n, m in res.items():
        if n == "shipped":
            m["extra"]["symgap_mean_abs"] = 319.0  # measured with datagen label on 35k x 8 (critique)
        m["pass_B"] = bool(m["ply20-40"]["corr"] >= 0.75 and m["ply>=40"]["corr"] >= 0.50)
        m["pass_critique_gate"] = bool(
            m["gate:ply>=20,played"]["vs_shipped_pct"] <= -15
            and m["gate:ply20-40,played"]["corr"] >= sh["gate:ply20-40,played"]["corr"] + 0.05
            and m["gate:ply>=40,played"]["corr"] >= sh["gate:ply>=40,played"]["corr"] + 0.05)
    path = probe.OUT / "buckets.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    allres.update(res)
    path.write_text(json.dumps(allres, indent=1))
    for n in names:
        j = probe.OUT / f"{n}.json"
        if j.exists():
            d = json.loads(j.read_text())
            if d.get("arm") == "gen":
                d["buckets"] = res[n]
                j.write_text(json.dumps(d, indent=1))
    # table
    w = max(len(n) for n in names) + 1
    head = "".join(f"{b + ' (' + format(masks[b].mean(), '.0%') + ')':>22s}" for b in MAIN)
    print(f"\n{'model':{w}s}{head}   pass(B) gate")
    for n in names:
        m = res[n]
        cells = "".join(f"{m[b]['vs_shipped_pct']:+8.1f}% / {m[b]['corr']:.3f}".rjust(22) for b in MAIN)
        print(f"{n:{w}s}{cells}   {'yes' if m['pass_B'] else 'no':>7s} {'yes' if m['pass_critique_gate'] else 'no'}")
    print(f"\n(cell: loss vs shipped / non-mate correlation with the label; shipped absolute losses: "
          + ", ".join(f"{b} {sh[b]['loss']:.5f}" for b in MAIN) + ")")
    print(f"\n{'model':{w}s}{'std':>7s}{'S (all)':>9s}{'S (nonmate)':>12s}{'calib slope':>12s}{'sym gap':>9s}"
          f"{'gate rows loss':>15s}{'gate c20-40':>12s}{'gate c>=40':>11s}")
    for n in names:
        ex, m = res[n]["extra"], res[n]
        sg = ex.get("symgap_mean_abs")
        print(f"{n:{w}s}{ex['std']:7.0f}{ex['scale_to_shipped']:9.3f}{ex['scale_to_shipped_nonmate']:12.3f}"
              f"{ex['calib_slope_nonmate']:12.3f}{(f'{sg:.0f}' if sg is not None else '-'):>9s}"
              f"{m['gate:ply>=20,played']['vs_shipped_pct']:+14.1f}%{m['gate:ply20-40,played']['corr']:12.3f}"
              f"{m['gate:ply>=40,played']['corr']:11.3f}")


if __name__ == "__main__":
    main()
