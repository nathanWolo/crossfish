#!/usr/bin/env python3
"""Streaming trainer for gen_nnue.py's pattern-generator ("B") nets on ALL the depth-8 self-play data.

Same model (gen_nnue.Gen), loss (win-probability MSE at K 1600 + 0.1 x the PSQT lane's), D4
augmentation (gen_nnue.augment_np's stratified per-batch maps), AdamW (first layer no decay, dense
1e-5) with 2% warmup and cosine to 1e-5, gradient clip 1.0, batch 16384 and log format as gen_nnue.py
train. The difference: the 176M depth-8 rows (22.5 GB) never sit in RAM. They are read from disk in
blocks of 32,768 records, every pass in a fresh random block order; 128 blocks at a time (about 4.2M
rows, from all four files and the eval2 training rows) are filtered, compacted to the 91-byte rows
(compact_fast, identical to probe.compact on every row: `survey` checks all 176M), labeled,
shuffled row by row, augmented, and handed to the GPU loop by a prefetch thread while the next 128
blocks load on a thread pool. The eval2 training rows (V2's other 90%) are blocks of the same stream.

Holdouts:
  V2   exactly gen_nnue.py's: 10% of the eval2 games (nnue_train_blend.game_holdout), 482,136 rows;
       "vs shipped" is against their recorded static eval (0.046817), as in every earlier run.
  D8H  1% of the depth-8 GAMES (d8_holdout(): a splitmix64 hash of (file, game id) % 100 == 0),
       never streamed. Scored without the games inside the first 5M records of d8_a.cfdg, the rows
       the lr1e2 baselines trained on, so the baselines are scored on unseen games too.
       "vs shipped" is against the files' static_eval column (the generating engine's evaluate()).

Subcommands:
  survey              scan every d8 record (and eval2): flags, label presence, clamps, sources, plies,
                      games, byte validity, compact_fast vs probe.compact; writes the D8H cache
                      (datasets/nnue2/stream/d8_holdout.npz), per-record position hashes
                      (stream/hash_<file>.u64) and probe/stream_survey.json
  dups [--out dups_sym] distinct positions; D8H and V2 holdout rows whose position occurs (in any of its
                      8 D4 images) in the training rows; stream/OUT.npz and probe/stream_OUT.json
                      (the first round's stream/dups.npz matched D8H rows by exact position only)
  bench [--passes P]  loader only (no GPU): rows/s, and one full pass's row count and label checksum
                      against the survey's counts
  check               the stream's rows are D4 images of probe.compact(record) with the record's label,
                      stratified like augment_np, and contain no D8H game
  train --run NAME,A,DENSE,LR[,seed=N][,passes=P][,e2rep=R][,steps=N][,forced=0|1][,sharedcon=0|1] [--passes 3]
                      writes datasets/nnue2/probe/<NAME>.{pt,json,log}; one epoch = one pass. e2rep=R streams
                      every eval2 training row R times per pass; steps=N sets the total step budget (the last
                      epoch is then partial). --overwrite replaces an existing <NAME>.log
  eval NAME... [--out stream_eval.json]
                      V2 and D8H metrics (ply buckets, novel rows) of probe checkpoints; merged into
                      probe/OUT. "novel": in no training set of any run, up to symmetry (dups_sym.npz)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import queue  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_nnue  # noqa: E402
import probe  # noqa: E402
from gen_nnue import COLPERM, Gen, AdamW, lr_lambda, n_params, payload, parse_run, get_dev, log  # noqa: E402

import eval_data  # noqa: E402  (probe put tools/ on sys.path)
import nnue_train_blend as ntb  # noqa: E402

ROOT = probe.ROOT
OUT = probe.OUT
SDIR = ROOT / "datasets/nnue2/stream"
D8 = [ROOT / "datasets/nnue2" / f for f in ("d8_a.cfdg", "d8_b.cfdg", "d8_laptop.cfdg", "d8_laptop_b.cfdg")]
REC_BYTES = eval_data.REC.itemsize
BASE_ROWS = 5_000_000  # the lr1e2 baselines trained on d8_a.cfdg records [0, 5M)
K_DEFAULT = 1600.0


# ---------------------------------------------------------------- rows
def compact_fast(raw):
    """(n, 128) uint8 DgRec bytes -> (n, 91) uint8 compact rows, probe.compact's encoding: cells and board
    states from the side to move's view (1 <-> 2 swapped when s[90] != '1'), constraint clipped to 0..9."""
    d = raw[:, :90] - np.uint8(48)
    swap = ((d - np.uint8(1)) < 2) & (raw[:, 90] != 49)[:, None]  # a 1 or a 2, player 2 to move
    out = np.empty((len(raw), 91), np.uint8)
    np.bitwise_xor(d, swap * np.uint8(3), out=out[:, :90])
    out[:, 90] = np.clip(raw[:, 91].astype(np.int16) - 48, 0, 9)
    return out


def sigmoid_target(search, k):
    """gen_nnue's label: sigmoid(search / K) in float64, stored as float32."""
    return (1 / (1 + np.exp(-search.astype(np.float64) / k))).astype(np.float32)


def d8_holdout(game, fidx):
    """1% of the depth-8 games: splitmix64 of (file index + 1) << 32 | game id, % 100 == 0."""
    z = game.astype(np.uint64) + np.uint64(fidx + 1) * np.uint64(1 << 32)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(100)) == 0


HASH_C = np.random.default_rng(20260926).integers(1, 2 ** 63 - 1, size=12, dtype=np.uint64) | np.uint64(1)


def pos_hash(raw):
    """(n, >=96) uint8 rows whose first 92 bytes are the state -> (n,) uint64 position hash (state chars
    0..91: cells, board states, side to move, constraint; s[92] and everything after ignored)."""
    w = np.ascontiguousarray(raw[:, :96]).view(np.uint64)  # (n, 12)
    h = np.zeros(len(raw), np.uint64)
    for i in range(12):
        v = w[:, i] if i < 11 else w[:, i] & np.uint64(0xFFFFFFFF)  # word 11: s[88..91] only
        h = (h ^ (v * HASH_C[i])) * np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(33)
    return h


def read_raw(path, a, b, fh=None):
    """Records [a, b) of a .cfdg as (n, 128) uint8 (plain file reads release the GIL)."""
    n = b - a
    buf = np.empty(n * REC_BYTES, np.uint8)
    f = fh or open(path, "rb")
    try:
        f.seek(a * REC_BYTES)
        got = f.readinto(memoryview(buf))
    finally:
        if fh is None:
            f.close()
    assert got == n * REC_BYTES, (path, a, b, got)
    return buf.reshape(n, REC_BYTES)


def rec_view(raw):
    return raw.reshape(-1).view(eval_data.REC)


def n_records(path):
    return Path(path).stat().st_size // REC_BYTES


# ---------------------------------------------------------------- eval2 (V2 holdout + its training rows)
def load_eval2():
    """probe.load_data (as gen_nnue.load_train_data without extra rows): compact rows, labels, V2 mask."""
    rec, _, hold = ntb.load_all([str(p) for p in probe.DATA], None)
    out = dict(x=probe.compact(rec), search=rec["search"].astype(np.float32), static=rec["static_eval"].astype(np.float32),
               hold=hold, ply=rec["ply"].copy(), flags=rec["flags"].copy())
    del rec
    return out


# ---------------------------------------------------------------- survey
def survey_chunk(fidx, a, b, want_hash):
    raw = read_raw(D8[fidx], a, b)
    r = rec_view(raw)
    fl = r["flags"]
    lab = (fl & eval_data.F_SEARCH) != 0
    s = r["search"][lab]
    st = raw[:, :93]
    digits_ok = ((st[:, :92] >= 48) & (st[:, :92] <= 57)).all(1)
    cells_ok = (st[:, :81] <= 50).all(1)  # '0'..'2'
    sup_ok = (st[:, 81:90] <= 51).all(1)  # '0'..'3'
    stm_ok = (st[:, 90] == 49) | (st[:, 90] == 50)
    xc = compact_fast(raw)
    ref = probe.compact(r)
    g = r["game"]
    hold = d8_holdout(g, fidx)
    out = dict(
        n=len(r), labeled=int(lab.sum()), flags=np.bincount(fl, minlength=256),
        source=np.bincount(r["source"], minlength=8), ply=np.bincount(r["ply"], minlength=128),
        result=np.bincount(r["result"].astype(np.int64) + 1, minlength=3),
        s_min=int(s.min()) if len(s) else 0, s_max=int(s.max()) if len(s) else 0,
        s_clamped=int((np.abs(s) >= 20000).sum()), s_over=int((np.abs(s) > 20000).sum()),
        s_sum=float(s.astype(np.float64).sum()), s_sq=float((s.astype(np.float64) ** 2).sum()),
        st_min=int(r["static_eval"].min()), st_max=int(r["static_eval"].max()),
        gs_eq_search=int((r["game_score"][lab] == s).sum()),
        uttt_flags=int(((fl & (eval_data.F_UTTT_V | eval_data.F_UTTT_Q)) != 0).sum()),
        bad_digits=int((~digits_ok).sum()), bad_cells=int((~cells_ok).sum()), bad_sup=int((~sup_ok).sum()),
        bad_stm=int((~stm_ok).sum()), s92=np.bincount(st[:, 92], minlength=256),
        compact_mismatch=int((xc != ref).any(1).sum()),
        runs=int(1 + (g[1:] != g[:-1]).sum()), first_game=int(g[0]), last_game=int(g[-1]),
        ply_step_bad=int(((g[1:] == g[:-1]) & (r["ply"][1:] != r["ply"][:-1] + 1)).sum()),
        games=np.unique(g), hold_rows=int((hold & lab).sum()),
        hold_unlabeled=int((hold & ~lab).sum()),
        base_hold_rows=int((hold & lab)[: max(0, min(b, BASE_ROWS) - a)].sum()) if fidx == 0 else 0,
    )
    keep = hold & lab
    idx = np.flatnonzero(keep)
    out["hold"] = dict(x=xc[idx], search=r["search"][idx].astype(np.float32), static=r["static_eval"][idx].astype(np.float32),
                       ply=r["ply"][idx].copy(), flags=fl[idx].copy(), game=g[idx].copy(),
                       fidx=np.full(len(idx), fidx, np.int8), recidx=(a + idx).astype(np.int64))
    if want_hash:
        out["hash"] = pos_hash(raw)
    return out


def cmd_survey(args):
    probe.below_normal_priority()
    SDIR.mkdir(parents=True, exist_ok=True)
    log.to(OUT / "stream_survey.log")
    t0 = time.time()
    res = dict(files={}, hold_rule="splitmix64((file+1)<<32 | game) % 100 == 0", base_rows=BASE_ROWS)
    holds = []
    total = sum(n_records(p) for p in D8)
    done = 0
    for fidx, path in enumerate(D8):
        n = n_records(path)
        chunks = [(a, min(n, a + args.chunk)) for a in range(0, n, args.chunk)]
        agg = None
        games, hparts = [], []
        hf = open(SDIR / f"hash_{path.stem}.u64", "wb")
        prev_last = None
        joins = 0
        with ThreadPoolExecutor(args.workers) as ex:
            for i, c in enumerate(ex.map(lambda ab: survey_chunk(fidx, ab[0], ab[1], True), chunks)):
                c["hash"].tofile(hf)
                holds.append(c.pop("hold"))
                games.append(c.pop("games"))
                if prev_last is not None and c["first_game"] == prev_last:
                    joins += 1  # one game split across the chunk boundary: one run, counted twice
                prev_last = c["last_game"]
                del c["hash"]
                if agg is None:
                    agg = c
                else:
                    for k2, v in c.items():
                        if k2 in ("s_min", "st_min"):
                            agg[k2] = min(agg[k2], v)
                        elif k2 in ("s_max", "st_max"):
                            agg[k2] = max(agg[k2], v)
                        elif k2 in ("first_game", "last_game"):
                            pass
                        else:
                            agg[k2] = agg[k2] + v
                done += chunks[i][1] - chunks[i][0]
                if i % 4 == 3 or i == len(chunks) - 1:
                    el = time.time() - t0
                    log(f"survey {path.name}: {chunks[i][1]:,}/{n:,} records; all files {done:,}/{total:,}"
                        f" ({100 * done / total:.1f}%) elapsed {el / 60:.1f}m eta {el / done * (total - done) / 60:.1f}m")
        hf.close()
        ug = np.unique(np.concatenate(games))
        lab = agg["labeled"]
        mean = agg["s_sum"] / max(1, lab)
        f = dict(
            records=agg["n"], labeled=lab, unlabeled=agg["n"] - lab,
            flags={str(i): int(v) for i, v in enumerate(agg["flags"]) if v},
            source={eval_data.SOURCES.get(i, str(i)): int(v) for i, v in enumerate(agg["source"]) if v},
            result={"loss": int(agg["result"][0]), "draw": int(agg["result"][1]), "win": int(agg["result"][2])},
            ply_min=int(np.flatnonzero(agg["ply"])[0]), ply_max=int(np.flatnonzero(agg["ply"])[-1]),
            ply_mean=float((agg["ply"] * np.arange(128)).sum() / agg["n"]),
            search_min=agg["s_min"], search_max=agg["s_max"], search_mean=mean,
            search_std=math.sqrt(agg["s_sq"] / max(1, lab) - mean ** 2),
            search_abs_ge_20000=agg["s_clamped"], search_abs_gt_20000=agg["s_over"],
            static_min=agg["st_min"], static_max=agg["st_max"],
            game_score_equals_search=agg["gs_eq_search"], uttt_flag_rows=agg["uttt_flags"],
            bad_digit_rows=agg["bad_digits"], bad_cell_rows=agg["bad_cells"], bad_board_state_rows=agg["bad_sup"],
            bad_side_to_move_rows=agg["bad_stm"], s92={chr(i): int(v) for i, v in enumerate(agg["s92"]) if v},
            compact_fast_mismatches=agg["compact_mismatch"],
            games=int(len(ug)), game_min=int(ug[0]), game_max=int(ug[-1]),
            game_runs=int(agg["runs"] - joins), ply_step_violations=agg["ply_step_bad"],
            hold_rows=agg["hold_rows"], hold_unlabeled_rows=agg["hold_unlabeled"],
            hold_games=int(d8_holdout(ug, fidx).sum()), train_rows=lab - agg["hold_rows"],
        )
        if fidx == 0:
            f["base_hold_rows"] = agg["base_hold_rows"]
        res["files"][path.name] = f
        log(f"{path.name}: {json.dumps(f)}")
    h = {k2: np.concatenate([d[k2] for d in holds]) for k2 in holds[0]}
    np.savez(SDIR / "d8_holdout.npz", **h)
    # eval2, for comparison (gen_nnue's own rows)
    e2 = {}
    for p in probe.DATA:
        r = np.array(eval_data.load(str(p)))
        lab = (r["flags"] & eval_data.F_SEARCH) != 0
        s = r["search"][lab]
        e2[p.name] = dict(records=len(r), labeled=int(lab.sum()), flags={str(i): int(v) for i, v in enumerate(np.bincount(r["flags"])) if v},
                          search_min=int(s.min()), search_max=int(s.max()),
                          search_abs_ge_20000=int((np.abs(s) >= 20000).sum()), search_std=float(s.std()),
                          compact_fast_mismatches=int((compact_fast(r.view(np.uint8).reshape(-1, 128)) != probe.compact(r)).any(1).sum()))
    res["eval2"] = e2
    ev = load_eval2()
    res["eval2_train_rows"] = int((~ev["hold"]).sum())
    res["v2_rows"] = int(ev["hold"].sum())
    res["d8_train_rows"] = int(sum(f["train_rows"] for f in res["files"].values()))
    res["rows_per_pass"] = res["d8_train_rows"] + res["eval2_train_rows"]
    res["d8h_rows"] = int(len(h["search"]))
    res["d8h_scored_rows"] = int((~((h["fidx"] == 0) & (h["recidx"] < BASE_ROWS))).sum())
    res["minutes"] = (time.time() - t0) / 60
    (OUT / "stream_survey.json").write_text(json.dumps(res, indent=1))
    log(f"survey done: {res['d8_train_rows']:,} d8 training rows + {res['eval2_train_rows']:,} eval2 training rows ="
        f" {res['rows_per_pass']:,} rows per pass; D8H {res['d8h_rows']:,} rows ({res['d8h_scored_rows']:,} scored);"
        f" V2 {res['v2_rows']:,}; {res['minutes']:.1f} min")
    log.to(None)


def survey_info():
    p = OUT / "stream_survey.json"
    if not p.exists():
        sys.exit("run `gen_nnue_stream.py survey` first")
    return json.loads(p.read_text())


# ---------------------------------------------------------------- duplicates / leakage
def dups_path():
    """The dups cache cmd_eval / load_d8h read: dups_sym.npz (D8H up to symmetry, review 5) when present,
    else the first round's dups.npz (D8H exact positions only)."""
    p = SDIR / "dups_sym.npz"
    return p if p.exists() else SDIR / "dups.npz"


def cmd_dups(args):
    """Position overlap of the holdouts with the training rows. Every holdout row is hashed in all 8 of
    its D4 images (probe.transform_states) and counts as seen when any image is a training position:
    V2 always was; D8H is too since review 5 (the first round matched D8H rows by exact position only,
    which missed the mirrored / rotated repeats). Writes stream/<out>.npz, probe/stream_<out>.json and
    probe/stream_<out>.log."""
    probe.below_normal_priority()
    log.to(OUT / f"stream_{args.out}.log")
    t0 = time.time()
    info = survey_info()
    hold = np.load(SDIR / "d8_holdout.npz")
    # per-record hash files, row aligned with the .cfdg; the training rows are labeled, non-holdout rows
    files = [(fidx, SDIR / f"hash_{p.stem}.u64", p) for fidx, p in enumerate(D8)]
    pad = lambda st: np.concatenate([st, np.zeros((len(st), 128 - 93), np.uint8)], 1)  # noqa: E731

    def images(st):  # (n, 93) state chars -> (8, n) position hashes of the 8 D4 images (image 0 = identity)
        return np.stack([pos_hash(pad(probe.transform_states(st, np.full(len(st), g)))) for g in range(8)])
    # D8H: the held-out records' states in all 8 images; image 0 must equal the survey's per-record hash
    nh = len(hold["search"])
    hold_h = np.empty((8, nh), np.uint64)
    id_bad = 0
    for fidx, hp, p in files:
        sel = np.flatnonzero(hold["fidx"] == fidx)
        ri = hold["recidx"][sel]
        recs = np.array(eval_data.load(str(p))[ri])
        hold_h[:, sel] = images(np.frombuffer(recs["s"].tobytes(), dtype=np.uint8).reshape(-1, 93))
        hh = np.fromfile(hp, dtype=np.uint64)
        id_bad += int((hh[ri] != hold_h[0, sel]).sum())
        del hh, recs
    if id_bad:
        sys.exit(f"D8H identity-image hashes differ from stream/hash_*.u64 on {id_bad:,} rows")
    # V2 holdout rows and their 7 images, hashed the same way; eval2 training rows (exact, as stored)
    rec, _, hv = ntb.load_all([str(p) for p in probe.DATA], None)
    v2 = rec[np.flatnonzero(hv)]
    tr2 = rec[np.flatnonzero(~hv)]
    del rec
    v2_h = images(np.frombuffer(v2["s"].tobytes(), dtype=np.uint8).reshape(-1, 93))  # (8, n)
    e2_h = np.unique(pos_hash(pad(np.frombuffer(tr2["s"].tobytes(), dtype=np.uint8).reshape(-1, 93))))
    del tr2
    log(f"hashed {nh:,} D8H rows and {v2_h.shape[1]:,} V2 rows x 8 symmetries (D8H identity images equal the"
        f" survey hashes on all rows) ({time.time() - t0:.0f}s)")
    # training-row mask per file: labeled and not held out (flags/game from the records, in chunks)
    P = args.parts
    shift = np.uint64(64 - int(math.log2(P)))
    res = dict(parts=P, holdout_matching="any of the 8 D4 images of a holdout row (D8H and V2); image 0 = exact")
    d8h_seen = np.zeros(hold_h.shape, bool)
    d8h_seen_base = np.zeros(hold_h.shape, bool)
    v2_seen_all = np.zeros(v2_h.shape, bool)
    v2_seen_base = np.zeros(v2_h.shape, bool)
    distinct_all, distinct_train, rows_train = 0, 0, 0
    per_file_distinct = {}
    masks = {}
    for fidx, hp, p in files:  # the training masks once (flags + game), kept packed
        n = n_records(p)
        m = np.zeros(n, bool)
        with open(p, "rb") as fh:
            for a in range(0, n, 4_000_000):
                r = rec_view(read_raw(p, a, min(n, a + 4_000_000), fh))
                m[a:a + len(r)] = ((r["flags"] & eval_data.F_SEARCH) != 0) & ~d8_holdout(r["game"], fidx)
        masks[fidx] = np.packbits(m)
        log(f"training mask {p.name}: {m.sum():,} rows ({time.time() - t0:.0f}s)")
    for part in range(P):
        allp, trp, basep = [], [], []
        for fidx, hp, p in files:
            hh = np.fromfile(hp, dtype=np.uint64)
            m = np.unpackbits(masks[fidx], count=len(hh)).astype(bool)
            sel = (hh >> shift) == np.uint64(part)
            allp.append(hh[sel])
            per_file_distinct[p.name] = per_file_distinct.get(p.name, 0) + len(np.unique(allp[-1]))
            trp.append(hh[sel & m])
            if fidx == 0:
                bm = np.zeros(len(hh), bool)
                bm[:BASE_ROWS] = True
                basep.append(hh[sel & m & bm])  # what the baselines trained on (their split had no D8H)
            del hh, m, sel
        ua = np.unique(np.concatenate(allp))
        tr = np.concatenate(trp)
        ut = np.unique(tr)
        ub = np.unique(np.concatenate(basep))
        distinct_all += len(ua)
        distinct_train += len(ut)
        rows_train += len(tr)
        for g in range(8):
            hs = (hold_h[g] >> shift) == np.uint64(part)
            d8h_seen[g, hs] = np.isin(hold_h[g, hs], ut)
            d8h_seen_base[g, hs] = np.isin(hold_h[g, hs], ub)
            vs = (v2_h[g] >> shift) == np.uint64(part)
            v2_seen_all[g, vs] = np.isin(v2_h[g, vs], ut)
            v2_seen_base[g, vs] = np.isin(v2_h[g, vs], ub)
        del allp, trp, basep, ua, tr, ut, ub
        el = time.time() - t0
        log(f"dups part {part + 1}/{P}: distinct training positions so far {distinct_train:,} of {rows_train:,} rows"
            f" (elapsed {el / 60:.1f}m, eta {el / (part + 1) * (P - part - 1) / 60:.1f}m)")
    v2_e2 = np.stack([np.isin(v2_h[g], e2_h) for g in range(8)])
    d8h_e2 = np.stack([np.isin(hold_h[g], e2_h) for g in range(8)])
    base_d8h = (hold["fidx"] == 0) & (hold["recidx"] < BASE_ROWS)
    sc = ~base_d8h  # the scored D8H rows (load_d8h)
    ply = hold["ply"]
    d8s, d8x = d8h_seen.any(0), d8h_seen[0]
    bands = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 128))
    res.update(
        records=int(sum(n_records(p) for p in D8)), distinct_positions_all_records=int(distinct_all),
        training_rows=int(rows_train), distinct_training_positions=int(distinct_train),
        distinct_per_file=per_file_distinct,
        d8h_rows=int(nh), d8h_scored_rows=int(sc.sum()),
        d8h_seen_in_training=float(d8s.mean()), d8h_seen_in_training_exact=float(d8x.mean()),
        d8h_scored_seen_in_training=float(d8s[sc].mean()), d8h_scored_seen_in_training_exact=float(d8x[sc].mean()),
        d8h_seen_by_ply={f"{lo}-{hi - 1}": float(d8s[(ply >= lo) & (ply < hi)].mean()) for lo, hi in bands},
        d8h_seen_by_ply_exact={f"{lo}-{hi - 1}": float(d8x[(ply >= lo) & (ply < hi)].mean()) for lo, hi in bands},
        d8h_scored_in_d8_base5M=float(d8h_seen_base.any(0)[sc].mean()),
        d8h_scored_in_eval2_training=float(d8h_e2.any(0)[sc].mean()),
        d8h_scored_novel_for_all_runs=float((~(d8s | d8h_e2.any(0)))[sc].mean()),
        d8h_scored_novel_for_d5M_runs=float((~(d8h_seen_base.any(0) | d8h_e2.any(0)))[sc].mean()),
        v2_rows=int(v2_h.shape[1]),
        v2_in_d8_training_exact=float(v2_seen_all[0].mean()), v2_in_d8_training_sym=float(v2_seen_all.any(0).mean()),
        v2_in_d8_base5M_exact=float(v2_seen_base[0].mean()), v2_in_d8_base5M_sym=float(v2_seen_base.any(0).mean()),
        v2_in_eval2_training_exact=float(v2_e2[0].mean()), v2_in_eval2_training_sym=float(v2_e2.any(0).mean()),
        v2_novel_for_all_runs=float((~(v2_seen_all.any(0) | v2_e2.any(0))).mean()),
        v2_novel_for_baselines=float((~(v2_seen_base.any(0) | v2_e2.any(0))).mean()),
        minutes=(time.time() - t0) / 60,
    )
    # d8h_seen: in the d8 training rows up to symmetry (dups.npz: exact); d8h_in_eval2_sym: in eval2's training rows
    np.savez(SDIR / f"{args.out}.npz", d8h_seen=d8s, d8h_seen_exact=d8x, d8h_in_d8_base_sym=d8h_seen_base.any(0),
             d8h_in_eval2_sym=d8h_e2.any(0), v2_in_d8_sym=v2_seen_all.any(0), v2_in_d8_base_sym=v2_seen_base.any(0),
             v2_in_eval2_sym=v2_e2.any(0))
    (OUT / f"stream_{args.out}.json").write_text(json.dumps(res, indent=1))
    log(f"dups: {json.dumps(res)}")
    log.to(None)


# ---------------------------------------------------------------- the stream
class Stream:
    """Endless shuffled, augmented (x uint8 (B, 91), y float32 (B,)) batches: every pass visits every
    training row once (d8 blocks minus unlabeled and D8H rows, plus the eval2 training rows)."""

    def __init__(self, e2x, e2y, k, batch, block=32768, chunk_blocks=128, workers=6, seed=1, aug=True,
                 qsize=1, debug=False, files=None, e2_repeat=1):
        self.k, self.batch, self.aug, self.debug = k, batch, aug, debug
        self.e2x, self.e2y = e2x, e2y
        self.blocks = []
        for fidx, p in enumerate(D8):
            if files is not None and fidx not in files:
                continue
            n = n_records(p)
            self.blocks += [(fidx, a, min(n, a + block)) for a in range(0, n, block)]
        if e2x is not None:
            for _ in range(e2_repeat):  # e2_repeat > 1: every eval2 training row e2_repeat times per pass
                self.blocks += [(-1, a, min(len(e2x), a + block)) for a in range(0, len(e2x), block)]
        self.chunk_blocks = chunk_blocks
        self.rng = np.random.default_rng(seed + 1000)
        self.pool = ThreadPoolExecutor(workers)
        self.q = queue.Queue(maxsize=qsize)
        self.stop = False
        self.wait_s = 0.0
        self.build_s = 0.0
        self.chunks = 0
        self.passes_started = 0
        self.err = None
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def n_chunks(self):
        """Shuffle chunks per pass: ceil(blocks / chunk_blocks), filled evenly by np.array_split."""
        return max(1, -(-len(self.blocks) // self.chunk_blocks))

    def chunk_sizes(self):
        """Blocks per shuffle chunk of one pass (for logs and checks)."""
        return [len(c) for c in np.array_split(np.arange(len(self.blocks)), self.n_chunks())]

    def _load(self, blk):
        fidx, a, b = blk
        if fidx < 0:
            x, y = self.e2x[a:b], self.e2y[a:b]
            key = np.full(b - a, -1, np.int64) if self.debug else None
            return x, y, key
        raw = read_raw(D8[fidx], a, b)
        r = rec_view(raw)
        keep = np.flatnonzero(((r["flags"] & eval_data.F_SEARCH) != 0) & ~d8_holdout(r["game"], fidx))
        x = compact_fast(raw[keep])
        y = sigmoid_target(r["search"][keep], self.k)
        key = (np.int64(fidx) << 40) + a + keep if self.debug else None  # (file, record index)
        return x, y, key

    def _augment(self, x, sym_out=None):
        """augment_np per batch: slice j of batch i gets symmetry (j + shift_i) % 8 (rows already shuffled)."""
        n = len(x)
        B = self.batch
        nb = n // B
        shift = self.rng.integers(0, 8, nb)
        j = (np.arange(n) % B) * 8 // B  # slice of the row inside its batch: j * B // 8 .. (j + 1) * B // 8
        sym = (j + np.repeat(shift, B)[:n]) % 8
        for g in range(8):
            idx = np.flatnonzero(sym == g)
            if g:
                x[idx] = x[idx][:, COLPERM[g]]
                x[idx, 90] = probe.CONMAP[g][x[idx, 90]]
        if sym_out is not None:
            sym_out.append(sym)

    def _run(self):
        try:
            left = None
            while not self.stop:
                order = self.rng.permutation(len(self.blocks))
                self.passes_started += 1
                # equal-sized shuffle chunks (review 5): slicing order[c0:c0 + chunk_blocks] left a last
                # chunk of len(order) % chunk_blocks blocks every pass (1 block with the full d8 set: two
                # batches drawn from one 32,768-record stretch of one file). array_split spreads the
                # blocks over the same number of chunks, sizes differing by at most one block.
                for chunk in np.array_split(order, self.n_chunks()):
                    if self.stop:
                        return
                    t0 = time.time()
                    parts = list(self.pool.map(self._load, [self.blocks[i] for i in chunk]))
                    if left is not None:
                        parts.insert(0, left)
                    x = np.concatenate([p[0] for p in parts])
                    y = np.concatenate([p[1] for p in parts])
                    key = np.concatenate([p[2] for p in parts]) if self.debug else None
                    del parts
                    perm = self.rng.permutation(len(x))
                    x, y = x[perm], y[perm]
                    if self.debug:
                        key = key[perm]
                    nb = len(x) // self.batch
                    cut = nb * self.batch
                    left = (x[cut:].copy(), y[cut:].copy(), key[cut:].copy() if self.debug else None)
                    x, y = x[:cut], y[:cut]
                    syms = [] if self.debug else None
                    if self.aug:
                        self._augment(x, syms)
                    self.build_s += time.time() - t0
                    self.chunks += 1
                    item = (x, y, nb, (key[:cut], syms[0] if syms else None) if self.debug else None, self.passes_started)
                    while not self.stop:
                        try:
                            self.q.put(item, timeout=0.5)
                            break
                        except queue.Full:
                            pass
        except BaseException as e:  # surfaced in the consumer
            self.err = e
            self.q.put(None)

    def batches(self):
        while True:
            t0 = time.time()
            item = self.q.get()
            self.wait_s += time.time() - t0
            if item is None:
                raise RuntimeError(f"stream loader failed: {self.err!r}")
            x, y, nb, dbg, pz = item
            for i in range(nb):
                sl = slice(i * self.batch, (i + 1) * self.batch)
                yield (x[sl], y[sl]) if dbg is None else (x[sl], y[sl], dbg[0][sl], None if dbg[1] is None else dbg[1][sl], pz)

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.pool.shutdown(wait=False, cancel_futures=True)


def eval2_train(e2, k):
    tr = np.flatnonzero(~e2["hold"])
    return np.ascontiguousarray(e2["x"][tr]), sigmoid_target(e2["search"][tr], k)


# ---------------------------------------------------------------- loader bench / check
def cmd_bench(args):
    probe.below_normal_priority()
    info = survey_info()
    t0 = time.time()
    e2 = load_eval2()
    e2x, e2y = eval2_train(e2, args.k)
    del e2
    log(f"eval2 loaded ({time.time() - t0:.0f}s); stream: {args.workers} workers, chunks of {args.chunk_blocks} x {args.block}")
    st = Stream(e2x, e2y, args.k, args.batch, args.block, args.chunk_blocks, args.workers, seed=1)
    per_pass = info["rows_per_pass"]
    target = int(args.passes * per_pass)
    t1 = time.time()
    rows, ysum, nxt = 0, 0.0, 0
    for xb, yb in st.batches():
        rows += len(xb)
        ysum += float(yb.sum(dtype=np.float64))
        if rows >= nxt:
            el = time.time() - t1
            log(f"bench: {rows:,}/{target:,} rows, {rows / max(el, 1e-9):,.0f} rows/s, consumer waited {st.wait_s:.0f}s,"
                f" chunk build {st.build_s / max(1, st.chunks):.2f}s/chunk, eta {el / max(rows, 1) * (target - rows) / 60:.1f}m")
            nxt += 20_000_000
        if rows + args.batch > target:
            break
    el = time.time() - t1
    st.close()
    exp_rows = per_pass * args.passes
    log(f"bench done: {rows:,} rows in {el:.0f}s = {rows / el:,.0f} rows/s; expected {exp_rows:,.0f} rows for {args.passes} passes"
        f" (leftover < one batch per chunk carried over); mean label {ysum / rows:.6f}; consumer waited {st.wait_s:.1f}s")


INV_COLPERM = np.argsort(COLPERM, axis=1)
INV_CONMAP = np.argsort(probe.CONMAP, axis=1)


def sym_image(x, sym):
    """Rows x (n, 91) under per-row symmetries sym, as augment_np maps them."""
    out = np.empty_like(x)
    for g in range(8):
        i = np.flatnonzero(sym == g)
        out[i] = x[i][:, COLPERM[g]]
        out[i, 90] = probe.CONMAP[g][x[i, 90]]
    return out


def row_key(x, y):
    """64-bit key of a compact row (91 bytes) and its float32 label."""
    pad = np.zeros((len(x), 96), np.uint8)
    pad[:, :91] = x
    pad[:, 92:96] = y.view(np.uint8).reshape(-1, 4)
    w = pad.view(np.uint64)
    h = np.zeros(len(x), np.uint64)
    for i in range(12):
        h = (h ^ (w[:, i] * HASH_C[i])) * np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(33)
    return h


def cmd_check(args):
    """Rows out of the stream (a few chunks, debug keys) against the records they came from."""
    probe.below_normal_priority()
    t0 = time.time()
    e2 = load_eval2()
    e2x, e2y = eval2_train(e2, args.k)
    st = Stream(e2x, e2y, args.k, args.batch, 32768, 32, args.workers, seed=7, debug=True)
    res = dict(rows=0, e2_rows=0, d8_rows=0, holdout_rows=0, row_mismatch=0, e2_row_missing=0, label_mismatch=0, unlabeled=0,
               per_batch_sym_counts_bad=0, sym_hist=[0] * 8, dup_keys=0)
    seen = set()
    mms = [eval_data.load(str(p)) for p in D8]
    e2keys = np.unique(row_key(e2x, e2y))
    it = st.batches()
    for bi in range(args.batches):
        xb, yb, key, sym, _ = next(it)
        res["rows"] += len(xb)
        c = np.bincount(sym, minlength=8)
        res["per_batch_sym_counts_bad"] += int((c != len(xb) // 8).any())
        for g in range(8):
            res["sym_hist"][g] += int(c[g])
        d8 = key >= 0
        res["e2_rows"] += int((~d8).sum())
        res["d8_rows"] += int(d8.sum())
        ks = key[d8]
        for kk in ks.tolist():
            if kk in seen:
                res["dup_keys"] += 1
            seen.add(kk)
        fid = (ks >> 40).astype(np.int64)
        ri = ks & ((1 << 40) - 1)
        xs, ys, ss = xb[d8], yb[d8], sym[d8]
        for f in np.unique(fid):
            sel = np.flatnonzero(fid == f)
            recs = np.array(mms[f][ri[sel]])  # the source records, by (file, record index)
            res["holdout_rows"] += int(d8_holdout(recs["game"], int(f)).sum())
            res["unlabeled"] += int(((recs["flags"] & eval_data.F_SEARCH) == 0).sum())
            ref = probe.compact(recs)
            res["row_mismatch"] += int((sym_image(ref, ss[sel]) != xs[sel]).any(1).sum())
            yref = (1 / (1 + np.exp(-recs["search"].astype(np.float64) / args.k))).astype(np.float32)
            res["label_mismatch"] += int((yref != ys[sel]).sum())
        # eval2 rows: undo the recorded symmetry, then (row, label) must be an eval2 training row
        if (~d8).any():
            xe, ye, se = xb[~d8], yb[~d8], sym[~d8]
            orig = np.empty_like(xe)
            for g in range(8):
                i = np.flatnonzero(se == g)
                orig[i] = xe[i][:, INV_COLPERM[g]]
                orig[i, 90] = INV_CONMAP[g][xe[i, 90]]
            res["e2_row_missing"] += int((~np.isin(row_key(orig, ye), e2keys)).sum())
    st.close()
    cs = st.chunk_sizes()
    full = Stream.__new__(Stream)  # the training default (128 blocks per chunk) on the full block list
    full.blocks, full.chunk_blocks = st.blocks, 128
    fcs = full.chunk_sizes()
    res.update(chunks_per_pass=len(cs), chunk_blocks_min=min(cs), chunk_blocks_max=max(cs),
               default_chunks_per_pass=len(fcs), default_chunk_blocks_min=min(fcs), default_chunk_blocks_max=max(fcs))
    ok = res["row_mismatch"] == 0 and res["e2_row_missing"] == 0 and res["label_mismatch"] == 0 and res["holdout_rows"] == 0 and res["unlabeled"] == 0 \
        and res["per_batch_sym_counts_bad"] == 0 and res["dup_keys"] == 0 and res["e2_rows"] > 0 \
        and max(cs) - min(cs) <= 1 and max(fcs) - min(fcs) <= 1
    res["PASS"] = bool(ok)
    res["secs"] = time.time() - t0
    (OUT / f"stream_check{'_' + args.tag if args.tag else ''}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------- validation sets
def load_d8h(k):
    """D8H minus the games inside the lr1e2 baselines' first 5M records of d8_a (scored set)."""
    h = np.load(SDIR / "d8_holdout.npz")
    keep = ~((h["fidx"] == 0) & (h["recidx"] < BASE_ROWS))
    d = {n: h[n][keep] for n in h.files}
    dp = dups_path()
    if dp.exists():
        z = np.load(dp)
        seen = z["d8h_seen"]  # dups_sym.npz: up to symmetry, in the d8 training rows
        if "d8h_in_eval2_sym" in z.files:  # ... or in eval2's training rows (V2's "novel" rule)
            seen = seen | z["d8h_in_eval2_sym"]
        d["seen"] = seen[keep]
    return d


def score(e, ps, search, static, k, ship=None):
    y = 1 / (1 + np.exp(-search.astype(np.float64) / k))
    p = 1 / (1 + np.exp(-e / k))
    nm = np.abs(search) < 8000
    val = float(((p - y) ** 2).mean())
    ship = ship if ship is not None else float(((1 / (1 + np.exp(-static.astype(np.float64) / k)) - y) ** 2).mean())
    out = dict(val=val, vs_shipped_pct=100 * (val / ship - 1), shipped=ship, std=float(e.std()),
               corr_search=float(np.corrcoef(e[nm], search[nm])[0, 1]),
               corr_static=float(np.corrcoef(e, static)[0, 1]))
    if ps is not None:
        out["val_psqt"] = float(((1 / (1 + np.exp(-ps / k)) - y) ** 2).mean())
    return out


def predict(model, X, dev, want_acc=False):
    model.eval()
    out, dead = [], None
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            us, them = model.accumulators_fast(X[i:i + 65536].to(dev))
            e, ps = model.head(us, them)
            out.append(torch.stack([e, ps], 1).cpu())
            if want_acc and i == 0:
                a = torch.cat([us[:, :model.A], them[:, :model.A]], 0)
                dead = dict(dead=float(((a <= 0).all(0)).float().mean()),
                            saturated=float(((a >= 1).all(0)).float().mean()),
                            active_frac=float(((a > 0) & (a < 1)).float().mean()))
    model.train()
    o = torch.cat(out).numpy().astype(np.float64)
    return o[:, 0], o[:, 1], dead


# ---------------------------------------------------------------- training
def cmd_train(args):
    probe.below_normal_priority()
    torch.set_num_threads(args.threads)
    info = survey_info()
    t_load = time.time()
    e2 = load_eval2()
    e2x, e2y = eval2_train(e2, args.k)
    va = np.flatnonzero(e2["hold"])
    V2 = dict(x=torch.from_numpy(np.ascontiguousarray(e2["x"][va])), search=e2["search"][va].astype(np.float64),
              static=e2["static"][va].astype(np.float64))
    del e2
    D8H = load_d8h(args.k)
    D8H["X"] = torch.from_numpy(np.ascontiguousarray(D8H["x"]))
    load_secs = time.time() - t_load
    dev = get_dev(args.device)
    fx = probe.Feats(dev)
    for spec in args.run:
        train_one(parse_run(spec), args, info, e2x, e2y, V2, D8H, dev, fx, load_secs)


def train_one(r, args, info, e2x, e2y, V2, D8H, dev, fx, load_secs):
    name = r["name"]
    if (OUT / f"{name}.log").exists() and not args.overwrite:
        sys.exit(f"{OUT / (name + '.log')} exists: pick another name (or --overwrite)")
    if (OUT / f"{name}.log").exists():
        (OUT / f"{name}.log").unlink()
    log.to(OUT / f"{name}.log")
    k = args.k
    seed = r.get("seed", args.seed)
    forced = bool(r.get("forced", 1))
    shared_con = bool(r.get("sharedcon", 0))
    passes = r.get("passes", args.passes)
    e2rep = r.get("e2rep", 1)
    aug = bool(r.get("aug", 1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows_per_pass = info["d8_train_rows"] + e2rep * info["eval2_train_rows"]
    n_steps = rows_per_pass // args.batch  # one epoch = one pass over the training rows
    steps = r.get("steps", passes * n_steps)  # steps=N: a total step budget; the last epoch is then partial
    passes = -(-steps // n_steps)
    log(f"data: {rows_per_pass:,} training rows per pass ({info['d8_train_rows']:,} depth-8 rows streamed from"
        f" {len(D8)} files, {info['eval2_train_rows']:,} eval2 rows" + (f" x {e2rep}" if e2rep != 1 else "") + f"); V2 {len(V2['search']):,} rows, D8H"
        f" {len(D8H['search']):,} rows; loaded in {load_secs:.0f}s")
    model = Gen(fx, r["A"], r["dense"], tuple(args.enc), forced, args.row_std, shared_con)
    params = n_params(model)
    log(f"{name}: B generator, A {r['A']}, dense {2 * r['A']}->{'->'.join(map(str, r['dense']))}->1, enc 27->"
        f"{'->'.join(map(str, args.enc))}, forced row {forced}, shared constraint rows {shared_con}, seed {seed};"
        f" {params:,} params; payload {payload(params)} units; AdamW lr {r['lr']} cosine (warmup {args.warmup:.0%},"
        f" floor {args.lr_floor}), batch {args.batch}, {passes} epochs, psqt aux {args.psqt_w}, aug {aug};"
        f" streamed: 1 epoch = 1 pass = {n_steps:,} steps, {steps:,} steps in all, blocks of {args.block:,} records,"
        f" {args.chunk_blocks} blocks per shuffle chunk, {args.workers} loader threads")
    yv = 1 / (1 + np.exp(-V2["search"] / k))
    ship_loss = float(((1 / (1 + np.exp(-V2["static"] / k)) - yv) ** 2).mean())
    d8_ship = float(((1 / (1 + np.exp(-D8H["static"].astype(np.float64) / k))
                      - 1 / (1 + np.exp(-D8H["search"].astype(np.float64) / k))) ** 2).mean())
    log(f"train {rows_per_pass:,} rows ({info['d8_train_rows']:,} extra), holdout {len(yv):,}; shipped static_eval loss"
        f" {ship_loss:.6f}; D8H shipped static_eval loss {d8_ship:.6f}")

    def v2_stats(e, ps):
        s = V2["search"]
        p = 1 / (1 + np.exp(-e / k))
        nm = np.abs(s) < 8000
        return dict(val=float(((p - yv) ** 2).mean()),
                    val_psqt=float(((1 / (1 + np.exp(-ps / k)) - yv) ** 2).mean()),
                    std=float(e.std()), mae=float(np.abs(e - s).clip(max=20000).mean()),
                    corr_search=float(np.corrcoef(e[nm], s[nm])[0, 1]),
                    corr_static=float(np.corrcoef(e, V2["static"])[0, 1]),
                    r2_target=float(1 - ((e - s) ** 2).mean() / s.var()))

    def evaluate():
        for attempt in range(4):  # DirectML once failed a holdout batch transfer ("The parameter is incorrect")
            try:
                e, ps, health = predict(model, V2["x"], dev, want_acc=True)
                e8, ps8, _ = predict(model, D8H["X"], dev)
                break
            except RuntimeError as ex:
                if attempt == 3:
                    raise
                log(f"{name}: holdout prediction failed ({ex}); retrying in 10s")
                time.sleep(10)
        st = v2_stats(e, ps)
        d8 = score(e8, ps8, D8H["search"], D8H["static"], k, ship=d8_ship)
        return st, d8, health

    first = [p for n, p in model.named_parameters() if not n.startswith("dense.")]
    dense = [p for n, p in model.named_parameters() if n.startswith("dense.")]
    opt = AdamW([dict(params=first, weight_decay=0.0), dict(params=dense, weight_decay=args.wd)], betas=(0.9, 0.999))
    lr_at = lr_lambda(steps, args.warmup, args.lr_floor / r["lr"])
    stream = Stream(e2x, e2y, k, args.batch, args.block, args.chunk_blocks, args.workers, seed=seed, aug=aug,
                    e2_repeat=e2rep)
    cs = stream.chunk_sizes()
    log(f"stream: {len(stream.blocks):,} blocks per pass in {len(cs)} shuffle chunks of {min(cs)}-{max(cs)} blocks"
        f" (np.array_split)")
    it = stream.batches()
    best, hist, checks = None, [], []
    t_start = time.time()
    train_secs, rows_done = 0.0, 0
    n_log = max(1, n_steps // args.log_per_epoch)
    n_check = max(1, n_steps // args.checks_per_epoch) if args.checks_per_epoch > 1 else 0
    for epoch in range(1, passes + 1):
        t0 = time.time()
        ep_train_secs = 0.0
        run = torch.zeros(2, device=dev)
        seen = 0
        ts = time.time()
        for step in range(min(n_steps, steps - (epoch - 1) * n_steps)):
            if step and step % n_log == 0:
                rl = run.cpu().numpy() / seen
                dt = time.time() - ts
                train_secs += dt
                ep_train_secs += dt
                done = (epoch - 1) * n_steps + step
                el = time.time() - t_start
                eta = el / done * (steps - done)
                log(f"{name} epoch {epoch} step {step}/{n_steps} train {rl[0]:.6f} psqt {rl[1]:.6f}"
                    f" lr {r['lr'] * lr_at(done):.2e} {(rows_done + seen) / max(1e-9, train_secs):,.0f} rows/s"
                    f" elapsed {el / 60:.1f}m eta {eta / 60:.1f}m")
                ts = time.time()
            if n_check and step and step % n_check == 0 and step < n_steps - n_check // 2:
                tc = time.time()
                st, d8, _ = evaluate()
                done = (epoch - 1) * n_steps + step
                checks.append(dict(step=done, epoch_frac=done / n_steps, v2=st["val"], d8h=d8["val"],
                                   v2_corr=st["corr_search"], d8h_corr=d8["corr_search"]))
                log(f"{name} check at step {done}/{steps} ({done / n_steps:.2f} passes): V2 val {st['val']:.6f}"
                    f" ({100 * (st['val'] / ship_loss - 1):+.2f}% vs shipped) corr(search) {st['corr_search']:.3f};"
                    f" D8H val {d8['val']:.6f} ({d8['vs_shipped_pct']:+.2f}%) corr(search) {d8['corr_search']:.3f}"
                    f" ({time.time() - tc:.0f}s)")
                ts += time.time() - tc  # not training time
            xb_np, yb_np = next(it)
            xb = torch.from_numpy(xb_np).to(dev)
            y = torch.from_numpy(yb_np).to(dev)
            e, ps = model.fast(xb, with_psqt=True)
            l_main = (torch.sigmoid(e / k) - y).pow(2).mean()
            l_ps = (torch.sigmoid(ps / k) - y).pow(2).mean()
            loss = l_main + args.psqt_w * l_ps
            for q in model.parameters():
                q.grad = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, foreach=False)
            opt.step(r["lr"] * lr_at((epoch - 1) * n_steps + step))
            run += torch.stack([l_main.detach(), l_ps.detach()]) * len(yb_np)
            seen += len(yb_np)
        rl = run.cpu().numpy() / max(1, seen)
        dt = time.time() - ts
        train_secs += dt
        ep_train_secs += dt
        rows_done += seen
        st, d8, health = evaluate()
        st.update(epoch=epoch, train=float(rl[0]), train_psqt=float(rl[1]),
                  secs=time.time() - t0, rows_per_s=seen / max(1e-9, ep_train_secs), **(health or {}))
        st["d8h"] = d8
        hist.append(st)
        log(f"{name} epoch {epoch}: train {st['train']:.6f} val {st['val']:.6f} ({100 * (st['val'] / ship_loss - 1):+.2f}%"
            f" vs shipped) psqt-only val {st['val_psqt']:.6f} std {st['std']:.0f} corr(search) {st['corr_search']:.3f}"
            f" corr(static) {st['corr_static']:.3f} R2(target) {st['r2_target']:.4f}; acc lanes dead {st.get('dead', 0):.1%} saturated"
            f" {st.get('saturated', 0):.1%}; {st['rows_per_s']:,.0f} rows/s ({st['secs']:.0f}s)")
        log(f"{name} D8H after epoch {epoch}: val {d8['val']:.6f} ({d8['vs_shipped_pct']:+.2f}% vs its static_eval"
            f" {d8_ship:.6f}) psqt-only val {d8['val_psqt']:.6f} std {d8['std']:.0f} corr(search) {d8['corr_search']:.3f};"
            f" loader: consumer waited {stream.wait_s:.0f}s so far, {stream.build_s / max(1, stream.chunks):.2f}s per"
            f" chunk build, {stream.passes_started} passes started")
        if not math.isfinite(st["val"]):
            break
        if best is None or st["val"] < best["val"]:
            best = st
            torch.save({n: p.detach().cpu() for n, p in model.state_dict().items()}, OUT / f"{name}.pt")
    stream.close()
    a = dict(arm="gen", A=r["A"], dense=list(r["dense"]), lr=r["lr"], enc=list(args.enc), forced=forced, shared_con=shared_con,
             batch=args.batch, epochs=passes, k=k, psqt_w=args.psqt_w, wd_dense=args.wd, clip=args.clip,
             warmup=args.warmup, lr_floor=args.lr_floor, row_std=args.row_std, aug=aug, target="search", seed=seed,
             e2rep=e2rep,
             stream=dict(files=[p.name for p in D8], eval2_train_rows=info["eval2_train_rows"], block=args.block,
                         chunk_blocks=args.chunk_blocks, workers=args.workers, d8_holdout="1% of games, d8_holdout()"))
    summary = dict(name=name, arm="gen", aug=aug, best=best, hist=hist, checks=checks, params=params,
                   payload_units=payload(params), train_rows=int(rows_per_pass), extra_rows=int(info["d8_train_rows"]),
                   holdout_rows=int(len(yv)), d8h_rows=int(len(D8H["search"])), steps=steps, steps_per_epoch=n_steps,
                   rows_seen=int(steps * args.batch), shipped_loss=ship_loss, d8h_shipped_loss=d8_ship,
                   vs_shipped_pct=100 * (best["val"] / ship_loss - 1),
                   rows_per_s=rows_done / max(1e-9, train_secs), train_minutes=(time.time() - t_start) / 60,
                   loader_wait_s=stream.wait_s, args=a)
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1))
    log(f"RESULT {name}: best val {best['val']:.6f} (epoch {best['epoch']}), {summary['vs_shipped_pct']:+.2f}% vs shipped"
        f" {ship_loss:.6f}; std {best['std']:.0f}; {params:,} params; {summary['rows_per_s']:,.0f} rows/s;"
        f" train {summary['train_minutes']:.1f} min")
    log.to(None)


# ---------------------------------------------------------------- eval of saved checkpoints
def cmd_eval(args):
    import buckets
    probe.below_normal_priority()
    torch.set_num_threads(args.threads)
    k = K_DEFAULT
    dev = get_dev(args.device)
    fx = probe.Feats(dev)
    D8H = load_d8h(k)
    X8 = torch.from_numpy(np.ascontiguousarray(D8H["x"]))
    r = buckets.holdout()
    x2 = probe.compact(r)
    s2 = r["search"].astype(np.float64)
    st2 = r["static_eval"].astype(np.float64)
    dups = np.load(dups_path()) if dups_path().exists() else None
    novel = None
    if dups is not None:
        novel = ~(dups["v2_in_d8_sym"] | dups["v2_in_eval2_sym"])  # in no training set of any run, up to symmetry
    d8_ship = None
    path = OUT / args.out
    allres = json.loads(path.read_text()) if path.exists() else {}
    ply8 = D8H["ply"].astype(np.int64)
    s8 = D8H["search"].astype(np.float64)
    st8 = D8H["static"].astype(np.float64)
    b8 = {"all": np.ones(len(s8), bool), "ply<20": ply8 < 20, "ply20-40": (ply8 >= 20) & (ply8 < 40), "ply>=40": ply8 >= 40,
          "played": (D8H["flags"] & eval_data.F_RESULT) != 0, "|s|<8000": np.abs(s8) < 8000}
    if "seen" in D8H:
        b8["novel"] = ~D8H["seen"]
    ship8 = {bn: float(((1 / (1 + np.exp(-st8[m] / k)) - 1 / (1 + np.exp(-s8[m] / k))) ** 2).mean()) for bn, m in b8.items()}
    ply2 = r["ply"].astype(np.int64)
    b2 = {"all": np.ones(len(s2), bool), "ply<20": ply2 < 20, "ply20-40": (ply2 >= 20) & (ply2 < 40), "ply>=40": ply2 >= 40}
    if novel is not None:
        b2["novel"] = novel
    ship2 = {bn: float(((1 / (1 + np.exp(-st2[m] / k)) - 1 / (1 + np.exp(-s2[m] / k))) ** 2).mean()) for bn, m in b2.items()}
    allres["_novel_rule"] = f"{dups_path().name}: position (any of its 8 images) in no training set of any run"
    allres["_shipped"] = dict(v2={bn: dict(loss=v, frac=float(b2[bn].mean())) for bn, v in ship2.items()},
                              d8h={bn: dict(loss=v, frac=float(b8[bn].mean())) for bn, v in ship8.items()},
                              d8h_rows=int(len(s8)), v2_rows=int(len(s2)))
    for name in args.names:
        m, meta = buckets.load_model(name, fx)
        e2, _, _ = predict(m, torch.from_numpy(x2), dev)
        e8, _, _ = predict(m, X8, dev)
        res = dict(v2={}, d8h={})
        for bn, msk in b2.items():
            sc = score(e2[msk], None, s2[msk], st2[msk], k, ship=ship2[bn])
            res["v2"][bn] = dict(loss=sc["val"], vs_shipped_pct=sc["vs_shipped_pct"], corr=sc["corr_search"], frac=float(msk.mean()))
        for bn, msk in b8.items():
            sc = score(e8[msk], None, s8[msk], st8[msk], k, ship=ship8[bn])
            res["d8h"][bn] = dict(loss=sc["val"], vs_shipped_pct=sc["vs_shipped_pct"], corr=sc["corr_search"], frac=float(msk.mean()))
        allres[name] = res
        print(f"{name}: V2 " + ", ".join(f"{bn} {v['vs_shipped_pct']:+.2f}% / {v['corr']:.3f}" for bn, v in res["v2"].items())
              + "\n    D8H " + ", ".join(f"{bn} {v['loss']:.5f} ({v['vs_shipped_pct']:+.2f}%) / {v['corr']:.3f}"
                                   for bn, v in res["d8h"].items()), flush=True)
    path.write_text(json.dumps(allres, indent=1))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("survey")
    p.add_argument("--chunk", type=int, default=2_000_000)
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(fn=cmd_survey)
    p = sub.add_parser("dups")
    p.add_argument("--parts", type=int, default=8)
    p.add_argument("--out", default="dups_sym", help="stream/OUT.npz, probe/stream_OUT.{json,log}")
    p.set_defaults(fn=cmd_dups)
    for nm, fn in (("bench", cmd_bench), ("check", cmd_check)):
        p = sub.add_parser(nm)
        p.add_argument("--k", type=float, default=K_DEFAULT)
        p.add_argument("--batch", type=int, default=16384)
        p.add_argument("--block", type=int, default=32768)
        p.add_argument("--chunk-blocks", type=int, default=128)
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--passes", type=float, default=1.0)
        p.add_argument("--batches", type=int, default=40)
        p.add_argument("--tag", default="", help="check: write probe/stream_check_TAG.json")
        p.set_defaults(fn=fn)
    p = sub.add_parser("train")
    p.add_argument("--run", action="append", required=True, help="NAME,A,DENSE,LR[,passes=P][,seed=N] e.g. B64_all,64,16x32,1e-2")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--k", type=float, default=K_DEFAULT)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--enc", type=int, nargs="+", default=[64, 64, 32])
    p.add_argument("--psqt-w", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--lr-floor", type=float, default=1e-5)
    p.add_argument("--row-std", type=float, default=0.05)
    p.add_argument("--block", type=int, default=32768)
    p.add_argument("--chunk-blocks", type=int, default=128)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-per-epoch", type=int, default=20)
    p.add_argument("--checks-per-epoch", type=int, default=4, help="V2/D8H checks inside each pass (1 = epoch ends only)")
    p.add_argument("--device", default="dml")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("eval")
    p.add_argument("names", nargs="+")
    p.add_argument("--out", default="stream_eval.json", help="merged results file in datasets/nnue2/probe")
    p.add_argument("--device", default="dml")
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
