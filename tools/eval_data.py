#!/usr/bin/env python3
"""Eval training data in cpp_impl/datagen.cpp's 128-byte DgRec format.

Subcommands:
  stats FILE...                       summary per source
  openings OUT NPZ_DIR... [--min-ply A --max-ply B]
                                      93-char opening states from uttt.ai self-play
  import-utttai OUT NPZ_DIR...        uttt.ai self-play rows as DgRec (source 3)
  uttt-value FILE ONNX [--batch N]    fill uttt_v with uttt.ai's network value (in place)
  concat OUT FILE...                  concatenate, renumbering game ids so they stay unique

The uttt.ai self-play npz files come from the fork's cg/train/selfplay.py:
state (N, 93) uint8 digits 0..9, policy, q (root value) and z (result), both
for the side to move. uttt-value imports the fork's cg/train/onnx_weights.py
from UTTTAI_TRAIN_DIR (default: the fork checked out next to this repository,
../utttai/cg/train), else from the ONNX file's directory.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

REC = np.dtype([
    ("s", "S93"), ("source", "u1"), ("ply", "u1"), ("result", "i1"), ("flags", "u1"),
    ("pad", "V3"), ("game", "<u4"), ("hce", "<i4"), ("static_eval", "<i4"), ("search", "<i4"),
    ("game_score", "<i4"), ("uttt_q", "<f4"), ("uttt_v", "<f4"),
])
assert REC.itemsize == 128

F_RESULT, F_SEARCH, F_UTTT_V, F_UTTT_Q = 1, 2, 4, 8
SOURCES = {0: "cf light-random", 1: "cf uttt-opening", 2: "cf heavy-random", 3: "uttt self-play"}
UTTTAI_TRAIN_DIR = str(Path(__file__).resolve().parents[2] / "utttai" / "cg" / "train")  # env var default


def load(path, mode="r"):
    return np.memmap(path, dtype=REC, mode=mode)


def npz_files(dirs):
    out = []
    for d in dirs:
        out += sorted(glob.glob(os.path.join(d, "*.npz"))) if os.path.isdir(d) else [d]
    return out


def cmd_stats(args):
    for path in args.files:
        r = load(path)
        print(f"{path}: {len(r):,} records, {len(np.unique(r['game'])):,} games")
        for src in np.unique(r["source"]):
            x = r[r["source"] == src]
            ok = (x["flags"] & F_RESULT) != 0
            lab = (x["flags"] & F_SEARCH) != 0
            res = x["result"][ok]
            line = (f"  {SOURCES.get(int(src), src)}: {len(x):,} rows, result-valid {ok.mean():.1%}"
                    f" (W/D/L {np.mean(res == 1):.1%}/{np.mean(res == 0):.1%}/{np.mean(res == -1):.1%}),"
                    f" mean ply {x['ply'].mean():.1f}")
            if lab.any():
                s = x["search"][lab].astype(np.float64)
                e = x["static_eval"][lab].astype(np.float64)
                line += (f", searched {lab.mean():.1%} (std {s.std():.0f}, |s|=20000 {np.mean(np.abs(s) >= 20000):.1%},"
                         f" corr static {np.corrcoef(s, e)[0, 1]:.3f})")
            if ((x["flags"] & F_UTTT_V) != 0).any():
                line += f", uttt_v mean {x['uttt_v'][(x['flags'] & F_UTTT_V) != 0].mean():+.3f}"
            print(line)


def _game_ids(states):
    """uttt.ai self-play rows are written game by game; a new game starts when
    the stone count does not increase."""
    stones = (states[:, :81] != 0).sum(axis=1)
    new = np.ones(len(states), dtype=bool)
    new[1:] = stones[1:] <= stones[:-1]
    return np.cumsum(new) - 1, stones


def cmd_openings(args):
    lines = set()
    for f in npz_files(args.dirs):
        z = np.load(f)
        st = z["state"]
        stones = (st[:, :81] != 0).sum(axis=1)
        keep = (stones >= args.min_ply) & (stones <= args.max_ply) & (st[:, 92] == 0)
        for row in st[keep]:
            lines.add((row + 48).tobytes().decode())
    lines = sorted(lines)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines):,} distinct openings (ply {args.min_ply}..{args.max_ply}) -> {args.out}")


def cmd_import(args):
    chunks, game_base = [], 0
    for f in npz_files(args.dirs):
        z = np.load(f)
        st = z["state"]
        gid, stones = _game_ids(st)
        keep = st[:, 92] == 0
        rec = np.zeros(int(keep.sum()), dtype=REC)
        rec["s"] = [(row + 48).tobytes() for row in st[keep]]
        rec["source"] = 3
        rec["ply"] = stones[keep]
        rec["result"] = np.rint(z["z"][keep]).astype(np.int8)
        rec["flags"] = F_RESULT | F_UTTT_Q
        rec["game"] = gid[keep] + game_base
        rec["uttt_q"] = z["q"][keep]
        game_base += int(gid.max()) + 1
        chunks.append(rec)
    out = np.concatenate(chunks)
    out.tofile(args.out)
    print(f"{len(out):,} uttt.ai self-play rows from {game_base:,} games -> {args.out}")


# uttt.ai's 4x9x9 input (cg/train/selfplay.py encode()).
ROW = np.array([3 * (s // 27) + (s % 9) // 3 for s in range(81)])
COL = np.array([3 * ((s // 9) % 3) + s % 3 for s in range(81)])
WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def encode_batch(s):
    """s: (B, 93) ASCII digit bytes -> (B, 4, 9, 9) float32 planes."""
    d = s.astype(np.int64) - 48
    cells, sup, nxt, con = d[:, :81], d[:, 81:90], d[:, 90], d[:, 91]
    b = len(s)
    x = np.zeros((b, 4, 9, 9), dtype=np.float32)
    me = np.where(nxt == 1, 1, 2)[:, None]
    opp = 3 - me
    grid = np.zeros((b, 81), dtype=np.int64)
    grid[:, ROW * 9 + COL] = cells
    x[:, 0] = (grid == me).reshape(b, 9, 9)
    x[:, 1] = (grid == opp).reshape(b, 9, 9)
    x[:, 2] = np.where(nxt == 1, 1.0, -1.0)[:, None, None]
    empty = cells == 0
    live = (sup == 0)
    allowed = np.repeat(live, 9, axis=1)  # free move: any live miniboard
    forced = con < 9
    mb_of = np.arange(81) // 9
    allowed[forced] = (mb_of[None, :] == con[forced, None]) & np.repeat(live[forced], 9, axis=1)
    legal = empty & allowed
    lg = np.zeros((b, 81), dtype=np.float32)
    lg[:, ROW * 9 + COL] = legal
    x[:, 3] = lg.reshape(b, 9, 9)
    return x


def cmd_uttt_value(args):
    import tempfile
    import onnxruntime as ort
    # The fork's cg/train/onnx_weights.py (set_dynamic_batch): UTTTAI_TRAIN_DIR,
    # else the ONNX file's own directory.
    sys.path[:0] = [os.environ.get("UTTTAI_TRAIN_DIR", UTTTAI_TRAIN_DIR), str(Path(args.onnx).resolve().parent)]
    import onnx_weights
    tmp = Path(tempfile.gettempdir()) / f"_dyn_{os.getpid()}.onnx"
    try:
        onnx_weights.set_dynamic_batch(args.onnx, tmp)
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        sess = ort.InferenceSession(str(tmp), sess_options=opts, providers=["DmlExecutionProvider"])
    finally:
        tmp.unlink(missing_ok=True)
    r = load(args.file, "r+")
    n = len(r)
    import time
    t0 = time.time()
    for i in range(0, n, args.batch):
        blk = r[i:i + args.batch]
        s = np.frombuffer(blk["s"].tobytes(), dtype=np.uint8).reshape(-1, 93)
        _, v = sess.run(["policy_logits", "state_value"], {"input": encode_batch(s)})
        blk["uttt_v"] = v.reshape(-1)
        blk["flags"] |= F_UTTT_V
        if (i // args.batch) % 50 == 0:
            done = i + len(blk)
            rate = done / max(1e-9, time.time() - t0)
            print(f"[{time.strftime('%H:%M:%S')}] uttt value {done:,}/{n:,} {rate:.0f}/s"
                  f" eta {(n - done) / rate / 60:.1f}m", flush=True)
    r.flush()
    print(f"uttt value filled for {n:,} rows of {args.file}")


def cmd_concat(args):
    parts, base = [], 0
    for f in args.files:
        r = np.array(load(f))
        _, inv = np.unique(r["game"], return_inverse=True)
        r["game"] = inv + base
        base += int(inv.max()) + 1 if len(r) else 0
        parts.append(r)
    out = np.concatenate(parts)
    out.tofile(args.out)
    print(f"{len(out):,} rows, {base:,} games -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats"); p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("openings"); p.add_argument("out"); p.add_argument("dirs", nargs="+")
    p.add_argument("--min-ply", type=int, default=4); p.add_argument("--max-ply", type=int, default=24)
    p.set_defaults(fn=cmd_openings)
    p = sub.add_parser("import-utttai"); p.add_argument("out"); p.add_argument("dirs", nargs="+")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("uttt-value"); p.add_argument("file"); p.add_argument("onnx")
    p.add_argument("--batch", type=int, default=4096); p.set_defaults(fn=cmd_uttt_value)
    p = sub.add_parser("concat"); p.add_argument("out"); p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_concat)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
