#!/usr/bin/env python3
"""Turn a nnue_train_blend.py result into headers and an isolated A/B build.

  emit  PREFIX DIR [--recluster]
      DIR/mini_eval_d16.hpp and DIR/macro_eval.hpp from PREFIX.{codes,cents}.npy,
      PREFIX.cfm2 and PREFIX.macro.pt, plus DIR/hce_weights.json. Centroid-mode
      results are packed with their own codes (no re-clustering, so the header
      is exactly the trained model); --recluster runs the shipped emitter's
      projection-aware k-means on the full table instead.

  build DIR [--prev-dir OTHER] [--no-tools]
      DIR/test_bots.exe where Dev uses DIR's headers (and HCE weights, and
      the source patches in DIR/dev_patches.json if present) and Prev uses the
      shipped headers under renamed symbols, so the two engines can never
      share weights. Also DIR/datagen.exe on the candidate Dev. The source
      tree in DIR is disposable. With --prev-dir, Prev is candidate OTHER
      instead: OTHER's headers under the same renames, with OTHER's HCE
      weights and dev_patches.json applied to crossfish_prev.hpp exactly as
      DIR's are to Dev (DIR/prev_source.json records which); crossfish_dev.hpp
      and crossfish_prev.hpp must then be the same engine (comments and the
      class name aside). tools/round_robin.py builds its pairings this way.
      --no-tools compiles test_bots only.

  verify DIR [SAMPLE.cfdg] [--train-prefix PREFIX]
      Checks that the candidate's C++ static eval equals HCE + the MiniNet and
      macro decoded from DIR's headers, and that its HCE is the one the model
      was trained on. With --train-prefix (the run's nnue_train_blend.py
      --out), DIR/datagen.exe relabels the statics (depth 0) of the first
      20,000 rows of the run's first --data file, and the candidate's HCE must
      equal that data's HCE plus exactly the trained weight change, so a
      missing or wrong Dev patch fails. With SAMPLE (labeled at depth 1), the
      HCE is compared with the shipped HCE instead, which is skipped when
      SAMPLE is not shipped-labeled or DIR has dev_patches.json.

  match DIR NAME [--depth D | --ms MS] [--games N] [--offset K]
      Dev (candidate) vs Prev (shipped) with test_bots: an equal-depth screen
      with --depth, or the timed pentanomial SPRT (H0 0, H1 +5) with --ms.
      Logs to datasets/eval2/sprt/NAME.log, which the lab dashboard shows.
      Exits non-zero when test_bots fails or the log has no SPRT verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CPP = ROOT / "cpp_impl"
sys.path.insert(0, str(HERE))
import nnue_emit_mininet_cg as legacy  # noqa: E402
import nnue_emit_mininet_header as mh  # noqa: E402
from nnue_cjk14 import encode_cjk14, wrap_cjk14  # noqa: E402

TOOLCHAIN = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
CXXFLAGS = ["-O3", "-std=c++17", "-mavx2", "-mbmi", "-mbmi2", "-mlzcnt", "-mpopcnt", "-pthread",
            "-Wno-unknown-pragmas", "-Wno-ignored-attributes", "-Wl,--stack,16777216"]
SHIPPED_WEIGHTS = "{2410, 836, 464, 1316, 534, 424, 33, PAWN, 33, 112}"


def cmd_emit(args):
    prefix, out = Path(args.prefix), Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = lambda suffix: Path(str(prefix) + suffix)  # noqa: E731  (as nnue_train_blend.export writes them)
    d, h, emb, super_e, loc, constr, active, w1, b1, w2, b2 = mh.load_cfm2(ext(".cfm2"))
    if args.recluster:
        cents, codes = mh.projected_centroids(emb, w1, w2, d, h, True)
    else:
        codes = np.load(ext(".codes.npy")).astype(np.uint8)
        cents = np.load(ext(".cents.npy")).astype(np.float32)
        if cents.shape[0] != 256:
            raise SystemExit("not a centroid-mode result; pass --recluster")
    rec = cents[codes]
    full = (d, h, emb, super_e, loc, constr, active, w1, b1, w2, b2)
    original_empty = mh.empty_output(*full)
    packed_empty = mh.empty_output(d, h, rec, super_e, loc, constr, active, w1, b1, w2, b2)
    b2 = float(b2 + original_empty - packed_empty)
    blob = legacy.pack_blob(codes, cents, super_e, loc, constr, active, w1, b1, w2, b2)
    mh.emit_header(CPP / "mini_eval.hpp", out / "mini_eval_d16.hpp", wrap_cjk14(encode_cjk14(blob)), d, h, None)
    subprocess.run([sys.executable, str(HERE / "nnue_emit_macro_header.py"), str(ext(".macro.pt")),
                    "-o", str(out / "macro_eval.hpp"), "--scale", "1.0", "--clip", "2000"], check=True)
    meta = json.loads(ext(".json").read_text())
    (out / "hce_weights.json").write_text(json.dumps(meta.get("hce_weights")))
    print(f"emitted {out}: packed embedding MAE {np.abs(emb - rec).mean():.5f}, "
          f"empty {original_empty:.3f} -> {packed_empty:.3f}; HCE weights {meta.get('hce_weights')}")


PREV_RENAMES = [
    (r"\bD16_", "P16_"), (r"\bd16_", "p16_"), (r"\bMACRO_", "PMACRO_"),
    (r"\bevaluate_macro_", "p_evaluate_macro_"), (r"\bmacro_load_packed\b", "p_macro_load_packed"),
    (r"\bmacro_finish_hidden\b", "p_macro_finish_hidden"),
]


def rename_prev(text):
    for a, b in PREV_RENAMES:
        text = re.sub(a, b, text)
    return text


def patch_engine(text, cand, who, fname):
    """Apply candidate directory CAND's HCE weights (hce_weights.json) and source
    patches (dev_patches.json) to engine source TEXT. WHO ("Dev" or "Prev") and
    FNAME (the source's file name) only label the messages; a weight constant
    or patch that does not match TEXT exactly is an error."""
    weights = json.loads((cand / "hce_weights.json").read_text()) if (cand / "hce_weights.json").exists() else None
    if weights:
        w = list(weights)
        w[7] = "PAWN"
        new = "{" + ", ".join(str(x) for x in w) + "}"
        if SHIPPED_WEIGHTS not in text:
            raise SystemExit(f"{fname} eval_weights changed; update SHIPPED_WEIGHTS")
        text = text.replace(SHIPPED_WEIGHTS, new)
        # The local terms are baked into the miniboard LUT from their own constants.
        for name, idx, old in (("LUT_W_TIAR", 4, "534"), ("LUT_W_CENTER_SQ", 6, "33"), ("LUT_W_SQUARES", 8, "33")):
            decl = f"static constexpr int {name} = {old};"
            if decl not in text:
                raise SystemExit(f"{decl} not found in {fname}")
            text = text.replace(decl, f"static constexpr int {name} = {w[idx]};")
        print(f"{who} HCE weights {new}")
    # Optional source experiments: dev_patches.json is a list of
    # {"old": ..., "new": ..., "count": N} exact replacements.
    patches = cand / "dev_patches.json"
    if patches.exists():
        for p in json.loads(patches.read_text()):
            n = text.count(p["old"])
            if n != p["count"]:
                raise SystemExit(f"{who.lower()} patch expected {p['count']} matches, found {n}: {p['old']!r}")
            text = text.replace(p["old"], p["new"])
        print(f"applied {len(json.loads(patches.read_text()))} {who} patches from {patches}")
    return text


PREV_SOURCE = "prev_source.json"  # build --prev-dir: which candidate Prev was built from


def engine_code(text, cls):
    """Engine source TEXT without blank and comment-only lines, its class CLS written as CrossfishDev."""
    return [ln.rstrip() for ln in text.replace(cls, "CrossfishDev").splitlines()
            if ln.strip() and not ln.lstrip().startswith("//")]


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_build(args):
    out = Path(args.dir)
    for name in ("mini_eval_d16.hpp", "macro_eval.hpp"):
        if not (out / name).exists():
            raise SystemExit(f"{out / name} missing; run emit first")
    other = Path(args.prev_dir) if args.prev_dir else None
    if other:
        for name in ("mini_eval_d16.hpp", "macro_eval.hpp"):
            if not (other / name).exists():
                raise SystemExit(f"--prev-dir: {other / name} missing; run emit first")
    prev = (CPP / "crossfish_prev.hpp").read_text(encoding="utf-8")
    if other:  # Prev is OTHER's candidate: its HCE weights and source patches, as Dev would get them
        # ...which is OTHER as Dev only while the two engine sources are the same search.
        # crossfish_dev.hpp is where search experiments are made, so check rather than assume.
        if engine_code(prev, "CrossfishPrev") != engine_code((CPP / "crossfish_dev.hpp").read_text(encoding="utf-8"),
                                                             "CrossfishDev"):
            raise SystemExit("--prev-dir: cpp_impl/crossfish_dev.hpp and crossfish_prev.hpp differ beyond comments"
                             " and the class name, so Prev would not play OTHER as Dev plays it; make them the same"
                             " engine first")
        prev = patch_engine(prev, other, "Prev", "crossfish_prev.hpp")
    prev = prev.replace('#include "mini_eval_d16.hpp"', '#include "prev_mini_eval_d16.hpp"')
    prev = prev.replace('#include "macro_eval.hpp"', '#include "prev_macro_eval.hpp"')
    (out / "crossfish_prev.hpp").write_text(rename_prev(prev), encoding="utf-8", newline="\n")
    for name in ("mini_eval_d16.hpp", "macro_eval.hpp"):
        text = ((other or CPP) / name).read_text(encoding="utf-8")
        (out / f"prev_{name}").write_text(rename_prev(text), encoding="utf-8", newline="\n")
    if other:
        src = {name: file_digest(other / name) for name in ("mini_eval_d16.hpp", "macro_eval.hpp", "hce_weights.json",
                                                              "dev_patches.json") if (other / name).exists()}
        (out / PREV_SOURCE).write_text(json.dumps({"prev_dir": str(other.resolve()), "files": src}, indent=1))
        print(f"Prev from {other}")
    else:
        (out / PREV_SOURCE).unlink(missing_ok=True)  # a stale record from an earlier --prev-dir build
    dev = (CPP / "crossfish_dev.hpp").read_text(encoding="utf-8")
    dev = patch_engine(dev, out, "Dev", "crossfish_dev.hpp")
    (out / "crossfish_dev.hpp").write_text(dev, encoding="utf-8", newline="\n")
    shutil.copy(CPP / "test_bots.cpp", out / "test_bots.cpp")
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""))
    exe = out / "test_bots.exe"
    cmd = [str(TOOLCHAIN / "g++.exe"), *CXXFLAGS, f"-I{CPP}", "-o", str(exe), str(out / "test_bots.cpp")]
    subprocess.run(cmd, check=True, env=env)
    for tool in ("datagen", "bench_ab"):  # bench_ab equiv: an unchanged eval must compute Prev's exact tree
        shutil.copy(CPP / f"{tool}.cpp", out / f"{tool}.cpp")  # sprt_worker.py setup builds datagen from it
        if args.no_tools:
            (out / f"{tool}.exe").unlink(missing_ok=True)  # a stale build would not match these sources
            continue
        cmd = [str(TOOLCHAIN / "g++.exe"), *CXXFLAGS, f"-I{CPP}", "-o", str(out / f"{tool}.exe"), str(out / f"{tool}.cpp")]
        subprocess.run(cmd, check=True, env=env)
    print(f"built {exe}{'' if args.no_tools else ', datagen.exe and bench_ab.exe'} in {out}")


VERIFY_ROWS = 20000  # --train-prefix: rows of the run's first --data file to relabel


def cmd_verify(args):
    import torch
    import eval_data
    import nnue_train_blend as tb
    out = Path(args.dir)
    if not (args.sample or args.train_prefix):
        raise SystemExit("verify needs SAMPLE or --train-prefix")
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""))
    model = tb.Eval(tb.load_shipped_mini(out / "mini_eval_d16.hpp"), tb.load_shipped_macro(out / "macro_eval.hpp"))
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long()  # noqa: E731

    def label(src, name, depth, threads):
        """Label SRC with the candidate's datagen: its search-labeled rows, their eval_diffs and SRC's rows."""
        lab = out / f"{name}.cfdg"
        lab.unlink(missing_ok=True)
        subprocess.run([str(out / "datagen.exe"), "label", str(src), str(lab), str(depth), str(threads)], check=True,
                       env=env, stdout=subprocess.DEVNULL)
        subprocess.run([str(out / "datagen.exe"), "diffs", str(lab), str(out / f"{name}.diffs")], check=True, env=env,
                       stdout=subprocess.DEVNULL)
        full = np.array(eval_data.load(lab))
        keep = (full["flags"] & eval_data.F_SEARCH) != 0
        diffs = np.fromfile(out / f"{name}.diffs", dtype=np.int8).reshape(-1, 10).astype(np.int64)
        if not keep.any():
            raise SystemExit(f"no search-labeled rows in {src}")
        return full[keep], diffs[keep], np.array(eval_data.load(src))[keep]

    def check(rec, diffs, orig, dw):
        """Candidate static eval minus the Python model on the candidate's HCE, and
        max |candidate HCE - (ORIG's HCE + the weight change)|."""
        mini_idx, super_idx, constr, sgn = tb.features(rec)
        with torch.no_grad():
            e, _, _ = model(T(mini_idx), T(super_idx), T(constr), torch.from_numpy(rec["hce"].astype(np.float32)),
                            torch.from_numpy(sgn.astype(np.float32)))
        return e.numpy() - rec["static_eval"], int(np.abs(rec["hce"] - orig["hce"] - sgn * (diffs @ dw)).max())

    bad = False
    if args.sample:
        weights = json.loads((out / "hce_weights.json").read_text()) if (out / "hce_weights.json").exists() else None
        dw = np.array(weights) - np.array(tb.HCE_WEIGHTS) if weights else np.zeros(10, dtype=np.int64)
        rec, diffs, orig = label(args.sample, "verify", 1, 8)
        d, hce = check(rec, diffs, orig, dw)
        # The labeled sample's own hce column was written by whichever datagen labeled it first,
        # and a patched Dev (dev_patches.json) changes the HCE by more than its weights.
        if not (orig["flags"] & eval_data.F_SEARCH).all() or (out / "dev_patches.json").exists():
            hce = None
        print(f"verify {out}: {len(rec):,} positions, static eval vs python: mean|d| {np.abs(d).mean():.2f},"
              f" max|d| {np.abs(d).max():.0f}; HCE change vs shipped + weight delta: "
              f"{'max|d| ' + str(hce) if hce is not None else 'skipped (sample not shipped-labeled, or Dev patched)'}")
        bad |= bool(np.abs(d).max() > 2 or (hce is not None and hce > 0))
    if args.train_prefix:
        # The HCE the net was trained on is the training data's own hce column (whatever
        # Dev, patches included, labeled it), shifted by the trained weight change.
        meta = json.loads(Path(str(args.train_prefix) + ".json").read_text())
        src = Path(meta["args"]["data"][0])
        src = src if src.is_absolute() else ROOT / src
        part = out / "verify_train_in.cfdg"
        np.array(eval_data.load(src)[:VERIFY_ROWS]).tofile(part)
        rec, diffs, orig = label(part, "verify_train", 0, 2)  # depth 0: statics only, flags kept
        d, hce = check(rec, diffs, orig, np.array(meta["hce_weights"]) - np.array(tb.HCE_WEIGHTS))
        print(f"verify {out} on {src.name}: {len(rec):,} positions, static eval vs python: mean|d|"
              f" {np.abs(d).mean():.2f}, max|d| {np.abs(d).max():.0f}; HCE change vs training + weight delta:"
              f" max|d| {hce}")
        bad |= bool(np.abs(d).max() > 2 or hce > 0)
    if bad:
        raise SystemExit("candidate build does not match the trained model")


def cmd_match(args):
    out = Path(args.dir)
    log = ROOT / "datasets" / "eval2" / "sprt" / f"{args.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""))
    if args.games:
        env["SPRT_MAX_GAMES"] = str(args.games)
    if args.offset:
        env["SPRT_GAME_OFFSET"] = str(args.offset)
    if args.depth:
        cmd = [str((out / "test_bots.exe").resolve()), "depth", str(args.depth)]
        env.setdefault("SPRT_LLR_BOUND", "100")  # a screen: fixed game count, no early stop
    else:
        cmd = [str((out / "test_bots.exe").resolve())]
        env["SPRT_THINK_MS"] = str(args.ms)
        # test_bots counts logical CPUs as physical on this machine (8 cores / 16
        # threads), so a timed run must set its thread count or it oversubscribes.
        env["SPRT_THREADS"] = str(args.threads)
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=CPP).returncode
    verdict = re.search(r"^SPRT (PASS|FAIL|INCONCLUSIVE)", log.read_text(encoding="utf-8", errors="replace"), re.M)
    print(f"{args.name}: exit {rc}{'' if verdict else ', no SPRT verdict'}; see {log}")
    if rc != 0 or not verdict:
        sys.exit(rc if 0 < rc < 256 else 1)  # a crash's Windows status (0xC0000005) overflows sys.exit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("emit"); p.add_argument("prefix"); p.add_argument("dir")
    p.add_argument("--recluster", action="store_true"); p.set_defaults(fn=cmd_emit)
    p = sub.add_parser("build"); p.add_argument("dir")
    p.add_argument("--prev-dir", help="Prev plays this candidate DIR (headers, HCE weights, dev_patches.json)"
                                      " instead of the shipped eval")
    p.add_argument("--no-tools", action="store_true", help="compile test_bots only, not datagen and bench_ab")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("verify"); p.add_argument("dir"); p.add_argument("sample", nargs="?")
    p.add_argument("--train-prefix", help="nnue_train_blend.py --out of the run: check the HCE against its data")
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("match"); p.add_argument("dir"); p.add_argument("name")
    p.add_argument("--depth", type=int, default=0); p.add_argument("--ms", type=int, default=90)
    p.add_argument("--games", type=int, default=0); p.add_argument("--offset", type=int, default=0)
    p.add_argument("--threads", type=int, default=6, help="timed runs: game threads (SPRT_THREADS)")
    p.set_defaults(fn=cmd_match)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
