#!/usr/bin/env python3
"""Write cpp_impl/bin/cand_fastb/ (or DIR): the fast NNUE wired into the Dev engine, with runtime format
dispatch (per-cell FNN1 or pattern-generator BGN1, chosen by the magic of the FASTNNUE_PATH file) or,
with --kind, one net kind fixed at compile time.

  make_cand_b.py [DIR] [--kind any|cell|b64|b128|b64s|b128s] [--keep-dead-hce] [--net NET]

Same call sites as stage 1's make_cand.py (and so as cand_fnn1: qsearch stand-pat, the interior static
eval, the depth-1 RFP MiniNet prefilter removed, evaluate()), the same make hook and root refreshes;
the differences are the type (fnnue::AnyStack from fast_nnue_any.hpp), the make hook passing the
FastBoard (the B stack prefetches the child's table rows and eval-cache entry from it) and the parent's
position key (MoveUndo.tt_hash), and evaluate() going through fnnue::evaluate_any.

Stage 3 additions:
  --kind K         #define FASTNNUE_KIND KIND_<K> before the include: AnyStack is that kind's stack with
                   no runtime switch (the net file must be of that kind). Default any: runtime dispatch.
  dead HCE work    with the NNUE eval, make_move_fast(FastBoard&)'s HCE local-score table lookup, its
                   global-term recomputation on a decided board, and the MiniNet code update are never
                   read (their only readers, evaluate_hce_incremental / evaluate_mini_cached /
                   evaluate_macro_cached, are replaced by the patches above). Two more patches drop them
                   and keep only the tiar maps (read by has_immediate_global_win and
                   has_forced_global_win_after_reply) and hce_mb_flags. --keep-dead-hce leaves them.
Stage 4 additions:
  --net NET        #define FASTNNUE_NET_FILE "<absolute NET>" before the include: the engine loads NET
                   unless FASTNNUE_PATH is set, so a two-net pairing (fast_pair.py, round robins) needs
                   no environment. The include also carries make_cand.ENGINE_GUARD.
Copies the Dev engine headers (make_cand.copy_engine_headers: cpp_impl/bin/cand_fnn1's, else the shipped
ones) and fast_nnue.hpp, fast_nnue_b.hpp, fast_nnue_any.hpp into DIR. Then: tools/eval_candidate.py build
DIR (build_cand.sh does both, plus the check builds).
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_cand  # noqa: E402  (stage 1: its PATCHES are the base)

ROOT = make_cand.ROOT
FNN1 = make_cand.FNN1

MAKE_HOOK = "            fnnue_stack.on_make(board, mb, move.square, stm, decided_state, before, u.tt_hash);\n"

SUBST = [  # (stage-1 text, stage-2 text), each must occur in exactly the patches noted
    ('#include "fast_nnue.hpp"', '#include "fast_nnue_any.hpp"'),
    ("        fnnue::Stack fnnue_stack;", "        fnnue::AnyStack fnnue_stack;"),
    ("            fnnue_stack.on_make(board, mb, move.square, stm, decided_state,\n"
     "                                before, board.mini_boards[mb].markers[stm ^ 1], u.tt_hash);\n",
     MAKE_HOOK),
    ("fnnue::evaluate_board(board, d16_mini_board_constraint(board))",
     "fnnue::evaluate_any(board, d16_mini_board_constraint(board))"),
]

KINDS = {"cell": "KIND_CELL", "b64": "KIND_B64", "b128": "KIND_B128", "b64s": "KIND_B64S", "b128s": "KIND_B128S"}

# Applied after the stage-1/2 patches (the first one's old text contains the make hook they insert).
DEAD_HCE = [
    {
        "old": MAKE_HOOK
               + "            board.n_moves++;\n"
                 "            if (hce_acc_ready) {\n"
                 "                set_hce_mb(board, mb);\n"
                 "                if (decided_state >= 0) {\n"
                 "                    hce_global_score = evaluate_hce_global(board);\n"
                 "                }\n"
                 "            }\n",
        "new": MAKE_HOOK
               + "            board.n_moves++;\n"
                 "            if (hce_acc_ready) {\n"
                 "                // EXPERIMENT (fast NNUE): only the tiar maps of the HCE state are still read\n"
                 "                // (has_immediate_global_win, has_forced_global_win_after_reply); the local\n"
                 "                // score lookup and the global terms were the HCE eval's.\n"
                 "                const int nn_bit = 1 << mb;\n"
                 "                const int nn_flags = (board.out_of_play & nn_bit) ? 0\n"
                 "                    : fast_tiar_flags[(board.mini_boards[mb].markers[0] << 9)\n"
                 "                                      | board.mini_boards[mb].markers[1]];\n"
                 "                hce_mb_flags[mb] = (uint8_t)nn_flags;\n"
                 "                hce_tiar_maps[0] = (hce_tiar_maps[0] & ~nn_bit) | ((nn_flags & 1) << mb);\n"
                 "                hce_tiar_maps[1] = (hce_tiar_maps[1] & ~nn_bit) | (((nn_flags >> 1) & 1) << mb);\n"
                 "            }\n",
        "count": 1,
    },
    {   # make_move_fast(FastBoard&) only (the template overload says move.mini_board)
        "old": "            board.mini_boards[mb].markers[stm] = before | bit;\n"
               "            update_mini_code(board, mb);\n"
               "            xor_move_combo(board, stm, mb, move.square);",
        "new": "            board.mini_boards[mb].markers[stm] = before | bit;\n"
               "            // EXPERIMENT (fast NNUE): the MiniNet codes are not read with the NNUE eval.\n"
               "            xor_move_combo(board, stm, mb, move.square);",
        "count": 1,
    },
]


def patches(kind="any", drop_dead_hce=True, net=None):
    out = copy.deepcopy(make_cand.PATCHES)
    for old, new in SUBST:
        hits = [p for p in out if old in p["new"]]
        assert len(hits) == 1, f"stage-1 patch text not found exactly once: {old!r}"
        hits[0]["new"] = hits[0]["new"].replace(old, new)
    for p in out:
        assert "fnnue::Stack " not in p["new"] and "evaluate_board(" not in p["new"], p
    inc = [p for p in out if '#include "fast_nnue_any.hpp"' in p["new"]]
    assert len(inc) == 1
    defines = ""
    if kind != "any":
        defines += f"#define FASTNNUE_KIND {KINDS[kind]}\n"
    if net:  # stage 4: the net file compiled in (FASTNNUE_PATH still overrides it)
        path = Path(net).resolve()
        assert path.is_file(), f"--net {path} not found"
        assert '"' not in path.as_posix() and "\\" not in path.as_posix()
        defines += f'#define FASTNNUE_NET_FILE "{path.as_posix()}"\n'
    inc[0]["new"] = inc[0]["new"].replace('#include "fast_nnue_any.hpp"', defines + '#include "fast_nnue_any.hpp"')
    if drop_dead_hce:
        out += copy.deepcopy(DEAD_HCE)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(ROOT / "cpp_impl" / "bin" / "cand_fastb"))
    ap.add_argument("--kind", default="any", choices=["any", *KINDS])
    ap.add_argument("--keep-dead-hce", action="store_true")
    ap.add_argument("--net", help="compile this net file in (FASTNNUE_NET_FILE; FASTNNUE_PATH still overrides it)")
    a = ap.parse_args()
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    fnn1 = make_cand.copy_engine_headers(out)
    for name in ("fast_nnue.hpp", "fast_nnue_b.hpp", "fast_nnue_any.hpp"):
        shutil.copy(HERE / name, out / name)
    ps = patches(a.kind, not a.keep_dead_hce, a.net)
    assert not fnn1 - {p["old"] for p in ps}, "cand_fnn1 patches a site this candidate does not"
    (out / "dev_patches.json").write_text(json.dumps(ps, indent=1))
    print(f"wrote {out}/dev_patches.json ({len(ps)} patches, kind {a.kind}, dead HCE work "
          f"{'kept' if a.keep_dead_hce else 'dropped'}, net {Path(a.net).resolve().as_posix() if a.net else 'from FASTNNUE_PATH'})"
          f" and copied the fast_nnue headers")


if __name__ == "__main__":
    main()
