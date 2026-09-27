#!/usr/bin/env python3
"""Write cpp_impl/bin/cand_fast/ (or DIR): the fast integer NNUE wired into the Dev engine.

  make_cand.py [DIR]

Copies fast_nnue.hpp and the Dev engine headers (mini_eval_d16.hpp, macro_eval.hpp, hce_weights.json of
the local build cpp_impl/bin/cand_fnn1, or the shipped cpp_impl/ headers without it: see
copy_engine_headers) into DIR and writes DIR/dev_patches.json. The eval call sites are cand_fnn1's
(qsearch stand-pat, the interior static eval, the depth-1 RFP MiniNet prefilter removed,
evaluate()), so results are comparable with the float per-cell hook; on top of those the patch
records every make_move_fast(FastBoard&) for the lazy accumulator stack (with the parent's
position key, MoveUndo.tt_hash, since stage 3) and refreshes the root accumulators wherever a
search builds its FastBoard (getMove, search_fixed_depth). Since stage 4 the include is guarded
(ENGINE_GUARD), so two fast engines cannot silently share one net in a two-engine build.
Then: tools/eval_candidate.py build DIR. make_cand_b.py (the B nets and compile-time kinds) builds on these
patches; build_cand.sh does the whole candidate build.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FNN1 = ROOT / "cpp_impl" / "bin" / "cand_fnn1"
SHIPPED = ROOT / "cpp_impl"


def copy_engine_headers(out):
    """The Dev engine's MiniNet / macro headers and HCE weights, which tools/eval_candidate.py build needs in
    every candidate directory (the NNUE patches leave them unread). From cpp_impl/bin/cand_fnn1 when that
    local build exists (every recorded candidate was made so), else the shipped cpp_impl/ headers: the same
    weights, and no hce_weights.json, which eval_candidate reads as the shipped HCE weights."""
    if (FNN1 / "dev_patches.json").exists():
        for name in ("mini_eval_d16.hpp", "macro_eval.hpp", "hce_weights.json"):
            shutil.copy(FNN1 / name, out / name)
        return {p["old"] for p in json.loads((FNN1 / "dev_patches.json").read_text())}
    for name in ("mini_eval_d16.hpp", "macro_eval.hpp"):
        shutil.copy(SHIPPED / name, out / name)
    (out / "hce_weights.json").unlink(missing_ok=True)
    return set()

# Stage 4: two engines of one build (test_bots's Dev and Prev) must not include the same fast NNUE
# headers, or #pragma once would hand the second engine the first one's net. fast_pair.py gives Prev a
# renamed copy (FASTNNUE_PREV_ENGINE, FNNUE_PREV_KERNEL, prev_fast_nnue*.hpp, namespace fnnue_prev); any
# other two-engine build of two fast candidates stops here instead of silently sharing one net.
# Stage 5: the guard also trips on FNNUE_KERNEL, which fast_nnue.hpp itself defines. A candidate made
# before the guard existed (stages 1-3) defines no FASTNNUE_ENGINE, so a stage-4 candidate paired after
# one of them used to compile; now it stops, whichever side the older candidate is on (the stage-1 to 3
# candidate directories were given the guard too, by a one-off stage-5 script).
ENGINE_GUARD = ('#if defined(FASTNNUE_ENGINE) || defined(FNNUE_KERNEL)\n'
                '#error "a second engine includes this fast NNUE: build two-net pairings with'
                ' tools/experiments/fast_nnue/fast_pair.py"\n'
                '#endif\n'
                '#define FASTNNUE_ENGINE 1\n')

PATCHES = [
    {
        "old": '#include "macro_eval.hpp"',
        "new": '#include "macro_eval.hpp"\n' + ENGINE_GUARD + '#include "fast_nnue.hpp"',
        "count": 1,
    },
    {   # the accumulator stack lives next to the other per-ply undo state
        "old": "        std::array<MoveUndo, 128> move_undo{};",
        "new": "        std::array<MoveUndo, 128> move_undo{};\n"
               "        // EXPERIMENT: fast integer NNUE accumulators (tools/experiments/fast_nnue).\n"
               "        fnnue::Stack fnnue_stack;",
        "count": 1,
    },
    {   # make_move_fast(FastBoard&) only (the template overload says move.mini_board)
        "old": "            board.n_moves++;\n"
               "            if (hce_acc_ready) {\n"
               "                set_hce_mb(board, mb);",
        "new": "            fnnue_stack.on_make(board, mb, move.square, stm, decided_state,\n"
               "                                before, board.mini_boards[mb].markers[stm ^ 1], u.tt_hash);\n"
               "            board.n_moves++;\n"
               "            if (hce_acc_ready) {\n"
               "                set_hce_mb(board, mb);",
        "count": 1,
    },
    {   # getMove and search_fixed_depth: the only places a search's FastBoard is built
        "old": "            init_macro_key(board);\n"
               "            sync_terminal(board);",
        "new": "            init_macro_key(board);\n"
               "            sync_terminal(board);\n"
               "            fnnue_stack.refresh_root(board);",
        "count": 2,
    },
    {
        "old": "            int hce = evaluate_hce_incremental(board)\n"
               "                    + qstruct.applied;\n"
               "            if (hce - QHCE_FAIL_HIGH_MARGIN >= beta) {\n"
               "                return beta;\n"
               "            }\n"
               "            int stand_pat;\n"
               "            if (hce + MINI_MAX + MACRO_CLIP < alpha) {\n"
               "                stand_pat = hce + MINI_MAX + MACRO_CLIP;\n"
               "            } else {\n"
               "                stand_pat = hce + evaluate_mini_cached(board)\n"
               "                          + evaluate_macro_cached(board);\n"
               "            }",
        "new": "            // EXPERIMENT: the fast NNUE replaces HCE + MiniNet + macro.\n"
               "            int stand_pat = fnnue_stack.evaluate_keyed(\n"
               "                                board, active_board_index(board), board.tt_hash)\n"
               "                          + qstruct.applied;",
        "count": 1,
    },
    {
        "old": "                static_eval = corrected_eval(\n"
               "                    evaluate_hce_incremental(board), static_corr_refs);",
        "new": "                static_eval = corrected_eval(\n"
               "                    fnnue_stack.evaluate_keyed(\n"
               "                        board, active_board_index(board), board.tt_hash),\n"
               "                    static_corr_refs);",
        "count": 1,
    },
    {
        "old": "                    if (depth == 1\n"
               "                        && static_eval + 2500 - reverse_futility_margin >= beta\n"
               "                        && static_eval + evaluate_mini_cached(board)\n"
               "                           - reverse_futility_margin >= beta) {\n"
               "                        return beta;\n"
               "                    }\n",
        "new": "",
        "count": 1,
    },
    {
        "old": "            return evaluate_hce(board)\n"
               "                 + d16_evaluate_mini_fast(board)\n"
               "                 + evaluate_macro_fast(board);",
        "new": "            return fnnue::evaluate_board(board, d16_mini_board_constraint(board));",
        "count": 1,
    },
]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cpp_impl" / "bin" / "cand_fast"
    out.mkdir(parents=True, exist_ok=True)
    fnn1 = copy_engine_headers(out)
    shutil.copy(HERE / "fast_nnue.hpp", out / "fast_nnue.hpp")
    # The cand_fnn1 call sites (when that build exists) must still be the ones this patch replaces.
    mine = {p["old"] for p in PATCHES}
    assert len(fnn1 - mine) == 0, "cand_fnn1 patches a site this candidate does not"
    (out / "dev_patches.json").write_text(json.dumps(PATCHES, indent=1))
    print(f"wrote {out}/dev_patches.json ({len(PATCHES)} patches) and copied fast_nnue.hpp")


if __name__ == "__main__":
    main()
