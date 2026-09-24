"""Write crossfish_leaf.hpp: Dev with a sampler at every qsearch entry.

usage: python tools/experiments/full_nnue/make_leaf_engine.py OUT_DIR
Each qsearch node is recorded with probability leaf_threshold / 4096 in the
93-byte UTTTAI state format that test_bots' dump tools read.
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
s = (root / "cpp_impl" / "crossfish_dev.hpp").read_text()
s = s.replace("CrossfishDev", "CrossfishLeaf")
old = """        int qsearch(FastBoard &board, int alpha, int beta, int ply) {
            if (time_up()) return min_val;
            nodes++;
"""
new = old + """            if (leaf_sink && (leaf_rng = leaf_rng * 6364136223846793005ull
                                  + 1442695040888963407ull) >> 52 < leaf_threshold) {
                std::string st(93, '0');
                for (int mb = 0; mb < 9; mb++)
                    for (int sq = 0; sq < 9; sq++) {
                        if (board.mini_boards[mb].markers[0] >> sq & 1) st[mb * 9 + sq] = '1';
                        else if (board.mini_boards[mb].markers[1] >> sq & 1) st[mb * 9 + sq] = '2';
                    }
                for (int mb = 0; mb < 9; mb++) {
                    if (board.mini_board_states[0] >> mb & 1) st[81 + mb] = '1';
                    else if (board.mini_board_states[1] >> mb & 1) st[81 + mb] = '2';
                    else if (board.mini_board_states[2] >> mb & 1) st[81 + mb] = '3';
                }
                st[90] = (board.n_moves & 1) ? '2' : '1';
                st[91] = (char)('0' + active_board_index(board));
                leaf_sink->push_back(st);
            }
"""
assert s.count(old) == 1, "qsearch entry changed; update the anchor"
s = s.replace(old, new)
s = s.replace("    public:\n        int root_score;",
              "    public:\n        std::vector<std::string> *leaf_sink = nullptr;\n"
              "        uint64_t leaf_rng = 88172645463325252ull;\n"
              "        uint64_t leaf_threshold = 0;\n        int root_score;", 1)
out = Path(sys.argv[1]) / "crossfish_leaf.hpp"
out.write_text(s)
print("wrote", out)
