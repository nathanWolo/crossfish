// Writes the shipped book back out as a text book ("S <seq> <move>" lines,
// play_book_text.hpp), so it can be re-packed after a change to the coder
// without the generator's toolchain (the uttt.ai fork).
//
//   play_book_text_dump > play_book.txt
#include <cstdio>
#include <vector>

#include "global_board.hpp"
#include "crossfish_dev.hpp"
#include "play_book_text.hpp"
#define PLAY_BOOK_NO_DATA
#include "play_book.hpp"
#include "play_book_data.hpp"

int main() {
    crossfish_nnue_load_once();
    static unsigned char buf[PLAY_BOOK_BYTES + 16];
    int n = d16_mini_cjk_decode(PLAY_BOOK_CJK, buf, (int)sizeof(buf));
    if (n < PLAY_BOOK_BYTES) { std::fprintf(stderr, "payload too short\n"); return 1; }
    struct Hooks {
        int (*eval)(const PbView &, int);
        PbDecoder dec;
        PbModels m;
        int choose(GlobalBoard &b, Move *legal, const int *order, int n_legal) {
            int r = pb_code_rank([&](uint16_t &p, int) { return dec.bit(p); }, m, n_legal, 0);
            auto h = b.move_history;
            std::vector<int> seq;
            while (!h.empty()) { seq.push_back(h.top().mini_board * 9 + h.top().square); h.pop(); }
            std::printf("S ");
            if (seq.empty()) std::printf("-");
            for (int i = (int)seq.size() - 1; i >= 0; i--) std::printf("%d%s", seq[i], i ? "," : "");
            Move m = legal[order[r]];
            std::printf(" %d\n", m.mini_board * 9 + m.square);
            return r;
        }
        bool more(GlobalBoard &b) { return dec.bit(m.cont[PbModels::ply_ctx(b.n_moves)]) != 0; }
        bool covers(GlobalBoard &b, int rank) {
            return dec.bit(m.cover[PbModels::ply_ctx(b.n_moves)][rank < 7 ? rank : 7]) != 0;
        }
    } hooks{[](const PbView &v, int c) { return b64::evaluate_board(v, c); },
            PbDecoder{buf, n}, PbModels{}};
    hooks.dec.init();
    PbWalker<GlobalBoard, Move, Hooks> walker{hooks};
    walker.run();
    if (walker.entries != PLAY_BOOK_ENTRIES) { std::fprintf(stderr, "decoded %d entries, expected %d\n", walker.entries, PLAY_BOOK_ENTRIES); return 1; }
    std::fprintf(stderr, "wrote %d positions\n", walker.entries);
    return 0;
}
