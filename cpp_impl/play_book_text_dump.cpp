// Writes the shipped book back out as a text book ("S <seq> <move>" lines,
// play_book_text.hpp), so it can be re-packed after a change to the coder
// without the generators' toolchains (the uttt.ai fork, the P2 book builder).
// A position with several book moves (payload format 2) gets one line per
// move, the primary first, which is how the packer reads them back: packing
// the output again gives the same payload.
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

static_assert(PLAY_BOOK_FORMAT == 2, "the dump decodes payload format 2 only");

static void print_line(GlobalBoard &b, const Move &m) {
    auto h = b.move_history;
    std::vector<int> seq;
    while (!h.empty()) { seq.push_back(h.top().mini_board * 9 + h.top().square); h.pop(); }
    std::printf("S ");
    if (seq.empty()) std::printf("-");
    for (int i = (int)seq.size() - 1; i >= 0; i--) std::printf("%d%s", seq[i], i ? "," : "");
    std::printf(" %d\n", m.mini_board * 9 + m.square);
}

int main() {
    crossfish_nnue_load_once();
    static unsigned char buf[PLAY_BOOK_BYTES + 16];
    int n = d16_mini_cjk_decode(PLAY_BOOK_CJK, buf, (int)sizeof(buf));
    if (n < PLAY_BOOK_BYTES) { std::fprintf(stderr, "payload too short\n"); return 1; }
    struct Hooks {
        int (*eval)(const PbView &, int);
        PbDecoder dec;
        PbModels m;
        int further = 0;  // non-primary moves written
        int choose(GlobalBoard &b, Move *legal, const int *order, int n_legal) {
            int r = pb_code_rank([&](uint16_t &p, int) { return dec.bit(p); }, m, n_legal, 0);
            print_line(b, legal[order[r]]);
            return r;
        }
        bool another(GlobalBoard &, int k) { return dec.bit(m.another[k - 1]) != 0; }
        int choose_alt(GlobalBoard &b, Move *legal, const int *order, int n_legal, int) {
            further++;
            return choose(b, legal, order, n_legal);
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
    std::fprintf(stderr, "wrote %d positions (%d further moves)\n", walker.entries, hooks.further);
    return 0;
}
