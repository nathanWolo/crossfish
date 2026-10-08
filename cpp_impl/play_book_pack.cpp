// Packs a text book (play_book_gen, or tools that write "S" lines) into
// play_book_data.hpp, payload format 2 (several moves per position; see
// play_book.hpp and play_book_text.hpp).
//
// Walks the book with the same PbWalker the CodinGame bot uses to decode it,
// so the stored order cannot drift from the runtime order, and codes each
// decision with the same adaptive binary models and the same NNUE evaluator
// (play_book.hpp), so the probabilities cannot drift either. The text book only
// lists our positions: an opponent reply is covered when it leads to one of
// them, and the book goes on after one of our moves when any reply is covered.
// A position listed on several "S" lines stores each listed move (at most
// PB_MAX_MOVES; the first is the primary) and the walk follows each of them.
// Fails if a position lists more than PB_MAX_MOVES moves, if a line's move is
// illegal, if the walk reaches a position the text book lacks, or if any
// text-book entry is never reached (for example a line that does not start
// from center-center).
//
//   play_book_pack <book.txt> <out.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "global_board.hpp"
#include "crossfish_dev.hpp"
#include "play_book_text.hpp"
#define PLAY_BOOK_NO_DATA
#include "play_book.hpp"

// Binary range encoder, the mirror of PbDecoder (LZMA's carry handling).
struct PbEncoder {
    std::vector<unsigned char> out;
    uint64_t low = 0, cache_size = 1;
    uint32_t range = 0xFFFFFFFFu;
    unsigned char cache = 0;
    void shift_low() {
        if ((uint32_t)low < 0xFF000000u || (low >> 32) != 0) {
            unsigned char carry = (unsigned char)(low >> 32), temp = cache;
            do { out.push_back((unsigned char)(temp + carry)); temp = 0xFF; } while (--cache_size != 0);
            cache = (unsigned char)(low >> 24);
        }
        cache_size++;
        low = (low & 0x00FFFFFFu) << 8;
    }
    int bit(uint16_t &p, int b) {
        uint32_t bound = (range >> 12) * p;
        if (b == 0) range = bound; else { low += bound; range -= bound; }
        pb_adapt(p, b);
        while (range < (1u << 24)) { range <<= 8; shift_low(); }
        return b;
    }
    void finish() { for (int i = 0; i < 5; i++) shift_low(); }
};

// Same bit layout as tools/nnue_cjk14.py encode_u15: 15-bit groups, MSB first, on the U15 alphabet.
static std::string encode_u15(const std::vector<unsigned char> &data) {
    std::string out;
    uint32_t acc = 0;
    int bits = 0, column = 0;
    auto emit = [&](uint32_t v) {
        uint32_t code = v < 27648u ? 0x3400 + v : 0xE000 + v - 27648u;
        out += (char)(0xE0 | (code >> 12));
        out += (char)(0x80 | ((code >> 6) & 63));
        out += (char)(0x80 | (code & 63));
        if (++column == 64) { out += '\n'; column = 0; }
    };
    for (unsigned char byte : data) {
        acc = (acc << 8) | byte;
        bits += 8;
        while (bits >= 15) {
            bits -= 15;
            emit((acc >> bits) & 0x7FFF);
        }
        acc &= (1u << bits) - 1;
    }
    if (bits) emit((acc << (15 - bits)) & 0x7FFF);
    if (column == 0 && !out.empty()) out.pop_back();
    return out;
}

static int pb_nnue_eval(const PbView &v, int c) { return b64::evaluate_board(v, c); }

// The packer's walk decisions, coded as they are made.
struct PackHooks {
    Book &book;
    int (*eval)(const PbView &, int) = pb_nnue_eval;
    PbEncoder enc;
    PbModels m;
    int missing = 0, expanded = 0, positions = 0, covered_bits = 0;
    int not_legal = 0;   // book moves that are not among the walk's legal moves
    int alt_coded = 0;   // further (non-primary) moves coded
    int by_count[PB_MAX_MOVES + 1] = {};  // positions by number of stored moves
    double uniform_bits = 0;  // what the plain digits would have cost (the "another" bits are not counted)
    std::vector<Move> cur;    // the current position's book moves, real orientation, primary first

    static int position_in(const Move &bm, const Move *legal, const int *order, int n) {
        for (int r = 0; r < n; r++)
            if (legal[order[r]].mini_board == bm.mini_board && legal[order[r]].square == bm.square) return r;
        return -1;
    }

    int choose(GlobalBoard &b, Move *legal, const int *order, int n) {
        int rank = 0;
        if (book.lookup_all(b, cur)) {
            int r = position_in(cur[0], legal, order, n);
            if (r >= 0) rank = r; else not_legal++;
            by_count[(int)cur.size() < PB_MAX_MOVES ? (int)cur.size() : PB_MAX_MOVES]++;
        } else {
            cur.clear();
            missing++;
        }
        positions++;
        uniform_bits += std::log2((double)n);
        return pb_code_rank([&](uint16_t &p, int bit) { return enc.bit(p, bit); }, m, n, rank);
    }
    bool another(GlobalBoard &, int k) {  // k moves stored so far
        bool yes = (int)cur.size() > k;
        return enc.bit(m.another[k - 1], yes) != 0;
    }
    // Rank in `order` of the k-th (0-based) book move, coded like the primary's.
    int choose_alt(GlobalBoard &, Move *legal, const int *order, int n, int k) {
        int rank = position_in(cur[k], legal, order, n);
        if (rank < 0) { not_legal++; rank = 0; }
        alt_coded++;
        uniform_bits += std::log2((double)n);
        return pb_code_rank([&](uint16_t &p, int bit) { return enc.bit(p, bit); }, m, n, rank);
    }
    bool more(GlobalBoard &b) {  // opponent to move in b: is any reply covered?
        Move legal[81];
        int n = b.fillLegalMoves(legal);
        bool any = false;
        for (int i = 0; i < n && !any; i++) {
            b.makeMove(legal[i]);
            any = b.checkWinner() == -1 && book.has(b);
            b.unmakeMove();
        }
        uniform_bits += 1;
        expanded += any;
        return enc.bit(m.cont[PbModels::ply_ctx(b.n_moves)], any) != 0;
    }
    bool covers(GlobalBoard &b, int rank) {  // b: after the opponent's reply
        bool yes = book.has(b);
        uniform_bits += 1;
        covered_bits++;
        return enc.bit(m.cover[PbModels::ply_ctx(b.n_moves)][rank < 7 ? rank : 7], yes) != 0;
    }
};

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: play_book_pack <book.txt> <out.hpp>\n");
        return 2;
    }
    Book book;
    if (!book.load(argv[1])) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    if (book.illegal) {
        std::fprintf(stderr, "%d book line(s) with an illegal move; the first: %s\n", book.illegal,
                     book.first_illegal.c_str());
        return 1;
    }
    // At most PB_MAX_MOVES distinct moves per position; a position is reported by its lines.
    int too_many = 0, alt_total = 0;
    for (auto &kv : book.entries) {
        int k = (int)kv.second.moves.size();
        alt_total += k > 1 ? k - 1 : 0;
        if (k > PB_MAX_MOVES && too_many++ < 5) {
            std::string ls;
            for (const std::string &l : kv.second.lines) ls += (ls.empty() ? "\"" : "\", \"") + l;
            std::fprintf(stderr, "one position has %d distinct moves, on the lines %s\"\n", k, ls.c_str());
        }
    }
    if (too_many) {
        std::fprintf(stderr, "%d position(s) list more than %d moves; a position may store at most %d. Not packed.\n",
                     too_many, PB_MAX_MOVES, PB_MAX_MOVES);
        return 1;
    }
    crossfish_nnue_load_once();

    PackHooks hooks{book};
    PbWalker<GlobalBoard, Move, PackHooks> walker{hooks};
    walker.run();
    if (hooks.missing || walker.entries != (int)book.entries.size()) {
        std::fprintf(stderr, "walk/book mismatch: walk visited %d positions, text book has %zu, %d missing\n",
                     walker.entries, book.entries.size(), hooks.missing);
        return 1;
    }
    if (hooks.not_legal || hooks.alt_coded != alt_total) {
        std::fprintf(stderr, "move mismatch: %d book move(s) not legal in the walk; %d of %d further moves coded\n",
                     hooks.not_legal, hooks.alt_coded, alt_total);
        return 1;
    }

    hooks.enc.finish();
    std::vector<unsigned char> &bytes = hooks.enc.out;
    double info_bits = hooks.uniform_bits;
    std::string text = encode_u15(bytes);
    size_t chars = 0;
    for (unsigned char ch : text) chars += (ch & 0xC0) != 0x80 && ch != '\n';

    FILE *f = std::fopen(argv[2], "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }
    std::fprintf(f,
        "#pragma once\n"
        "// Generated by cpp_impl/play_book_pack.cpp (payload format 2). Do not edit; see documentation/play_book.md.\n"
        "// %d of our positions (%d with 2 moves, %d with 3 moves) and %d opponent positions whose\n"
        "// replies it covers, after the first player's center-center.\n\n"
        "static constexpr int PLAY_BOOK_FORMAT = 2;\n"
        "static constexpr int PLAY_BOOK_ENTRIES = %d;\n"
        "static constexpr int PLAY_BOOK_BYTES = %zu;\n"
        "// pb_eval_fingerprint of the net the moves were ordered with; pb_init refuses the payload under another.\n"
        "static constexpr uint64_t PLAY_BOOK_EVAL_FINGERPRINT = %lluull;\n\n"
        "static const char PLAY_BOOK_CJK[] = R\"~(\n%s\n)~\";\n",
        walker.entries, hooks.by_count[2], hooks.by_count[3], hooks.expanded + 1, walker.entries, bytes.size(),
        (unsigned long long)pb_eval_fingerprint(hooks.eval), text.c_str());
    std::fclose(f);
    std::printf("packed %d positions (%d expanded opponent positions; %d with 2 moves, %d with 3 moves): %.0f bits "
                "as plain digits, coded to %zu bytes, %zu payload characters -> %s\n",
                walker.entries, hooks.expanded + 1, hooks.by_count[2], hooks.by_count[3], info_bits, bytes.size(),
                chars, argv[2]);
    return 0;
}
