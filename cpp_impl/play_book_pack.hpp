#pragma once
// The packer's coding side (play_book_pack.cpp), shared with the unit tests:
// the range encoder, the walk hooks that code each decision, the checks a text
// book must pass before it is packed, and the walk itself. Include after
// play_book.hpp (the packer defines PLAY_BOOK_NO_DATA first) and the net
// (crossfish_dev.hpp).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "global_board.hpp"
#include "play_book_text.hpp"
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

// Why a loaded text book cannot be packed, or "" when it can: a line with an
// illegal move, a line past the 25-ply cap, or a position with more than
// PB_MAX_MOVES distinct moves (reported by its lines, at most 5 positions).
// `alt_total` gets the number of further (non-primary) moves.
static std::string pb_pack_refusal(const Book &book, int &alt_total) {
    std::string err;
    char buf[160];
    if (book.illegal) {
        std::snprintf(buf, sizeof buf, "%d book line(s) with an illegal move; the first: ", book.illegal);
        err += buf + book.first_illegal + "\n";
    }
    if (book.too_deep) {
        std::snprintf(buf, sizeof buf,
                      "%d book line(s) past the 25-ply cap (a seq of more than %d moves); the first: ",
                      book.too_deep, PB_TEXT_MAX_INDEX);
        err += buf + book.first_too_deep + "\n";
    }
    int too_many = 0;
    alt_total = 0;
    for (auto &kv : book.entries) {
        int k = (int)kv.second.moves.size();
        alt_total += k > 1 ? k - 1 : 0;
        if (k > PB_MAX_MOVES && too_many++ < 5) {
            std::string ls;
            for (const std::string &l : kv.second.lines) ls += (ls.empty() ? "\"" : "\", \"") + l;
            std::snprintf(buf, sizeof buf, "one position has %d distinct moves, on the lines ", k);
            err += buf + ls + "\"\n";
        }
    }
    if (too_many) {
        std::snprintf(buf, sizeof buf, "%d position(s) list more than %d moves; a position may store at most %d.\n",
                      too_many, PB_MAX_MOVES, PB_MAX_MOVES);
        err += buf;
    }
    return err;
}

// Walks the book from both roots and codes every decision into hooks.enc
// (finished on success; PB_TABLE holds the walk's table as a side effect).
// False, with a message, when the walk and the text book disagree.
static bool pb_pack(PackHooks &hooks, int alt_total, int &entries, std::string &err) {
    PbWalker<GlobalBoard, Move, PackHooks> walker{hooks};
    walker.run();
    entries = walker.entries;
    char buf[200];
    if (hooks.missing || walker.entries != (int)hooks.book.entries.size()) {
        std::snprintf(buf, sizeof buf, "walk/book mismatch: walk visited %d positions, text book has %zu, %d missing",
                      walker.entries, hooks.book.entries.size(), hooks.missing);
        err = buf;
        return false;
    }
    if (hooks.not_legal || hooks.alt_coded != alt_total) {
        std::snprintf(buf, sizeof buf, "move mismatch: %d book move(s) not legal in the walk; %d of %d further moves coded",
                      hooks.not_legal, hooks.alt_coded, alt_total);
        err = buf;
        return false;
    }
    hooks.enc.finish();
    return true;
}
