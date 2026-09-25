#pragma once
// Opening book for the CodinGame bot.
//
// The book is a tree. Both roots assume the first player opens center-center:
// when we move first the bot plays it and the book starts at the opponent's
// reply; when we move second the book starts at our reply to it. At each of our
// positions the book stores our move and whether the book goes on after it; at
// each opponent position it stores which replies it covers. Positions need no
// keys: PbWalker visits the book in one fixed order and the payload holds only
// mixed-radix digits (our move's index among the legal moves, then a 0/1
// "continues" digit; one 0/1 "covered" digit per non-terminal opponent reply).
// The packer (play_book_pack.cpp) and the runtime drive the same walk, so they
// cannot disagree about the order. See documentation/play_book.md.
//
// Positions equivalent under the 8 board symmetries share one entry. The
// runtime table maps a symmetry-canonical 64-bit hash to the book move in that
// canonical orientation.
//
// Requires d16_mini_cjk_decode (mini_eval_d16.hpp) and the board type's
// fillLegalMoves / makeMove / unmakeMove / checkWinner.

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#ifndef PLAY_BOOK_NO_DATA  // defined only by the packer, which generates the data
#include "play_book_data.hpp"
#endif

static uint64_t PB_ZCELL[81][2];
static uint64_t PB_ZACTIVE[10];
static uint8_t PB_CELL_MAP[8][81];   // cell index under each symmetry
static uint8_t PB_BOARD_MAP[8][10];  // miniboard index under each symmetry; 9 = free choice
static bool PB_TABLES_READY = false;
static std::unordered_map<uint64_t, uint8_t> PB_TABLE;  // canonical hash -> mb * 9 + sq
static bool PB_READY = false;

static void pb_sym(int t, int m, int r, int c, int &orow, int &ocol) {
    switch (t) {
        case 0: orow = r; ocol = c; break;
        case 1: orow = c; ocol = m - r; break;
        case 2: orow = m - r; ocol = m - c; break;
        case 3: orow = m - c; ocol = r; break;
        case 4: orow = r; ocol = m - c; break;
        case 5: orow = m - r; ocol = c; break;
        case 6: orow = c; ocol = r; break;
        default: orow = m - c; ocol = m - r; break;
    }
}

static int pb_cell(int mb, int sq) {
    return ((mb / 3) * 3 + sq / 3) * 9 + (mb % 3) * 3 + sq % 3;
}

static void pb_init_tables() {
    if (PB_TABLES_READY) return;
    uint64_t x = 0x9E3779B97F4A7C15ull;
    auto next = [&x]() {  // splitmix64
        uint64_t z = (x += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    };
    for (int c = 0; c < 81; c++) {
        PB_ZCELL[c][0] = next();
        PB_ZCELL[c][1] = next();
    }
    for (int b = 0; b < 10; b++) PB_ZACTIVE[b] = next();
    for (int t = 0; t < 8; t++) {
        for (int c = 0; c < 81; c++) {
            int orow, ocol;
            pb_sym(t, 8, c / 9, c % 9, orow, ocol);
            PB_CELL_MAP[t][c] = (uint8_t)(orow * 9 + ocol);
        }
        for (int b = 0; b < 9; b++) {
            int orow, ocol;
            pb_sym(t, 2, b / 3, b % 3, orow, ocol);
            PB_BOARD_MAP[t][b] = (uint8_t)(orow * 3 + ocol);
        }
        PB_BOARD_MAP[t][9] = 9;
    }
    PB_TABLES_READY = true;
}

template <typename Board>
static int pb_active(Board &b) {
    if (b.n_moves == 0 || b.prev_move_was_pass) return 9;
    int last = b.move_history.top().square;
    int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    return (oop >> last & 1) ? 9 : last;
}

// Smallest hash over the 8 orientations; `t` receives the orientation used.
template <typename Board>
static uint64_t pb_canonical(Board &b, int &t) {
    uint64_t h[8];
    int active = pb_active(b);
    for (int s = 0; s < 8; s++) h[s] = PB_ZACTIVE[PB_BOARD_MAP[s][active]];
    for (int mb = 0; mb < 9; mb++) {
        for (int p = 0; p < 2; p++) {
            int marks = b.mini_boards[mb].markers[p];
            while (marks) {
                int c = pb_cell(mb, __builtin_ctz(marks));
                marks &= marks - 1;
                for (int s = 0; s < 8; s++) h[s] ^= PB_ZCELL[PB_CELL_MAP[s][c]][p];
            }
        }
    }
    t = 0;
    for (int s = 1; s < 8; s++) if (h[s] < h[t]) t = s;
    return h[t];
}

// Packed move (mb * 9 + sq) mapped through orientation t, or its inverse.
static int pb_map_move(int t, int packed, bool inverse) {
    int c = pb_cell(packed / 9, packed % 9);
    int mapped = c;
    if (!inverse) {
        mapped = PB_CELL_MAP[t][c];
    } else {
        for (int k = 0; k < 81; k++) if (PB_CELL_MAP[t][k] == c) { mapped = k; break; }
    }
    int r = mapped / 9, col = mapped % 9;
    return ((r / 3) * 3 + col / 3) * 9 + (r % 3) * 3 + col % 3;
}

// Walks every book position in the fixed order shared by packer and runtime.
// The hooks supply the decisions (the runtime reads them from the payload, the
// packer from the text book):
//   choose(b, legal, n)  index of our book move among the legal moves
//   more(b)              after our move (opponent to move in b): does the book go on?
//   covers(b, reply)     is this non-terminal opponent reply in the book?
template <typename Board, typename MoveT, typename Hooks>
struct PbWalker {
    Hooks &hooks;
    std::unordered_set<uint64_t> seen_opponent;
    int entries = 0;

    void ours(Board &b) {
        int t;
        uint64_t h = pb_canonical(b, t);
        if (PB_TABLE.count(h)) return;
        MoveT legal[81];
        int n = b.fillLegalMoves(legal);
        MoveT m = legal[hooks.choose(b, legal, n)];
        PB_TABLE[h] = (uint8_t)pb_map_move(t, m.mini_board * 9 + m.square, false);
        entries++;
        b.makeMove(m);
        if (b.checkWinner() == -1 && hooks.more(b)) opponent(b);
        b.unmakeMove();
    }

    void opponent(Board &b) {
        int t;
        if (!seen_opponent.insert(pb_canonical(b, t)).second) return;
        MoveT legal[81];
        int n = b.fillLegalMoves(legal);
        for (int i = 0; i < n; i++) {
            b.makeMove(legal[i]);
            if (b.checkWinner() == -1 && hooks.covers(b, legal[i])) ours(b);
            b.unmakeMove();
        }
    }

    // Roots, both after the first player's center-center: we move first (the
    // bot opened, the opponent replies), then we move second.
    void run() {
        pb_init_tables();
        PB_TABLE.clear();
        Board we_first;
        we_first.makeMove(MoveT{4, 4});
        opponent(we_first);
        Board we_second;
        we_second.makeMove(MoveT{4, 4});
        ours(we_second);
    }
};

// Mixed-radix reader: 56-bit chunks, 7 little-endian payload bytes each.
struct PbReader {
    const unsigned char *bytes;
    int n_bytes;
    int pos = 0;
    uint64_t value = 0;
    uint64_t range = ~0ull;  // forces a chunk load on the first read
    int next(int radix) {
        if (range > (1ull << 56) / (uint64_t)radix) {
            value = 0;
            for (int i = 0; i < 7; i++) {
                uint64_t byte = pos < n_bytes ? bytes[pos] : 0;
                value |= byte << (8 * i);
                pos++;
            }
            range = 1;
        }
        int digit = (int)(value % (uint64_t)radix);
        value /= (uint64_t)radix;
        range *= (uint64_t)radix;
        return digit;
    }
};

#ifndef PLAY_BOOK_NO_DATA
// Decodes the payload into PB_TABLE. Returns false if the payload is malformed.
template <typename Board, typename MoveT>
static bool pb_init() {
    if (PB_READY) return true;
    static unsigned char buf[PLAY_BOOK_BYTES + 16];
    int n = d16_mini_cjk_decode(PLAY_BOOK_CJK, buf, (int)sizeof(buf));
    if (n < PLAY_BOOK_BYTES) return false;
    struct Hooks {
        PbReader reader;
        int choose(Board &, MoveT *, int n_legal) { return reader.next(n_legal); }
        bool more(Board &) { return reader.next(2) != 0; }
        bool covers(Board &, const MoveT &) { return reader.next(2) != 0; }
    } hooks{PbReader{buf, n}};
    PbWalker<Board, MoveT, Hooks> walker{hooks};
    walker.run();
    PB_READY = walker.entries == PLAY_BOOK_ENTRIES;
    if (!PB_READY) PB_TABLE.clear();
    return PB_READY;
}
#endif

// Book move for the side to move, or false when out of book. The returned move
// is always one of the position's legal moves.
template <typename Board, typename MoveT>
static bool pb_lookup(Board &b, MoveT &out) {
    if (!PB_READY) return false;
    int t;
    auto it = PB_TABLE.find(pb_canonical(b, t));
    if (it == PB_TABLE.end()) return false;
    int packed = pb_map_move(t, it->second, true);
    MoveT legal[81];
    int n = b.fillLegalMoves(legal);
    for (int i = 0; i < n; i++) {
        if (legal[i].mini_board * 9 + legal[i].square == packed) {
            out = legal[i];
            return true;
        }
    }
    return false;
}
