#pragma once
// Opening book for the CodinGame bot.
//
// The book is a tree. Both roots assume the first player opens center-center:
// when we move first the bot plays it and the book starts at the opponent's
// reply; when we move second the book starts at our reply to it. At each of our
// positions the book stores our move (up to three, see format 2 below) and
// whether the book goes on after it; at each opponent position it stores which
// replies it covers. Positions need no
// keys: PbWalker visits the book in one fixed order and the payload is one
// adaptive binary arithmetic-coded stream of its decisions: our move's rank
// among the legal moves ordered by the NNUE's static evaluation, then a 0/1
// "continues" bit; one 0/1 "covered" bit per non-terminal opponent reply, in
// the same evaluation order. Every bit has a context (the ply, the rank), so
// the stream costs about half the bits of the plain digits. The packer
// (play_book_pack.cpp) and the runtime drive the same walk with the same
// evaluator, so they cannot disagree about the order or the probabilities.
// See documentation/play_book.md.
//
// Positions equivalent under the 8 board symmetries share one entry. The
// runtime table maps a symmetry-canonical 64-bit hash to the book moves in that
// canonical orientation.
//
// Payload format 2 (multi-move): one of our positions may store up to
// PB_MAX_MOVES moves, the first being the primary. After the primary's rank the
// stream has, for k = 1, 2, an adaptive "another move" bit (context k) and, when
// it is 1, the next move's rank in the same order, coded like the primary's
// (same rank models). Then the walk follows each stored move in order (its
// "continues" bit, then the opponent's covered replies). pb_lookup picks one of
// a position's moves uniformly at random (a per-process PRNG); with one move it
// draws nothing. A single-move book costs one "another" bit per position (about
// 4 bytes in all). The payload carries PLAY_BOOK_FORMAT = 2, and pb_init does
// not compile with a payload without it: format 1, the single-move stream of
// the books before 2026-10-08, has no "another" bits and would decode wrongly.
//
// No book (the shipped configuration since 2026-10-09): play_book_pack writes
// PLAY_BOOK_ENTRIES 0, PLAY_BOOK_BYTES 0, fingerprint 0 and an empty payload
// from an empty text book. pb_init then returns false before it decodes
// anything, PB_TABLE stays empty and pb_lookup never returns a move, so the bot
// searches every move after its hard-coded center-center opener. Without that
// early return a zero-byte payload would decode as one bogus entry (an all-0
// bit stream). The empty book belongs to no net (fingerprint 0), so a net swap
// needs no book step; packing a non-empty text book brings a book back.
//
// The runtime code is kept short on purpose: every character of it counts
// against the CodinGame paste limit (cg_minify strips the comments).
//
// Requires d16_mini_cjk_decode (mini_eval_d16.hpp), the board type's
// fillLegalMoves / makeMove / unmakeMove / checkWinner, and an evaluator
// `int eval(const PbView &, int constraint)` for the side to move (the bot
// passes the NNUE's b64::evaluate_board); its values must be identical in the
// packer and the bot.

#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#ifndef PLAY_BOOK_NO_DATA  // defined only by the packer, which generates the data
#include "play_book_data.hpp"
#endif

static constexpr int PB_MAX_MOVES = 3;
static uint64_t PB_ZCELL[81][2];
static uint64_t PB_ZACTIVE[10];
static uint8_t PB_CELL_MAP[8][81];   // cell index under each symmetry
static uint8_t PB_BOARD_MAP[8][10];  // miniboard index under each symmetry; 9 = free choice
static bool PB_TABLES_READY = false;
// canonical hash -> the position's moves: bits 8i..8i+7 = move i (mb * 9 + sq,
// canonical orientation; move 0 is the primary), bits 24..31 = the move count.
// The low byte is the primary, as in format 1's uint8_t table, so copying an
// entry into a uint8_t keeps the primary (cg_selfcheck's table_checksum).
static std::unordered_map<uint64_t, uint32_t> PB_TABLE;
static bool PB_READY = false;

// splitmix64 step, shared by the Zobrist keys, the evaluator fingerprint and pb_random.
static uint64_t pb_splitmix(uint64_t &x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Per-process random numbers for the choice between a position's book moves:
// splitmix64 seeded on first use from the clock and a stack address (CodinGame
// starts a new process per game, so each game draws its own choices).
static uint64_t PB_RNG = 0;
static uint64_t pb_random() {
    if (!PB_RNG) {
        int local;
        PB_RNG = (uint64_t)(std::chrono::high_resolution_clock::now() -
                            std::chrono::high_resolution_clock::time_point()).count() ^
                 (uint64_t)(uintptr_t)&local << 20;
    }
    return pb_splitmix(PB_RNG);
}

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
    for (int c = 0; c < 81; c++) {
        PB_ZCELL[c][0] = pb_splitmix(x);
        PB_ZCELL[c][1] = pb_splitmix(x);
    }
    for (int b = 0; b < 10; b++) PB_ZACTIVE[b] = pb_splitmix(x);
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

// The coder orders moves by the evaluation of the position after each one. A
// child is built as this light view rather than through the board's own
// makeMove (whose move-history stack is slow when CodinGame compiles without
// -O); the packer and the runtime build it identically, which is all the
// ordering needs. The evaluator sees the fields nnue_b64's evaluate_board reads.
struct PbView {
    struct { int markers[2]; } mini_boards[9];
    int mini_board_states[3];
    int n_moves;
};

static bool pb_three(int m) {
    static const int lines[8] = {0007, 0070, 0700, 0111, 0222, 0444, 0421, 0124};
    for (int l : lines) if ((m & l) == l) return true;
    return false;
}

// order[r] = index in `legal` of the r-th best move: a move that wins the game
// first, then by eval(child, constraint) for the side to move in the child (so
// lower is better for the mover); ties keep the generation order.
template <typename Board, typename MoveT, typename Eval>
static void pb_order(const Board &b, const MoveT *legal, int n, Eval &eval, int *order) {
    PbView base;
    for (int mb = 0; mb < 9; mb++) {
        base.mini_boards[mb].markers[0] = b.mini_boards[mb].markers[0];
        base.mini_boards[mb].markers[1] = b.mini_boards[mb].markers[1];
    }
    for (int k = 0; k < 3; k++) base.mini_board_states[k] = b.mini_board_states[k];
    base.n_moves = b.n_moves + 1;
    const int stm = b.n_moves & 1;
    int score[81];
    for (int i = 0; i < n; i++) {
        PbView v = base;
        const int mb = legal[i].mini_board, sq = legal[i].square, bit = 1 << mb;
        v.mini_boards[mb].markers[stm] |= 1 << sq;
        order[i] = i;
        if (pb_three(v.mini_boards[mb].markers[stm])) {
            v.mini_board_states[stm] |= bit;
            if (pb_three(v.mini_board_states[stm])) { score[i] = -1000000; continue; }
        } else if ((v.mini_boards[mb].markers[0] | v.mini_boards[mb].markers[1]) == 511) {
            v.mini_board_states[2] |= bit;
        }
        const int oop = v.mini_board_states[0] | v.mini_board_states[1] | v.mini_board_states[2];
        score[i] = eval(v, (oop >> sq & 1) ? 9 : sq);
    }
    for (int i = 1; i < n; i++) {  // insertion sort: n <= 81, stable
        int k = order[i], j = i;
        while (j > 0 && score[order[j - 1]] > score[k]) { order[j] = order[j - 1]; j--; }
        order[j] = k;
    }
}

// Adaptive binary models: probabilities of a 0 bit in 1/4096, LZMA-style update.
struct PbModels {
    uint16_t cont[32];      // "the book continues", by ply
    uint16_t cover[32][8];  // "this reply is covered", by ply and the reply's rank
    uint16_t rank[16][8];   // "our move ranks below k", by legal-move count bucket and k
    uint16_t another[PB_MAX_MOVES - 1] = {2048, 2048};  // "another book move", by moves stored so far - 1
    PbModels() {
        for (auto &p : cont) p = 2048;
        for (auto &r : cover) for (auto &p : r) p = 2048;
        for (auto &r : rank) for (auto &p : r) p = 2048;
    }
    static int ply_ctx(int ply) { return ply < 31 ? ply : 31; }
    static int n_ctx(int n) { return n <= 9 ? n : 10 + ((n - 10) / 12 < 5 ? (n - 10) / 12 : 5); }
};

static inline void pb_adapt(uint16_t &p, int bit) {
    if (bit) p -= p >> 5; else p += (4096 - p) >> 5;
}

// Fingerprint of the evaluator the book was packed with: the evaluations of 64
// fixed pseudo-positions. A payload coded with another net would be decoded
// by a different move ordering into a huge tree of nonsense, so pb_init
// compares this first and refuses the payload instead.
template <typename Eval>
static uint64_t pb_eval_fingerprint(Eval &eval) {
    uint64_t x = 0x243F6A8885A308D3ull, h = 1469598103934665603ull;
    for (int i = 0; i < 64; i++) {
        PbView v{};
        for (int mb = 0; mb < 9; mb++) {
            uint64_t r = pb_splitmix(x);
            int a = (int)(r & 511), b = (int)(r >> 9 & 511) & ~a;  // disjoint markers
            v.mini_boards[mb].markers[0] = a;
            v.mini_boards[mb].markers[1] = b;
        }
        uint64_t r = pb_splitmix(x);
        v.mini_board_states[0] = (int)(r & 511);
        v.mini_board_states[1] = (int)(r >> 9 & 511) & ~v.mini_board_states[0];
        v.mini_board_states[2] = (int)(r >> 18 & 511) & ~(v.mini_board_states[0] | v.mini_board_states[1]);
        v.n_moves = (int)(r >> 27 & 63);
        int score = eval(v, (int)(r >> 33) % 10);
        h ^= (uint64_t)(uint32_t)score;
        h *= 1099511628211ull;
    }
    return h;
}

// Our move's rank as truncated unary: bit k says "rank > k". `code` encodes or
// decodes one bit and returns its value, so the packer and the runtime share
// this binarization.
template <typename Code>
static int pb_code_rank(Code &&code, PbModels &m, int n, int rank) {
    int nb = PbModels::n_ctx(n), k = 0;
    while (k < n - 1 && code(m.rank[nb][k < 7 ? k : 7], rank > k)) k++;
    return k;
}

// Walks every book position in the fixed order shared by packer and runtime.
// The hooks supply the decisions (the runtime reads them from the payload, the
// packer from the text book) and the evaluator:
//   eval(view, constraint)       static evaluation of a PbView for its side to move
//   choose(b, legal, order, n)   rank of our primary book move in `order`
//   another(b, k)                with k moves stored: does the position have another book move?
//   choose_alt(b, legal, order, n, k)
//                                rank in `order` of the next (k-th, 0-based) book move
//   more(b)                      after our move (opponent to move in b): does the book go on?
//   covers(b, rank)              is this non-terminal opponent reply (of that rank) in the book?
template <typename Board, typename MoveT, typename Hooks>
struct PbWalker {
    Hooks &hooks;
    std::unordered_set<uint64_t> seen_opponent;
    int entries = 0;
    int limit = 1 << 30;  // the runtime sets the expected entry count: a stale payload stops here

    void ours(Board &b) {
        int t;
        if (entries > limit) return;
        uint64_t h = pb_canonical(b, t);
        if (PB_TABLE.count(h)) return;
        MoveT legal[81], ms[PB_MAX_MOVES];
        int order[81];
        int n = b.fillLegalMoves(legal), k = 0;
        pb_order(b, legal, n, hooks.eval, order);
        uint32_t e = 0;
        do {  // the primary, then up to PB_MAX_MOVES - 1 more while the payload says so
            MoveT m = ms[k] = legal[order[k ? hooks.choose_alt(b, legal, order, n, k) : hooks.choose(b, legal, order, n)]];
            e |= (uint32_t)pb_map_move(t, m.mini_board * 9 + m.square, false) << (8 * k);
            k++;
        } while (k < PB_MAX_MOVES && hooks.another(b, k));
        PB_TABLE[h] = e | (uint32_t)k << 24;
        entries++;
        for (int i = 0; i < k; i++) {
            b.makeMove(ms[i]);
            if (b.checkWinner() == -1 && hooks.more(b)) opponent(b);
            b.unmakeMove();
        }
    }

    void opponent(Board &b) {
        int t;
        if (entries > limit) return;
        if (!seen_opponent.insert(pb_canonical(b, t)).second) return;
        MoveT legal[81];
        int order[81];
        int n = b.fillLegalMoves(legal);
        pb_order(b, legal, n, hooks.eval, order);
        for (int r = 0; r < n; r++) {
            b.makeMove(legal[order[r]]);
            if (b.checkWinner() == -1 && hooks.covers(b, r)) ours(b);
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

// Binary range decoder (the LZMA scheme: 32-bit range, 12-bit probabilities).
struct PbDecoder {
    const unsigned char *bytes;
    int n_bytes;
    int pos = 0;
    uint32_t range = 0xFFFFFFFFu, code = 0;
    unsigned next_byte() { return pos < n_bytes ? bytes[pos++] : 0; }
    void init() { for (int i = 0; i < 5; i++) code = code << 8 | next_byte(); }
    int bit(uint16_t &p) {
        uint32_t bound = (range >> 12) * p;
        int b;
        if (code < bound) { range = bound; b = 0; } else { range -= bound; code -= bound; b = 1; }
        pb_adapt(p, b);
        while (range < (1u << 24)) { range <<= 8; code = code << 8 | next_byte(); }
        return b;
    }
};

#ifndef PLAY_BOOK_NO_DATA
// Decodes the payload into PB_TABLE. Returns false if the payload is malformed,
// was packed with another evaluator, or is the empty no-book payload.
// `eval` must be the evaluator the book was packed with.
template <typename Board, typename MoveT, typename Eval>
static bool pb_init(Eval eval) {
    static_assert(PLAY_BOOK_FORMAT == 2);  // fails on a format-1 payload: repack with play_book_pack
    if (PLAY_BOOK_ENTRIES == 0) return false;  // no book (an empty text book): nothing to decode
    if (PB_READY) return true;
    if (pb_eval_fingerprint(eval) != PLAY_BOOK_EVAL_FINGERPRINT) return false;  // packed with another net
    static unsigned char buf[PLAY_BOOK_BYTES + 16];
    int n = d16_mini_cjk_decode(PLAY_BOOK_CJK, buf, (int)sizeof(buf));
    if (n < PLAY_BOOK_BYTES) return false;
    struct Hooks {
        Eval &eval;
        PbDecoder dec;
        PbModels m;
        int choose(Board &b, MoveT *, const int *, int n_legal) {
            return pb_code_rank([&](uint16_t &p, int) { return dec.bit(p); }, m, n_legal, 0);
        }
        bool another(Board &, int k) { return dec.bit(m.another[k - 1]); }
        int choose_alt(Board &b, MoveT *l, const int *o, int n_legal, int) { return choose(b, l, o, n_legal); }
        bool more(Board &b) { return dec.bit(m.cont[PbModels::ply_ctx(b.n_moves)]) != 0; }
        bool covers(Board &b, int rank) {
            return dec.bit(m.cover[PbModels::ply_ctx(b.n_moves)][rank < 7 ? rank : 7]) != 0;
        }
    } hooks{eval, PbDecoder{buf, n}, PbModels{}};
    hooks.dec.init();
    PbWalker<Board, MoveT, Hooks> walker{hooks};
    walker.limit = PLAY_BOOK_ENTRIES;
    walker.run();
    PB_READY = walker.entries == PLAY_BOOK_ENTRIES;
    if (!PB_READY) PB_TABLE.clear();
    return PB_READY;
}
#endif

// Book move for the side to move, or false when out of book. The returned move
// is always one of the position's legal moves. A position with several book
// moves returns one of them uniformly at random; with one move no random
// number is drawn.
template <typename Board, typename MoveT>
static bool pb_lookup(Board &b, MoveT &out) {
    if (!PB_READY) return false;
    int t;
    auto it = PB_TABLE.find(pb_canonical(b, t));
    if (it == PB_TABLE.end()) return false;
    uint32_t e = it->second, count = e >> 24;
    int packed = pb_map_move(t, e >> (8 * (count > 1 ? pb_random() % count : 0)) & 255, true);
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
