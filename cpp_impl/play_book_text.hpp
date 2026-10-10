#pragma once
// Shared by the book tools (packer, checker, generator, text dump):
// symmetry-canonical position keys and a plain-text book file.
//
// Several "S <seq> <move>" lines with the same seq list that position's book
// moves in file order, the first being the primary (payload format 2,
// play_book.hpp). A move that is the same as an earlier one of the position,
// or leads to the same canonical position (a symmetric twin at a symmetric
// position), is ignored. A line that reaches an already listed position by
// another seq (a transposition, or a symmetric image of the position) is
// merged into it as in format 1: the first line's moves stand, and a
// different move on such a line is counted in `transposed` (the packer warns)
// and never becomes an alternative. The loader keeps every distinct move of a
// position's seq; the packer refuses a position with more than PB_MAX_MOVES
// (3) moves, any illegal move and any line past the 25-ply cap. A book with
// one line per position is a single-move book, as before.
//
// Blank lines and comment lines (first non-blank character '#') are skipped.
// Any other line that is not a book line (an "S" line without a seq and a
// move or with a seq that is not cell indices 0..80, a key line with missing
// fields, prose) is skipped too but counted in `unparsed`, and the packer and
// play_book_check refuse a text book with one. So a text book of only blank
// and comment lines, and nothing else, is the no-book configuration
// (play_book_pack.hpp): another file passed as the book is an error.
#include <array>
#include <cstdio>
#include <fstream>
#include <istream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "global_board.hpp"

// The 25-ply cap: no stored move deeper than half-move index 24, the number of
// moves in its line's seq. Our moves are at even indices moving first and at
// odd ones moving second, so that is index 24 moving first and 23 moving second.
static constexpr int PB_TEXT_MAX_INDEX = 24;

// The 8 symmetries of the 9x9 cell grid. Applying one to the grid moves
// miniboards and squares coherently, so it maps legal UTTT states to legal ones.
inline void sym_rc(int t, int n, int r, int c, int &orow, int &ocol) {
    const int m = n - 1;
    switch (t) {
        case 0: orow = r;     ocol = c;     break;
        case 1: orow = c;     ocol = m - r; break;  // rotate 90
        case 2: orow = m - r; ocol = m - c; break;  // rotate 180
        case 3: orow = m - c; ocol = r;     break;  // rotate 270
        case 4: orow = r;     ocol = m - c; break;  // mirror columns
        case 5: orow = m - r; ocol = c;     break;  // mirror rows
        case 6: orow = c;     ocol = r;     break;  // transpose
        default: orow = m - c; ocol = m - r; break; // anti-transpose
    }
}
inline int sym_inverse(int t) { return t == 1 ? 3 : (t == 3 ? 1 : t); }

inline Move sym_move(int t, Move m) {
    int r = (m.mini_board / 3) * 3 + m.square / 3;
    int c = (m.mini_board % 3) * 3 + m.square % 3;
    int orow, ocol;
    sym_rc(t, 9, r, c, orow, ocol);
    return Move{(orow / 3) * 3 + ocol / 3, (orow % 3) * 3 + ocol % 3};
}
inline int sym_board(int t, int b) {  // a 0..8 miniboard index, or 9 = free choice
    if (b == 9) return 9;
    int orow, ocol;
    sym_rc(t, 3, b / 3, b % 3, orow, ocol);
    return orow * 3 + ocol;
}

inline int active_board(GlobalBoard &b) {
    if (b.n_moves == 0 || b.prev_move_was_pass) return 9;
    int last = b.move_history.top().square;
    int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    return (oop & (1 << last)) ? 9 : last;
}

// Key in orientation t: 81 cell owners ('.', 'x' = first player, 'o') + constraint.
inline std::string key_in(GlobalBoard &b, int t) {
    std::string k(82, '.');
    for (int mb = 0; mb < 9; mb++) {
        for (int sq = 0; sq < 9; sq++) {
            char ch = '.';
            if (b.mini_boards[mb].markers[0] >> sq & 1) ch = 'x';
            else if (b.mini_boards[mb].markers[1] >> sq & 1) ch = 'o';
            if (ch == '.') continue;
            Move tm = sym_move(t, Move{mb, sq});
            int r = (tm.mini_board / 3) * 3 + tm.square / 3;
            int c = (tm.mini_board % 3) * 3 + tm.square % 3;
            k[r * 9 + c] = ch;
        }
    }
    k[81] = (char)('0' + sym_board(t, active_board(b)));
    return k;
}

// Canonical key = lexicographically smallest orientation; `t` maps real -> canonical.
inline std::string canonical_key(GlobalBoard &b, int &t) {
    std::string best = key_in(b, 0);
    t = 0;
    for (int s = 1; s < 8; s++) {
        std::string k = key_in(b, s);
        if (k < best) { best = k; t = s; }
    }
    return best;
}

struct BookEntry {
    Move move;         // the primary move, in the canonical orientation
    int score = 0;     // deep-search score for the side to move
    int ply = 0;
    double prob = 0;   // estimated probability of reaching this position
    std::vector<Move> moves;              // all book moves, primary first, canonical orientation
    std::vector<std::string> child_keys;  // canonical key after each of `moves` ("S" lines only)
    std::vector<std::string> lines;       // the "S" line of each of `moves`, for messages
    std::vector<int> seq;                 // the cells of the position's first "S" line
    bool has_seq = false;                 // false for the key form
};

struct Book {
    std::unordered_map<std::string, BookEntry> entries;
    int illegal = 0;     // "S" lines whose move is not legal in their position (skipped)
    std::string first_illegal;
    int too_deep = 0;    // "S" lines past the 25-ply cap: seq longer than PB_TEXT_MAX_INDEX (kept)
    std::string first_too_deep;
    int transposed = 0;  // lines for a listed position by another seq, with another move (ignored)
    std::string first_transposed;
    int unparsed = 0;    // non-blank, non-comment lines that are not book lines (skipped)
    std::string first_unparsed;

    // The primary move, in the real orientation of b.
    bool lookup(GlobalBoard &b, Move &out) const {
        int t;
        std::string k = canonical_key(b, t);
        auto it = entries.find(k);
        if (it == entries.end()) return false;
        out = sym_move(sym_inverse(t), it->second.move);
        return true;
    }
    // All book moves of b's position, primary first, in the real orientation of b.
    bool lookup_all(GlobalBoard &b, std::vector<Move> &out) const {
        int t;
        std::string k = canonical_key(b, t);
        auto it = entries.find(k);
        out.clear();
        if (it == entries.end()) return false;
        const std::vector<Move> &ms = it->second.moves;
        if (ms.empty()) out.push_back(sym_move(sym_inverse(t), it->second.move));
        for (const Move &m : ms) out.push_back(sym_move(sym_inverse(t), m));
        return true;
    }
    bool save(const std::string &path) const {
        std::ofstream f(path);
        if (!f) return false;
        for (auto &kv : entries) {
            const BookEntry &e = kv.second;
            f << kv.first << ' ' << e.move.mini_board << ' ' << e.move.square << ' '
              << e.score << ' ' << e.ply << ' ' << e.prob << '\n';
        }
        return (bool)f;
    }
    // Two line forms. "<key> <mb> <sq> <score> <ply> <prob>": a canonical key
    // with the move in canonical orientation (play_book_gen). "S <seq> <move>":
    // the position reached from the empty board by seq (comma-separated cell
    // indices mb * 9 + sq, or "-" for none) and our move there as a cell index,
    // in real orientation; it is replayed and keyed here, so an external
    // generator never has to reproduce the canonical key. Several "S" lines of
    // one position with the same seq list its book moves in file order (see the
    // top of the file). An "S" line with an illegal move is counted in `illegal`
    // and skipped; one past the 25-ply cap is counted in `too_deep` and kept.
    bool load(const std::string &path) {
        std::ifstream f(path);
        if (!f) return false;
        return read(f);
    }
    bool read(std::istream &f) {
        std::string line;
        auto unparsable = [&](const std::string &l) {
            if (!unparsed++) first_unparsed = l;
        };
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream in(line);
            std::string k;
            if (!(in >> k)) continue;   // a blank line
            if (k[0] == '#') continue;  // a comment line
            if (k == "S") {
                std::string seq;
                int move;
                if (!(in >> seq >> move)) { unparsable(line); continue; }
                GlobalBoard b;
                std::vector<int> cells;
                bool bad_cell = false;  // not a cell index 0..80
                if (seq != "-") {
                    std::istringstream ms(seq);
                    std::string cell;
                    while (std::getline(ms, cell, ',')) {
                        size_t used = 0;
                        int c = -1;
                        try { c = std::stoi(cell, &used); } catch (...) { used = 0; }
                        bad_cell = used == 0 || used != cell.size() || c < 0 || c > 80;
                        if (bad_cell) break;
                        cells.push_back(c);
                        b.makeMove(Move{c / 9, c % 9});
                    }
                }
                if (bad_cell) { unparsable(line); continue; }
                Move real{move / 9, move % 9}, legal[81];
                bool ok = false;
                int n_legal = b.fillLegalMoves(legal);
                for (int i = 0; i < n_legal; i++)
                    ok |= legal[i].mini_board == real.mini_board && legal[i].square == real.square;
                if (!ok) {
                    if (!illegal++) first_illegal = line;
                    continue;
                }
                if (b.n_moves > PB_TEXT_MAX_INDEX && !too_deep++) first_too_deep = line;
                int t, tc;
                std::string key = canonical_key(b, t);
                b.makeMove(real);
                std::string child = canonical_key(b, tc);
                b.unmakeMove();
                Move cm = sym_move(t, real);
                auto it = entries.find(key);
                if (it == entries.end()) {
                    BookEntry e;
                    e.move = cm;
                    e.ply = b.n_moves;
                    e.moves.push_back(cm);
                    e.child_keys.push_back(child);
                    e.lines.push_back(line);
                    e.seq = cells;
                    e.has_seq = true;
                    entries.emplace(key, e);
                    continue;
                }
                // Another line for a known position. A repeat of one of its moves (the same
                // canonical move, or a move to the same canonical position) is ignored; any
                // other move is a further book move when the line has the position's seq.
                // By another seq the first line's moves stand (format 1's merge).
                BookEntry &e = it->second;
                bool dup = false;
                for (const Move &m : e.moves) dup |= m.mini_board == cm.mini_board && m.square == cm.square;
                for (const std::string &ck : e.child_keys) dup |= ck == child;
                if (dup) continue;
                if (!e.has_seq || cells != e.seq) {
                    if (!transposed++)
                        first_transposed = "\"" + line + "\" (the position's first line: \"" +
                                           (e.lines.empty() ? key : e.lines[0]) + "\")";
                    continue;
                }
                e.moves.push_back(cm);
                e.child_keys.push_back(child);
                e.lines.push_back(line);
                continue;
            }
            BookEntry e;
            if (in >> e.move.mini_board >> e.move.square >> e.score >> e.ply >> e.prob) {
                e.moves.push_back(e.move);
                entries[k] = e;
            } else {
                unparsable(line);
            }
        }
        return true;
    }

    bool has(GlobalBoard &b) const {
        int t;
        return entries.count(canonical_key(b, t)) != 0;
    }
};
