#pragma once
// Shared by the pilot book generator and match harness: symmetry-canonical
// position keys and a plain-text book file.
#include <array>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "global_board.hpp"

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
    Move move;         // in the canonical orientation
    int score = 0;     // deep-search score for the side to move
    int ply = 0;
    double prob = 0;   // estimated probability of reaching this position
};

struct Book {
    std::unordered_map<std::string, BookEntry> entries;

    bool lookup(GlobalBoard &b, Move &out) const {
        int t;
        std::string k = canonical_key(b, t);
        auto it = entries.find(k);
        if (it == entries.end()) return false;
        out = sym_move(sym_inverse(t), it->second.move);
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
    // generator never has to reproduce the canonical key.
    bool load(const std::string &path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream in(line);
            std::string k;
            if (!(in >> k)) continue;
            if (k == "S") {
                std::string seq;
                int move;
                if (!(in >> seq >> move)) continue;
                GlobalBoard b;
                if (seq != "-") {
                    std::istringstream ms(seq);
                    std::string cell;
                    while (std::getline(ms, cell, ',')) {
                        int c = std::stoi(cell);
                        b.makeMove(Move{c / 9, c % 9});
                    }
                }
                int t;
                std::string key = canonical_key(b, t);
                BookEntry e;
                e.move = sym_move(t, Move{move / 9, move % 9});
                e.ply = b.n_moves;
                entries.emplace(key, e);  // a symmetric duplicate keeps the first
                continue;
            }
            BookEntry e;
            if (in >> e.move.mini_board >> e.move.square >> e.score >> e.ply >> e.prob) entries[k] = e;
        }
        return true;
    }

    bool has(GlobalBoard &b) const {
        int t;
        return entries.count(canonical_key(b, t)) != 0;
    }
};
