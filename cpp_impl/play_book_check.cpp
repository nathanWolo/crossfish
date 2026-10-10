// Checks play_book_data.hpp (payload format 2) against the text book it was
// packed from.
//
// Decodes the payload through the runtime path (pb_init) and then walks the
// book with an independent recursion keyed by play_book_text's canonical
// strings. At every position the runtime table must hold the text book's
// moves, in the same order (primary first), each one legal and leading to the
// same canonical position as the text move; the walk then follows every move.
// At positions with several moves it also draws pb_lookup 3,000 times and
// prints how often each move came out; a move more than 20% off its uniform
// share is a failure ("bad draws"). A text book with a line past the 25-ply
// cap fails before the walk (the packer refuses one).
//
// It prints two checksums of the decoded table: table_checksum over the hashes
// and the primary moves (the value cg_selfcheck prints, and the value a
// format-1 table with the same primaries gave), and moves_checksum over the
// hashes and every stored move (test_play_book pins both for a booked build).
//
// With the no-book payload (PLAY_BOOK_ENTRIES 0, from an empty text book) it
// prints "pb_init: none" and passes only if pb_init decoded nothing and the
// text book is empty too.
//
//   play_book_check <book.txt>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

#include "global_board.hpp"
#include "crossfish_dev.hpp"
#include "play_book_text.hpp"
#include "play_book.hpp"

// FNV-1a over the decoded table in key order: the hash and the primary move
// (the entry's low byte). cg_selfcheck computes the same value (its uint8_t
// copy of an entry keeps the low byte), and for a single-move book it equals
// the format-1 runtime's checksum over the same positions.
static uint64_t primary_table_checksum() {
    std::vector<std::pair<uint64_t, uint32_t>> rows(PB_TABLE.begin(), PB_TABLE.end());
    std::sort(rows.begin(), rows.end());
    uint64_t h = 1469598103934665603ull;
    for (auto &r : rows) {
        for (int i = 0; i < 8; i++) { h ^= (r.first >> (8 * i)) & 0xff; h *= 1099511628211ull; }
        h ^= r.second & 0xff; h *= 1099511628211ull;
    }
    return h;
}

// The same over the whole entry: the hash, then the entry's 4 bytes (moves 0,
// 1, 2 and the move count), so the alternatives and their order count too.
static uint64_t moves_table_checksum() {
    std::vector<std::pair<uint64_t, uint32_t>> rows(PB_TABLE.begin(), PB_TABLE.end());
    std::sort(rows.begin(), rows.end());
    uint64_t h = 1469598103934665603ull;
    for (auto &r : rows) {
        for (int i = 0; i < 8; i++) { h ^= (r.first >> (8 * i)) & 0xff; h *= 1099511628211ull; }
        for (int i = 0; i < 4; i++) { h ^= (r.second >> (8 * i)) & 0xff; h *= 1099511628211ull; }
    }
    return h;
}

static Book g_text;
static std::set<std::string> g_seen_ours, g_seen_opp;
static int g_checked = 0, g_bad = 0, g_multi = 0, g_draw_bad = 0;

static std::string child_key(GlobalBoard b, Move m) {
    b.makeMove(m);
    int t;
    return canonical_key(b, t);
}

static std::string rc(const Move &m) {
    return std::to_string((m.mini_board / 3) * 3 + m.square / 3) + " " + std::to_string((m.mini_board % 3) * 3 + m.square % 3);
}

static void opp(GlobalBoard &b);

static void ours(GlobalBoard &b) {
    int t;
    if (!g_seen_ours.insert(canonical_key(b, t)).second) return;
    g_checked++;
    std::vector<Move> want;
    bool has_text = g_text.lookup_all(b, want);
    int pt;
    auto it = PB_TABLE.find(pb_canonical(b, pt));
    bool has_rt = it != PB_TABLE.end();
    std::vector<Move> got;
    if (has_rt) {
        for (int i = 0; i < (int)(it->second >> 24); i++) {
            int packed = pb_map_move(pt, (int)(it->second >> (8 * i) & 255), true);
            got.push_back(Move{packed / 9, packed % 9});
        }
    }
    bool same = has_text && has_rt && want.size() == got.size();
    Move legal[81];
    int n = b.fillLegalMoves(legal);
    for (size_t i = 0; same && i < want.size(); i++) {
        bool is_legal = false;
        for (int j = 0; j < n; j++) is_legal |= legal[j].mini_board == got[i].mini_board && legal[j].square == got[i].square;
        same = is_legal && child_key(b, want[i]) == child_key(b, got[i]);
    }
    if (!same) {
        if (g_bad++ < 5)
            std::printf("MISMATCH at ply %d: text %s %zu moves, runtime %s %zu moves\n", b.n_moves,
                        has_text ? "has" : "lacks", want.size(), has_rt ? "has" : "lacks", got.size());
        return;
    }
    if (want.size() > 1) {  // the runtime choice: uniform over the stored moves
        g_multi++;
        std::vector<int> hits(want.size(), 0);
        const int draws = 3000;
        for (int d = 0; d < draws; d++) {
            Move m;
            if (!pb_lookup(b, m)) { g_draw_bad++; continue; }
            int k = -1;
            for (size_t i = 0; i < got.size(); i++)
                if (got[i].mini_board == m.mini_board && got[i].square == m.square) k = (int)i;
            if (k < 0) g_draw_bad++; else hits[k]++;
        }
        std::printf("ply %d, %zu moves:", b.n_moves, want.size());
        for (size_t i = 0; i < want.size(); i++) std::printf("  %s %d/%d", rc(want[i]).c_str(), hits[i], draws);
        std::printf("\n");
        for (size_t i = 0; i < want.size(); i++) {  // more than 20% off uniform (over 7 sigma) is a failure
            double share = hits[i] * (double)want.size() / draws;
            if (share < 0.8 || share > 1.2) g_draw_bad++;
        }
    }
    for (const Move &m : want) {
        b.makeMove(m);
        if (b.checkWinner() == -1) opp(b);  // opp() follows only the replies the text book covers
        b.unmakeMove();
    }
}

static void opp(GlobalBoard &b) {
    int t;
    if (!g_seen_opp.insert(canonical_key(b, t)).second) return;
    Move legal[81];
    int n = b.fillLegalMoves(legal);
    for (int i = 0; i < n; i++) {
        b.makeMove(legal[i]);
        if (b.checkWinner() == -1 && g_text.has(b)) ours(b);
        b.unmakeMove();
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: play_book_check <book.txt>\n"); return 2; }
    if (!g_text.load(argv[1])) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    if (g_text.too_deep) {
        std::printf("%d text-book line(s) past the 25-ply cap (a seq of more than %d moves); the first: %s\n",
                    g_text.too_deep, PB_TEXT_MAX_INDEX, g_text.first_too_deep.c_str());
        return 1;
    }
    auto t0 = std::chrono::steady_clock::now();
    crossfish_nnue_load_once();
    bool ok = pb_init<GlobalBoard, Move>([](const PbView &v, int c) { return b64::evaluate_board(v, c); });
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (PLAY_BOOK_ENTRIES == 0) {  // the no-book payload: right only for an empty text book
        std::printf("pb_init: none (no book, PLAY_BOOK_ENTRIES 0), %zu table entries; text book %zu positions\n",
                    PB_TABLE.size(), g_text.entries.size());
        return (!ok && PB_TABLE.empty() && g_text.entries.empty()) ? 0 : 1;
    }
    std::printf("pb_init: %s, %zu table entries, %.1f ms\n", ok ? "ok" : "FAILED", PB_TABLE.size(), ms);
    if (!ok) return 1;
    std::printf("table_checksum=%llu (hashes and primary moves)\n", (unsigned long long)primary_table_checksum());
    std::printf("moves_checksum=%llu (hashes and every stored move)\n", (unsigned long long)moves_table_checksum());
    GlobalBoard first;
    first.makeMove(Move{4, 4});
    opp(first);
    GlobalBoard second;  // both roots follow the first player's center-center
    second.makeMove(Move{4, 4});
    ours(second);
    std::printf("checked %d positions (text book %zu, %d with several moves): %d mismatches, %d bad draws\n", g_checked,
                g_text.entries.size(), g_multi, g_bad, g_draw_bad);
    return (g_bad == 0 && g_draw_bad == 0 && g_checked == (int)g_text.entries.size() &&
            (int)PB_TABLE.size() == g_checked) ? 0 : 1;
}
