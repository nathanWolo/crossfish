// Checks play_book_data.hpp against the text book it was packed from.
//
// Decodes the payload through the runtime path (pb_init) and then walks the
// book with an independent recursion keyed by play_book_text's canonical
// strings. At every position the runtime lookup must return a legal move that
// is the text book's move or a symmetric equivalent (same canonical child).
//
//   play_book_check <book.txt>
#include <chrono>
#include <cstdio>
#include <set>
#include <string>

#include "global_board.hpp"
#include "mini_eval_d16.hpp"
#include "play_book_text.hpp"
#include "play_book.hpp"
#include <algorithm>
#include <vector>

// FNV-1a over the decoded book table in key order, so builds can be compared.
inline uint64_t play_book_table_checksum() {
    std::vector<std::pair<uint64_t, uint8_t>> rows(PB_TABLE.begin(), PB_TABLE.end());
    std::sort(rows.begin(), rows.end());
    uint64_t h = 1469598103934665603ull;
    for (auto &r : rows) {
        for (int i = 0; i < 8; i++) { h ^= (r.first >> (8 * i)) & 0xff; h *= 1099511628211ull; }
        h ^= r.second; h *= 1099511628211ull;
    }
    return h;
}

static Book g_text;
static std::set<std::string> g_seen_ours, g_seen_opp;
static int g_checked = 0, g_bad = 0;

static std::string child_key(GlobalBoard b, Move m) {
    b.makeMove(m);
    int t;
    return canonical_key(b, t);
}

static void opp(GlobalBoard &b, int idx, int max_idx);

static void ours(GlobalBoard &b, int idx, int max_idx) {
    int t;
    if (!g_seen_ours.insert(canonical_key(b, t)).second) return;
    Move want, got;
    bool has_text = g_text.lookup(b, want);
    bool has_rt = pb_lookup(b, got);
    g_checked++;
    if (!has_text || !has_rt || child_key(b, want) != child_key(b, got)) {
        if (g_bad++ < 5)
            std::printf("MISMATCH at ply %d: text %s %d/%d, runtime %s %d/%d\n", b.n_moves,
                        has_text ? "has" : "lacks", want.mini_board, want.square,
                        has_rt ? "has" : "lacks", got.mini_board, got.square);
        return;
    }
    if (idx + 1 >= max_idx) return;
    b.makeMove(want);
    if (b.checkWinner() == -1) opp(b, idx + 1, max_idx);
    b.unmakeMove();
}

static void opp(GlobalBoard &b, int idx, int max_idx) {
    int t;
    if (!g_seen_opp.insert(canonical_key(b, t)).second) return;
    Move legal[81];
    int n = b.fillLegalMoves(legal);
    for (int i = 0; i < n; i++) {
        b.makeMove(legal[i]);
        if (b.checkWinner() == -1) ours(b, idx, max_idx);
        b.unmakeMove();
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: play_book_check <book.txt>\n"); return 2; }
    if (!g_text.load(argv[1])) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    bool ok = pb_init<GlobalBoard, Move>();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("pb_init: %s, %zu table entries, %.1f ms\n", ok ? "ok" : "FAILED", PB_TABLE.size(), ms);
    if (!ok) return 1;
    std::printf("table_checksum=%llu\n", (unsigned long long)play_book_table_checksum());
    GlobalBoard first;
    first.makeMove(Move{4, 4});
    opp(first, 0, PLAY_BOOK_DEPTH_FIRST);
    GlobalBoard second;
    opp(second, 0, PLAY_BOOK_DEPTH_SECOND);
    std::printf("checked %d positions (text book %zu): %d mismatches\n", g_checked, g_text.entries.size(), g_bad);
    return (g_bad == 0 && g_checked == (int)g_text.entries.size() && (int)PB_TABLE.size() == g_checked) ? 0 : 1;
}
