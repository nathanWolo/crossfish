#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <stdlib.h>
#include <chrono>
#include <array>
#include <cmath>
#include <string>
#include <random>
#include <stack>
#include <future>
#include <numeric>
#include <thread>
#include <bitset>
#include <limits>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#if defined(__linux__)
#include <sched.h>
#endif
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include <immintrin.h>

// CodinGame UTTT: 1000ms first execute per player, 100ms per later move.
// SPRT searches for 90ms and independently forfeits any move returned after
// the real 100ms deadline. Fixed-depth eval tests intentionally have no clock.
static int g_sprt_think_ms = 90;
static constexpr int REFEREE_MOVE_TIMEOUT_MS = 100;
static double g_sprt_elo0 = 0;
static double g_sprt_elo1 = 5;
static double g_sprt_llr_bound = 3;
static int g_sprt_max_games = 0;
static unsigned int g_sprt_threads = 0;
static bool g_sprt_pair_model = true;
static bool g_sprt_allow_book_wrap = false;
static int g_sprt_resume_wins = 0;
static int g_sprt_resume_draws = 0;
static int g_sprt_resume_losses = 0;
static std::array<int, 5> g_sprt_resume_pentanomial{};
static bool g_sprt_resume_pentanomial_set = false;
static int g_sprt_game_offset = -1;
#pragma GCC optimize("O3")
#pragma GCC optimization("Ofast,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include "global_board.hpp"
#include "nnue.hpp"
// Both engines can be swapped at compile time so a candidate can be measured
// without editing the tracked headers:
//   -DCROSSFISH_DEV_HEADER='"/path/candidate.hpp"'
//   -DCROSSFISH_PREV_HEADER='"/path/accepted.hpp"'
#ifndef CROSSFISH_PREV_HEADER
#define CROSSFISH_PREV_HEADER "crossfish_prev.hpp"
#endif
#ifndef CROSSFISH_DEV_HEADER
#define CROSSFISH_DEV_HEADER "crossfish_dev.hpp"
#endif
#include CROSSFISH_PREV_HEADER
#include CROSSFISH_DEV_HEADER

struct EloResult {
    double elo_diff;
    double ci;
};

double norm_ppf(double p) {
    // An approximation of the inverse of the cumulative distribution function for the standard normal distribution.
    // Constants are from a simplified version of the Abramowitz and Stegun formula (26.2.23).
    // This approximation is not as accurate as those provided by statistical libraries but is sufficient for basic needs.
    const double a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
    const double a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
    const double b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
    const double b4 = 66.8013118877197, b5 = -13.2806815528857, c1 = -7.78489400243029E-03;
    const double c2 = -0.322396458041136, c3 = -2.40075827716184, c4 = -2.54973253934373;
    const double c5 = 4.37466414146497, c6 = 2.93816398269878, d1 = 7.78469570904146E-03;
    const double d2 = 0.32246712907004, d3 = 2.445134137143, d4 = 3.75440866190742;
    const double p_low = 0.02425, p_high = 1 - p_low;
    double q, r;

    if (p < 0 || p > 1) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (p < p_low) {
        q = sqrt(-2*log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q*q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
    } else {
        q = sqrt(-2*log(1-p));
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }
}


EloResult calc_elo_diff(int wins, int losses, int draws) {
    int total_games = wins + losses + draws;
    double win_rate = static_cast<double>(wins) / total_games;
    double draw_rate = static_cast<double>(draws) / total_games;
    double loss_rate = static_cast<double>(losses) / total_games;
    double E = win_rate + 0.5 * draw_rate;
    double elo_diff;

    try {
        if (E == 1) {
            elo_diff = std::numeric_limits<double>::infinity();
        } else {
            elo_diff = -400 * log10(1 / E - 1);
        }
    } catch (...) {
        elo_diff = std::numeric_limits<double>::infinity();
    }

    // CI formula
    double percentage = (wins + static_cast<double>(draws) / 2) / total_games;
    
    double wins_dev = win_rate * std::pow(1 - percentage, 2);
    double draws_dev = draw_rate * std::pow(0.5 - percentage, 2);
    double losses_dev = loss_rate * std::pow(0 - percentage, 2);

    double std_dev = sqrt(wins_dev + draws_dev + losses_dev) / sqrt(total_games);

    double confidence = 0.95;
    double min_confidence = (1 - confidence) / 2;
    double max_confidence = 1 - min_confidence;

    double min_dev = percentage + norm_ppf(min_confidence) * std_dev;
    double max_dev = percentage + norm_ppf(max_confidence) * std_dev;

    double diff;

    try {
        if (max_dev == 1 || min_dev == 1) {
            diff = std::numeric_limits<double>::infinity();
        } else {
            diff = ((-400 * log10(1 / max_dev - 1)) - (-400 * log10(1 / min_dev - 1)))/2;
        }
    } catch (...) {
        diff = std::numeric_limits<double>::infinity();
    }

    return {elo_diff, diff};
}


class RandomMover {
    public:
        Move getMove(GlobalBoard board) {
            std::vector<Move> legal_moves = board.getLegalMoves();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(0, legal_moves.size() - 1);
            return legal_moves[dis(gen)];
        }
};

class HumanPlayer {
    public:
        Move getMove(GlobalBoard board) {
            int mini_board;
            int square;
            std::cout << "Enter mini board and square: ";
            std::cin >> mini_board >> square;
            Move move = {mini_board, square};
            return move;
        }
};

std::array<int, 3> global_total = {0, 0, 0}; //wins, draws, losses
// Opening-pair score from Dev's perspective:
// LL, LD+DL, LW+DD+WL, DW+WD, WW.
std::array<int, 5> global_pentanomial = {0, 0, 0, 0, 0};
std::mutex global_mutex;
std::atomic<int> completed_tasks(0);
std::atomic<int> prev_timeout_losses{0};
std::atomic<int> dev_timeout_losses{0};
std::atomic<int64_t> prev_max_move_ns{0};
std::atomic<int64_t> dev_max_move_ns{0};
std::atomic<int> tune_games_done{0};
static int tune_games_total = 0;
static std::chrono::steady_clock::time_point tune_t0;

static void log_tune_progress() {
    int done = ++tune_games_done;
    if (done != 100 && done != 500 && done % 1000 != 0 && done != tune_games_total) {
        return;
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tune_t0).count();
    if (sec < 0.001) sec = 0.001;
    double gps = done / sec;
    int left = (int)((tune_games_total - done) / gps + 0.5);
    int pct = (int)((100.0 * done) / (double)tune_games_total);
    std::lock_guard<std::mutex> lock(global_mutex);
    std::cout << "self-play " << done << "/" << tune_games_total << " (" << pct << "%)  "
              << (int)(sec + 0.5) << "s elapsed  ~" << left << "s left  "
              << (int)(gps + 0.5) << " games/s" << std::endl;
}

std::array<double, 3> eloToWDL(double elo, double dlo) {
    std::array<double, 3> probabilities;
    
    double w = 1 / (1 + std::pow(10, (-elo + dlo) / 400)); // win probability
    double l = 1 / (1 + std::pow(10, (+elo + dlo) / 400)); // loss probability
    double d = 1 - w - l;                                  // draw probability

    probabilities[0] = w;
    probabilities[1] = d;
    probabilities[2] = l;
    
    return probabilities;
}

std::pair<double, double> wdlToElo(double w, double d, double l) {
    double elo = 200 * std::log10((w / l) * ((1 - l) / (1 - w)));
    double dlo = 200 * std::log10(((1 - l) / l) * ((1 - w) / w));
    return {elo, dlo};
}

double sprt_trinomial(int wins, int draws, int losses) {
    if (wins == 0 || losses == 0 || draws == 0) {
        return 0;
    }
    double n = wins + draws + losses;

    double dlo = wdlToElo(wins / n, draws / n, losses / n).second;

    std::array<double, 3> probabilities0 = eloToWDL(g_sprt_elo0, dlo);
    std::array<double, 3> probabilities1 = eloToWDL(g_sprt_elo1, dlo);

    return (double)wins * log(probabilities1[0] / probabilities0[0]) 
        + (double)draws * log(probabilities1[1] / probabilities0[1])
        + (double)losses * log(probabilities1[2] / probabilities0[2]); 
}

static double logistic_score(double elo) {
    return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0));
}

static std::array<double, 5> pentanomial_mle_expected(
    const std::array<int, 5> &results,
    double expected_score) {
    std::array<double, 5> counts{};
    std::array<double, 5> probabilities{};
    std::array<double, 5> deltas{};
    double total = 0;
    for (size_t i = 0; i < results.size(); i++) {
        // Match Fishtest's small prior for an unobserved outcome bin.
        counts[i] = results[i] ? (double)results[i] : 1e-3;
        total += counts[i];
    }
    double min_delta = std::numeric_limits<double>::infinity();
    double max_delta = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < results.size(); i++) {
        probabilities[i] = counts[i] / total;
        deltas[i] = i / 4.0 - expected_score;
        min_delta = std::min(min_delta, deltas[i]);
        max_delta = std::max(max_delta, deltas[i]);
    }
    if (!(min_delta < 0 && max_delta > 0)) {
        std::cerr << "invalid pentanomial expectation "
                  << expected_score << std::endl;
        std::abort();
    }
    double lower = -1.0 / max_delta;
    double upper = -1.0 / min_delta;
    const double lower_nudge =
        1e-12 * std::max(1.0, std::abs(lower));
    const double upper_nudge =
        1e-12 * std::max(1.0, std::abs(upper));
    lower += lower_nudge;
    upper -= upper_nudge;
    auto secular = [&](double x) {
        double value = 0;
        for (size_t i = 0; i < results.size(); i++) {
            value += probabilities[i] * deltas[i]
                / (1.0 + x * deltas[i]);
        }
        return value;
    };
    // The secular function is strictly decreasing on this interval.
    for (int iteration = 0; iteration < 200; iteration++) {
        double middle = (lower + upper) * 0.5;
        if (secular(middle) > 0) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    double root = (lower + upper) * 0.5;
    std::array<double, 5> mle{};
    for (size_t i = 0; i < results.size(); i++) {
        mle[i] = probabilities[i]
            / (1.0 + root * deltas[i]);
    }
    return mle;
}

static double sprt_pentanomial(
    const std::array<int, 5> &results,
    double elo0,
    double elo1) {
    int pairs = std::accumulate(results.begin(), results.end(), 0);
    if (pairs == 0) return 0;
    std::array<double, 5> mle0 =
        pentanomial_mle_expected(results, logistic_score(elo0));
    std::array<double, 5> mle1 =
        pentanomial_mle_expected(results, logistic_score(elo1));
    double llr = 0;
    for (size_t i = 0; i < results.size(); i++) {
        double count = results[i] ? (double)results[i] : 1e-3;
        llr += count * std::log(mle1[i] / mle0[i]);
    }
    return llr;
}

static double current_sprt_llr() {
    return g_sprt_pair_model
        ? sprt_pentanomial(
              global_pentanomial, g_sprt_elo0, g_sprt_elo1)
        : sprt_trinomial(
              global_total[0], global_total[1], global_total[2]);
}

static EloResult calc_pentanomial_elo(
    const std::array<int, 5> &results) {
    int pairs = std::accumulate(results.begin(), results.end(), 0);
    if (pairs == 0) return {0, 0};
    double games = 2.0 * pairs;
    double score_sum = 0;
    for (size_t i = 0; i < results.size(); i++) {
        score_sum += results[i] * (i / 2.0);
    }
    double mean = score_sum / games;
    double pair_mean = 2.0 * mean;
    double variance = 0;
    for (size_t i = 0; i < results.size(); i++) {
        double delta = i / 2.0 - pair_mean;
        variance += results[i] * delta * delta;
    }
    variance /= games;
    double standard_error = std::sqrt(variance / games);
    double lower_score =
        mean + norm_ppf(0.025) * standard_error;
    double upper_score =
        mean + norm_ppf(0.975) * standard_error;
    auto score_to_elo = [](double score) {
        score = std::max(1e-9, std::min(1.0 - 1e-9, score));
        return -400.0 * std::log10(1.0 / score - 1.0);
    };
    double elo = score_to_elo(mean);
    double ci =
        (score_to_elo(upper_score) - score_to_elo(lower_score)) / 2.0;
    return {elo, ci};
}

// Deterministic openings. Each game index derives its own generator, so a run
// replays the same book of start positions every time and two candidates are
// compared over identical openings rather than independent random ones. Set
// SPRT_BOOK=0 to fall back to the unseeded rand() openings.
static bool g_sprt_book = true;
// Opening-heuristic experiments need to retain the early game instead of
// consuming it with the normal 4-8 random plies. This alternate book starts
// with center-center, then enumerates three legal replies from the game index.
// It yields hundreds of deterministic paired openings while always handing
// the engines a position at ply four.
static bool g_sprt_center_enum = false;

static constexpr int OPENING_BOOK_MAX_PLIES = 12;
struct OpeningLine {
    uint8_t n_moves = 0;
    std::array<uint8_t, OPENING_BOOK_MAX_PLIES> moves{};
    int16_t baseline_score = 0;
};

struct OpeningBookMeta {
    uint8_t format_version = 0;
    uint32_t seed = 0;
    uint16_t balance_limit = 0;
    uint16_t move_margin = 0;
    uint8_t guide_depth = 0;
    uint8_t shallow_depth = 0;
    uint8_t prefilter_depth = 0;
    uint8_t score_depth = 0;
    uint8_t min_ply = 0;
    uint8_t max_ply = 0;
    uint16_t prefilter_limit = 0;
};

static std::vector<OpeningLine> g_sprt_openings;
static std::vector<size_t> g_sprt_opening_order;
static OpeningBookMeta g_sprt_opening_meta;
static std::string g_sprt_opening_source;
static constexpr uint32_t OPENING_ORDER_SALT = 0x6D2B79F5u;

static std::vector<size_t> make_opening_order(
    size_t count, uint32_t book_seed) {
    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), size_t{0});
    // Spell out Fisher-Yates instead of std::shuffle so the traversal is
    // reproducible across standard-library implementations. mt19937 output
    // and the modulo operation are fully specified for this book size.
    std::mt19937 rng(book_seed ^ OPENING_ORDER_SALT);
    for (size_t remaining = count; remaining > 1; remaining--) {
        size_t other = (size_t)rng() % remaining;
        std::swap(order[remaining - 1], order[other]);
    }
    return order;
}

static uint64_t opening_order_fingerprint(
    const std::vector<size_t> &order) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index : order) {
        uint64_t value = (uint64_t)index;
        for (int byte = 0; byte < 8; byte++) {
            hash ^= (value >> (byte * 8)) & 255;
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static void write_u16_le(std::ostream &out, uint16_t value) {
    char bytes[2] = {
        (char)(value & 255),
        (char)((value >> 8) & 255)
    };
    out.write(bytes, sizeof(bytes));
}

static void write_u32_le(std::ostream &out, uint32_t value) {
    char bytes[4] = {
        (char)(value & 255),
        (char)((value >> 8) & 255),
        (char)((value >> 16) & 255),
        (char)((value >> 24) & 255)
    };
    out.write(bytes, sizeof(bytes));
}

static bool read_u16_le(std::istream &in, uint16_t &value) {
    unsigned char bytes[2];
    if (!in.read((char *)bytes, sizeof(bytes))) return false;
    value = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
    return true;
}

static bool read_u32_le(std::istream &in, uint32_t &value) {
    unsigned char bytes[4];
    if (!in.read((char *)bytes, sizeof(bytes))) return false;
    value = (uint32_t)bytes[0]
          | ((uint32_t)bytes[1] << 8)
          | ((uint32_t)bytes[2] << 16)
          | ((uint32_t)bytes[3] << 24);
    return true;
}

static bool opening_move_is_legal(
    GlobalBoard &board, uint8_t packed) {
    if (packed >= 81) return false;
    int mb = packed / 9;
    int sq = packed % 9;
    std::vector<Move> legal = board.getLegalMoves();
    for (const Move &move : legal) {
        if (move.mini_board == mb && move.square == sq) return true;
    }
    return false;
}

static bool replay_opening(
    const OpeningLine &line, GlobalBoard &board) {
    if (line.n_moves == 0
        || line.n_moves > OPENING_BOOK_MAX_PLIES) {
        return false;
    }
    board = GlobalBoard();
    for (int i = 0; i < line.n_moves; i++) {
        uint8_t packed = line.moves[i];
        if (!opening_move_is_legal(board, packed)) return false;
        board.makeMove(Move{packed / 9, packed % 9});
        if (i + 1 < line.n_moves && board.checkWinner() != -1) {
            return false;
        }
    }
    return board.checkWinner() == -1;
}

static bool load_opening_book(
    const std::string &path, bool quiet = false) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8];
    if (!in.read(magic, sizeof(magic))
        || std::memcmp(magic, "CFBOOK", 6) != 0
        || (magic[6] != '1' && magic[6] != '2')
        || magic[7] != '\0') {
        if (!quiet) std::cerr << "bad opening-book magic: " << path << std::endl;
        return false;
    }
    uint32_t count = 0;
    OpeningBookMeta meta;
    meta.format_version = (uint8_t)(magic[6] - '0');
    if (!read_u32_le(in, count)
        || !read_u32_le(in, meta.seed)
        || !read_u16_le(in, meta.balance_limit)
        || !read_u16_le(in, meta.move_margin)) {
        return false;
    }
    uint8_t encoded_max_plies = 0;
    if (meta.format_version == 1) {
        char fields[5];
        char reserved[7];
        if (!in.read(fields, sizeof(fields))
            || !in.read(reserved, sizeof(reserved))) {
            return false;
        }
        meta.guide_depth = (uint8_t)fields[0];
        meta.shallow_depth = (uint8_t)fields[1];
        meta.score_depth = (uint8_t)fields[1];
        meta.min_ply = (uint8_t)fields[2];
        meta.max_ply = (uint8_t)fields[3];
        encoded_max_plies = (uint8_t)fields[4];
        for (char value : reserved) {
            if (value != 0) return false;
        }
    } else {
        char fields[7];
        char reserved[3];
        if (!in.read(fields, sizeof(fields))
            || !read_u16_le(in, meta.prefilter_limit)
            || !in.read(reserved, sizeof(reserved))) {
            return false;
        }
        meta.guide_depth = (uint8_t)fields[0];
        meta.shallow_depth = (uint8_t)fields[1];
        meta.prefilter_depth = (uint8_t)fields[2];
        meta.score_depth = (uint8_t)fields[3];
        meta.min_ply = (uint8_t)fields[4];
        meta.max_ply = (uint8_t)fields[5];
        encoded_max_plies = (uint8_t)fields[6];
        for (char value : reserved) {
            if (value != 0) return false;
        }
    }
    if (encoded_max_plies != OPENING_BOOK_MAX_PLIES
        || count == 0 || count > 1000000
        || meta.balance_limit == 0
        || meta.shallow_depth == 0
        || meta.score_depth == 0
        || meta.min_ply == 0
        || meta.max_ply < meta.min_ply
        || meta.max_ply > OPENING_BOOK_MAX_PLIES) {
        return false;
    }

    std::vector<OpeningLine> loaded;
    loaded.reserve(count);
    std::set<std::string> seen_states;
    NnueNet encoder;
    for (uint32_t i = 0; i < count; i++) {
        OpeningLine line;
        char n_moves;
        char moves[OPENING_BOOK_MAX_PLIES];
        uint16_t raw_score = 0;
        char record_reserved;
        if (!in.get(n_moves)
            || !in.read(moves, sizeof(moves))
            || !read_u16_le(in, raw_score)
            || !in.get(record_reserved)) {
            return false;
        }
        if (record_reserved != 0) return false;
        line.n_moves = (uint8_t)n_moves;
        for (int j = 0; j < OPENING_BOOK_MAX_PLIES; j++) {
            line.moves[j] = (uint8_t)moves[j];
        }
        line.baseline_score = (int16_t)raw_score;
        if (line.n_moves < meta.min_ply
            || line.n_moves > meta.max_ply
            || std::abs((int)line.baseline_score)
                   > meta.balance_limit) {
            if (!quiet) {
                std::cerr << "opening " << i
                          << " violates book metadata in "
                          << path << std::endl;
            }
            return false;
        }
        for (int j = line.n_moves;
             j < OPENING_BOOK_MAX_PLIES;
             j++) {
            if (line.moves[j] != 0) {
                if (!quiet) {
                    std::cerr << "nonzero opening tail " << i
                              << " in " << path << std::endl;
                }
                return false;
            }
        }
        GlobalBoard board;
        if (!replay_opening(line, board)) {
            if (!quiet) {
                std::cerr << "illegal opening " << i
                          << " in " << path << std::endl;
            }
            return false;
        }
        char state[93];
        encoder.encode_state(board, state);
        if (!seen_states.insert(
                std::string(state, sizeof(state))).second) {
            if (!quiet) {
                std::cerr << "duplicate opening state " << i
                          << " in " << path << std::endl;
            }
            return false;
        }
        loaded.push_back(line);
    }
    if (in.peek() != std::char_traits<char>::eof()) {
        if (!quiet) {
            std::cerr << "trailing opening-book data: "
                      << path << std::endl;
        }
        return false;
    }
    std::vector<size_t> order =
        make_opening_order(loaded.size(), meta.seed);
    g_sprt_openings = std::move(loaded);
    g_sprt_opening_order = std::move(order);
    g_sprt_opening_meta = meta;
    g_sprt_opening_source = path;
    return true;
}

static bool load_default_opening_book(bool quiet = false) {
    static constexpr const char *paths[] = {
        "cpp_impl/opening_book.bin",
        "opening_book.bin",
        "../opening_book.bin"
    };
    for (const char *path : paths) {
        if (load_opening_book(path, true)) return true;
    }
    if (!quiet) {
        std::cerr
            << "failed to find the shipped opening book; tried"
            << " cpp_impl/opening_book.bin, opening_book.bin, and"
            << " ../opening_book.bin" << std::endl;
    }
    return false;
}

static bool referee_move_timed_out(
    std::chrono::steady_clock::duration elapsed) {
    return g_fixed_search_depth <= 0
        && elapsed > std::chrono::milliseconds(REFEREE_MOVE_TIMEOUT_MS);
}

static void record_move_duration(
    std::atomic<int64_t> &maximum,
    std::chrono::steady_clock::duration elapsed) {
    int64_t observed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    int64_t current = maximum.load(std::memory_order_relaxed);
    while (observed > current
           && !maximum.compare_exchange_weak(
               current, observed, std::memory_order_relaxed)) {
    }
}

struct CpuTopology {
    unsigned int logical;
    unsigned int physical;
};

static CpuTopology detect_cpu_topology() {
    unsigned int fallback =
        std::max(1u, std::thread::hardware_concurrency());
#if defined(__linux__)
    std::vector<int> cpus;
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &affinity)) {
                cpus.push_back(cpu);
            }
        }
    }
    if (cpus.empty()) {
        for (unsigned int cpu = 0; cpu < fallback; ++cpu) {
            cpus.push_back((int)cpu);
        }
    }

    std::set<std::pair<int, int>> physical_cores;
    for (int cpu : cpus) {
        std::string topology =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
            + "/topology/";
        std::ifstream package_file(topology + "physical_package_id");
        std::ifstream core_file(topology + "core_id");
        int package_id;
        int core_id;
        if (!(package_file >> package_id) || !(core_file >> core_id)) {
            return {(unsigned int)cpus.size(), (unsigned int)cpus.size()};
        }
        physical_cores.insert({package_id, core_id});
    }
    return {
        (unsigned int)cpus.size(),
        std::max(1u, (unsigned int)physical_cores.size())
    };
#else
    return {fallback, fallback};
#endif
}

void play_game(int idx){
    //play two games from the same start position, alternating who goes first
    RandomMover random_mover;
    std::mt19937 book(0x9E3779B9u * (unsigned int)(idx + 1));

    //update these bots to test new changes
    CrossfishDev bot2;
    CrossfishPrev bot1;

    GlobalBoard board;
    if (!g_sprt_openings.empty()) {
        size_t traversal_index = (size_t)idx;
        if (g_sprt_allow_book_wrap) {
            traversal_index %= g_sprt_opening_order.size();
        }
        if (traversal_index >= g_sprt_opening_order.size()) {
            std::cerr << "opening book exhausted at pair " << idx
                      << " of " << g_sprt_opening_order.size()
                      << std::endl;
            std::abort();
        }
        size_t opening_index =
            g_sprt_opening_order[traversal_index];
        const OpeningLine &line =
            g_sprt_openings[opening_index];
        if (!replay_opening(line, board)) {
            std::cerr << "failed to replay traversal opening " << idx
                      << " (book record " << opening_index << ')'
                      << std::endl;
            std::abort();
        }
    } else if (g_sprt_center_enum) {
        board.makeMove(Move{4, 4});
        unsigned int opening = (unsigned int)idx;
        for (int i = 1; i < 4; i++) {
            std::vector<Move> legal_moves = board.getLegalMoves();
            unsigned int choice = opening % legal_moves.size();
            opening /= legal_moves.size();
            board.makeMove(legal_moves[choice]);
        }
    } else {
        // First 4-8 moves are random.
        int num_random_moves =
            4 + (int)(g_sprt_book ? book() % 5 : (unsigned int)rand() % 5);
        for (int i = 0; i < num_random_moves; i++) {
            if (i == 0) {
                // 30% chance of first move being very center.
                unsigned int roll =
                    g_sprt_book ? book() % 10 : (unsigned int)rand() % 10;
                if (roll < 3) {
                    board.makeMove(Move{4, 4});
                    continue;
                }
            }
            Move m;
            if (g_sprt_book) {
                std::vector<Move> legal_moves = board.getLegalMoves();
                m = legal_moves[book() % legal_moves.size()];
            } else {
                m = random_mover.getMove(board);
            }
            board.makeMove(m);
        }
    }
    GlobalBoard startpos = GlobalBoard(board);
    std::array<int, 3> pair_wdl = {0, 0, 0};
    int pair_score_index = 0;
    //play two games, alternating who goes first
    for (int i = 0; i < 2; i++) {
        const int bot1_player = i;
        const int bot2_player = i ^ 1;
        int forced_winner = -1;
        while (board.checkWinner() == -1){
            if (board.n_moves % 2 == i) {
                auto move_start = std::chrono::steady_clock::now();
                Move m = bot1.getMove(board, std::chrono::milliseconds(g_sprt_think_ms));
                auto elapsed = std::chrono::steady_clock::now() - move_start;
                record_move_duration(prev_max_move_ns, elapsed);
                if (referee_move_timed_out(elapsed)) {
                    prev_timeout_losses.fetch_add(1, std::memory_order_relaxed);
                    forced_winner = bot2_player;
                    break;
                }
                board.makeMove(m);
            }
            else {
                auto move_start = std::chrono::steady_clock::now();
                Move best_move = bot2.getMove(board, std::chrono::milliseconds(g_sprt_think_ms));
                auto elapsed = std::chrono::steady_clock::now() - move_start;
                record_move_duration(dev_max_move_ns, elapsed);
                if (referee_move_timed_out(elapsed)) {
                    dev_timeout_losses.fetch_add(1, std::memory_order_relaxed);
                    forced_winner = bot1_player;
                    break;
                }
                board.makeMove(best_move);
            }
        }
        // Update this opening pair after both colors have been played.
        int winner =
            forced_winner >= 0 ? forced_winner : board.checkWinner();
        if (winner == bot1_player) {
            pair_wdl[2]++; // loss
        } else if (winner == bot2_player) {
            pair_wdl[0]++; // win
            pair_score_index += 2;
        } else {
            pair_wdl[1]++; // draw
            pair_score_index += 1;
        }
        board = GlobalBoard(startpos);
    }
    {
        std::lock_guard<std::mutex> lock(global_mutex);
        for (size_t i = 0; i < global_total.size(); i++) {
            global_total[i] += pair_wdl[i];
        }
        global_pentanomial[pair_score_index]++;
    }
}

static bool same_moves(const std::vector<Move>& v, Move* buf, int n) {
    if ((int)v.size() != n) return false;
    for (int i = 0; i < n; i++) {
        if (v[i].mini_board != buf[i].mini_board || v[i].square != buf[i].square) {
            return false;
        }
    }
    return true;
}

static void verify_fill_movegen() {
    std::mt19937 rng(12345);
    Move buf[81];
    Move cbuf[81];
    for (int game = 0; game < 200; game++) {
        GlobalBoard board;
        for (int ply = 0; ply < 90; ply++) {
            if (board.checkWinner() != -1) break;
            std::vector<Move> v = board.getLegalMoves();
            int n = board.fillLegalMoves(buf);
            if (!same_moves(v, buf, n)) {
                std::cerr << "fillLegalMoves mismatch at ply " << ply << std::endl;
                std::exit(1);
            }
            if (board.n_moves > 0) {
                std::vector<Move> c = board.get_captures();
                int cn = board.fillCaptures(cbuf);
                if (!same_moves(c, cbuf, cn)) {
                    std::cerr << "fillCaptures mismatch at ply " << ply << std::endl;
                    std::exit(1);
                }
            }
            if (v.empty()) break;
            board.makeMove(v[rng() % v.size()]);
        }
    }
    std::cout << "movegen fill vs vector: OK" << std::endl;
}

static void verify_referee_timeout() {
    const auto limit =
        std::chrono::milliseconds(REFEREE_MOVE_TIMEOUT_MS);
    if (referee_move_timed_out(limit)
        || !referee_move_timed_out(limit + std::chrono::nanoseconds(1))) {
        std::cerr << "referee timeout boundary mismatch" << std::endl;
        std::exit(1);
    }
    std::cout << "referee 100ms deadline: OK" << std::endl;
}

static void verify_opening_order() {
    std::vector<size_t> first = make_opening_order(257, 0x12345678u);
    std::vector<size_t> repeat = make_opening_order(257, 0x12345678u);
    std::vector<size_t> other = make_opening_order(257, 0x12345679u);
    std::vector<size_t> sorted = first;
    std::sort(sorted.begin(), sorted.end());
    bool permutation = sorted.size() == 257;
    for (size_t i = 0; permutation && i < sorted.size(); i++) {
        permutation = sorted[i] == i;
    }
    if (!permutation || first != repeat || first == other
        || make_opening_order(0, 1).size() != 0
        || make_opening_order(1, 1) != std::vector<size_t>{0}
        || opening_order_fingerprint(
               make_opening_order(10000, 3237998146u))
               != 8698397342672575767ULL
        || opening_order_fingerprint(
               make_opening_order(50000, 3237998146u))
               != 5306913481027657611ULL) {
        std::cerr << "opening traversal permutation mismatch" << std::endl;
        std::exit(1);
    }
    std::cout << "opening traversal permutation: OK" << std::endl;
}

static void verify_pentanomial_sprt() {
    const std::array<int, 5> reference = {
        10789, 19328, 33806, 19402, 10543
    };
    // Pinned against official-stockfish/fishtest LLRcalc.py.
    double llr = sprt_pentanomial(reference, -3, 1);
    if (std::abs(llr - 2.1310678117942596) > 1e-9) {
        std::cerr << "pentanomial SPRT mismatch: " << llr << std::endl;
        std::exit(1);
    }
    std::cout << "pentanomial SPRT: OK" << std::endl;
}

static void verify_mini_lut() {
    CrossfishDev::init_mini_lut();
    const CrossfishDev::MiniLut &empty = CrossfishDev::mini_lut[0];
    if (empty.dead || empty.p0_tiar || empty.p1_tiar || empty.p0_win1 || empty.p0_sq) {
        std::cerr << "mini lut empty-board mismatch" << std::endl;
        std::exit(1);
    }
    // P0 on squares 0 and 1: ternary index 1 + 3 = 4, wins by playing 2.
    const CrossfishDev::MiniLut &row = CrossfishDev::mini_lut[4];
    if (row.dead || row.p0_win1 == 0 || row.p0_tiar < 1) {
        std::cerr << "mini lut win-in-one mismatch" << std::endl;
        std::exit(1);
    }
    // Same row blocked by P1 on square 2: index 1 + 3 + 2*9 = 22.
    const CrossfishDev::MiniLut &blocked = CrossfishDev::mini_lut[22];
    if (blocked.p0_win1 || blocked.p0_tiar) {
        std::cerr << "mini lut blocked-line mismatch" << std::endl;
        std::exit(1);
    }
    std::cout << "mini lut: OK" << std::endl;
}

static bool load_utttai_state(GlobalBoard &board, const char *s);
static void clear_board_pos(GlobalBoard &board);

static void verify_utttai_state() {
    GlobalBoard board;
    const char *s =
        "100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200";
    if (!load_utttai_state(board, s) || board.n_moves != 1) {
        std::cerr << "utttai depth-1 parse failed" << std::endl;
        std::exit(1);
    }
    if ((board.mini_boards[0].markers[0] & 1) == 0 || (board.n_moves % 2) != 1) {
        std::cerr << "utttai depth-1 markers/stm mismatch" << std::endl;
        std::exit(1);
    }
    // Round trip through the 93-digit encoding along random games: a loaded
    // position must have the same legal moves as the original, and so must
    // every child (a free-move position once loaded as a pass, which left
    // every later move free).
    NnueNet enc;
    std::mt19937 rng(4242);
    int free_checked = 0;
    for (int g = 0; g < 300; g++) {
        GlobalBoard orig;
        while (orig.checkWinner() == -1) {
            char s93[93];
            enc.encode_state(orig, s93);
            GlobalBoard loaded;
            if (!load_utttai_state(loaded, s93)) {
                std::cerr << "utttai round trip: load failed at ply " << orig.n_moves << std::endl;
                std::exit(1);
            }
            free_checked += s93[91] == '9' && orig.n_moves > 0;
            std::vector<Move> a = orig.getLegalMoves();
            std::vector<Move> b = loaded.getLegalMoves();
            auto same = [](std::vector<Move> x, std::vector<Move> y) {
                auto key = [](const Move &m) { return m.mini_board * 9 + m.square; };
                std::vector<int> kx, ky;
                for (auto &m : x) kx.push_back(key(m));
                for (auto &m : y) ky.push_back(key(m));
                std::sort(kx.begin(), kx.end());
                std::sort(ky.begin(), ky.end());
                return kx == ky;
            };
            if (!same(a, b)) {
                std::cerr << "utttai round trip: legal moves differ at ply " << orig.n_moves << std::endl;
                std::exit(1);
            }
            for (const Move &m : a) {
                GlobalBoard oc = orig, lc = loaded;
                oc.makeMove(m);
                lc.makeMove(m);
                if (oc.checkWinner() == -1 && !same(oc.getLegalMoves(), lc.getLegalMoves())) {
                    std::cerr << "utttai round trip: child legal moves differ at ply " << orig.n_moves << std::endl;
                    std::exit(1);
                }
            }
            orig.makeMove(a[rng() % a.size()]);
        }
    }
    std::cout << "utttai state parse: OK (round trip, " << free_checked << " free-move positions)" << std::endl;
}

static void verify_eval_linear() {
    CrossfishDev dev;
    CrossfishDev::init_mini_lut();
    std::mt19937 rng(999);
    for (int g = 0; g < 200; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 90; ply++) {
            if (board.checkWinner() != -1) break;
            int d[CrossfishDev::N_EVAL_WEIGHTS];
            dev.eval_diffs(board, d);
            int stm = (board.n_moves % 2 == 0) ? 1 : -1;
            int val = 0;
            for (int i = 0; i < 9; i++) {
                val += dev.eval_weights[i] * d[i];
            }
            val += stm * dev.eval_weights[9];
            int extra = dev.eval_extra(board);
            int ev = dev.evaluate_hce(board);
            if (ev != stm * val + extra) {
                std::cerr << "eval linear mismatch at ply " << ply
                          << " eval=" << ev << " linear=" << stm * val << " extra=" << extra << std::endl;
                std::exit(1);
            }
            int16_t idx[9];
            int n = 0;
            int base = 0;
            dev.eval_parts(board, idx, n, base);
            int local = 0;
            for (int i = 0; i < n; i++) {
                local += CrossfishDev::mini_score[idx[i]];
            }
            if (ev != base + stm * local + extra) {
                std::cerr << "eval parts mismatch at ply " << ply << std::endl;
                std::exit(1);
            }
            std::vector<Move> moves = board.getLegalMoves();
            if (moves.empty()) break;
            board.makeMove(moves[rng() % moves.size()]);
        }
    }
    std::cout << "eval linear combo: OK" << std::endl;
}

static void verify_nnue_incremental() {
    NnueNet net;
    net.init_random(20260814u);
    NnueNet fresh;
    fresh.copy_weights_from(net);
    std::mt19937 rng(424242);
    int positions = 0;
    for (int g = 0; g < 400; g++) {
        GlobalBoard board;
        net.refresh(board);
        fresh.refresh(board);
        if (!net.acc_equal(fresh) || net.evaluate(board) != fresh.evaluate(board)) {
            std::cerr << "nnue refresh mismatch on empty" << std::endl;
            std::exit(1);
        }
        for (int ply = 0; ply < 90; ply++) {
            if (board.checkWinner() != -1) break;
            std::vector<Move> moves = board.getLegalMoves();
            if (moves.empty()) break;
            Move m = moves[rng() % moves.size()];
            board.makeMove(m);
            net.make(board, m);
            fresh.refresh(board);
            if (!net.acc_equal(fresh)) {
                std::cerr << "nnue incremental mismatch at ply " << ply
                          << " game " << g << " constr inc=" << net.constraint
                          << " refresh=" << fresh.constraint << std::endl;
                std::exit(1);
            }
            if (net.evaluate(board) != fresh.evaluate(board)) {
                std::cerr << "nnue eval mismatch at ply " << ply << std::endl;
                std::exit(1);
            }
            net.unmake();
            board.unmakeMove();
            fresh.refresh(board);
            if (!net.acc_equal(fresh)) {
                std::cerr << "nnue unmake mismatch at ply " << ply << std::endl;
                std::exit(1);
            }
            board.makeMove(m);
            net.make(board, m);
            positions++;
        }
    }
    std::cout << "nnue incremental vs refresh: OK (" << positions << " positions)" << std::endl;
}

struct NnueDumpPos {
    char s[93];
    float y;
    int32_t hce;
};

static void write_nnue_dump(const char *path, const std::vector<NnueDumpPos> &data);

static void play_nnue_games(int n_games, int think_ms,
                            std::vector<NnueDumpPos> &out, uint32_t seed,
                            bool record_root_score) {
    std::mt19937 rng(seed);
    RandomMover random_mover;
    CrossfishDev bot;
    CrossfishDev::init_mini_lut();
    NnueNet enc;
    std::vector<NnueDumpPos> local;
    local.reserve((size_t)n_games * 40);
    for (int g = 0; g < n_games; g++) {
        GlobalBoard board;
        int n_random = 4 + (int)(rng() % 5);
        for (int i = 0; i < n_random; i++) {
            if (board.checkWinner() != -1) break;
            Move m = random_mover.getMove(board);
            board.makeMove(m);
        }
        std::vector<NnueDumpPos> game_pos;
        game_pos.reserve(64);
        while (board.checkWinner() == -1) {
            NnueDumpPos p{};
            enc.encode_state(board, p.s);
            p.y = 0;
            p.hce = bot.evaluate_hce(board);
            game_pos.push_back(p);
            Move m = bot.getMove(board, std::chrono::milliseconds(think_ms));
            if (record_root_score) {
                game_pos.back().y = (float)bot.completed_root_depth;
                game_pos.back().hce = bot.completed_root_score;
            }
            board.makeMove(m);
        }
        int winner = board.checkWinner();
        for (size_t i = 0; i < game_pos.size(); i++) {
            if (record_root_score) continue;
            int stm_player = (n_random + (int)i) % 2;
            if (winner == 2) game_pos[i].y = 0.5f;
            else if (winner == stm_player) game_pos[i].y = 1.0f;
            else game_pos[i].y = 0.0f;
        }
        local.insert(local.end(), game_pos.begin(), game_pos.end());
    }
    global_mutex.lock();
    out.insert(out.end(), local.begin(), local.end());
    global_mutex.unlock();
}

static void dump_nnue_wdl(int n_games, int think_ms, const char *path,
                          bool record_root_score = false) {
    const unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "NNUE " << (record_root_score ? "root-score " : "")
              << "dump: " << n_games << " games at " << think_ms
              << "ms on " << n_threads << " threads -> " << path << std::endl;
    fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
    std::vector<NnueDumpPos> data;
    data.reserve((size_t)n_games * 40);
    int per = n_games / (int)n_threads;
    int extra = n_games % (int)n_threads;
    std::vector<std::future<void>> futures;
    for (unsigned int t = 0; t < n_threads; t++) {
        int n = per + (t < (unsigned)extra ? 1 : 0);
        uint32_t seed = 9000u + t * 9973u;
        futures.push_back(std::async(std::launch::async, play_nnue_games, n,
                                     think_ms, std::ref(data), seed,
                                     record_root_score));
    }
    for (auto &f : futures) f.get();
    std::cout << "positions: " << data.size() << std::endl;
    std::ofstream out(path, std::ios::binary);
    char magic[8] = {'N','N','U','E','W','D','L','1'};
    uint64_t n = data.size();
    out.write(magic, 8);
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const NnueDumpPos &pos : data) {
        out.write(pos.s, 93);
        out.write(reinterpret_cast<const char *>(&pos.y), 4);
        out.write(reinterpret_cast<const char *>(&pos.hce), 4);
    }
    if (!out) {
        std::cerr << "failed to write " << path << std::endl;
        std::exit(1);
    }
    std::cout << "wrote " << path << std::endl;
}

static void play_hce_random_games(int n_games, std::vector<NnueDumpPos> &out, uint32_t seed) {
    std::mt19937 rng(seed);
    CrossfishDev bot;
    NnueNet enc;
    Move buf[81];
    std::vector<NnueDumpPos> local;
    local.reserve((size_t)n_games * 45);
    for (int g = 0; g < n_games; g++) {
        GlobalBoard board;
        while (board.checkWinner() == -1) {
            NnueDumpPos p{};
            enc.encode_state(board, p.s);
            p.y = 0.5f;
            p.hce = bot.evaluate_hce(board);
            local.push_back(p);
            int n = board.fillLegalMoves(buf);
            if (n <= 0) break;
            board.makeMove(buf[rng() % n]);
        }
    }
    global_mutex.lock();
    out.insert(out.end(), local.begin(), local.end());
    global_mutex.unlock();
}

static void dump_nnue_hce(int n_pos, const char *path) {
    const unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());
    int n_games = std::max(n_pos / 35, (int)n_threads);
    std::cout << "HCE dump: ~" << n_pos << " positions from " << n_games
              << " random games on " << n_threads << " threads -> " << path << std::endl;
    CrossfishDev::init_mini_lut();
    fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
    std::vector<NnueDumpPos> data;
    data.reserve((size_t)n_games * 45);
    int per = n_games / (int)n_threads;
    int extra = n_games % (int)n_threads;
    std::vector<std::future<void>> futures;
    for (unsigned int t = 0; t < n_threads; t++) {
        int n = per + (t < (unsigned)extra ? 1 : 0);
        uint32_t seed = 4242u + t * 9973u;
        futures.push_back(std::async(std::launch::async, play_hce_random_games, n, std::ref(data), seed));
    }
    for (auto &f : futures) f.get();
    if ((int)data.size() > n_pos) data.resize((size_t)n_pos);
    std::cout << "positions: " << data.size() << std::endl;
    std::ofstream out(path, std::ios::binary);
    char magic[8] = {'N','N','U','E','W','D','L','1'};
    uint64_t n = data.size();
    out.write(magic, 8);
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const NnueDumpPos &pos : data) {
        out.write(pos.s, 93);
        out.write(reinterpret_cast<const char *>(&pos.y), 4);
        out.write(reinterpret_cast<const char *>(&pos.hce), 4);
    }
    if (!out) {
        std::cerr << "failed to write " << path << std::endl;
        std::exit(1);
    }
    std::cout << "wrote " << path << std::endl;
}

static uint64_t rebuild_zobrist(const GlobalBoard &b) {
    uint64_t h = 0;
    for (int mb = 0; mb < 9; mb++) {
        for (int sq = 0; sq < 9; sq++) {
            if (b.mini_boards[mb].markers[0] & (1 << sq)) {
                h ^= b.move_hashes[0][mb][sq];
            }
            if (b.mini_boards[mb].markers[1] & (1 << sq)) {
                h ^= b.move_hashes[1][mb][sq];
            }
        }
    }
    for (int st = 0; st < 3; st++) {
        for (int mb = 0; mb < 9; mb++) {
            if (b.mini_board_states[st] & (1 << mb)) {
                h ^= b.mini_board_hashes[st][mb];
            }
        }
    }
    if (b.n_moves % 2 == 1) {
        h ^= b.player_to_move_hash;
    }
    if (b.n_moves > 0 && !b.move_history.empty()) {
        h ^= b.legal_mini_board_hashes[b.move_history.top().square];
    }
    return h;
}

static std::string find_data_file(const char *name) {
    const char *cands[] = {
        name,
        nullptr,
    };
    for (const char *p : cands) {
        if (p && fs::exists(p)) return p;
    }
    std::string rels[] = {
        std::string("datasets/") + name,
        std::string("../datasets/") + name,
        std::string("../../datasets/") + name,
        std::string("datasets/") + fs::path(name).filename().string(),
        std::string("../../datasets/") + fs::path(name).filename().string(),
    };
    for (const std::string &p : rels) {
        if (fs::exists(p)) return p;
    }
    return {};
}

struct NnueDumpState {
    char s[93];
    int32_t old_hce;
};

static bool read_nnue_dump_pos(const std::string &path, std::vector<NnueDumpPos> &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "NNUEWDL1", 8) != 0) return false;
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in) return false;
    out.reserve(out.size() + (size_t)n);
    for (uint64_t i = 0; i < n; i++) {
        NnueDumpPos rec{};
        in.read(rec.s, 93);
        in.read(reinterpret_cast<char *>(&rec.y), 4);
        in.read(reinterpret_cast<char *>(&rec.hce), 4);
        if (!in) return false;
        out.push_back(rec);
    }
    return true;
}

static void dump_nnue_annotate(const char *in_path, const char *out_path) {
    g_force_hce_eval = true;
    CrossfishDev::init_mini_lut();
    std::vector<NnueDumpPos> data;
    if (!read_nnue_dump_pos(in_path, data)) {
        std::cerr << "failed to read " << in_path << std::endl;
        std::exit(1);
    }
    CrossfishDev bot;
    GlobalBoard board;
    int ok = 0, bad = 0;
    double sum_s = 0, sum_h = 0, sum_ss = 0, sum_hh = 0, sum_sh = 0;
    int n_corr = 0, n_clamp = 0;
    for (NnueDumpPos &p : data) {
        if (!load_utttai_state(board, p.s) || board.checkWinner() != -1) {
            bad++;
            continue;
        }
        board.zobrist_hash = rebuild_zobrist(board);
        int h = bot.evaluate_hce(board);
        p.y = (float)h;
        ok++;
        double s = (double)p.hce;
        double hv = (double)h;
        sum_s += s;
        sum_h += hv;
        sum_ss += s * s;
        sum_hh += hv * hv;
        sum_sh += s * hv;
        n_corr++;
        if (std::abs(p.hce) == CrossfishDev::SEARCH_SCORE_CLAMP) n_clamp++;
    }
    double corr = 0;
    if (n_corr > 2) {
        double mean_s = sum_s / n_corr;
        double mean_h = sum_h / n_corr;
        double var_s = sum_ss / n_corr - mean_s * mean_s;
        double var_h = sum_hh / n_corr - mean_h * mean_h;
        double cov = sum_sh / n_corr - mean_s * mean_h;
        if (var_s > 1 && var_h > 1) corr = cov / std::sqrt(var_s * var_h);
    }
    std::cout << "annotate " << in_path << " -> " << out_path
              << " n=" << data.size() << " hce_ok=" << ok << " skip=" << bad
              << " search_vs_hce_corr=" << corr
              << " clamped=" << n_clamp << std::endl;
    write_nnue_dump(out_path, data);
}

static void dump_nnue_distill(const char *teacher_bin, const char *out_path,
                              const std::vector<std::string> &in_paths) {
    g_force_hce_eval = false;
    g_nnue_mode = 2;
    g_nnue_residual = 1;
    std::snprintf(g_nnue_bin_path, sizeof(g_nnue_bin_path), "%s", teacher_bin);
    if (!nnue_init_runtime()) {
        std::exit(1);
    }
    CrossfishDev::init_mini_lut();
    std::vector<NnueDumpPos> data;
    for (const std::string &p : in_paths) {
        size_t before = data.size();
        if (!read_nnue_dump_pos(p, data)) {
            std::cerr << "failed to read " << p << std::endl;
            std::exit(1);
        }
        std::cout << "read " << p << " +" << (data.size() - before) << std::endl;
    }
    CrossfishDev bot;
    GlobalBoard board;
    int ok = 0, bad = 0;
    double sum_t = 0, sum_h = 0, sum_tt = 0, sum_hh = 0, sum_th = 0, sum_ae = 0;
    for (NnueDumpPos &p : data) {
        if (!load_utttai_state(board, p.s) || board.checkWinner() != -1) {
            bad++;
            continue;
        }
        int h = bot.evaluate_hce(board);
        int t = bot.evaluate(board);
        p.y = (float)h;
        p.hce = t;
        ok++;
        double tv = (double)t;
        double hv = (double)h;
        sum_t += tv;
        sum_h += hv;
        sum_tt += tv * tv;
        sum_hh += hv * hv;
        sum_th += tv * hv;
        sum_ae += std::fabs(tv - hv);
    }
    double corr = 0;
    if (ok > 2) {
        double mt = sum_t / ok;
        double mh = sum_h / ok;
        double vt = sum_tt / ok - mt * mt;
        double vh = sum_hh / ok - mh * mh;
        double cov = sum_th / ok - mt * mh;
        if (vt > 1 && vh > 1) corr = cov / std::sqrt(vt * vh);
    }
    GlobalBoard empty;
    std::cout << "distill teacher=" << teacher_bin << " n=" << data.size()
              << " ok=" << ok << " skip=" << bad
              << " teacher_vs_hce_corr=" << corr
              << " mean|teacher-hce|=" << (ok ? sum_ae / ok : 0)
              << " empty hce=" << bot.evaluate_hce(empty)
              << " teacher=" << bot.evaluate(empty) << std::endl;
    write_nnue_dump(out_path, data);
}

static bool read_nnue_dump_states(const std::string &path, std::vector<NnueDumpState> &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "NNUEWDL1", 8) != 0) return false;
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in) return false;
    out.reserve(out.size() + (size_t)n);
    for (uint64_t i = 0; i < n; i++) {
        NnueDumpState rec{};
        float y = 0;
        in.read(rec.s, 93);
        in.read(reinterpret_cast<char *>(&y), 4);
        in.read(reinterpret_cast<char *>(&rec.old_hce), 4);
        if (!in) return false;
        out.push_back(rec);
    }
    return true;
}

static void subsample_dump_states(std::vector<NnueDumpState> &v, size_t n, uint32_t seed) {
    if (v.size() <= n) return;
    std::mt19937 rng(seed);
    for (size_t i = 0; i < n; i++) {
        size_t remain = v.size() - i;
        size_t j = i + (size_t)(rng() % (unsigned)remain);
        std::swap(v[i], v[j]);
    }
    v.resize(n);
}

static bool prepare_board_for_search(GlobalBoard &board, const char *s) {
    if (!load_utttai_state(board, s)) return false;
    if (board.checkWinner() != -1) return false;
    board.zobrist_hash = rebuild_zobrist(board);
    Move buf[81];
    if (board.fillLegalMoves(buf) <= 0) return false;
    return true;
}

static void write_nnue_dump(const char *path, const std::vector<NnueDumpPos> &data) {
    fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    char magic[8] = {'N','N','U','E','W','D','L','1'};
    uint64_t n = data.size();
    out.write(magic, 8);
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const NnueDumpPos &pos : data) {
        out.write(pos.s, 93);
        out.write(reinterpret_cast<const char *>(&pos.y), 4);
        out.write(reinterpret_cast<const char *>(&pos.hce), 4);
    }
    if (!out) {
        std::cerr << "failed to write " << path << std::endl;
        std::exit(1);
    }
    std::cout << "wrote " << path << " positions=" << data.size() << std::endl;
}

// Relabel existing dumps with a full-window fixed-depth HCE search score.
// mix: random-legal + self-play 50/50. play: self-play boards only.
static void dump_nnue_search(
    int depth, int n_pos, const char *path, bool play_only,
    const char *play_path_override = nullptr, bool force_hce = true) {
    g_force_hce_eval = force_hce;
    std::string rand_path = find_data_file("nnue_hce_rand.bin");
    std::string play_path = play_path_override
        ? std::string(play_path_override)
        : find_data_file("nnue_pos.bin");
    if (rand_path.empty()) rand_path = find_data_file("datasets/nnue_hce_rand.bin");
    if (play_path.empty()) play_path = find_data_file("datasets/nnue_pos.bin");
    if (play_path.empty() || (!play_only && rand_path.empty())) {
        std::cerr << "need datasets/nnue_pos.bin"
                  << (play_only ? "" : " and datasets/nnue_hce_rand.bin") << std::endl;
        std::exit(1);
    }
    std::vector<NnueDumpState> rand_states, play_states;
    if (!play_only && !read_nnue_dump_states(rand_path, rand_states)) {
        std::cerr << "failed to read " << rand_path << std::endl;
        std::exit(1);
    }
    if (!read_nnue_dump_states(play_path, play_states)) {
        std::cerr << "failed to read " << play_path << std::endl;
        std::exit(1);
    }
    size_t want = (size_t)std::max(1, n_pos);
    size_t n_rand = play_only ? 0 : want / 2;
    size_t n_play = want - n_rand;
    if (play_states.size() < n_play) {
        n_play = play_states.size();
        n_rand = std::min(rand_states.size(), want - n_play);
    }
    if (rand_states.size() < n_rand) {
        n_rand = rand_states.size();
        n_play = std::min(play_states.size(), want - n_rand);
    }
    subsample_dump_states(rand_states, n_rand, 20260814u);
    subsample_dump_states(play_states, n_play, 20260815u);
    std::vector<NnueDumpState> states;
    states.reserve(n_rand + n_play);
    states.insert(states.end(), rand_states.begin(), rand_states.end());
    states.insert(states.end(), play_states.begin(), play_states.end());
    {
        std::mt19937 rng(20260816u);
        std::shuffle(states.begin(), states.end(), rng);
    }
    const unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "search dump depth=" << depth << " play_only=" << (int)play_only
              << " eval=" << (force_hce ? "HCE" : "current")
              << " from " << rand_path
              << " (" << n_rand << ") + " << play_path << " (" << n_play
              << ") on " << n_threads << " threads -> " << path << std::endl;
    CrossfishDev::init_mini_lut();

    {
        CrossfishDev bot;
        GlobalBoard board;
        int n_bench = (int)std::min((size_t)64, states.size());
        auto t0 = std::chrono::high_resolution_clock::now();
        long long nodes = 0;
        int ok = 0;
        for (int i = 0; i < n_bench; i++) {
            if (!prepare_board_for_search(board, states[(size_t)i].s)) continue;
            int score = 0;
            if (!bot.search_fixed_depth(board, depth, score)) continue;
            nodes += bot.nodes;
            ok++;
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::cout << "bench n=" << ok << "/" << n_bench << " " << ms << "ms"
                  << " avg_nodes=" << (ok ? nodes / ok : 0)
                  << " ms/pos=" << (ok ? (double)ms / ok : 0) << std::endl;
    }

    std::vector<NnueDumpPos> labeled(states.size());
    std::vector<char> ok(states.size(), 0);
    std::vector<int32_t> old_hce(states.size(), 0);
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::atomic<size_t> discarded{0};
    auto worker = [&]() {
        CrossfishDev bot;
        GlobalBoard board;
        for (;;) {
            size_t i = next.fetch_add(1);
            if (i >= states.size()) break;
            old_hce[i] = states[i].old_hce;
            if (!prepare_board_for_search(board, states[i].s)) {
                discarded.fetch_add(1);
                done.fetch_add(1);
                continue;
            }
            int static_hce = bot.evaluate_hce(board);
            int score = 0;
            if (!bot.search_fixed_depth(board, depth, score)) {
                discarded.fetch_add(1);
                done.fetch_add(1);
                continue;
            }
            std::memcpy(labeled[i].s, states[i].s, 93);
            labeled[i].y = (float)static_hce;
            labeled[i].hce = score;
            ok[i] = 1;
            done.fetch_add(1);
        }
    };
    std::vector<std::future<void>> futures;
    for (unsigned int t = 0; t < n_threads; t++) {
        futures.push_back(std::async(std::launch::async, worker));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    while (done.load() < states.size()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        size_t d = done.load();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        double rate = (ms > 0) ? 1000.0 * (double)d / (double)ms : 0;
        double eta = (rate > 0) ? (states.size() - d) / rate : 0;
        std::cout << "labeled " << d << "/" << states.size()
                  << " discarded=" << discarded.load()
                  << " " << rate << "/s eta=" << eta << "s" << std::endl;
        if (d >= states.size()) break;
    }
    for (auto &f : futures) f.get();

    std::vector<NnueDumpPos> data;
    data.reserve(states.size());
    double sum_s = 0, sum_h = 0, sum_ss = 0, sum_hh = 0, sum_sh = 0;
    int n_corr = 0, n_clamp = 0;
    for (size_t i = 0; i < states.size(); i++) {
        if (!ok[i]) continue;
        data.push_back(labeled[i]);
        double s = (double)labeled[i].hce;
        double h = (double)old_hce[i];
        sum_s += s;
        sum_h += h;
        sum_ss += s * s;
        sum_hh += h * h;
        sum_sh += s * h;
        n_corr++;
        if (abs(labeled[i].hce) == CrossfishDev::SEARCH_SCORE_CLAMP) n_clamp++;
    }
    double corr = 0;
    if (n_corr > 2) {
        double mean_s = sum_s / n_corr;
        double mean_h = sum_h / n_corr;
        double var_s = sum_ss / n_corr - mean_s * mean_s;
        double var_h = sum_hh / n_corr - mean_h * mean_h;
        double cov = sum_sh / n_corr - mean_s * mean_h;
        if (var_s > 1 && var_h > 1) corr = cov / std::sqrt(var_s * var_h);
        std::cout << "kept=" << data.size() << " discarded=" << discarded.load()
                  << " search_mean=" << mean_s << " search_std=" << std::sqrt(std::max(0.0, var_s))
                  << " hce_std=" << std::sqrt(std::max(0.0, var_h))
                  << " search_vs_hce_corr=" << corr
                  << " clamped=" << n_clamp << std::endl;
    }
    write_nnue_dump(path, data);
}

struct NnueRankPos {
    uint32_t group;
    uint8_t move;
    uint8_t n_legal;
    char s[93];
    float hce;
    int32_t search;
};

// Label every non-terminal child of the same root. Training on score
// differences within each group removes root-specific score calibration and
// directly targets the move ordering induced by the leaf evaluator.
static void dump_nnue_rank(
    int depth, int n_roots, const char *path,
    const char *source_override = nullptr) {
    std::string play_path = source_override
        ? std::string(source_override)
        : find_data_file("nnue_pos.bin");
    if (play_path.empty()) play_path = find_data_file("datasets/nnue_pos.bin");
    std::vector<NnueDumpState> states;
    if (play_path.empty() || !read_nnue_dump_states(play_path, states)) {
        std::cerr << "need datasets/nnue_pos.bin" << std::endl;
        std::exit(1);
    }
    subsample_dump_states(states, (size_t)std::max(1, n_roots), 20260912u);
    CpuTopology topology = detect_cpu_topology();
    const unsigned int n_threads =
        topology.physical ? topology.physical
                          : std::max(1u, std::thread::hardware_concurrency());
    std::cout << "rank dump depth=" << depth << " roots=" << states.size()
              << " from " << play_path << " on " << n_threads
              << " threads -> " << path << std::endl;
    CrossfishPrev::init_mini_lut();

    std::vector<NnueRankPos> data;
    data.reserve(states.size() * 10);
    std::mutex data_mutex;
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    std::atomic<size_t> kept_groups{0};
    std::atomic<size_t> skipped_terminal{0};
    std::atomic<size_t> skipped_bad{0};
    auto worker = [&]() {
        CrossfishPrev bot;
        NnueNet enc;
        GlobalBoard board;
        Move moves[81];
        std::vector<NnueRankPos> local;
        local.reserve(4096);
        for (;;) {
            size_t group = next.fetch_add(1);
            if (group >= states.size()) break;
            if (!prepare_board_for_search(board, states[group].s)) {
                skipped_bad.fetch_add(1);
                done.fetch_add(1);
                continue;
            }
            int n = board.fillLegalMoves(moves);
            if (n < 2) {
                skipped_bad.fetch_add(1);
                done.fetch_add(1);
                continue;
            }
            bool has_terminal = false;
            for (int i = 0; i < n; i++) {
                GlobalBoard child = board;
                child.makeMove(moves[i]);
                child.prev_move_was_pass = false;
                if (child.checkWinner() != -1) {
                    has_terminal = true;
                    break;
                }
            }
            if (has_terminal) {
                skipped_terminal.fetch_add(1);
                done.fetch_add(1);
                continue;
            }
            size_t begin = local.size();
            for (int i = 0; i < n; i++) {
                GlobalBoard child = board;
                child.makeMove(moves[i]);
                child.prev_move_was_pass = false;
                int score = 0;
                if (!bot.search_fixed_depth(child, depth, score)) continue;
                NnueRankPos rec{};
                rec.group = (uint32_t)group;
                rec.move = (uint8_t)(moves[i].mini_board * 9 + moves[i].square);
                rec.n_legal = (uint8_t)n;
                enc.encode_state(child, rec.s);
                rec.hce = (float)bot.evaluate_hce(child);
                rec.search = score;
                local.push_back(rec);
            }
            if (local.size() - begin >= 2) {
                kept_groups.fetch_add(1);
            } else {
                local.resize(begin);
                skipped_bad.fetch_add(1);
            }
            if (local.size() >= 4096) {
                std::lock_guard<std::mutex> lock(data_mutex);
                data.insert(data.end(), local.begin(), local.end());
                local.clear();
            }
            done.fetch_add(1);
        }
        if (!local.empty()) {
            std::lock_guard<std::mutex> lock(data_mutex);
            data.insert(data.end(), local.begin(), local.end());
        }
    };
    std::vector<std::future<void>> futures;
    for (unsigned int t = 0; t < n_threads; t++) {
        futures.push_back(std::async(std::launch::async, worker));
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    while (done.load() < states.size()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        size_t d = done.load();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        double rate = ms ? 1000.0 * d / ms : 0.0;
        std::cout << "rank roots " << d << "/" << states.size()
                  << " groups=" << kept_groups.load()
                  << " terminal=" << skipped_terminal.load()
                  << " bad=" << skipped_bad.load()
                  << " " << rate << "/s" << std::endl;
    }
    for (auto &f : futures) f.get();
    std::sort(data.begin(), data.end(), [](const NnueRankPos &a,
                                          const NnueRankPos &b) {
        if (a.group != b.group) return a.group < b.group;
        return a.move < b.move;
    });
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream out(path, std::ios::binary);
    const char magic[8] = {'N','N','U','E','R','N','K','1'};
    uint64_t records = data.size();
    uint64_t groups = kept_groups.load();
    out.write(magic, 8);
    out.write(reinterpret_cast<const char *>(&records), sizeof(records));
    out.write(reinterpret_cast<const char *>(&groups), sizeof(groups));
    out.write(reinterpret_cast<const char *>(&depth), sizeof(depth));
    for (const NnueRankPos &rec : data) {
        out.write(reinterpret_cast<const char *>(&rec.group), 4);
        out.write(reinterpret_cast<const char *>(&rec.move), 1);
        out.write(reinterpret_cast<const char *>(&rec.n_legal), 1);
        out.write(rec.s, 93);
        out.write(reinterpret_cast<const char *>(&rec.hce), 4);
        out.write(reinterpret_cast<const char *>(&rec.search), 4);
    }
    if (!out) {
        std::cerr << "failed to write " << path << std::endl;
        std::exit(1);
    }
    std::cout << "wrote " << path << " records=" << records
              << " groups=" << groups
              << " terminal_skips=" << skipped_terminal.load()
              << " bad_skips=" << skipped_bad.load() << std::endl;
}

static void pin_nnue_scale() {
    CrossfishDev::init_mini_lut();
    CrossfishDev hce;
    nnue_load_compiled(g_nnue_sparse);
    std::mt19937 rng(7);
    std::vector<int> hs, ns;
    hs.reserve(8000);
    ns.reserve(8000);
    int saved = g_nnue_sparse.scale;
    g_nnue_sparse.scale = NnueNet::QA * NnueNet::QB;
    for (int g = 0; g < 250; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 80; ply++) {
            if (board.checkWinner() != -1) break;
            g_nnue_sparse.refresh(board);
            int h = hce.evaluate_hce(board);
            int n = g_nnue_sparse.evaluate(board);
            if (h != 0 && n != 0) {
                hs.push_back(std::abs(h));
                ns.push_back(std::abs(n));
            }
            std::vector<Move> moves = board.getLegalMoves();
            if (moves.empty()) break;
            board.makeMove(moves[rng() % moves.size()]);
        }
    }
    if (hs.size() < 100) {
        std::cerr << "not enough pin samples" << std::endl;
        std::exit(1);
    }
    std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
    std::nth_element(ns.begin(), ns.begin() + ns.size() / 2, ns.end());
    double med_h = hs[hs.size() / 2];
    double med_n = std::max(1, ns[ns.size() / 2]);
    int scale = (int)std::lround(med_h * (double)(NnueNet::QA * NnueNet::QB) / med_n);
    if (scale < 1) scale = 1;
    std::cout << "nnue pin: samples=" << hs.size()
              << " median|hce|=" << med_h
              << " median|raw|=" << med_n
              << " old_scale=" << saved
              << " new_scale=" << scale << std::endl;
    GlobalBoard empty;
    g_nnue_sparse.scale = scale;
    g_nnue_sparse.refresh(empty);
    std::cout << "empty HCE=" << hce.evaluate_hce(empty)
              << " NNUE=" << g_nnue_sparse.evaluate(empty) << std::endl;
}

static void report_nnue_hce_fit() {
    CrossfishDev::init_mini_lut();
    CrossfishDev dev;
    nnue_load_compiled(g_nnue_sparse);
    std::mt19937 rng(7);
    std::vector<double> hs, ns;
    hs.reserve(12000);
    ns.reserve(12000);
    for (int g = 0; g < 400; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 80; ply++) {
            if (board.checkWinner() != -1) break;
            g_nnue_sparse.refresh(board);
            hs.push_back((double)dev.evaluate_hce(board));
            ns.push_back((double)g_nnue_sparse.evaluate(board));
            std::vector<Move> moves = board.getLegalMoves();
            if (moves.empty()) break;
            board.makeMove(moves[rng() % moves.size()]);
        }
    }
    const int n = (int)hs.size();
    if (n < 100) {
        std::cerr << "not enough fit samples" << std::endl;
        std::exit(1);
    }
    double sum_h = 0, sum_n = 0, sum_e = 0, sum_ae = 0, sum_hh = 0, sum_nn = 0, sum_hn = 0;
    std::vector<double> ae(n);
    for (int i = 0; i < n; i++) {
        double e = ns[i] - hs[i];
        ae[i] = std::fabs(e);
        sum_e += e;
        sum_ae += ae[i];
        sum_h += hs[i];
        sum_n += ns[i];
        sum_hh += hs[i] * hs[i];
        sum_nn += ns[i] * ns[i];
        sum_hn += hs[i] * ns[i];
    }
    std::nth_element(ae.begin(), ae.begin() + n / 2, ae.end());
    std::nth_element(ae.begin() + n / 2, ae.begin() + n * 9 / 10, ae.end());
    double mean_h = sum_h / n;
    double mean_n = sum_n / n;
    double var_h = sum_hh / n - mean_h * mean_h;
    double var_n = sum_nn / n - mean_n * mean_n;
    double cov = sum_hn / n - mean_h * mean_n;
    double corr = (var_h > 1 && var_n > 1) ? cov / std::sqrt(var_h * var_n) : 0;
    GlobalBoard empty;
    g_nnue_sparse.refresh(empty);
    std::cout << "nnue vs HCE fit: n=" << n
              << " mae=" << (sum_ae / n)
              << " median|e|=" << ae[n / 2]
              << " p90|e|=" << ae[n * 9 / 10]
              << " bias=" << (sum_e / n)
              << " corr=" << corr << std::endl;
    std::cout << "empty HCE=" << dev.evaluate_hce(empty)
              << " NNUE=" << g_nnue_sparse.evaluate(empty)
              << " scale=" << g_nnue_sparse.scale
              << " crelu_max=" << g_nnue_sparse.crelu_max
              << " asinh_s=" << g_nnue_sparse.asinh_s << std::endl;
}

static void report_nnue_search_fit(int depth) {
    CrossfishDev::init_mini_lut();
    CrossfishDev dev;
    nnue_load_compiled(g_nnue_sparse);
    std::mt19937 rng(7);
    std::vector<double> ss, ns, hs;
    Move buf[81];
    for (int g = 0; g < 80; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 80; ply++) {
            if (board.checkWinner() != -1) break;
            int nmoves = board.fillLegalMoves(buf);
            if (nmoves <= 0) break;
            int score = 0;
            GlobalBoard search_board = board;
            if (dev.search_fixed_depth(search_board, depth, score)) {
                g_nnue_sparse.refresh(board);
                ss.push_back((double)score);
                ns.push_back((double)g_nnue_sparse.evaluate(board));
                hs.push_back((double)dev.evaluate_hce(board));
            }
            board.makeMove(buf[rng() % nmoves]);
        }
    }
    auto corr_of = [](const std::vector<double> &a, const std::vector<double> &b) {
        const int n = (int)a.size();
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < n; i++) {
            sa += a[i];
            sb += b[i];
            saa += a[i] * a[i];
            sbb += b[i] * b[i];
            sab += a[i] * b[i];
        }
        double ma = sa / n, mb = sb / n;
        double va = saa / n - ma * ma, vb = sbb / n - mb * mb;
        double cov = sab / n - ma * mb;
        return (va > 1 && vb > 1) ? cov / std::sqrt(va * vb) : 0.0;
    };
    auto mae_of = [](const std::vector<double> &a, const std::vector<double> &b) {
        double s = 0;
        for (size_t i = 0; i < a.size(); i++) s += std::fabs(a[i] - b[i]);
        return s / (double)a.size();
    };
    const int n = (int)ss.size();
    if (n < 50) {
        std::cerr << "not enough search-fit samples" << std::endl;
        std::exit(1);
    }
    GlobalBoard empty;
    int empty_s = 0;
    dev.search_fixed_depth(empty, depth, empty_s);
    g_nnue_sparse.refresh(empty);
    std::cout << "nnue vs search d=" << depth << " fit: n=" << n
              << " mae=" << mae_of(ns, ss)
              << " corr=" << corr_of(ns, ss)
              << " nnue_vs_hce_corr=" << corr_of(ns, hs)
              << " search_vs_hce_corr=" << corr_of(ss, hs) << std::endl;
    std::cout << "empty HCE=" << dev.evaluate_hce(empty)
              << " search=" << empty_s
              << " NNUE=" << g_nnue_sparse.evaluate(empty) << std::endl;
}

struct TexelPos {
    int16_t f[CrossfishDev::N_EVAL_WEIGHTS];
    float y;
};

static double texel_sigmoid(double x) {
    if (x > 20) return 1.0;
    if (x < -20) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

static double texel_loss(const std::vector<TexelPos> &data, const double *w, double K) {
    double loss = 0;
    for (const TexelPos &p : data) {
        double e = 0;
        for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
            e += w[i] * p.f[i];
        }
        double pred = texel_sigmoid(e / K);
        pred = std::min(1.0 - 1e-12, std::max(1e-12, pred));
        loss += -p.y * std::log(pred) - (1.0 - p.y) * std::log(1.0 - pred);
    }
    return loss / (double)data.size();
}

static void texel_pin_pawn(double *w) {
    w[CrossfishDev::PAWN_IDX] = (double)CrossfishDev::PAWN;
}

static void play_tune_games(int n_games, int think_ms, std::vector<TexelPos> &out, uint32_t seed) {
    std::mt19937 rng(seed);
    RandomMover random_mover;
    CrossfishDev bot;
    CrossfishDev::init_mini_lut();
    std::vector<TexelPos> local;
    local.reserve(n_games * 40);
    for (int g = 0; g < n_games; g++) {
        GlobalBoard board;
        int n_random = 4 + (int)(rng() % 5);
        for (int i = 0; i < n_random; i++) {
            if (board.checkWinner() != -1) break;
            Move m = random_mover.getMove(board);
            board.makeMove(m);
        }
        std::vector<TexelPos> game_pos;
        game_pos.reserve(64);
        while (board.checkWinner() == -1) {
            int d[CrossfishDev::N_EVAL_WEIGHTS];
            bot.eval_diffs(board, d);
            int stm = (board.n_moves % 2 == 0) ? 1 : -1;
            TexelPos p{};
            for (int i = 0; i < 9; i++) {
                p.f[i] = (int16_t)(stm * d[i]);
            }
            p.f[9] = 1;
            p.y = 0;
            game_pos.push_back(p);

            Move m = bot.getMove(board, std::chrono::milliseconds(think_ms));
            board.makeMove(m);
        }
        int winner = board.checkWinner();
        for (size_t i = 0; i < game_pos.size(); i++) {
            int stm_player = (n_random + (int)i) % 2;
            if (winner == 2) {
                game_pos[i].y = 0.5f;
            } else if (winner == stm_player) {
                game_pos[i].y = 1.0f;
            } else {
                game_pos[i].y = 0.0f;
            }
        }
        local.insert(local.end(), game_pos.begin(), game_pos.end());
    }
    global_mutex.lock();
    out.insert(out.end(), local.begin(), local.end());
    global_mutex.unlock();
}

static const char *TEXEL_POS_PATH = "texel_pos.bin";

static bool save_texel_pos(const char *path, const std::vector<TexelPos> &data) {
    std::ofstream out(path, std::ios::binary);
    uint64_t n = data.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(n * sizeof(TexelPos)));
    return (bool)out;
}

static bool load_texel_pos(const char *path, std::vector<TexelPos> &data) {
    std::ifstream in(path, std::ios::binary);
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in || n < 1000 || n > 5000000) return false;
    data.resize((size_t)n);
    in.read(reinterpret_cast<char *>(data.data()), (std::streamsize)(n * sizeof(TexelPos)));
    return (bool)in;
}

static void run_texel(bool load_saved) {
    const int think_ms = 20;
    const int n_games = 4800;
    const int n_epochs = 100;
    const unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());

    std::vector<TexelPos> data;
    if (load_saved) {
        if (!load_texel_pos(TEXEL_POS_PATH, data)) {
            std::cerr << "failed to load " << TEXEL_POS_PATH << std::endl;
            std::exit(1);
        }
        std::cout << "loaded " << data.size() << " positions from " << TEXEL_POS_PATH << std::endl;
    } else {
        std::cout << "Texel self-play: " << n_games << " games at " << think_ms
                  << "ms on " << n_threads << " threads" << std::endl;
        data.reserve(n_games * 40);
        int per = n_games / (int)n_threads;
        int extra = n_games % (int)n_threads;
        std::vector<std::future<void>> futures;
        for (unsigned int t = 0; t < n_threads; t++) {
            int n = per + (t < (unsigned)extra ? 1 : 0);
            uint32_t seed = 5000u + t * 9973u;
            futures.push_back(std::async(std::launch::async, play_tune_games, n, think_ms, std::ref(data), seed));
        }
        for (auto &f : futures) {
            f.get();
        }
        std::cout << "positions: " << data.size() << std::endl;
        if (data.size() < 1000) {
            std::cerr << "not enough texel positions" << std::endl;
            std::exit(1);
        }
        if (save_texel_pos(TEXEL_POS_PATH, data)) {
            std::cout << "wrote " << TEXEL_POS_PATH << std::endl;
        }
    }

    std::mt19937 rng(42);
    std::shuffle(data.begin(), data.end(), rng);
    size_t split = data.size() * 9 / 10;
    std::vector<TexelPos> train(data.begin(), data.begin() + split);
    std::vector<TexelPos> val(data.begin() + split, data.end());

    double w[CrossfishDev::N_EVAL_WEIGHTS];
    CrossfishDev start_bot;
    int start_w[CrossfishDev::N_EVAL_WEIGHTS];
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        start_w[i] = start_bot.eval_weights[i];
        w[i] = (double)start_w[i];
    }

    double best_k = 400;
    double best_k_loss = 1e100;
    for (double K = 200; K <= 3000; K += 100) {
        double loss = texel_loss(val, w, K);
        if (loss < best_k_loss) {
            best_k_loss = loss;
            best_k = K;
        }
    }
    std::cout << "K=" << best_k << " val_loss=" << best_k_loss << " (frozen start weights)" << std::endl;
    std::cout << "frozen pawn corner_sq=" << CrossfishDev::PAWN << " (search margins in pawns)" << std::endl;

    double m[CrossfishDev::N_EVAL_WEIGHTS] = {};
    double v[CrossfishDev::N_EVAL_WEIGHTS] = {};
    double step_scale[CrossfishDev::N_EVAL_WEIGHTS];
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        step_scale[i] = std::max(std::fabs((double)start_w[i]), 10.0);
    }
    // Smaller relative steps than round 1: start weights are already a Texel local max.
    double lr = 0.005;
    const double wmin[CrossfishDev::N_EVAL_WEIGHTS] = {500, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const double wmax[CrossfishDev::N_EVAL_WEIGHTS] = {4000, 2500, 1500, 3000, 1500, 1500, 80, 40, 60, 120};
    double best_val = best_k_loss;
    double best_w[CrossfishDev::N_EVAL_WEIGHTS];
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        best_w[i] = w[i];
    }
    int stale = 0;
    int tstep = 0;
    const char *names[CrossfishDev::N_EVAL_WEIGHTS] = {
        "miniboards", "center_board", "corner_boards", "global_tiar", "local_tiar",
        "tiar_lined", "center_sq", "corner_sq", "squares", "tempo"
    };

    for (int epoch = 1; epoch <= n_epochs; epoch++) {
        std::shuffle(train.begin(), train.end(), rng);
        for (const TexelPos &p : train) {
            tstep++;
            double e = 0;
            for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
                e += w[i] * p.f[i];
            }
            double pred = texel_sigmoid(e / best_k);
            pred = std::min(1.0 - 1e-12, std::max(1e-12, pred));
            double gscale = (pred - p.y) / best_k;
            for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
                if (i == CrossfishDev::PAWN_IDX) continue;
                double g = gscale * p.f[i] + 4e-5 * (w[i] - (double)start_w[i]) / step_scale[i];
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                double mhat = m[i] / (1.0 - std::pow(0.9, tstep));
                double vhat = v[i] / (1.0 - std::pow(0.999, tstep));
                w[i] -= lr * mhat / (std::sqrt(vhat) + 1e-8) * step_scale[i];
                if (w[i] < wmin[i]) w[i] = wmin[i];
                if (w[i] > wmax[i]) w[i] = wmax[i];
            }
            texel_pin_pawn(w);
        }
        double tr = texel_loss(train, w, best_k);
        double va = texel_loss(val, w, best_k);
        std::cout << "epoch " << epoch << " train=" << tr << " val=" << va << " w=";
        for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
            std::cout << (int)std::lround(w[i]);
            if (i + 1 < CrossfishDev::N_EVAL_WEIGHTS) std::cout << ",";
        }
        std::cout << std::endl;
        if (va + 1e-6 < best_val) {
            best_val = va;
            stale = 0;
            for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
                best_w[i] = w[i];
            }
        } else {
            stale++;
            if (stale >= 15) {
                std::cout << "early stop" << std::endl;
                break;
            }
        }
    }

    std::cout << "best val_loss=" << best_val << " (start " << best_k_loss << ")" << std::endl;
    std::cout << "start: {";
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        std::cout << start_w[i];
        if (i + 1 < CrossfishDev::N_EVAL_WEIGHTS) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << "tuned: {";
    bool changed = false;
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        int rounded = (int)std::lround(best_w[i]);
        if (rounded != start_w[i]) changed = true;
        std::cout << rounded;
        if (i + 1 < CrossfishDev::N_EVAL_WEIGHTS) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    if (!changed) {
        std::cout << "kept start weights (no val improvement)" << std::endl;
    }
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        std::cout << names[i] << ": " << start_w[i] << " -> " << (int)std::lround(best_w[i])
                  << " (" << (100.0 * (best_w[i] - start_w[i]) / step_scale[i]) << "%)" << std::endl;
    }
}

static void clear_board_pos(GlobalBoard &board) {
    for (int i = 0; i < 9; i++) {
        board.mini_boards[i].markers[0] = 0;
        board.mini_boards[i].markers[1] = 0;
    }
    board.mini_board_states[0] = 0;
    board.mini_board_states[1] = 0;
    board.mini_board_states[2] = 0;
    board.n_moves = 0;
    board.prev_move_was_pass = false;
    while (!board.move_history.empty()) {
        board.move_history.pop();
    }
}

// utttai 93-digit state: 81 squares, 9 supergame cells, next-symbol, constraint, result.
// X=1 is our p0 (first player), O=2 is p1. Skip terminated positions.
static bool load_utttai_state(GlobalBoard &board, const char *s) {
    for (int i = 0; i < 93; i++) {
        if (s[i] < '0' || s[i] > '9') return false;
    }
    if (s[92] != '0') return false;
    clear_board_pos(board);
    int occupied = 0;
    for (int i = 0; i < 81; i++) {
        int mb = i / 9;
        int sq = i % 9;
        if (s[i] == '1') {
            board.mini_boards[mb].markers[0] |= (1 << sq);
            occupied++;
        } else if (s[i] == '2') {
            board.mini_boards[mb].markers[1] |= (1 << sq);
            occupied++;
        } else if (s[i] != '0') {
            return false;
        }
    }
    for (int mb = 0; mb < 9; mb++) {
        char c = s[81 + mb];
        if (c == '1') board.mini_board_states[0] |= (1 << mb);
        else if (c == '2') board.mini_board_states[1] |= (1 << mb);
        else if (c == '3') board.mini_board_states[2] |= (1 << mb);
        else if (c != '0') return false;
    }
    int next = s[90] - '0';
    if (next == 1) {
        if (occupied % 2 != 0) return false;
    } else if (next == 2) {
        if (occupied % 2 != 1) return false;
    } else {
        return false;
    }
    board.n_moves = occupied;
    int constraint = s[91] - '0';
    if (constraint == 9) {
        // A free move with stones on the board: the last move sent the player
        // to a decided miniboard, so record that as the last move. Setting
        // prev_move_was_pass instead would make every move searched from here
        // free as well (only pass()/unpass() ever clear it).
        if (occupied > 0) {
            int decided = board.mini_board_states[0] | board.mini_board_states[1]
                        | board.mini_board_states[2];
            if (decided == 0) return false;
            board.move_history.push(Move{0, __builtin_ctz(decided)});
        }
    } else if (constraint >= 0 && constraint <= 8) {
        board.move_history.push(Move{0, constraint});
    } else {
        return false;
    }
    return true;
}

static void collect_depth_files(const fs::path &root, std::vector<fs::path> &out) {
    if (!fs::exists(root)) return;
    for (const auto &ent : fs::recursive_directory_iterator(root)) {
        if (!fs::is_regular_file(ent.path())) continue;
        std::string name = ent.path().filename().string();
        if (name.size() >= 5 && name.compare(0, 5, "depth") == 0 && ent.path().extension() == ".txt") {
            out.push_back(ent.path());
        }
    }
    std::sort(out.begin(), out.end());
}

static bool parse_nmcts_texel_line(const std::string &line, GlobalBoard &board, CrossfishDev &bot, TexelPos &out) {
    const char *p = std::strstr(line.c_str(), "evaluatedState{");
    if (!p) return false;
    p += 15;
    if (!load_utttai_state(board, p)) return false;
    const char *q = p + 93;
    if (*q != ' ') return false;
    q++;
    while (*q && *q != ' ') q++;
    if (*q != ' ') return false;
    q++;
    char *end = nullptr;
    double v = std::strtod(q, &end);
    if (end == q) return false;
    float y = (float)((v + 1.0) * 0.5);
    if (y < 0.f) y = 0.f;
    if (y > 1.f) y = 1.f;
    int d[CrossfishDev::N_EVAL_WEIGHTS];
    bot.eval_diffs(board, d);
    int stm = (board.n_moves % 2 == 0) ? 1 : -1;
    for (int i = 0; i < 9; i++) {
        out.f[i] = (int16_t)(stm * d[i]);
    }
    out.f[9] = 1;
    out.y = y;
    return true;
}

static void run_texel_utttai(const char *dir_arg, bool load_saved) {
    const size_t cap = 1500000;
    const int n_epochs = 60;
    std::vector<TexelPos> data;
    CrossfishDev bot;
    CrossfishDev::init_mini_lut();

    if (load_saved) {
        if (!load_texel_pos("utttai_texel_pos.bin", data)) {
            std::cerr << "failed to load utttai_texel_pos.bin" << std::endl;
            std::exit(1);
        }
        std::cout << "loaded " << data.size() << " positions from utttai_texel_pos.bin" << std::endl;
    } else {
    std::vector<std::string> cands;
    if (dir_arg && dir_arg[0]) cands.push_back(dir_arg);
    cands.push_back("datasets/stage2-nmcts");
    cands.push_back("../datasets/stage2-nmcts");
    cands.push_back("../../datasets/stage2-nmcts");
    cands.push_back("C:/Users/natha/crossfish/datasets/stage2-nmcts");

    std::vector<fs::path> files;
    std::string used;
    for (const std::string &c : cands) {
        files.clear();
        collect_depth_files(c, files);
        if (!files.empty()) {
            used = c;
            break;
        }
    }
    if (files.empty()) {
        std::cerr << "no utttai depth*.txt files found (pass dir after tune hce utttai)" << std::endl;
        std::exit(1);
    }
    std::cout << "utttai NMCTS Texel: " << files.size() << " files in " << used
              << " cap=" << cap << std::endl;

    GlobalBoard board;
    data.reserve(cap);
    std::mt19937 parse_rng(123);
    size_t seen = 0;
    size_t skipped = 0;
    int fi = 0;
    for (const auto &path : files) {
        fi++;
        std::ifstream in(path);
        if (!in) {
            std::cerr << "failed to open " << path << std::endl;
            continue;
        }
        std::string line;
        while (std::getline(in, line)) {
            TexelPos p{};
            if (!parse_nmcts_texel_line(line, board, bot, p)) {
                skipped++;
                continue;
            }
            seen++;
            if (data.size() < cap) {
                data.push_back(p);
            } else {
                size_t j = (size_t)(parse_rng() % seen);
                if (j < cap) data[j] = p;
            }
        }
        if (fi % 5 == 0 || fi == (int)files.size()) {
            std::cout << "  file " << fi << "/" << files.size()
                      << " parsed=" << seen << " kept=" << data.size()
                      << " skipped=" << skipped << std::endl;
        }
    }
    std::cout << "parsed " << seen << " kept " << data.size() << " skipped " << skipped << std::endl;
    if (data.size() < 1000) {
        std::cerr << "not enough utttai positions" << std::endl;
        std::exit(1);
    }
    if (save_texel_pos("utttai_texel_pos.bin", data)) {
        std::cout << "wrote utttai_texel_pos.bin" << std::endl;
    }
    }

    std::mt19937 rng(42);
    std::shuffle(data.begin(), data.end(), rng);
    size_t split = data.size() * 9 / 10;
    std::vector<TexelPos> train(data.begin(), data.begin() + split);
    std::vector<TexelPos> val(data.begin() + split, data.end());

    double w[CrossfishDev::N_EVAL_WEIGHTS];
    int start_w[CrossfishDev::N_EVAL_WEIGHTS];
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        start_w[i] = bot.eval_weights[i];
        w[i] = (double)start_w[i];
    }

    double best_k = 400;
    double best_k_loss = 1e100;
    for (double K = 200; K <= 20000; K += 200) {
        double loss = texel_loss(val, w, K);
        if (loss < best_k_loss) {
            best_k_loss = loss;
            best_k = K;
        }
    }
    std::cout << "K=" << best_k << " val_loss=" << best_k_loss << " (frozen start weights)" << std::endl;
    std::cout << "frozen pawn corner_sq=" << CrossfishDev::PAWN << " (search margins in pawns)" << std::endl;

    double m[CrossfishDev::N_EVAL_WEIGHTS] = {};
    double v[CrossfishDev::N_EVAL_WEIGHTS] = {};
    double step_scale[CrossfishDev::N_EVAL_WEIGHTS];
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        step_scale[i] = std::max(std::fabs((double)start_w[i]), 10.0);
    }
    double lr = 0.01;
    const double wmin[CrossfishDev::N_EVAL_WEIGHTS] = {200, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const double wmax[CrossfishDev::N_EVAL_WEIGHTS] = {5000, 3000, 2000, 4000, 2500, 2500, 200, 80, 150, 400};
    double best_val = best_k_loss;
    double best_w[CrossfishDev::N_EVAL_WEIGHTS];
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        best_w[i] = w[i];
    }
    int stale = 0;
    int tstep = 0;
    const char *names[CrossfishDev::N_EVAL_WEIGHTS] = {
        "miniboards", "center_board", "corner_boards", "global_tiar", "local_tiar",
        "tiar_lined", "center_sq", "corner_sq", "squares", "tempo"
    };

    for (int epoch = 1; epoch <= n_epochs; epoch++) {
        std::shuffle(train.begin(), train.end(), rng);
        for (const TexelPos &p : train) {
            tstep++;
            double e = 0;
            for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
                e += w[i] * p.f[i];
            }
            double pred = texel_sigmoid(e / best_k);
            pred = std::min(1.0 - 1e-12, std::max(1e-12, pred));
            double gscale = (pred - p.y) / best_k;
            for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
                if (i == CrossfishDev::PAWN_IDX) continue;
                double g = gscale * p.f[i] + 1e-5 * (w[i] - (double)start_w[i]) / step_scale[i];
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                double mhat = m[i] / (1.0 - std::pow(0.9, tstep));
                double vhat = v[i] / (1.0 - std::pow(0.999, tstep));
                w[i] -= lr * mhat / (std::sqrt(vhat) + 1e-8) * step_scale[i];
                if (w[i] < wmin[i]) w[i] = wmin[i];
                if (w[i] > wmax[i]) w[i] = wmax[i];
            }
            texel_pin_pawn(w);
        }
        double tr = texel_loss(train, w, best_k);
        double va = texel_loss(val, w, best_k);
        std::cout << "epoch " << epoch << " train=" << tr << " val=" << va << " w=";
        for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
            std::cout << (int)std::lround(w[i]);
            if (i + 1 < CrossfishDev::N_EVAL_WEIGHTS) std::cout << ",";
        }
        std::cout << std::endl;
        if (va + 1e-6 < best_val) {
            best_val = va;
            stale = 0;
            for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
                best_w[i] = w[i];
            }
        } else {
            stale++;
            if (stale >= 12) {
                std::cout << "early stop" << std::endl;
                break;
            }
        }
    }

    std::cout << "best val_loss=" << best_val << " (start " << best_k_loss << ")" << std::endl;
    std::cout << "start: {";
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        std::cout << start_w[i];
        if (i + 1 < CrossfishDev::N_EVAL_WEIGHTS) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << "tuned: {";
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        std::cout << (int)std::lround(best_w[i]);
        if (i + 1 < CrossfishDev::N_EVAL_WEIGHTS) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) {
        std::cout << names[i] << ": " << start_w[i] << " -> " << (int)std::lround(best_w[i])
                  << " (" << (100.0 * (best_w[i] - start_w[i]) / step_scale[i]) << "%)" << std::endl;
    }
}

#pragma pack(push, 1)
struct LutTexelPos {
    int16_t idx[9];
    int8_t n;
    int8_t stm;
    float base;
    float y;
};
#pragma pack(pop)

static const char *LUT_POS_PATH = "lut_texel_pos.bin";
static const char *MINI_SCORE_PATH = "mini_score.bin";

static bool save_lut_pos(const char *path, const std::vector<LutTexelPos> &data) {
    std::ofstream out(path, std::ios::binary);
    uint64_t n = data.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(n * sizeof(LutTexelPos)));
    return (bool)out;
}

static bool load_lut_pos(const char *path, std::vector<LutTexelPos> &data) {
    std::ifstream in(path, std::ios::binary);
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!in || n < 1000 || n > 5000000) return false;
    data.resize((size_t)n);
    in.read(reinterpret_cast<char *>(data.data()), (std::streamsize)(n * sizeof(LutTexelPos)));
    return (bool)in;
}

static bool save_mini_scores(const char *path, const double *score) {
    std::ofstream out(path, std::ios::binary);
    for (int i = 0; i < CrossfishDev::MINI_LUT_SIZE; i++) {
        int v = (int)std::lround(score[i]);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        int16_t s = (int16_t)v;
        out.write(reinterpret_cast<const char *>(&s), sizeof(s));
    }
    return (bool)out;
}

static bool load_mini_scores(const char *path) {
    CrossfishDev::init_mini_lut();
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char *>(CrossfishDev::mini_score),
            (std::streamsize)(CrossfishDev::MINI_LUT_SIZE * sizeof(int16_t)));
    return (bool)in && in.gcount() == (std::streamsize)(CrossfishDev::MINI_LUT_SIZE * sizeof(int16_t));
}

static double lut_eval_e(const LutTexelPos &p, const double *score) {
    double local = 0;
    for (int i = 0; i < p.n; i++) {
        local += score[p.idx[i]];
    }
    return (double)p.base + (double)p.stm * local;
}

static double lut_texel_loss(const std::vector<LutTexelPos> &data, const double *score, double K) {
    double loss = 0;
    for (const LutTexelPos &p : data) {
        double pred = texel_sigmoid(lut_eval_e(p, score) / K);
        pred = std::min(1.0 - 1e-12, std::max(1e-12, pred));
        loss += -p.y * std::log(pred) - (1.0 - p.y) * std::log(1.0 - pred);
    }
    return loss / (double)data.size();
}

static void play_lut_tune_games(int n_games, int think_ms, std::vector<LutTexelPos> &out, uint32_t seed) {
    std::mt19937 rng(seed);
    RandomMover random_mover;
    CrossfishDev bot;
    CrossfishDev::init_mini_lut();
    std::vector<LutTexelPos> local;
    local.reserve(n_games * 40);
    for (int g = 0; g < n_games; g++) {
        GlobalBoard board;
        int n_random = 12;
        for (int i = 0; i < n_random; i++) {
            if (board.checkWinner() != -1) break;
            Move m = random_mover.getMove(board);
            board.makeMove(m);
        }
        std::vector<LutTexelPos> game_pos;
        game_pos.reserve(64);
        while (board.checkWinner() == -1) {
            LutTexelPos p{};
            int n = 0;
            int base = 0;
            bot.eval_parts(board, p.idx, n, base);
            p.n = (int8_t)n;
            p.stm = (board.n_moves % 2 == 0) ? 1 : -1;
            p.base = (float)base;
            p.y = 0;
            game_pos.push_back(p);

            Move m = bot.getMove(board, std::chrono::milliseconds(think_ms));
            board.makeMove(m);
        }
        int winner = board.checkWinner();
        for (size_t i = 0; i < game_pos.size(); i++) {
            int stm_player = (n_random + (int)i) % 2;
            if (winner == 2) {
                game_pos[i].y = 0.5f;
            } else if (winner == stm_player) {
                game_pos[i].y = 1.0f;
            } else {
                game_pos[i].y = 0.0f;
            }
        }
        local.insert(local.end(), game_pos.begin(), game_pos.end());
        log_tune_progress();
    }
    global_mutex.lock();
    out.insert(out.end(), local.begin(), local.end());
    global_mutex.unlock();
}

static void run_lut_texel(bool load_saved) {
    const int think_ms = 5;
    const int n_games = 100000;
    const int n_epochs = 40;
    const unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());
    const int N = CrossfishDev::MINI_LUT_SIZE;

    std::vector<LutTexelPos> data;
    if (load_saved) {
        if (!load_lut_pos(LUT_POS_PATH, data)) {
            std::cerr << "failed to load " << LUT_POS_PATH << std::endl;
            std::exit(1);
        }
        std::cout << "loaded " << data.size() << " positions from " << LUT_POS_PATH << std::endl;
    } else {
        std::cout << "LUT Texel self-play: " << n_games << " games at " << think_ms
                  << "ms, random prefix 12, on " << n_threads << " threads" << std::endl;
        data.reserve(n_games * 40);
        tune_games_done = 0;
        tune_games_total = n_games;
        tune_t0 = std::chrono::steady_clock::now();
        int per = n_games / (int)n_threads;
        int extra = n_games % (int)n_threads;
        std::vector<std::future<void>> futures;
        for (unsigned int t = 0; t < n_threads; t++) {
            int n = per + (t < (unsigned)extra ? 1 : 0);
            uint32_t seed = 9000u + t * 9973u;
            futures.push_back(std::async(std::launch::async, play_lut_tune_games, n, think_ms, std::ref(data), seed));
        }
        for (auto &f : futures) {
            f.get();
        }
        std::cout << "positions: " << data.size() << std::endl;
        if (data.size() < 1000) {
            std::cerr << "not enough texel positions" << std::endl;
            std::exit(1);
        }
        if (save_lut_pos(LUT_POS_PATH, data)) {
            std::cout << "wrote " << LUT_POS_PATH << std::endl;
        }
    }

    CrossfishDev::init_mini_lut();
    std::vector<double> score(N), init_score(N), best_score(N);
    std::vector<double> m(N, 0.0), v(N, 0.0), acc(N, 0.0);
    std::vector<int> tcount(N, 0);
    std::vector<uint8_t> seen_mark(N, 0);
    for (int i = 0; i < N; i++) {
        init_score[i] = (double)CrossfishDev::mini_score[i];
        score[i] = init_score[i];
        best_score[i] = init_score[i];
    }

    std::vector<uint8_t> appeared(N, 0);
    int unique = 0;
    for (const LutTexelPos &p : data) {
        for (int i = 0; i < p.n; i++) {
            int idx = p.idx[i];
            if (idx < 0 || idx >= N) {
                std::cerr << "bad lut index " << idx << std::endl;
                std::exit(1);
            }
            if (!appeared[idx]) {
                appeared[idx] = 1;
                unique++;
            }
        }
    }
    std::cout << "unique 3x3 states: " << unique << " / " << N << std::endl;

    std::mt19937 rng(42);
    std::shuffle(data.begin(), data.end(), rng);
    size_t split = data.size() * 9 / 10;
    std::vector<LutTexelPos> train(data.begin(), data.begin() + split);
    std::vector<LutTexelPos> val(data.begin() + split, data.end());

    double best_k = 400;
    double best_k_loss = 1e100;
    for (double K = 200; K <= 3000; K += 100) {
        double loss = lut_texel_loss(val, score.data(), K);
        if (loss < best_k_loss) {
            best_k_loss = loss;
            best_k = K;
        }
    }
    std::cout << "K=" << best_k << " val_loss=" << best_k_loss << " (linear-init scores)" << std::endl;

    const int BATCH = 64;
    const double lr = 0.4;
    const double l2 = 1e-6;
    const double clamp_r = 2000.0;
    double best_val = best_k_loss;
    int stale = 0;
    int b = 0;
    std::vector<int> touched;
    touched.reserve(512);

    auto flush_batch = [&]() {
        if (touched.empty()) return;
        for (int idx : touched) {
            tcount[idx]++;
            double g = acc[idx] / (double)BATCH + l2 * (score[idx] - init_score[idx]);
            m[idx] = 0.9 * m[idx] + 0.1 * g;
            v[idx] = 0.999 * v[idx] + 0.001 * g * g;
            double mhat = m[idx] / (1.0 - std::pow(0.9, tcount[idx]));
            double vhat = v[idx] / (1.0 - std::pow(0.999, tcount[idx]));
            score[idx] -= lr * mhat / (std::sqrt(vhat) + 1e-8);
            double lo = init_score[idx] - clamp_r;
            double hi = init_score[idx] + clamp_r;
            if (score[idx] < lo) score[idx] = lo;
            if (score[idx] > hi) score[idx] = hi;
            if (score[idx] > 32767.0) score[idx] = 32767.0;
            if (score[idx] < -32768.0) score[idx] = -32768.0;
            acc[idx] = 0;
            seen_mark[idx] = 0;
        }
        touched.clear();
        b = 0;
    };

    for (int epoch = 1; epoch <= n_epochs; epoch++) {
        std::shuffle(train.begin(), train.end(), rng);
        for (const LutTexelPos &p : train) {
            double e = lut_eval_e(p, score.data());
            double pred = texel_sigmoid(e / best_k);
            pred = std::min(1.0 - 1e-12, std::max(1e-12, pred));
            double gscale = (pred - p.y) / best_k;
            for (int i = 0; i < p.n; i++) {
                int idx = p.idx[i];
                if (!seen_mark[idx]) {
                    seen_mark[idx] = 1;
                    touched.push_back(idx);
                }
                acc[idx] += gscale * (double)p.stm;
            }
            b++;
            if (b >= BATCH) {
                flush_batch();
            }
        }
        flush_batch();
        double tr = lut_texel_loss(train, score.data(), best_k);
        double va = lut_texel_loss(val, score.data(), best_k);
        int moved = 0;
        double max_abs = 0;
        double sum_abs = 0;
        for (int i = 0; i < N; i++) {
            double dlt = std::fabs(score[i] - init_score[i]);
            if (dlt >= 1.0) moved++;
            if (dlt > max_abs) max_abs = dlt;
            sum_abs += dlt;
        }
        std::cout << "epoch " << epoch << " train=" << tr << " val=" << va
                  << " moved=" << moved << " max|d|=" << max_abs
                  << " mean|d|=" << (sum_abs / (double)N) << std::endl;
        if (va + 1e-6 < best_val) {
            best_val = va;
            stale = 0;
            best_score = score;
        } else {
            stale++;
            if (stale >= 8) {
                std::cout << "early stop" << std::endl;
                break;
            }
        }
    }

    std::cout << "best val_loss=" << best_val << " (start " << best_k_loss << ")" << std::endl;
    int moved = 0;
    double max_abs = 0;
    int max_i = 0;
    for (int i = 0; i < N; i++) {
        double dlt = std::fabs(best_score[i] - init_score[i]);
        if (dlt >= 1.0) moved++;
        if (dlt > max_abs) {
            max_abs = dlt;
            max_i = i;
        }
    }
    std::cout << "entries moved by >=1: " << moved << " max|d|=" << max_abs
              << " at idx " << max_i << " " << init_score[max_i] << " -> " << best_score[max_i] << std::endl;

    if (best_val + 1e-7 >= best_k_loss) {
        std::cout << "kept linear-init scores (no val improvement)" << std::endl;
        return;
    }
    if (save_mini_scores(MINI_SCORE_PATH, best_score.data())) {
        std::cout << "wrote " << MINI_SCORE_PATH << std::endl;
    }
    for (int i = 0; i < N; i++) {
        int v = (int)std::lround(best_score[i]);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        CrossfishDev::mini_score[i] = (int16_t)v;
    }
    std::cout << "applied scores to Dev in this process" << std::endl;
}

static bool parse_nnue_mode(const char *mode, const char *bin) {
    if (std::strcmp(mode, "sparse") == 0) {
        g_nnue_mode = 1;
        g_nnue_residual = 0;
    } else if (std::strcmp(mode, "residual") == 0 || std::strcmp(mode, "sparse-res") == 0) {
        g_nnue_mode = 1;
        g_nnue_residual = 1;
    } else if (std::strcmp(mode, "mini") == 0) {
        g_nnue_mode = 2;
        g_nnue_residual = 0;
    } else if (std::strcmp(mode, "minires") == 0 || std::strcmp(mode, "mini-res") == 0) {
        g_nnue_mode = 2;
        g_nnue_residual = 1;
    } else {
        return false;
    }
    if (bin) {
        std::snprintf(g_nnue_bin_path, sizeof(g_nnue_bin_path), "%s", bin);
    }
    return true;
}

struct GeneratedOpening {
    bool accepted = false;
    bool prefilter_passed = false;
    bool deep_accepted = false;
    uint32_t candidate_id = 0;
    uint8_t target_ply = 0;
    OpeningLine line;
    std::string state;
};

static GeneratedOpening make_opening_candidate(
    uint32_t candidate_id,
    uint32_t base_seed,
    int target_ply,
    int move_margin,
    int shallow_limit,
    int shallow_depth,
    int forced_first_move,
    CrossfishPrev &bot,
    NnueNet &encoder) {
    GeneratedOpening result;
    result.candidate_id = candidate_id;
    result.target_ply = (uint8_t)target_ply;
    if (target_ply < 1 || target_ply > OPENING_BOOK_MAX_PLIES) {
        return result;
    }

    std::mt19937 rng(
        base_seed ^ (0x9E3779B9u * (candidate_id + 1)));
    GlobalBoard board;
    OpeningLine line;
    for (int ply = 0; ply < target_ply; ply++) {
        std::vector<Move> legal = board.getLegalMoves();
        if (legal.empty()) return result;
        std::vector<int> values(legal.size());
        int best = std::numeric_limits<int>::min();
        for (size_t i = 0; i < legal.size(); i++) {
            GlobalBoard child(board);
            child.makeMove(legal[i]);
            // Child evaluation is from the opponent's perspective.
            values[i] = -bot.evaluate(child);
            best = std::max(best, values[i]);
        }
        std::vector<size_t> reasonable;
        for (size_t i = 0; i < legal.size(); i++) {
            if (values[i] >= best - move_margin) {
                reasonable.push_back(i);
            }
        }
        if (reasonable.empty()) return result;
        size_t selected = reasonable[rng() % reasonable.size()];
        if (ply == 0 && forced_first_move >= 0) {
            bool found = false;
            for (size_t i : reasonable) {
                int packed =
                    legal[i].mini_board * 9 + legal[i].square;
                if (packed == forced_first_move) {
                    selected = i;
                    found = true;
                    break;
                }
            }
            if (!found) return result;
        }
        Move move = legal[selected];
        board.makeMove(move);
        line.moves[ply] =
            (uint8_t)(move.mini_board * 9 + move.square);
        line.n_moves++;
        if (board.checkWinner() != -1) return result;
    }

    int score = 0;
    // Use a fresh search object so candidate admission is independent of the
    // worker count and of TT/history state left by earlier roots.
    CrossfishPrev shallow_scorer;
    if (!shallow_scorer.search_fixed_depth(
            board, shallow_depth, score)
        || std::abs(score) > shallow_limit) {
        return result;
    }
    line.baseline_score = (int16_t)score;
    char state[93];
    encoder.encode_state(board, state);
    result.accepted = true;
    result.line = line;
    result.state.assign(state, sizeof(state));
    return result;
}

static bool save_opening_book(
    const std::string &path,
    const std::vector<OpeningLine> &lines,
    const OpeningBookMeta &meta) {
    fs::path output(path);
    if (output.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(output.parent_path(), ec);
        if (ec) {
            std::cerr << "failed to create " << output.parent_path()
                      << ": " << ec.message() << std::endl;
            return false;
        }
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const char magic[8] = {'C','F','B','O','O','K','2','\0'};
    out.write(magic, sizeof(magic));
    write_u32_le(out, (uint32_t)lines.size());
    write_u32_le(out, meta.seed);
    write_u16_le(out, meta.balance_limit);
    write_u16_le(out, meta.move_margin);
    char fields[7] = {
        (char)meta.guide_depth,
        (char)meta.shallow_depth,
        (char)meta.prefilter_depth,
        (char)meta.score_depth,
        (char)meta.min_ply,
        (char)meta.max_ply,
        (char)OPENING_BOOK_MAX_PLIES
    };
    char reserved[3] = {};
    out.write(fields, sizeof(fields));
    write_u16_le(out, meta.prefilter_limit);
    out.write(reserved, sizeof(reserved));
    for (const OpeningLine &line : lines) {
        out.put((char)line.n_moves);
        out.write((const char *)line.moves.data(), line.moves.size());
        write_u16_le(out, (uint16_t)line.baseline_score);
        out.put(0);
    }
    return out.good();
}

static bool same_opening_meta(
    const OpeningBookMeta &a,
    const OpeningBookMeta &b) {
    return a.format_version == b.format_version
        && a.seed == b.seed
        && a.balance_limit == b.balance_limit
        && a.move_margin == b.move_margin
        && a.guide_depth == b.guide_depth
        && a.shallow_depth == b.shallow_depth
        && a.prefilter_depth == b.prefilter_depth
        && a.score_depth == b.score_depth
        && a.min_ply == b.min_ply
        && a.max_ply == b.max_ply
        && a.prefilter_limit == b.prefilter_limit;
}

static bool save_opening_generation_checkpoint(
    const std::string &path,
    const std::vector<OpeningLine> &lines,
    const OpeningBookMeta &meta,
    int requested,
    uint32_t next_candidate) {
    const std::string book_path = path + ".partial";
    const std::string book_tmp = book_path + ".tmp";
    const std::string state_path = path + ".next";
    const std::string state_tmp = state_path + ".tmp";
    if (!save_opening_book(book_tmp, lines, meta)) {
        std::cerr << "failed to write opening checkpoint "
                  << book_tmp << std::endl;
        return false;
    }
    if (std::rename(book_tmp.c_str(), book_path.c_str()) != 0) {
        std::perror("failed to install opening checkpoint");
        return false;
    }
    {
        std::ofstream out(state_tmp);
        if (!out) return false;
        out << "CFBGEN1 " << requested << ' '
            << next_candidate << '\n';
        if (!out.good()) return false;
    }
    if (std::rename(state_tmp.c_str(), state_path.c_str()) != 0) {
        std::perror("failed to install opening checkpoint cursor");
        return false;
    }
    return true;
}

static bool load_opening_generation_checkpoint(
    const std::string &path,
    const OpeningBookMeta &expected_meta,
    int requested,
    std::vector<OpeningLine> &lines,
    uint32_t &next_candidate) {
    const std::string book_path = path + ".partial";
    const std::string state_path = path + ".next";
    std::ifstream state(state_path);
    if (!state) return false;
    std::string magic;
    int stored_requested = 0;
    uint32_t stored_next = 0;
    if (!(state >> magic >> stored_requested >> stored_next)
        || magic != "CFBGEN1"
        || stored_requested != requested) {
        std::cerr << "ignoring incompatible opening checkpoint state "
                  << state_path << std::endl;
        return false;
    }
    if (!load_opening_book(book_path, true)) {
        std::cerr << "ignoring invalid opening checkpoint book "
                  << book_path << std::endl;
        return false;
    }
    if (!same_opening_meta(g_sprt_opening_meta, expected_meta)
        || g_sprt_openings.size() > (size_t)requested) {
        std::cerr << "ignoring incompatible opening checkpoint book "
                  << book_path << std::endl;
        g_sprt_openings.clear();
        g_sprt_opening_order.clear();
        g_sprt_opening_source.clear();
        return false;
    }
    lines = g_sprt_openings;
    next_candidate = stored_next;
    g_sprt_openings.clear();
    g_sprt_opening_order.clear();
    g_sprt_opening_source.clear();
    return true;
}

static void report_opening_book(
    const std::vector<OpeningLine> &lines,
    const OpeningBookMeta &meta,
    const std::string &source) {
    std::array<int, OPENING_BOOK_MAX_PLIES + 1> by_ply{};
    std::array<int, 81> first_moves{};
    std::vector<int> absolute_scores;
    absolute_scores.reserve(lines.size());
    int64_t score_sum = 0;
    int max_abs = 0;
    for (const OpeningLine &line : lines) {
        if (line.n_moves <= OPENING_BOOK_MAX_PLIES) {
            by_ply[line.n_moves]++;
        }
        if (line.n_moves > 0 && line.moves[0] < first_moves.size()) {
            first_moves[line.moves[0]]++;
        }
        score_sum += line.baseline_score;
        int av = std::abs((int)line.baseline_score);
        absolute_scores.push_back(av);
        max_abs = std::max(max_abs, av);
    }
    std::sort(absolute_scores.begin(), absolute_scores.end());
    auto percentile = [&](double q) {
        if (absolute_scores.empty()) return 0;
        size_t idx = (size_t)std::min<double>(
            absolute_scores.size() - 1,
            std::floor(q * (absolute_scores.size() - 1)));
        return absolute_scores[idx];
    };
    std::cout << "Opening book: " << source
              << " positions=" << lines.size()
              << " format=" << (int)meta.format_version
              << " seed=" << meta.seed
              << " move_margin=" << meta.move_margin
              << " shallow_depth=" << (int)meta.shallow_depth
              << " prefilter_depth=" << (int)meta.prefilter_depth
              << " prefilter_limit=" << meta.prefilter_limit
              << " score_depth=" << (int)meta.score_depth
              << " balance_limit=" << meta.balance_limit
              << " mean_score="
              << (lines.empty() ? 0.0
                                : score_sum / (double)lines.size())
              << " mean_abs=";
    int64_t abs_sum =
        std::accumulate(
            absolute_scores.begin(), absolute_scores.end(), int64_t{0});
    std::cout << (lines.empty() ? 0.0
                               : abs_sum / (double)lines.size())
              << " p50_abs=" << percentile(0.50)
              << " p90_abs=" << percentile(0.90)
              << " p99_abs=" << percentile(0.99)
              << " max_abs=" << max_abs << std::endl;
    std::cout << "Opening plies:";
    for (int ply = 0; ply <= OPENING_BOOK_MAX_PLIES; ply++) {
        if (by_ply[ply]) {
            std::cout << ' ' << ply << '=' << by_ply[ply];
        }
    }
    std::cout << std::endl;
    int distinct_first_moves = 0;
    int min_first_count = std::numeric_limits<int>::max();
    int max_first_count = 0;
    for (int count : first_moves) {
        if (count == 0) continue;
        distinct_first_moves++;
        min_first_count = std::min(min_first_count, count);
        max_first_count = std::max(max_first_count, count);
    }
    std::cout << "Opening first moves: distinct=" << distinct_first_moves
              << " center=" << first_moves[4 * 9 + 4]
              << " min_count="
              << (distinct_first_moves ? min_first_count : 0)
              << " max_count=" << max_first_count << std::endl;
    std::vector<size_t> order =
        make_opening_order(lines.size(), meta.seed);
    std::cout << "Opening traversal: fisher-yates-v1"
              << " seed=" << (meta.seed ^ OPENING_ORDER_SALT)
              << " fingerprint=" << opening_order_fingerprint(order)
              << " first_records=";
    size_t preview = std::min<size_t>(8, order.size());
    for (size_t i = 0; i < preview; i++) {
        if (i) std::cout << ',';
        std::cout << order[i];
    }
    std::cout << std::endl;
}

struct OpeningAuditResult {
    bool ok = false;
    int score = 0;
    uint64_t nodes = 0;
    int64_t elapsed_us = 0;
};

static int audit_opening_book(
    const std::string &path,
    int audit_depth,
    int requested,
    unsigned int threads,
    const std::string &score_output = "") {
    if (!load_opening_book(path)) return 1;
    if (audit_depth < 1) audit_depth = 20;
    size_t count = requested <= 0
        ? g_sprt_openings.size()
        : std::min<size_t>(
            g_sprt_openings.size(), (size_t)requested);
    threads = std::max(1u, std::min<unsigned int>(
        threads, (unsigned int)std::max<size_t>(1, count)));
    // The generated evaluator payload loaders predate concurrent tooling and
    // use simple ready flags. Warm them on this thread before constructors run
    // in parallel, establishing a happens-before edge through thread launch.
    {
        CrossfishPrev warmup;
        CrossfishPrev::init_mini_lut();
    }
    std::vector<OpeningAuditResult> results(count);
    std::atomic<size_t> next{0};
    std::atomic<size_t> completed{0};
    auto wall_start = std::chrono::steady_clock::now();
    std::vector<std::future<void>> futures;
    for (unsigned int worker = 0; worker < threads; worker++) {
        futures.push_back(std::async(
            std::launch::async,
            [&]() {
                while (true) {
                    size_t i = next.fetch_add(1);
                    if (i >= count) break;
                    GlobalBoard board;
                    OpeningAuditResult result;
                    if (replay_opening(g_sprt_openings[i], board)) {
                        // A fresh engine makes every score independent of TT,
                        // history, and counter-move state from earlier roots.
                        CrossfishPrev bot;
                        auto started = std::chrono::steady_clock::now();
                        result.ok = bot.search_fixed_depth(
                            board, audit_depth, result.score);
                        result.elapsed_us =
                            std::chrono::duration_cast<
                                std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
                        result.nodes = (uint64_t)bot.nodes;
                    }
                    results[i] = result;
                    size_t done = completed.fetch_add(1) + 1;
                    size_t interval = std::max<size_t>(1, count / 20);
                    if (done == count || done % interval == 0) {
                        double elapsed =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now()
                                - wall_start).count();
                        std::cout << "book audit: " << done << '/' << count
                                  << " depth=" << audit_depth
                                  << " elapsed=" << elapsed << "s"
                                  << std::endl;
                    }
                }
            }));
    }
    for (auto &future : futures) future.get();

    std::vector<int> absolute_scores;
    std::vector<int64_t> elapsed_us;
    absolute_scores.reserve(count);
    elapsed_us.reserve(count);
    int failed = 0;
    int within_100 = 0;
    int within_200 = 0;
    int within_300 = 0;
    int within_500 = 0;
    int stored_score_mismatches = 0;
    int max_stored_score_delta = 0;
    int64_t score_sum = 0;
    uint64_t node_sum = 0;
    for (size_t i = 0; i < results.size(); i++) {
        const OpeningAuditResult &result = results[i];
        if (!result.ok) {
            failed++;
            continue;
        }
        int av = std::abs(result.score);
        absolute_scores.push_back(av);
        elapsed_us.push_back(result.elapsed_us);
        score_sum += result.score;
        node_sum += result.nodes;
        within_100 += av <= 100;
        within_200 += av <= 200;
        within_300 += av <= 300;
        within_500 += av <= 500;
        int stored_delta = std::abs(
            result.score
            - (int)g_sprt_openings[i].baseline_score);
        if (stored_delta != 0) stored_score_mismatches++;
        max_stored_score_delta =
            std::max(max_stored_score_delta, stored_delta);
    }
    std::sort(absolute_scores.begin(), absolute_scores.end());
    std::sort(elapsed_us.begin(), elapsed_us.end());
    auto percentile = [](const auto &values, double q) {
        if (values.empty()) {
            return typename std::decay_t<decltype(values)>::value_type{};
        }
        size_t idx = (size_t)std::min<double>(
            values.size() - 1,
            std::floor(q * (values.size() - 1)));
        return values[idx];
    };
    size_t ok = absolute_scores.size();
    double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();
    std::cout << "Deep opening audit: source=" << path
              << " depth=" << audit_depth
              << " positions=" << count
              << " ok=" << ok
              << " failed=" << failed
              << " threads=" << threads
              << " wall_seconds=" << wall_seconds << std::endl;
    if (ok) {
        int64_t abs_sum = std::accumulate(
            absolute_scores.begin(), absolute_scores.end(), int64_t{0});
        std::cout << "Deep scores: mean="
                  << score_sum / (double)ok
                  << " mean_abs=" << abs_sum / (double)ok
                  << " p50_abs=" << percentile(absolute_scores, 0.50)
                  << " p90_abs=" << percentile(absolute_scores, 0.90)
                  << " p99_abs=" << percentile(absolute_scores, 0.99)
                  << " max_abs=" << absolute_scores.back()
                  << " within100=" << within_100
                  << " within200=" << within_200
                  << " within300=" << within_300
                  << " within500=" << within_500
                  << " stored_score_mismatches="
                  << stored_score_mismatches
                  << " max_stored_delta="
                  << max_stored_score_delta << std::endl;
        std::cout << "Deep search cost: mean_nodes="
                  << node_sum / (double)ok
                  << " median_ms="
                  << percentile(elapsed_us, 0.50) / 1000.0
                  << " p90_ms="
                  << percentile(elapsed_us, 0.90) / 1000.0
                  << " p99_ms="
                  << percentile(elapsed_us, 0.99) / 1000.0
                  << " max_ms=" << elapsed_us.back() / 1000.0
                  << std::endl;
    }
    if (!score_output.empty()) {
        std::ofstream out(score_output);
        if (!out) {
            std::cerr << "failed to write audit scores "
                      << score_output << std::endl;
            return 1;
        }
        out << "index\tplies\tbook_score\tdeep_score\tnodes\telapsed_us\n";
        for (size_t i = 0; i < count; i++) {
            const OpeningAuditResult &result = results[i];
            out << i << '\t'
                << (int)g_sprt_openings[i].n_moves << '\t'
                << g_sprt_openings[i].baseline_score << '\t';
            if (result.ok) {
                out << result.score << '\t'
                    << result.nodes << '\t'
                    << result.elapsed_us;
            } else {
                out << "NA\tNA\tNA";
            }
            out << '\n';
        }
        if (!out.good()) {
            std::cerr << "failed while writing audit scores "
                      << score_output << std::endl;
            return 1;
        }
        std::cout << "Wrote deep audit scores: "
                  << score_output << std::endl;
    }
    if (audit_depth == g_sprt_opening_meta.score_depth
        && stored_score_mismatches != 0) {
        std::cerr << "stored opening scores do not reproduce at depth "
                  << audit_depth << std::endl;
        return 1;
    }
    return failed ? 1 : 0;
}

static int generate_opening_book(
    const std::string &path,
    int requested,
    unsigned int threads) {
    const int min_ply = 4;
    const int max_ply = 10;
    const int move_margin = 250;
    const int shallow_depth = 4;
    const int shallow_limit = 300;
    const int prefilter_depth = 12;
    const int prefilter_limit = 525;
    const int balance_limit = 300;
    const int score_depth = 16;
    const uint32_t seed = 0xC0FFEE42u;
    const int n_ply_buckets = max_ply - min_ply + 1;
    if (requested < n_ply_buckets) requested = n_ply_buckets;
    threads = std::max(1u, threads);
    OpeningBookMeta meta;
    meta.format_version = 2;
    meta.seed = seed;
    meta.balance_limit = balance_limit;
    meta.move_margin = move_margin;
    meta.guide_depth = 0;  // full static HCE + D16 + macro evaluator
    meta.shallow_depth = shallow_depth;
    meta.prefilter_depth = prefilter_depth;
    meta.prefilter_limit = prefilter_limit;
    meta.score_depth = score_depth;
    meta.min_ply = min_ply;
    meta.max_ply = max_ply;
    {
        CrossfishPrev warmup;
        CrossfishPrev::init_mini_lut();
    }

    std::array<int, OPENING_BOOK_MAX_PLIES + 1> accepted_by_ply{};
    // Preserve broad early-game coverage without making the rarest ply bucket
    // dictate the runtime of the entire production build.
    const int minimum_per_ply =
        std::min(500, requested / n_ply_buckets);

    std::vector<OpeningLine> accepted;
    accepted.reserve(requested);
    std::set<std::string> seen_states;
    uint32_t next_candidate = 0;
    if (load_opening_generation_checkpoint(
            path, meta, requested, accepted, next_candidate)) {
        NnueNet checkpoint_encoder;
        char encoded[93];
        for (const OpeningLine &line : accepted) {
            if (line.n_moves <= OPENING_BOOK_MAX_PLIES) {
                accepted_by_ply[line.n_moves]++;
            }
            GlobalBoard board;
            if (!replay_opening(line, board)) {
                std::cerr << "illegal line in opening checkpoint"
                          << std::endl;
                return 1;
            }
            checkpoint_encoder.encode_state(board, encoded);
            seen_states.emplace(encoded, sizeof(encoded));
        }
        std::cout << "book generation resume: accepted="
                  << accepted.size() << '/' << requested
                  << " next_candidate=" << next_candidate
                  << std::endl;
    }
    const int batch_size = std::max(32u, threads * 8u);
    const uint32_t max_candidates =
        (uint32_t)std::max(100000, requested * 150);
    uint32_t last_report = 0;
    uint64_t unique_shallow = 0;
    uint64_t prefilter_passes = 0;
    uint64_t deep_searches = 0;
    uint64_t deep_passes = 0;
    // Every proposal slot is paired once with each target ply. All 81 legal
    // first moves get one slot; moves in the center miniboard get three extra
    // slots, and center-center gets eight more. This preserves broad coverage
    // while compensating for their much lower balance-filter acceptance.
    const int first_proposal_slots = 116;
    auto proposed_first_move = [&](uint32_t candidate_id) {
        int slot =
            (int)((candidate_id / n_ply_buckets)
                  % first_proposal_slots);
        if (slot < 81) return slot;
        if (slot < 108) return 36 + (slot - 81) / 3;
        return 40;
    };
    while ((int)accepted.size() < requested
           && next_candidate < max_candidates) {
        int current_batch = (int)std::min<uint32_t>(
            batch_size, max_candidates - next_candidate);
        std::vector<GeneratedOpening> results(current_batch);
        std::vector<std::future<void>> futures;
        for (unsigned int worker = 0; worker < threads; worker++) {
            futures.push_back(std::async(
                std::launch::async,
                [&, worker, current_batch]() {
                    CrossfishPrev bot;
                    NnueNet encoder;
                    for (int i = (int)worker;
                         i < current_batch;
                         i += (int)threads) {
                        uint32_t id = next_candidate + (uint32_t)i;
                        int target =
                            min_ply + (int)(id % n_ply_buckets);
                        int first_move = proposed_first_move(id);
                        results[i] = make_opening_candidate(
                            id, seed, target, move_margin, shallow_limit,
                            shallow_depth, first_move, bot, encoder);
                    }
                }));
        }
        for (auto &future : futures) future.get();

        std::vector<size_t> score_indices;
        score_indices.reserve(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            const GeneratedOpening &result = results[i];
            if (!result.accepted) continue;
            if (!seen_states.insert(result.state).second) continue;
            unique_shallow++;
            score_indices.push_back(i);
        }
        std::atomic<size_t> next_score{0};
        futures.clear();
        for (unsigned int worker = 0; worker < threads; worker++) {
            futures.push_back(std::async(
                std::launch::async,
                [&]() {
                    while (true) {
                        size_t slot = next_score.fetch_add(1);
                        if (slot >= score_indices.size()) break;
                        GeneratedOpening &result =
                            results[score_indices[slot]];
                        GlobalBoard board;
                        if (!replay_opening(result.line, board)) continue;
                        int score = 0;
                        CrossfishPrev prefilter_bot;
                        if (!prefilter_bot.search_fixed_depth(
                                board, prefilter_depth, score)
                            || std::abs(score) > prefilter_limit) {
                            continue;
                        }
                        result.prefilter_passed = true;
                        CrossfishPrev deep_bot;
                        if (!deep_bot.search_fixed_depth(
                                board, score_depth, score)
                            || std::abs(score) > balance_limit) {
                            continue;
                        }
                        result.line.baseline_score = (int16_t)score;
                        result.deep_accepted = true;
                    }
                }));
        }
        for (auto &future : futures) future.get();

        for (size_t i : score_indices) {
            GeneratedOpening &result = results[i];
            if (result.prefilter_passed) {
                prefilter_passes++;
                deep_searches++;
            }
            if (!result.deep_accepted) continue;
            deep_passes++;
            int ply = result.target_ply;
            if ((int)accepted.size() >= requested) continue;
            int missing_minimum = 0;
            for (int p = min_ply; p <= max_ply; p++) {
                missing_minimum += std::max(
                    0, minimum_per_ply - accepted_by_ply[p]);
            }
            int slots_left = requested - (int)accepted.size();
            if (accepted_by_ply[ply] >= minimum_per_ply
                && slots_left <= missing_minimum) {
                continue;
            }
            accepted.push_back(result.line);
            accepted_by_ply[ply]++;
        }
        next_candidate += (uint32_t)current_batch;
        if ((int)accepted.size() == requested
            || next_candidate - last_report
                   >= (uint32_t)(batch_size * 4)) {
            last_report = next_candidate;
            std::cout << "book generation: accepted=" << accepted.size()
                      << '/' << requested
                      << " candidates=" << next_candidate << " plies";
            for (int ply = min_ply; ply <= max_ply; ply++) {
                std::cout << ' ' << ply << '=' << accepted_by_ply[ply];
            }
            std::cout << " minimum_per_ply=" << minimum_per_ply
                      << " unique_shallow=" << unique_shallow
                      << " prefilter_passes=" << prefilter_passes
                      << " deep_searches=" << deep_searches
                      << " deep_passes=" << deep_passes
                      << std::endl;
            if (!save_opening_generation_checkpoint(
                    path, accepted, meta, requested, next_candidate)) {
                return 1;
            }
        }
    }
    if ((int)accepted.size() != requested) {
        std::cerr << "opening generation exhausted candidates at "
                  << accepted.size() << '/' << requested << std::endl;
        return 1;
    }

    if (!save_opening_book(path, accepted, meta)) {
        std::cerr << "failed to write opening book " << path << std::endl;
        return 1;
    }
    std::remove((path + ".partial").c_str());
    std::remove((path + ".next").c_str());
    report_opening_book(accepted, meta, path);
    return 0;
}

static bool parse_pentanomial_counts(
    const char *text,
    std::array<int, 5> &counts) {
    char trailing = 0;
    int parsed = std::sscanf(
        text, "%d,%d,%d,%d,%d%c",
        &counts[0], &counts[1], &counts[2], &counts[3], &counts[4],
        &trailing);
    if (parsed != 5) return false;
    return std::all_of(
        counts.begin(), counts.end(), [](int value) { return value >= 0; });
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "book") == 0) {
        if (std::strcmp(argv[2], "generate") == 0) {
            std::string path =
                argc >= 4 ? argv[3] : "cpp_impl/opening_book.bin";
            int count = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 50000;
            CpuTopology topology = detect_cpu_topology();
            unsigned int threads =
                topology.physical > 1 ? topology.physical - 1 : 1;
            if (argc >= 6) {
                threads =
                    (unsigned int)std::max(1, std::atoi(argv[5]));
            }
            return generate_opening_book(path, count, threads);
        }
        if (std::strcmp(argv[2], "inspect") == 0) {
            if (argc < 4) {
                if (!load_default_opening_book()) return 1;
                report_opening_book(
                    g_sprt_openings, g_sprt_opening_meta,
                    g_sprt_opening_source);
                return 0;
            }
            std::string path = argv[3];
            if (!load_opening_book(path)) return 1;
            report_opening_book(
                g_sprt_openings, g_sprt_opening_meta, path);
            return 0;
        }
        if (std::strcmp(argv[2], "audit") == 0) {
            std::string path =
                argc >= 4 ? argv[3] : "cpp_impl/opening_book.bin";
            int depth = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 20;
            int count = argc >= 6 ? std::max(0, std::atoi(argv[5])) : 0;
            CpuTopology topology = detect_cpu_topology();
            unsigned int threads =
                topology.physical > 1 ? topology.physical - 1 : 1;
            if (argc >= 7) {
                threads =
                    (unsigned int)std::max(1, std::atoi(argv[6]));
            }
            std::string score_output = argc >= 8 ? argv[7] : "";
            return audit_opening_book(
                path, depth, count, threads, score_output);
        }
        std::cerr << "usage: test_bots book generate|inspect|audit"
                  << " [path] [count/depth] [threads/count] [threads]"
                  << " [audit_scores.tsv]"
                  << std::endl;
        return 1;
    }
    if (argc >= 2 && std::strcmp(argv[1], "dump") == 0) {
        if (argc >= 3 && std::strcmp(argv[2], "hce") == 0) {
            int n_pos = 2000000;
            const char *path = "../../datasets/nnue_hce_rand.bin";
            if (argc >= 4) n_pos = std::atoi(argv[3]);
            if (argc >= 5) path = argv[4];
            dump_nnue_hce(n_pos, path);
            return 0;
        }
        if (argc >= 3 && std::strcmp(argv[2], "annotate") == 0) {
            if (argc < 5) {
                std::cerr << "usage: test_bots dump annotate in.bin out.bin" << std::endl;
                return 1;
            }
            dump_nnue_annotate(argv[3], argv[4]);
            return 0;
        }
        if (argc >= 3 && std::strcmp(argv[2], "distill") == 0) {
            if (argc < 6) {
                std::cerr << "usage: test_bots dump distill teacher.bin out.bin in1.bin [in2.bin ...]" << std::endl;
                return 1;
            }
            std::vector<std::string> ins;
            for (int i = 5; i < argc; i++) ins.push_back(argv[i]);
            dump_nnue_distill(argv[3], argv[4], ins);
            return 0;
        }
        if (argc >= 3 && std::strcmp(argv[2], "search") == 0) {
            int depth = 5;
            int n_pos = 800000;
            const char *path = "datasets/nnue_search.bin";
            bool play_only = false;
            if (argc >= 4) depth = std::atoi(argv[3]);
            if (argc >= 5) n_pos = std::atoi(argv[4]);
            if (argc >= 6) path = argv[5];
            if (argc >= 7 && std::strcmp(argv[6], "play") == 0) play_only = true;
            dump_nnue_search(depth, n_pos, path, play_only);
            return 0;
        }
        if (argc >= 3 && std::strcmp(argv[2], "relabel") == 0) {
            if (argc < 7) {
                std::cerr
                    << "usage: test_bots dump relabel depth n_pos in.bin out.bin"
                    << " [hce|current]" << std::endl;
                return 1;
            }
            int depth = std::atoi(argv[3]);
            int n_pos = std::atoi(argv[4]);
            bool force_hce =
                argc < 8 || std::strcmp(argv[7], "current") != 0;
            dump_nnue_search(
                depth, n_pos, argv[6], true, argv[5], force_hce);
            return 0;
        }
        if (argc >= 3 && std::strcmp(argv[2], "rank") == 0) {
            int depth = 5;
            int n_roots = 10000;
            const char *path = "datasets/nnue_rank.bin";
            const char *source = nullptr;
            if (argc >= 4) depth = std::atoi(argv[3]);
            if (argc >= 5) n_roots = std::atoi(argv[4]);
            if (argc >= 6) path = argv[5];
            if (argc >= 7) source = argv[6];
            dump_nnue_rank(depth, n_roots, path, source);
            return 0;
        }
        int n_games = 8000;
        int think_ms = 20;
        const char *path = "../../datasets/nnue_pos.bin";
        bool record_root_score = false;
        if (argc >= 3 && (std::strcmp(argv[2], "nnue") == 0
                          || std::strcmp(argv[2], "root") == 0)) {
            record_root_score = std::strcmp(argv[2], "root") == 0;
            if (argc >= 4) n_games = std::atoi(argv[3]);
            if (argc >= 5) think_ms = std::atoi(argv[4]);
            if (argc >= 6) path = argv[5];
        } else {
            if (argc >= 3) n_games = std::atoi(argv[2]);
            if (argc >= 4) path = argv[3];
        }
        dump_nnue_wdl(n_games, think_ms, path, record_root_score);
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "nnue") == 0 && argc >= 3 && std::strcmp(argv[2], "pin") == 0) {
        pin_nnue_scale();
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "nnue") == 0 && argc >= 3 && std::strcmp(argv[2], "fit") == 0) {
        if (argc >= 4 && std::strcmp(argv[3], "search") == 0) {
            int depth = 5;
            if (argc >= 5) depth = std::atoi(argv[4]);
            report_nnue_search_fit(depth);
            return 0;
        }
        report_nnue_hce_fit();
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "probe") == 0) {
        if (argc < 4 || std::strcmp(argv[2], "mini") != 0) {
            std::cerr << "usage: test_bots probe mini path.bin" << std::endl;
            return 1;
        }
        g_nnue_mode = 2;
        g_nnue_residual = 1;
        std::snprintf(g_nnue_bin_path, sizeof(g_nnue_bin_path), "%s", argv[3]);
        if (!nnue_init_runtime()) return 1;
        CrossfishDev::init_mini_lut();
        CrossfishDev bot;
        GlobalBoard empty;
        int h0 = bot.evaluate_hce(empty);
        int m0 = g_nnue_mini.evaluate(empty);
        std::cout << "empty hce=" << h0 << " mini=" << m0
                  << " combined=" << bot.evaluate(empty) << std::endl;
        std::mt19937 rng(7);
        Move buf[81];
        double sum_h = 0, sum_m = 0;
        int n = 0, mismatch = 0, unmake_mismatch = 0, pass_mismatch = 0;
        MiniNnue::Acc acc;
        for (int g = 0; g < 80; g++) {
            GlobalBoard board;
            g_nnue_mini.refresh(board, acc);
            for (int ply = 0; ply < 80; ply++) {
                if (board.checkWinner() != -1) break;
                int nmoves = board.fillLegalMoves(buf);
                if (nmoves <= 0) break;
                int hv = bot.evaluate_hce(board);
                int mv = g_nnue_mini.evaluate(board);
                int inc = acc.ok ? g_nnue_mini.evaluate_acc(acc, board.n_moves) : mv;
                if (inc != mv) mismatch++;
                if (n < 6) {
                    std::cout << "hce=" << hv << " mini=" << mv << " acc=" << inc << std::endl;
                }
                sum_h += hv;
                sum_m += mv;
                n++;
                if ((ply % 11) == 7 && !board.prev_move_was_pass) {
                    int before = inc;
                    board.pass();
                    g_nnue_mini.make(board, 0, acc);
                    int pinc = g_nnue_mini.evaluate_acc(acc, board.n_moves);
                    int psc = g_nnue_mini.evaluate(board);
                    if (pinc != psc) pass_mismatch++;
                    g_nnue_mini.unmake(acc);
                    board.unpass();
                    int restored = g_nnue_mini.evaluate_acc(acc, board.n_moves);
                    if (restored != before) unmake_mismatch++;
                }
                Move m = buf[rng() % nmoves];
                int before = inc;
                board.makeMove(m);
                g_nnue_mini.make(board, m.mini_board, acc);
                g_nnue_mini.unmake(acc);
                board.unmakeMove();
                if (g_nnue_mini.evaluate_acc(acc, board.n_moves) != before) unmake_mismatch++;
                board.makeMove(m);
                g_nnue_mini.make(board, m.mini_board, acc);
            }
        }
        std::cout << "random n=" << n
                  << " mean_hce=" << (sum_h / n)
                  << " mean_mini=" << (sum_m / n)
                  << " mean|mini|=" << std::fabs(sum_m / n)
                  << " acc_mismatch=" << mismatch
                  << " unmake_mismatch=" << unmake_mismatch
                  << " pass_mismatch=" << pass_mismatch << std::endl;
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "tune") == 0) {
        CrossfishDev::init_mini_lut();
        if (argc >= 3 && std::strcmp(argv[2], "hce") == 0) {
            if (argc >= 4 && std::strcmp(argv[3], "utttai") == 0) {
                bool load_saved = argc >= 5 && std::strcmp(argv[4], "load") == 0;
                const char *dir = (!load_saved && argc >= 5) ? argv[4] : nullptr;
                run_texel_utttai(dir, load_saved);
            } else {
                bool load_saved = argc >= 4 && std::strcmp(argv[3], "load") == 0;
                run_texel(load_saved);
            }
        } else {
            bool load_saved = argc >= 3 && std::strcmp(argv[2], "load") == 0;
            run_lut_texel(load_saved);
        }
        return 0;
    }
    verify_fill_movegen();
    verify_referee_timeout();
    verify_opening_order();
    verify_pentanomial_sprt();
    verify_mini_lut();
    verify_utttai_state();
    verify_eval_linear();
    verify_nnue_incremental();
    if (argc >= 2 && std::strcmp(argv[1], "verify") == 0) {
        return 0;
    }
    int argi = 1;
    if (argc >= 2 && (std::strcmp(argv[1], "90") == 0 || std::strcmp(argv[1], "90ms") == 0)) {
        g_sprt_think_ms = 90;
        argi = 2;
    } else if (argc >= 2 && (std::strcmp(argv[1], "95") == 0 || std::strcmp(argv[1], "95ms") == 0)) {
        g_sprt_think_ms = 95;
        argi = 2;
    } else if (argc >= 2 && (std::strcmp(argv[1], "20") == 0 || std::strcmp(argv[1], "20ms") == 0)) {
        g_sprt_think_ms = 20;
        argi = 2;
    }
    if (argi < argc
        && (std::strcmp(argv[argi], "depth") == 0
            || std::strcmp(argv[argi], "depth-prune") == 0)) {
        bool keep_eval_pruning =
            std::strcmp(argv[argi], "depth-prune") == 0;
        g_fixed_search_depth = (argc > argi + 1) ? std::atoi(argv[argi + 1]) : 4;
        if (g_fixed_search_depth < 1) g_fixed_search_depth = 4;
        g_disable_eval_prune = !keep_eval_pruning;
        argi += 2;
    }
    if (argi < argc) {
        const char *mode = argv[argi];
        if (std::strcmp(mode, "sparse") == 0 || std::strcmp(mode, "residual") == 0
            || std::strcmp(mode, "sparse-res") == 0 || std::strcmp(mode, "mini") == 0
            || std::strcmp(mode, "minires") == 0 || std::strcmp(mode, "mini-res") == 0) {
            const char *bin = (argc > argi + 1) ? argv[argi + 1] : nullptr;
            if (!parse_nnue_mode(mode, bin)) {
                std::cerr << "unknown nnue mode " << mode
                          << " (sparse|residual|mini|minires)" << std::endl;
                return 1;
            }
        }
    }
    if (const char *s = std::getenv("SPRT_THINK_MS")) {
        g_sprt_think_ms = std::max(1, std::atoi(s));
    }
    if (const char *s = std::getenv("SPRT_ELO0")) {
        g_sprt_elo0 = std::atof(s);
    }
    if (const char *s = std::getenv("SPRT_ELO1")) {
        g_sprt_elo1 = std::atof(s);
    }
    if (const char *s = std::getenv("SPRT_LLR_BOUND")) {
        g_sprt_llr_bound = std::max(0.1, std::atof(s));
    }
    if (const char *s = std::getenv("SPRT_MAX_GAMES")) {
        g_sprt_max_games = std::max(0, std::atoi(s));
    }
    if (const char *s = std::getenv("SPRT_PAIR_MODEL")) {
        g_sprt_pair_model = std::atoi(s) != 0;
    }
    if (const char *s = std::getenv("SPRT_ALLOW_BOOK_WRAP")) {
        g_sprt_allow_book_wrap = std::atoi(s) != 0;
    }
    if (const char *s = std::getenv("SPRT_BOOK")) {
        g_sprt_book = std::atoi(s) != 0;
    }
    if (const char *s = std::getenv("SPRT_CENTER_ENUM")) {
        g_sprt_center_enum = std::atoi(s) != 0;
    }
    if (const char *s = std::getenv("SPRT_THREADS")) {
        g_sprt_threads = (unsigned int)std::max(1, std::atoi(s));
    }
    if (const char *s = std::getenv("SPRT_RESUME_WINS")) {
        g_sprt_resume_wins = std::max(0, std::atoi(s));
    }
    if (const char *s = std::getenv("SPRT_RESUME_DRAWS")) {
        g_sprt_resume_draws = std::max(0, std::atoi(s));
    }
    if (const char *s = std::getenv("SPRT_RESUME_LOSSES")) {
        g_sprt_resume_losses = std::max(0, std::atoi(s));
    }
    if (const char *s = std::getenv("SPRT_RESUME_PENTA")) {
        if (!parse_pentanomial_counts(s, g_sprt_resume_pentanomial)) {
            std::cerr
                << "SPRT_RESUME_PENTA must be five nonnegative"
                << " comma-separated counts" << std::endl;
            return 1;
        }
        g_sprt_resume_pentanomial_set = true;
    }
    if (const char *s = std::getenv("SPRT_GAME_OFFSET")) {
        g_sprt_game_offset = std::max(0, std::atoi(s));
    }
    const char *opening_book_env =
        std::getenv("SPRT_OPENING_BOOK");
    bool opening_book_disabled =
        opening_book_env
        && (std::strcmp(opening_book_env, "0") == 0
            || std::strcmp(opening_book_env, "none") == 0);
    if (!g_sprt_center_enum && !opening_book_disabled) {
        if (opening_book_env && opening_book_env[0]) {
            if (!load_opening_book(opening_book_env)) {
                std::cerr << "failed to load SPRT_OPENING_BOOK="
                          << opening_book_env << std::endl;
                return 1;
            }
        } else if (g_sprt_book) {
            if (!load_default_opening_book()) {
                std::cerr
                    << "Set SPRT_OPENING_BOOK to an explicit path, or set"
                    << " SPRT_OPENING_BOOK=none to use the legacy seeded"
                    << " 4-8 ply opener." << std::endl;
                return 1;
            }
        }
    }
    if (!(g_sprt_elo1 > g_sprt_elo0)) {
        std::cerr << "SPRT_ELO1 must be greater than SPRT_ELO0" << std::endl;
        return 1;
    }
    const int resumed_games =
        g_sprt_resume_wins + g_sprt_resume_draws + g_sprt_resume_losses;
    if (resumed_games & 1) {
        std::cerr
            << "SPRT resume total must be even"
            << " (one opening produces two games)" << std::endl;
        return 1;
    }
    int resumed_pairs = 0;
    if (g_sprt_resume_pentanomial_set) {
        resumed_pairs = std::accumulate(
            g_sprt_resume_pentanomial.begin(),
            g_sprt_resume_pentanomial.end(), 0);
        if (resumed_pairs * 2 != resumed_games) {
            std::cerr
                << "SPRT_RESUME_PENTA does not match resumed W/D/L"
                << std::endl;
            return 1;
        }
    } else if (g_sprt_pair_model && resumed_games != 0) {
        std::cerr
            << "paired SPRT resume requires SPRT_RESUME_PENTA="
            << "LL,LD,MID,DW,WW" << std::endl;
        return 1;
    }
    if (g_sprt_game_offset < 0) {
        g_sprt_game_offset = resumed_games / 2;
    }
    global_total = {
        g_sprt_resume_wins,
        g_sprt_resume_draws,
        g_sprt_resume_losses
    };
    global_pentanomial = g_sprt_resume_pentanomial;
    if (!nnue_init_runtime()) {
        return 1;
    }
    std::cout << "SPRT think ms: " << g_sprt_think_ms
              << " H0=" << g_sprt_elo0
              << " H1=" << g_sprt_elo1
              << " bound=" << g_sprt_llr_bound
              << " model="
              << (g_sprt_pair_model ? "pentanomial-pairs"
                                    : "trinomial-games")
              << " referee_timeout="
              << (g_fixed_search_depth > 0
                      ? std::string("off")
                      : std::to_string(REFEREE_MOVE_TIMEOUT_MS) + "ms")
              << " eval=HCE+MiniNet"
              << " nnue_mode=" << g_nnue_mode
              << " residual=" << g_nnue_residual
              << " opening="
              << (!g_sprt_openings.empty()
                      ? "book"
                      : (g_sprt_center_enum
                             ? "center-enum-4ply"
                             : (g_sprt_book ? "seeded-4to8"
                                            : "random-4to8")));
    if (!g_sprt_openings.empty()) {
        std::cout << " book_wrap="
                  << (g_sprt_allow_book_wrap ? "explicit" : "disabled");
    }
    if (g_nnue_bin_path[0]) {
        std::cout << " nnue_bin=" << g_nnue_bin_path;
    }
    std::cout << std::endl;
    if (!g_sprt_openings.empty()) {
        report_opening_book(
            g_sprt_openings, g_sprt_opening_meta,
            g_sprt_opening_source);
    }
    if (argc >= 2 && std::strcmp(argv[1], "lut") == 0) {
        if (!load_mini_scores(MINI_SCORE_PATH)) {
            std::cerr << "failed to load " << MINI_SCORE_PATH << std::endl;
            return 1;
        }
        std::cout << "loaded " << MINI_SCORE_PATH << " into Dev" << std::endl;
    }
    CpuTopology topology = detect_cpu_topology();
    const unsigned int default_threads =
        topology.physical > 1 ? topology.physical - 1 : 1;
    const unsigned int n_threads = g_sprt_threads
        ? g_sprt_threads
        : default_threads;
    std::cout << "Number of threads: " << n_threads
              << " (" << topology.physical << " physical / "
              << topology.logical << " logical available"
              << (g_sprt_threads ? ", override" : ", one core reserved")
              << ")" << std::endl;
    double llr = current_sprt_llr();

    //benchmark NPS from startpos for Prev and Dev
    CrossfishPrev prev;
    CrossfishDev dev;
    GlobalBoard board;
    std::chrono::milliseconds thinking_time = std::chrono::milliseconds(1000);//1 second
    prev.getMove(board, thinking_time);
    int prev_nps = prev.nodes;
    dev.getMove(board, thinking_time);
    int dev_nps = dev.nodes;
    std::cout << "Prev NPS: " << prev_nps << " Dev NPS: " << dev_nps << std::endl;
    int total_games = resumed_games;
    int game_idx = g_sprt_game_offset;
    if (resumed_games) {
        std::cout << "SPRT resume: N=" << resumed_games
                  << " W=" << global_total[0]
                  << " D=" << global_total[1]
                  << " L=" << global_total[2]
                  << " Penta="
                  << global_pentanomial[0] << ','
                  << global_pentanomial[1] << ','
                  << global_pentanomial[2] << ','
                  << global_pentanomial[3] << ','
                  << global_pentanomial[4]
                  << " next opening=" << game_idx
                  << " LLR=" << llr << std::endl;
    }
    bool book_exhausted = false;
    while (std::abs(llr) < g_sprt_llr_bound
           && (g_sprt_max_games == 0 || total_games < g_sprt_max_games)) {
        unsigned int jobs = n_threads;
        if (g_sprt_max_games != 0) {
            int pairs_left =
                std::max(0, (g_sprt_max_games - total_games) / 2);
            jobs = std::min<unsigned int>(
                jobs, (unsigned int)pairs_left);
        }
        if (!g_sprt_openings.empty() && !g_sprt_allow_book_wrap) {
            size_t openings_left =
                game_idx < (int)g_sprt_opening_order.size()
                    ? g_sprt_opening_order.size() - (size_t)game_idx
                    : 0;
            jobs = std::min<unsigned int>(
                jobs, (unsigned int)openings_left);
            if (jobs == 0) {
                book_exhausted = true;
                break;
            }
        }
        if (jobs == 0) break;
        std::vector<std::future<void>> futures;
        for (unsigned int i = 0; i < jobs; ++i) {
            futures.push_back(std::async(std::launch::async, play_game, game_idx++));
        }
        for (auto& f : futures) {
            f.get();
        }
        total_games = global_total[0] + global_total[1] + global_total[2];
        EloResult elo = g_sprt_pair_model
            ? calc_pentanomial_elo(global_pentanomial)
            : calc_elo_diff(
                  global_total[0], global_total[2], global_total[1]);
        llr = current_sprt_llr();
        std::cout << "N: " << total_games << " W: " << global_total[0]
                << " D: " << global_total[1] << " L: " << global_total[2]
                << " Penta="
                << global_pentanomial[0] << ','
                << global_pentanomial[1] << ','
                << global_pentanomial[2] << ','
                << global_pentanomial[3] << ','
                << global_pentanomial[4]
                << " Elo diff: " << elo.elo_diff << " +/- " << elo.ci
                << " LLR: " << llr
                << " timeouts Prev=" << prev_timeout_losses.load()
                << " Dev=" << dev_timeout_losses.load()
                << " max_ms Prev="
                << prev_max_move_ns.load() / 1000000.0
                << " Dev=" << dev_max_move_ns.load() / 1000000.0
                << std::endl;
    }
    if (llr >= g_sprt_llr_bound) {
        std::cout << "SPRT PASS: H1 " << g_sprt_elo1
                  << " Elo favored over H0 " << g_sprt_elo0 << std::endl;
    } else if (llr <= -g_sprt_llr_bound) {
        std::cout << "SPRT FAIL: H0 " << g_sprt_elo0
                  << " Elo favored over H1 " << g_sprt_elo1 << std::endl;
    } else if (book_exhausted) {
        int tested_pairs = std::accumulate(
            global_pentanomial.begin(), global_pentanomial.end(), 0);
        std::cout
            << "SPRT INCONCLUSIVE: opening book exhausted after "
            << total_games << " games / " << tested_pairs
            << " unique opening pairs" << std::endl;
    } else {
        std::cout << "SPRT INCONCLUSIVE at " << total_games << " games" << std::endl;
    }
    return 0;
}
