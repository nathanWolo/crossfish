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
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif
#include <immintrin.h>

// CodinGame UTTT: 1000ms first execute per player, 100ms per later move
// (engine searches 800ms / 95ms). SPRT uses the per-move budget.
static int g_sprt_think_ms = 20;
static double g_sprt_elo0 = 0;
static double g_sprt_elo1 = 5;
static double g_sprt_llr_bound = 3;
static int g_sprt_max_games = 0;
static unsigned int g_sprt_threads = 0;
#pragma GCC optimize("O3")
#pragma GCC optimization("Ofast,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include "global_board.hpp"
#include "nnue.hpp"
#include "crossfish_prev.hpp"
#include "crossfish_dev.hpp"

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
std::mutex global_mutex;
std::atomic<int> completed_tasks(0);
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

double sprt(int wins, int draws, int losses) {
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

void play_game(){
    //play two games from the same start position, alternating who goes first
    RandomMover random_mover;

    //update these bots to test new changes
    CrossfishDev bot2;
    CrossfishPrev bot1;

    GlobalBoard board;
    //game loop
    //first 4-8 moves are random
    int num_random_moves = 4 + rand() % 5;
    for (int i = 0; i < num_random_moves; i++) {
        if (i == 0) {
            //30% chance of first move being very center
            if (rand() % 10 < 3) {
                Move m = {4, 4};
                board.makeMove(m);
                continue;
            }
        }
        Move m = random_mover.getMove(board);
        board.makeMove(m);
    }
    GlobalBoard startpos = GlobalBoard(board);
    //play two games, alternating who goes first
    for (int i = 0; i < 2; i++) {
        int bot1_player;
        int bot2_player;
        while (board.checkWinner() == -1){
            if (board.n_moves % 2 == i) {
                bot1_player = board.n_moves % 2;
                Move m = bot1.getMove(board, std::chrono::milliseconds(g_sprt_think_ms));
                board.makeMove(m);
            }
            else {
                bot2_player = board.n_moves % 2;
                Move best_move = bot2.getMove(board, std::chrono::milliseconds(g_sprt_think_ms));
                board.makeMove(best_move);
            }
        }
        //update global total
        int winner = board.checkWinner();
        if (winner  == bot1_player) {
            global_mutex.lock();
            global_total[2]++; //loss
            global_mutex.unlock();
        }
        else if (winner  == bot2_player) {
            global_mutex.lock();
            global_total[0]++;  //win
            global_mutex.unlock();
        }
        else {
            global_mutex.lock();
            global_total[1]++; //draw
            global_mutex.unlock();
        }
        board = GlobalBoard(startpos);
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
    std::cout << "utttai state parse: OK" << std::endl;
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

static void play_nnue_games(int n_games, int think_ms, std::vector<NnueDumpPos> &out, uint32_t seed) {
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
            board.makeMove(m);
        }
        int winner = board.checkWinner();
        for (size_t i = 0; i < game_pos.size(); i++) {
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

static void dump_nnue_wdl(int n_games, int think_ms, const char *path) {
    const unsigned int n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "NNUE dump: " << n_games << " games at " << think_ms
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
        futures.push_back(std::async(std::launch::async, play_nnue_games, n, think_ms, std::ref(data), seed));
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
static void dump_nnue_search(int depth, int n_pos, const char *path, bool play_only) {
    g_force_hce_eval = true;
    std::string rand_path = find_data_file("nnue_hce_rand.bin");
    std::string play_path = find_data_file("nnue_pos.bin");
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
        board.prev_move_was_pass = occupied > 0;
        // fillLegalMoves always reads move_history.top() when n_moves > 0.
        if (occupied > 0) board.move_history.push(Move{0, 0});
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

int main(int argc, char** argv) {
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
        int n_games = 8000;
        int think_ms = 20;
        const char *path = "../../datasets/nnue_pos.bin";
        if (argc >= 3 && std::strcmp(argv[2], "nnue") == 0) {
            if (argc >= 4) n_games = std::atoi(argv[3]);
            if (argc >= 5) think_ms = std::atoi(argv[4]);
            if (argc >= 6) path = argv[5];
        } else {
            if (argc >= 3) n_games = std::atoi(argv[2]);
            if (argc >= 4) path = argv[3];
        }
        dump_nnue_wdl(n_games, think_ms, path);
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
    verify_mini_lut();
    verify_utttai_state();
    verify_eval_linear();
    verify_nnue_incremental();
    if (argc >= 2 && std::strcmp(argv[1], "verify") == 0) {
        return 0;
    }
    int argi = 1;
    if (argc >= 2 && (std::strcmp(argv[1], "95") == 0 || std::strcmp(argv[1], "95ms") == 0)) {
        g_sprt_think_ms = 95;
        argi = 2;
    }
    if (argi < argc && std::strcmp(argv[argi], "depth") == 0) {
        g_fixed_search_depth = (argc > argi + 1) ? std::atoi(argv[argi + 1]) : 4;
        if (g_fixed_search_depth < 1) g_fixed_search_depth = 4;
        g_disable_eval_prune = true;
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
    if (const char *s = std::getenv("SPRT_THREADS")) {
        g_sprt_threads = (unsigned int)std::max(1, std::atoi(s));
    }
    if (!(g_sprt_elo1 > g_sprt_elo0)) {
        std::cerr << "SPRT_ELO1 must be greater than SPRT_ELO0" << std::endl;
        return 1;
    }
    if (!nnue_init_runtime()) {
        return 1;
    }
    std::cout << "SPRT think ms: " << g_sprt_think_ms
              << " H0=" << g_sprt_elo0
              << " H1=" << g_sprt_elo1
              << " bound=" << g_sprt_llr_bound
              << " eval=HCE+MiniNet"
              << " nnue_mode=" << g_nnue_mode
              << " residual=" << g_nnue_residual;
    if (g_nnue_bin_path[0]) {
        std::cout << " nnue_bin=" << g_nnue_bin_path;
    }
    std::cout << std::endl;
    if (argc >= 2 && std::strcmp(argv[1], "lut") == 0) {
        if (!load_mini_scores(MINI_SCORE_PATH)) {
            std::cerr << "failed to load " << MINI_SCORE_PATH << std::endl;
            return 1;
        }
        std::cout << "loaded " << MINI_SCORE_PATH << " into Dev" << std::endl;
    }
    const unsigned int n_threads = g_sprt_threads
        ? g_sprt_threads
        : std::max(1u, std::thread::hardware_concurrency());
    // const unsigned int n_threads = 6;
    std::cout << "Number of threads: " << n_threads << std::endl;
    double llr = 0;

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
    int total_games = 0;
    while (std::abs(llr) < g_sprt_llr_bound
           && (g_sprt_max_games == 0 || total_games < g_sprt_max_games)) {
        std::vector<std::future<void>> futures;
        for (unsigned int i = 0; i < n_threads; ++i) {
            futures.push_back(std::async(std::launch::async, play_game));
        }
        for (auto& f : futures) {
            f.get();
        }
        total_games = global_total[0] + global_total[1] + global_total[2];
        EloResult elo = calc_elo_diff(global_total[0], global_total[2], global_total[1]);
        llr = sprt(global_total[0], global_total[1], global_total[2]);
        std::cout << "N: " << total_games << " W: " << global_total[0]
                << " D: " << global_total[1] << " L: " << global_total[2]
                << " Elo diff: " << elo.elo_diff << " +/- " << elo.ci << " LLR: " << llr << std::endl;
    }
    if (llr >= g_sprt_llr_bound) {
        std::cout << "SPRT PASS: H1 " << g_sprt_elo1
                  << " Elo favored over H0 " << g_sprt_elo0 << std::endl;
    } else if (llr <= -g_sprt_llr_bound) {
        std::cout << "SPRT FAIL: H0 " << g_sprt_elo0
                  << " Elo favored over H1 " << g_sprt_elo1 << std::endl;
    } else {
        std::cout << "SPRT INCONCLUSIVE at " << total_games << " games" << std::endl;
    }
    return 0;
}
