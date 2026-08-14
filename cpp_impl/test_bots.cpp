#include <iostream>
#include <vector>
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
#include <immintrin.h>

// CodinGame UTTT: 1000ms first execute per player, 100ms per later move
// (engine searches 800ms / 95ms). SPRT uses the per-move budget.
static constexpr int SPRT_THINK_MS = 95;
#pragma GCC optimize("O3")
#pragma GCC optimization("Ofast,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include "global_board.hpp"
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
    //testing that we're gaining 5 or more elo
    double elo0 = 0;
    double elo1 = 5;
    
    double n = wins + draws + losses;

    double dlo = wdlToElo(wins / n, draws / n, losses / n).second;

    std::array<double, 3> probabilities0 = eloToWDL(elo0, dlo);
    std::array<double, 3> probabilities1 = eloToWDL(elo1, dlo);

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
                Move m = bot1.getMove(board, std::chrono::milliseconds(SPRT_THINK_MS));
                board.makeMove(m);
            }
            else {
                bot2_player = board.n_moves % 2;
                Move best_move = bot2.getMove(board, std::chrono::milliseconds(SPRT_THINK_MS));
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

static void verify_eval_linear() {
    CrossfishDev dev;
    CrossfishPrev prev;
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
            int ev = dev.evaluate(board);
            if (CrossfishDev::EVAL_EXPERIMENT != 3 && ev != stm * val + extra) {
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
            if (CrossfishDev::EVAL_EXPERIMENT == 0 && ev != prev.evaluate(board)) {
                std::cerr << "dev vs prev eval mismatch at ply " << ply
                          << " dev=" << ev << " prev=" << prev.evaluate(board) << std::endl;
                std::exit(1);
            }
            std::vector<Move> moves = board.getLegalMoves();
            if (moves.empty()) break;
            board.makeMove(moves[rng() % moves.size()]);
        }
    }
    if (CrossfishDev::EVAL_EXPERIMENT == 0) {
        std::cout << "eval linear combo: OK (matches Prev)" << std::endl;
    } else {
        std::cout << "eval linear combo: OK (experiment " << CrossfishDev::EVAL_EXPERIMENT << ")" << std::endl;
    }
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
                double g = gscale * p.f[i] + 4e-5 * (w[i] - (double)start_w[i]) / step_scale[i];
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                double mhat = m[i] / (1.0 - std::pow(0.9, tstep));
                double vhat = v[i] / (1.0 - std::pow(0.999, tstep));
                w[i] -= lr * mhat / (std::sqrt(vhat) + 1e-8) * step_scale[i];
                if (w[i] < wmin[i]) w[i] = wmin[i];
                if (w[i] > wmax[i]) w[i] = wmax[i];
            }
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

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "tune") == 0) {
        CrossfishDev::init_mini_lut();
        if (argc >= 3 && std::strcmp(argv[2], "hce") == 0) {
            bool load_saved = argc >= 4 && std::strcmp(argv[3], "load") == 0;
            run_texel(load_saved);
        } else {
            bool load_saved = argc >= 3 && std::strcmp(argv[2], "load") == 0;
            run_lut_texel(load_saved);
        }
        return 0;
    }
    verify_fill_movegen();
    verify_mini_lut();
    verify_eval_linear();
    if (argc >= 2 && std::strcmp(argv[1], "verify") == 0) {
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "lut") == 0) {
        if (!load_mini_scores(MINI_SCORE_PATH)) {
            std::cerr << "failed to load " << MINI_SCORE_PATH << std::endl;
            return 1;
        }
        std::cout << "loaded " << MINI_SCORE_PATH << " into Dev" << std::endl;
    }
    std::cout << "Dev extra experiment: " << CrossfishDev::EVAL_EXPERIMENT
              << " search experiment: " << CrossfishDev::SEARCH_EXPERIMENT << std::endl;
    const unsigned int n_threads = std::thread::hardware_concurrency(); // Get the number of threads supported by the hardware
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
    while(abs(llr) < 3) {
        std::vector<std::future<void>> futures;
        for (unsigned int i = 0; i < n_threads; ++i) {
            futures.push_back(std::async(std::launch::async, play_game));
        }
        for (auto& f : futures) {
            f.get();
        }
        int total_games = global_total[0] + global_total[1] + global_total[2];
        EloResult elo = calc_elo_diff(global_total[0], global_total[2], global_total[1]);
        llr = sprt(global_total[0], global_total[1], global_total[2]);
        std::cout << "N: " << total_games << " W: " << global_total[0]
                << " D: " << global_total[1] << " L: " << global_total[2]
                << " Elo diff: " << elo.elo_diff << " +/- " << elo.ci << " LLR: " << llr << std::endl;
    }
    return 0;
}
