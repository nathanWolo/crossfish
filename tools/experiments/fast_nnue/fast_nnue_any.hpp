#pragma once
// Dispatch between the fast NNUE formats (nnue2 stages 2-3): this side's net file (FASTNNUE_PATH, or
// the FASTNNUE_NET_FILE compiled in: fast_nnue.hpp net_path()) is opened once, its magic picks the
// implementation, and every engine's AnyStack forwards to that implementation's lazy accumulator stack.
// A two-engine build gives Prev a renamed copy of this header (fast_pair.py), whose compile-time kind
// macro is FASTNNUE_PREV_KIND.
//
//   FNN1  per-cell net (fast_nnue.hpp, stage 1), A=256, 2A -> 32 -> 1
//   BGN1  pattern-generator net (fast_nnue_b.hpp), instantiated for A in {64, 128} with a
//         2A -> 16 -> 32 -> 1 or a 2A -> 32 -> 1 head (the shapes in datasets/nnue2/probe)
//
// The engine patch (make_cand_b.py) declares `fnnue::AnyStack fnnue_stack;` and calls
//   fnnue_stack.on_make(board, mb, square, stm, decided_state, markers_before, u.tt_hash)
//                                                               in make_move_fast(FastBoard&)
//   fnnue_stack.refresh_root(board)                             where a search starts
//   fnnue_stack.evaluate_keyed(board, constraint, tt_hash)      stand-pat / static eval
//   fnnue::evaluate_any(board, constraint)                      evaluate(GlobalBoard&)
//
// Runtime dispatch (default): one predictable switch per call, for experiments that take any net.
// Compile-time kind (stage 3): #define FASTNNUE_KIND KIND_CELL / KIND_B64 / KIND_B128 / KIND_B64S /
// KIND_B128S before including this header (make_cand_b.py --kind) and AnyStack is that kind's stack
// with no switch; the net file must then be of that kind (checked once at start-up).
#include "fast_nnue.hpp"
#include "fast_nnue_b.hpp"

#include <optional>

namespace fnnue {

enum Kind { KIND_NONE = 0, KIND_CELL, KIND_B64, KIND_B128, KIND_B64S, KIND_B128S };
using B64 = bnn::Net<64, 16, 32>;
using B128 = bnn::Net<128, 16, 32>;
using B64S = bnn::Net<64, 32, 0>;
using B128S = bnn::Net<128, 32, 0>;

inline int g_kind = KIND_NONE;
inline std::once_flag g_kind_once;

inline int any_kind() {
    std::call_once(g_kind_once, [] {
        const char *path = net_path();  // this side's file, announced once (fast_nnue.hpp)
        FILE *f = std::fopen(path, "rb");
        char magic[4] = {0, 0, 0, 0};
        const bool got = f && std::fread(magic, 1, 4, f) == 4;
        if (f) std::fclose(f);
        if (!got) { std::fprintf(stderr, "fast_nnue: cannot read %s\n", path); std::exit(1); }
        if (std::memcmp(magic, "FNN1", 4) == 0) {
            net();  // stage 1 loader (exits on failure)
            g_kind = KIND_CELL;
            return;
        }
        bnn::Header h{};
        if (std::memcmp(magic, "BGN1", 4) != 0 || !bnn::read_header(path, h)) {
            std::fprintf(stderr, "fast_nnue: %s: unknown magic %.4s (supported: FNN1, BGN1)\n", path, magic);
            std::exit(1);
        }
        bool ok = false;
        if (h.A == 64 && h.L1 == 16 && h.L2 == 32) { ok = bnn::load(bnn::g_bnet<B64>, path); g_kind = KIND_B64; }
        else if (h.A == 128 && h.L1 == 16 && h.L2 == 32) { ok = bnn::load(bnn::g_bnet<B128>, path); g_kind = KIND_B128; }
        else if (h.A == 64 && h.L1 == 32 && h.L2 == 0) { ok = bnn::load(bnn::g_bnet<B64S>, path); g_kind = KIND_B64S; }
        else if (h.A == 128 && h.L1 == 32 && h.L2 == 0) { ok = bnn::load(bnn::g_bnet<B128S>, path); g_kind = KIND_B128S; }
        else std::fprintf(stderr, "fast_nnue: %s: BGN1 shape A=%d L1=%d L2=%d is not instantiated\n", path, h.A, h.L1, h.L2);
        if (!ok) { std::fprintf(stderr, "fast_nnue: cannot load %s\n", path); std::exit(1); }
    });
    return g_kind;
}

// Kind -> net type (KIND_CELL is the stage-1 Stack, whose net is the global g_net).
template <int K> struct KindNet;
template <> struct KindNet<KIND_B64> { using type = B64; };
template <> struct KindNet<KIND_B128> { using type = B128; };
template <> struct KindNet<KIND_B64S> { using type = B64S; };
template <> struct KindNet<KIND_B128S> { using type = B128S; };

// From scratch for a known kind.
template <int K, typename Board>
inline int evaluate_kind(const Board &b, int constraint) {
    if constexpr (K == KIND_CELL) return evaluate_board(b, constraint);
    else return bnn::evaluate_board(bnn::g_bnet<typename KindNet<K>::type>, b, constraint);
}

#if defined(FASTNNUE_KIND)

// The one stack of the compile-time kind.
template <int K>
struct KindStack : bnn::Stack<typename KindNet<K>::type> {
    KindStack() : bnn::Stack<typename KindNet<K>::type>(bnn::g_bnet<typename KindNet<K>::type>) {}
};
template <>
struct KindStack<KIND_CELL> : Stack {};

inline int checked_kind() {
    const int k = any_kind();  // loads the net
    if (k != FASTNNUE_KIND) {
        std::fprintf(stderr, "fast_nnue [%s]: %s is net kind %d but this build is compiled for kind %d\n", kSide,
                     net_path(), k, (int)FASTNNUE_KIND);
        std::exit(1);
    }
    return k;
}

struct AnyStack {
    int kind = checked_kind();  // declared before s: the net must be loaded first
    KindStack<FASTNNUE_KIND> s;

    template <typename Board>
    void refresh_root(const Board &b) { s.refresh_root(b); }

    template <typename Board>
    __attribute__((always_inline)) void on_make(const Board &b, int mb, int sq, int stm, int decided, int before_stm,
                                                uint64_t parent_key) {
        s.on_make(b, mb, sq, stm, decided, before_stm, b.mini_boards[mb].markers[stm ^ 1], parent_key);
    }

    template <typename Board>
    int evaluate_keyed(const Board &b, int constraint, uint64_t key) { return s.evaluate_keyed(b, constraint, key); }

    template <typename Board>
    int evaluate(const Board &b, int constraint) { return s.evaluate(b, constraint); }
};

template <typename Board>
inline int evaluate_any(const Board &b, int constraint) {
    (void)checked_kind();
    return evaluate_kind<FASTNNUE_KIND>(b, constraint);
}

#else  // runtime dispatch

#define FNNUE_ANY(CALL)                              \
    switch (kind) {                                  \
    case KIND_CELL: return cell->CALL;               \
    case KIND_B64: return b64->CALL;                 \
    case KIND_B128: return b128->CALL;               \
    case KIND_B64S: return b64s->CALL;               \
    default: return b128s->CALL;                     \
    }

struct AnyStack {
    int kind;
    // In place (no pointer chase on the per-node make hook); only the loaded kind is constructed.
    std::optional<Stack> cell;
    std::optional<bnn::Stack<B64>> b64;
    std::optional<bnn::Stack<B128>> b128;
    std::optional<bnn::Stack<B64S>> b64s;
    std::optional<bnn::Stack<B128S>> b128s;

    AnyStack() : kind(any_kind()) {
        switch (kind) {
        case KIND_CELL: cell.emplace(); break;
        case KIND_B64: b64.emplace(bnn::g_bnet<B64>); break;
        case KIND_B128: b128.emplace(bnn::g_bnet<B128>); break;
        case KIND_B64S: b64s.emplace(bnn::g_bnet<B64S>); break;
        default: b128s.emplace(bnn::g_bnet<B128S>); break;
        }
    }

    template <typename Board>
    void refresh_root(const Board &b) { FNNUE_ANY(refresh_root(b)) }

    // In make_move_fast(FastBoard&) just before board.n_moves++: the move's miniboard mb, square,
    // mover, decided state (-1, or the mini_board_states index it set), the mover's markers on mb
    // before the move and the parent's tt_hash (MoveUndo.tt_hash); the board is otherwise already
    // the child's.
    template <typename Board>
    void on_make(const Board &b, int mb, int sq, int stm, int decided, int before_stm, uint64_t parent_key) {
        const int other = b.mini_boards[mb].markers[stm ^ 1];
        switch (kind) {
        case KIND_CELL: cell->on_make(b, mb, sq, stm, decided, before_stm, other, parent_key); return;
        case KIND_B64: b64->on_make(b, mb, sq, stm, decided, before_stm, other, parent_key); return;
        case KIND_B128: b128->on_make(b, mb, sq, stm, decided, before_stm, other, parent_key); return;
        case KIND_B64S: b64s->on_make(b, mb, sq, stm, decided, before_stm, other, parent_key); return;
        default: b128s->on_make(b, mb, sq, stm, decided, before_stm, other, parent_key); return;
        }
    }

    template <typename Board>
    int evaluate_keyed(const Board &b, int constraint, uint64_t key) { FNNUE_ANY(evaluate_keyed(b, constraint, key)) }

    template <typename Board>
    int evaluate(const Board &b, int constraint) { FNNUE_ANY(evaluate(b, constraint)) }
};

#undef FNNUE_ANY

// From scratch (datagen label's evaluate(GlobalBoard&)).
template <typename Board>
inline int evaluate_any(const Board &b, int constraint) {
    switch (any_kind()) {
    case KIND_CELL: return evaluate_kind<KIND_CELL>(b, constraint);
    case KIND_B64: return evaluate_kind<KIND_B64>(b, constraint);
    case KIND_B128: return evaluate_kind<KIND_B128>(b, constraint);
    case KIND_B64S: return evaluate_kind<KIND_B64S>(b, constraint);
    default: return evaluate_kind<KIND_B128S>(b, constraint);
    }
}

#endif  // FASTNNUE_KIND

}  // namespace fnnue
