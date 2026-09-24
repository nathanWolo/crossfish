CXX ?= g++
UNAME_M := $(shell uname -m)
ifeq ($(filter x86_64 i386 i686 amd64,$(UNAME_M)),)
# Non-x86 host: compat/immintrin.h maps the AVX2 intrinsics onto NEON via simde.
ARCH_FLAGS ?= -Icompat -I$(shell brew --prefix simde 2>/dev/null)/include
else
ARCH_FLAGS ?= -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt
endif
CXXFLAGS ?= -O3 -std=c++17 $(ARCH_FLAGS) -pthread
CXXFLAGS += -Wno-unknown-pragmas -Wno-ignored-attributes
ifeq ($(OS),Windows_NT)
# The engine object (~1.2 MB) is a stack local in the unit tests and in the
# CodinGame main(). Windows' default 1 MB stack overflows silently; Linux (and
# CodinGame) default to 8 MB.
CXXFLAGS += -Wl,--stack,16777216
endif
FILESYSTEM_LIB := $(shell major=`$(CXX) -dumpversion | cut -d. -f1`; test "$$major" -lt 9 && printf '%s' '-lstdc++fs')
SRC_DIR := $(abspath .)
ROOT := $(abspath ..)
PYTHON ?= $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3)
BIN := $(SRC_DIR)/bin
# Opening book (documentation/play_book.md). The text book is a build artifact;
# only the packed play_book_data.hpp is committed.
PLAY_BOOK_TXT ?= $(BIN)/play_book.txt
PLAY_BOOK_FIRST ?= 5
PLAY_BOOK_SECOND ?= 4
PLAY_BOOK_SEARCH_MS ?= 2000

CG_CXX ?= $(shell command -v g++-11 >/dev/null 2>&1 && echo g++-11 || echo $(CXX))
# CodinGame's own C++ command line: g++ 11.2, no -O flag. Only the source's
# pragmas and always_inline attributes make the submission fast there.
CG_FLAGS := -std=gnu++17 -Werror=return-type -g -pthread

.PHONY: all test bench test-cpp cg-flags test-python sprt verify cg compile-cg compile-legend cg-input roundrobin opening-book book-inspect play-book-gen play-book play-book-check play-book-match play-book-protocol clean

all: test

$(BIN):
	mkdir -p $(BIN)

$(BIN)/unit_tests: unit_tests.cpp global_board.hpp crossfish_dev.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp play_book.hpp play_book_data.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ unit_tests.cpp

$(BIN)/test_bots: test_bots.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp nnue.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ test_bots.cpp $(FILESYSTEM_LIB)

$(BIN)/crossfish: crossfish.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ crossfish.cpp

$(BIN)/codingame_nnue: codingame_nnue.cpp mini_eval_d16.hpp macro_eval.hpp play_book.hpp play_book_data.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ codingame_nnue.cpp

$(BIN)/cg_input: cg_input.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_input.cpp

$(BIN)/cg_input_cgflags: cg_input.cpp | $(BIN)
	$(CG_CXX) $(CG_FLAGS) -o $@ cg_input.cpp -lm -lpthread -ldl -lcrypt

$(BIN)/cg_legend: cg_legend_hce.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_legend_hce.cpp

$(BIN)/bench_ab: bench_ab.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ bench_ab.cpp

$(BIN)/play_book_gen: play_book_gen.cpp play_book_text.hpp global_board.hpp crossfish_dev.hpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_gen.cpp

$(BIN)/play_book_pack: play_book_pack.cpp play_book.hpp play_book_text.hpp global_board.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_pack.cpp

$(BIN)/play_book_check: play_book_check.cpp play_book.hpp play_book_data.hpp play_book_text.hpp mini_eval_d16.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_check.cpp

$(BIN)/play_book_match: play_book_match.cpp play_book.hpp play_book_data.hpp crossfish_dev.hpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_match.cpp

test-cpp: $(BIN)/unit_tests
	$(BIN)/unit_tests

# Deterministic Dev-vs-Prev screen. `equiv` proves a speed-only candidate
# computes the same tree (identical scores AND node counts); `nodes` reports
# node counts and wall time. Unlike `test_bots depth N` this builds a fresh
# engine per position, so results do not depend on carried-over TT state.
bench: $(BIN)/bench_ab
	$(BIN)/bench_ab equiv 80 8 777
	$(BIN)/bench_ab nodes 120 8 4242

test-python:
	cd $(ROOT) && $(PYTHON) -m unittest python_impl.test_rules -v
	cd $(ROOT) && $(PYTHON) -m unittest discover -s tools -p 'test_*.py' -v

compile-cg: $(BIN)/crossfish $(BIN)/codingame_nnue $(BIN)/cg_input $(BIN)/cg_input_cgflags
cg-flags: $(BIN)/cg_input_cgflags
compile-legend: $(BIN)/cg_legend

roundrobin: compile-cg compile-legend
	cd $(ROOT) && python3 roundrobin.py --games 10000 --ms 20 --out /tmp/roundrobin.txt

opening-book: $(BIN)/test_bots
	$(BIN)/test_bots book generate $(SRC_DIR)/opening_book.bin 50000

book-inspect: $(BIN)/test_bots
	$(BIN)/test_bots book inspect $(SRC_DIR)/opening_book.bin

test: test-cpp test-python compile-cg compile-legend book-inspect

# Opening book. play-book-gen takes ~1.7 h on 7 threads; play-book packs the
# text book into play_book_data.hpp and checks the result against it.
play-book-gen: $(BIN)/play_book_gen
	$(BIN)/play_book_gen $(PLAY_BOOK_TXT) $(PLAY_BOOK_FIRST) $(PLAY_BOOK_SECOND) $(PLAY_BOOK_SEARCH_MS)

play-book: $(BIN)/play_book_pack
	$(BIN)/play_book_pack $(PLAY_BOOK_TXT) $(SRC_DIR)/play_book_data.hpp $(PLAY_BOOK_FIRST) $(PLAY_BOOK_SECOND)
	$(MAKE) play-book-check

play-book-check: $(BIN)/play_book_check
	$(BIN)/play_book_check $(PLAY_BOOK_TXT)

play-book-match: $(BIN)/play_book_match
	$(BIN)/play_book_match 3000 0

play-book-protocol: $(BIN)/cg_input
	cd $(ROOT) && $(PYTHON) tools/play_book_protocol_check.py $(BIN)/cg_input 40

verify: $(BIN)/test_bots
	$(BIN)/test_bots verify

sprt: $(BIN)/test_bots
	$(BIN)/test_bots

cg: compile-cg

cg-input:
	cd $(ROOT) && $(PYTHON) tools/cg_minify.py cpp_impl/codingame_nnue.cpp -o cpp_impl/cg_input.cpp --inline-local

clean:
	rm -rf $(BIN)
