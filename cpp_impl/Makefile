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

.PHONY: all test bench test-cpp test-python sprt verify cg compile-cg compile-legend cg-input roundrobin opening-book book-inspect clean

all: test

$(BIN):
	mkdir -p $(BIN)

$(BIN)/unit_tests: unit_tests.cpp global_board.hpp crossfish_dev.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ unit_tests.cpp

$(BIN)/test_bots: test_bots.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp nnue.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ test_bots.cpp $(FILESYSTEM_LIB)

$(BIN)/crossfish: crossfish.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ crossfish.cpp

$(BIN)/codingame_nnue: codingame_nnue.cpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ codingame_nnue.cpp

$(BIN)/cg_input: cg_input.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_input.cpp

$(BIN)/cg_legend: cg_legend_hce.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_legend_hce.cpp

$(BIN)/bench_ab: bench_ab.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ bench_ab.cpp

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

compile-cg: $(BIN)/crossfish $(BIN)/codingame_nnue $(BIN)/cg_input
compile-legend: $(BIN)/cg_legend

roundrobin: compile-cg compile-legend
	cd $(ROOT) && python3 roundrobin.py --games 10000 --ms 20 --out /tmp/roundrobin.txt

opening-book: $(BIN)/test_bots
	$(BIN)/test_bots book generate $(SRC_DIR)/opening_book.bin 50000

book-inspect: $(BIN)/test_bots
	$(BIN)/test_bots book inspect $(SRC_DIR)/opening_book.bin

test: test-cpp test-python compile-cg compile-legend book-inspect

verify: $(BIN)/test_bots
	$(BIN)/test_bots verify

sprt: $(BIN)/test_bots
	$(BIN)/test_bots

cg: compile-cg

cg-input:
	cd $(ROOT) && $(PYTHON) tools/cg_minify.py cpp_impl/codingame_nnue.cpp -o cpp_impl/cg_input.cpp --inline-local

clean:
	rm -rf $(BIN)
