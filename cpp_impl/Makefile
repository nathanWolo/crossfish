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
FILESYSTEM_LIB := $(shell major=`$(CXX) -dumpversion | cut -d. -f1`; test "$$major" -lt 9 && printf '%s' '-lstdc++fs')
SRC_DIR := $(abspath .)
ROOT := $(abspath ..)
PYTHON ?= $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3)
BIN := $(SRC_DIR)/bin

.PHONY: all test test-cpp test-python sprt verify cg compile-cg compile-legend cg-input roundrobin clean

all: test

$(BIN):
	mkdir -p $(BIN)

$(BIN)/unit_tests: unit_tests.cpp global_board.hpp crossfish_dev.hpp mini_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ unit_tests.cpp

$(BIN)/test_bots: test_bots.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp nnue.hpp mini_eval.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ test_bots.cpp $(FILESYSTEM_LIB)

$(BIN)/crossfish: crossfish.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ crossfish.cpp

$(BIN)/codingame_nnue: codingame_nnue.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ codingame_nnue.cpp

$(BIN)/cg_input: cg_input.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_input.cpp

$(BIN)/cg_legend: cg_legend_hce.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_legend_hce.cpp

test-cpp: $(BIN)/unit_tests
	$(BIN)/unit_tests

test-python:
	cd $(ROOT) && $(PYTHON) -m unittest python_impl.test_rules -v
	cd $(ROOT) && $(PYTHON) -m unittest discover -s tools -p 'test_*.py' -v

compile-cg: $(BIN)/crossfish $(BIN)/codingame_nnue $(BIN)/cg_input
compile-legend: $(BIN)/cg_legend

roundrobin: compile-cg compile-legend
	cd $(ROOT) && python3 roundrobin.py --games 10000 --ms 20 --workers $$(nproc) --out /tmp/roundrobin.txt

test: test-cpp test-python compile-cg compile-legend

verify: $(BIN)/test_bots
	$(BIN)/test_bots verify

sprt: $(BIN)/test_bots
	$(BIN)/test_bots

cg: compile-cg

cg-input:
	cd $(ROOT) && $(PYTHON) tools/cg_minify.py cpp_impl/codingame_nnue.cpp -o cpp_impl/cg_input.cpp

clean:
	rm -rf $(BIN)
