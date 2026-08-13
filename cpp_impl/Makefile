CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread
CXXFLAGS += -Wno-unknown-pragmas
SRC_DIR := $(abspath .)
ROOT := $(abspath ..)
BIN := $(SRC_DIR)/bin

.PHONY: all test test-cpp test-python sprt verify cg compile-cg compile-legend clean

all: test

$(BIN):
	mkdir -p $(BIN)

$(BIN)/unit_tests: unit_tests.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ unit_tests.cpp

$(BIN)/test_bots: test_bots.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ test_bots.cpp

$(BIN)/crossfish: crossfish.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ crossfish.cpp

$(BIN)/cg_legend: cg_legend_hce.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_legend_hce.cpp

test-cpp: $(BIN)/unit_tests
	$(BIN)/unit_tests

test-python:
	cd $(ROOT) && python3 -m unittest python_impl.test_rules -v

compile-cg: $(BIN)/crossfish
compile-legend: $(BIN)/cg_legend

test: test-cpp test-python compile-cg compile-legend

verify: $(BIN)/test_bots
	$(BIN)/test_bots verify

sprt: $(BIN)/test_bots
	$(BIN)/test_bots

cg: compile-cg

clean:
	rm -rf $(BIN)
