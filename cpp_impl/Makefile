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
# Opening book (documentation/play_book.md). Both the text book play_book.txt
# and its packed payload play_book_data.hpp (payload format 2) are committed.
# Since 2026-10-09 the text book is empty: no book (PLAY_BOOK_ENTRIES 0).
PLAY_BOOK_TXT ?= $(SRC_DIR)/play_book.txt
PLAY_BOOK_FIRST ?= 5
PLAY_BOOK_SECOND ?= 4
PLAY_BOOK_SEARCH_MS ?= 2000
# The evaluation: the NNUE runtime and its generated payload
# (tools/nnue_emit_b64_header.py), included by crossfish_dev.hpp and by the
# CodinGame bot, which also includes the helpers it keeps from mini_eval_d16.hpp.
NNUE_HDRS := nnue_b64.hpp nnue_b64_net.hpp
CG_HDRS := d16_helpers.hpp macro_eval.hpp $(NNUE_HDRS) play_book.hpp play_book_data.hpp

CG_CXX ?= $(shell command -v g++-11 >/dev/null 2>&1 && echo g++-11 || echo $(CXX))
# CodinGame's own C++ command line: g++ 11.2, no -O flag. Only the source's
# pragmas and always_inline attributes make the submission fast there.
CG_FLAGS := -std=gnu++17 -Werror=return-type -g -pthread

.PHONY: all test bench cg-gate cg-min-check port-check test-cpp nnue-parity cg-flags cg-speed test-python sprt verify cg compile-cg compile-legend cg-input roundrobin opening-book book-inspect play-book-gen play-book play-book-text play-book-check play-book-match play-book-protocol datagen cg-native cg-native-check clean

all: test

$(BIN):
	mkdir -p $(BIN)

$(BIN)/unit_tests: unit_tests.cpp global_board.hpp crossfish_dev.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) play_book.hpp play_book_data.hpp play_book_text.hpp play_book_pack.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ unit_tests.cpp

$(BIN)/test_bots: test_bots.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp nnue.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ test_bots.cpp $(FILESYSTEM_LIB)

$(BIN)/crossfish: crossfish.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ crossfish.cpp

$(BIN)/codingame_nnue: codingame_nnue.cpp $(CG_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ codingame_nnue.cpp

$(BIN)/cg_input: cg_input.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_input.cpp

$(BIN)/cg_input_cgflags: cg_input.cpp | $(BIN)
	$(CG_CXX) $(CG_FLAGS) -o $@ cg_input.cpp -lm -lpthread -ldl -lcrypt

$(BIN)/cg_selfcheck: cg_selfcheck.cpp codingame_nnue.cpp $(CG_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_selfcheck.cpp

$(BIN)/cg_selfcheck_cgflags: cg_selfcheck.cpp codingame_nnue.cpp $(CG_HDRS) | $(BIN)
	$(CG_CXX) $(CG_FLAGS) -o $@ cg_selfcheck.cpp

$(BIN)/cg_legend: cg_legend_hce.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cg_legend_hce.cpp

$(BIN)/bench_ab: bench_ab.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ bench_ab.cpp

$(BIN)/engine_selfcheck: engine_selfcheck.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp mini_eval.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ engine_selfcheck.cpp

# After porting a Dev change into codingame_nnue.cpp: the CG search must match
# Dev exactly (fixed-depth scores and node counts over cg_selfcheck's
# positions). This holds for tree-changing ports too, unlike comparing
# cg_selfcheck against itself before and after.
port-check: $(BIN)/cg_selfcheck $(BIN)/engine_selfcheck
	@for d in 5 7 9; do \
	  cg=$$($(BIN)/cg_selfcheck 120 $$d | head -n 1); \
	  dev=$$($(BIN)/engine_selfcheck dev 120 $$d); \
	  if [ "$$cg" = "$$dev" ]; then echo "port-check depth $$d: IDENTICAL ($$cg)"; \
	  else echo "port-check depth $$d: MISMATCH"; echo "  cg:  $$cg"; echo "  dev: $$dev"; exit 1; fi; \
	done

# Eval training data (documentation/eval_data.md).
$(BIN)/datagen: datagen.cpp test_bots.cpp global_board.hpp crossfish_dev.hpp crossfish_prev.hpp nnue.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ datagen.cpp $(FILESYSTEM_LIB)

datagen: $(BIN)/datagen

$(BIN)/play_book_gen: play_book_gen.cpp play_book_text.hpp global_board.hpp crossfish_dev.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_gen.cpp

$(BIN)/play_book_pack: play_book_pack.cpp play_book_pack.hpp play_book.hpp play_book_text.hpp global_board.hpp crossfish_dev.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_pack.cpp

$(BIN)/play_book_check: play_book_check.cpp play_book.hpp play_book_data.hpp play_book_text.hpp crossfish_dev.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_check.cpp

$(BIN)/play_book_match: play_book_match.cpp play_book.hpp play_book_data.hpp crossfish_dev.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_match.cpp

test-cpp: $(BIN)/unit_tests
	$(BIN)/unit_tests

# Static-eval parity (nnue_parity.cpp): every record's integer eval, from scratch or replayed through the
# stack, to compare with tools/nnue_emit_b64_header.py --int-eval.
$(BIN)/nnue_parity: nnue_parity.cpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ nnue_parity.cpp

nnue-parity: $(BIN)/nnue_parity

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

# Same tree at -O3 and with CodinGame's flags: the checksum lines must match,
# and the seconds ratio is how much speed CodinGame's compiler still loses.
# Minifier identity gate: minify cg_selfcheck.cpp itself (it bundles the CG
# file, so every rename and macro applies to the whole program) and compare
# its fixed-depth checksums with the readable build's.
# Both compilers: -O3, and CodinGame's flags (no -O), where the AVX intrinsics
# are function-like macros and the pragma does the optimizing.
cg-min-check: $(BIN)/cg_selfcheck $(BIN)/cg_selfcheck_cgflags
	cd $(ROOT) && $(PYTHON) tools/cg_minify.py cpp_impl/cg_selfcheck.cpp -o cpp_impl/bin/cg_selfcheck_min.cpp --inline-local
	$(CXX) $(CXXFLAGS) -o $(BIN)/cg_selfcheck_min $(BIN)/cg_selfcheck_min.cpp
	$(CG_CXX) $(CG_FLAGS) -o $(BIN)/cg_selfcheck_min_cgflags $(BIN)/cg_selfcheck_min.cpp -lm -lpthread -ldl -lcrypt
	@for d in 5 7 9; do \
	  b=$$($(BIN)/cg_selfcheck 120 $$d | grep -v seconds | sed 's/ book_ms=.*//'); \
	  for v in cg_selfcheck_min cg_selfcheck_min_cgflags; do \
	    a=$$($(BIN)/$$v 120 $$d | grep -v seconds | sed 's/ book_ms=.*//'); \
	    if [ "$$a" = "$$b" ]; then echo "cg-min-check $$v depth $$d: IDENTICAL"; \
	    else echo "cg-min-check $$v depth $$d: DIFFERENT"; echo "$$a"; echo "$$b"; exit 1; fi; \
	  done; \
	done

cg-speed: $(BIN)/cg_selfcheck $(BIN)/cg_selfcheck_cgflags
	$(BIN)/cg_selfcheck 30 13
	$(BIN)/cg_selfcheck_cgflags 30 13
compile-legend: $(BIN)/cg_legend

roundrobin: compile-cg compile-legend
	cd $(ROOT) && python3 roundrobin.py --games 10000 --ms 20 --out /tmp/roundrobin.txt

opening-book: $(BIN)/test_bots
	$(BIN)/test_bots book generate $(SRC_DIR)/opening_book.bin 50000

book-inspect: $(BIN)/test_bots
	$(BIN)/test_bots book inspect $(SRC_DIR)/opening_book.bin

test: test-cpp test-python compile-cg compile-legend book-inspect

# Opening book (documentation/play_book.md). No book ships since 2026-10-09:
# play_book.txt is empty, and play-book packs it into the no-book payload
# (PLAY_BOOK_ENTRIES 0, fingerprint 0; pb_init decodes nothing), which
# play-book-check, play-book-protocol and cg-native-check accept. The last
# book (the s5 book: the first player's half from the uttt.ai fork, the second
# player's from the deep-search P2 book builder, up to 3 moves per position) is
# in git history: `git show 4919ed7:cpp_impl/play_book.txt > play_book.txt`,
# then `make play-book` re-packs it under the current net. play-book-gen
# writes a full-coverage book with crossfish's own search (an older method,
# ~1.7 h on 7 threads). play-book packs a non-empty text book into
# play_book_data.hpp (payload format 2) and checks the result against it
# (every stored move, and the runtime's uniform choice between a position's moves).
play-book-gen: $(BIN)/play_book_gen
	$(BIN)/play_book_gen $(PLAY_BOOK_TXT) $(PLAY_BOOK_FIRST) $(PLAY_BOOK_SECOND) $(PLAY_BOOK_SEARCH_MS)

$(BIN)/play_book_text_dump: play_book_text_dump.cpp play_book.hpp play_book_data.hpp play_book_text.hpp global_board.hpp crossfish_dev.hpp mini_eval_d16.hpp macro_eval.hpp $(NNUE_HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ play_book_text_dump.cpp

# The shipped payload back as a text book (one line per stored move, a
# position's primary first), for re-packing after a coder change; packing it
# again gives the same payload.
play-book-text: $(BIN)/play_book_text_dump
	$(BIN)/play_book_text_dump > $(PLAY_BOOK_TXT).tmp && mv $(PLAY_BOOK_TXT).tmp $(PLAY_BOOK_TXT) || { rm -f $(PLAY_BOOK_TXT).tmp; exit 1; }

play-book: $(BIN)/play_book_pack
	$(BIN)/play_book_pack $(PLAY_BOOK_TXT) $(SRC_DIR)/play_book_data.hpp
	$(MAKE) play-book-check

play-book-check: $(BIN)/play_book_check
	$(BIN)/play_book_check $(PLAY_BOOK_TXT)

play-book-match: $(BIN)/play_book_match
	$(BIN)/play_book_match 3000 0

play-book-protocol: $(BIN)/cg_input
	cd $(ROOT) && $(PYTHON) tools/play_book_protocol_check.py $(BIN)/cg_input 40 24 --book $(PLAY_BOOK_TXT)

verify: $(BIN)/test_bots
	$(BIN)/test_bots verify

sprt: $(BIN)/test_bots
	$(BIN)/test_bots

cg: compile-cg

cg-input:
	cd $(ROOT) && $(PYTHON) tools/cg_minify.py cpp_impl/codingame_nnue.cpp -o cpp_impl/cg_input.cpp --inline-local

# The native submission (documentation/native_build.md): clang -O3 build of
# codingame_nnue.cpp + the cg_selfcheck driver, packed into the Python 3
# launcher cg_input_native.py (the live CodinGame submission). Linux x86-64
# only. CF_CLANG names the clang++ (the shipped binary: LLVM 23.1.2).
cg-native:
	cd $(ROOT) && PYTHON=$(PYTHON) bash tools/cg_native/build.sh

# Its checks: static (tools/test_cg_native.py), the d5/d7/d9 fingerprint through
# the launcher against the readable build, and CodinGame-protocol games with the
# exact book check. CF_PROTOCOL_GAMES (default 40), CF_RUN (command prefix).
cg-native-check: $(BIN)/cg_selfcheck $(BIN)/play_book_text_dump
	cd $(ROOT) && PYTHON=$(PYTHON) bash tools/cg_native/verify.sh

# CodinGame performance gate (tools/cg_perf_gate.py) in a g++ 11.2 Linux
# container, the compiler CodinGame uses. Compares against BASE (default
# origin/main). Runs on every pull request in CI (cg-perf-gate workflow).
CG_GATE_BASE ?= origin/main
cg-gate:
	cd $(ROOT) && docker build -t crossfish-cg-gate tools/cg_gate
	cd $(ROOT) && docker run --rm -v "$(ROOT):/repo" crossfish-cg-gate --base $(CG_GATE_BASE)

clean:
	rm -rf $(BIN)
