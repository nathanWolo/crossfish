#!/usr/bin/env bash
# Idempotent Cloud Agent setup for crossfish (Ultimate Tic-Tac-Toe engine).
# Runs after the repository is checked out. Safe to run repeatedly.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

# --- Python dependencies -------------------------------------------------
# The repo pins no versions; numpy is the core dependency, with scipy /
# matplotlib / joblib used by the benchmarking (test_bots.py) and
# visualization (vis_tools.py) tooling. pip skips already-satisfied packages.
python3 -m pip install --user numpy scipy matplotlib joblib

# --- C++ engines ---------------------------------------------------------
# Build the CodinGame submissions and the SPRT/tuning/verification harness.
# The sources force AVX2/BMI via #pragma GCC target(...); -march=native keeps
# the rest of the build consistent with the host CPU.
mkdir -p cpp_impl/bin
CXXFLAGS="-O3 -std=c++17 -pthread -march=native"
g++ $CXXFLAGS cpp_impl/crossfish.cpp     -o cpp_impl/bin/crossfish
g++ $CXXFLAGS cpp_impl/test_bots.cpp     -o cpp_impl/bin/test_bots
g++ $CXXFLAGS cpp_impl/cg_legend_hce.cpp -o cpp_impl/bin/cg_legend_hce

echo "crossfish install complete: python deps installed and C++ engines built in cpp_impl/bin/"
