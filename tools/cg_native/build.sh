#!/usr/bin/env bash
# Build the native CodinGame submission (documentation/native_build.md):
#   1. clang -O3 -march=haswell, thin LTO, of codingame_nnue.cpp plus the
#      cg_selfcheck driver (tools/cg_native/native_main.cpp), linked
#      dynamically against libstdc++;
#   2. a check that the binary needs nothing newer than CodinGame's runtime
#      (glibc 2.36, GLIBCXX_3.4.30, CXXABI_1.3.13) and only the four libraries
#      it has;
#   3. tools/cg_native/pack.py: xz + U15 into the Python 3 launcher
#      cpp_impl/cg_input_native.py, and the build record
#      tools/cg_native/manifest.json.
#
#   make cg-native                                     (from the repository root)
#   CF_CLANG=/opt/llvm-23.1.2/bin/clang++ make cg-native
#
# Linux x86-64 only: the binary is a Linux ELF executable, and Windows (MSVC,
# MinGW) cannot produce it. Then run `make cg-native-check`.
#
# Environment:
#   CF_CLANG            clang++ to use (default: clang++ on PATH). The shipped
#                       binary was built with LLVM 23.1.2 (official release tarball).
#   CF_GCC_INSTALL_DIR  the GCC whose libstdc++ headers clang compiles against
#                       (default /usr/lib/gcc/x86_64-linux-gnu/11: g++ 11, the same
#                       headers as CodinGame's g++ 11.2). libc++ is not an option:
#                       its std::uniform_int_distribution draws other Zobrist
#                       keys from the same seed, so the search would differ.
#   CF_OBJDUMP          objdump for the symbol-version check (default: objdump,
#                       else llvm-objdump next to CF_CLANG)
#   CF_RUN              prefix for the compiler, e.g. "taskset -c 0-3 nice -n 10"
#   CF_NATIVE_OUT       submission to write (default cpp_impl/cg_input_native.py)
#   PYTHON              python 3 (default python3)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/cpp_impl
B=$CPP/bin/native
CLANG=${CF_CLANG:-clang++}
GCC_DIR=${CF_GCC_INSTALL_DIR:-/usr/lib/gcc/x86_64-linux-gnu/11}
RUN=${CF_RUN:-}
OUT=${CF_NATIVE_OUT:-$CPP/cg_input_native.py}
PY=${PYTHON:-python3}
die() { echo "cg-native: $*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] ||
  die "needs a Linux x86-64 host (this is $(uname -s) $(uname -m)); see documentation/native_build.md"
command -v "$CLANG" >/dev/null || die "no clang++ at '$CLANG'; set CF_CLANG (documentation/native_build.md, 'Toolchain')"
[ -d "$GCC_DIR" ] || die "no GCC install at $GCC_DIR; install g++-11 or set CF_GCC_INSTALL_DIR"
OBJDUMP=${CF_OBJDUMP:-}
if [ -z "$OBJDUMP" ]; then
  if command -v objdump >/dev/null; then OBJDUMP=objdump
  else OBJDUMP=$(dirname "$(command -v "$CLANG")")/llvm-objdump; fi
fi
command -v "$OBJDUMP" >/dev/null || die "no objdump; install binutils or set CF_OBJDUMP"

TOOLCHAIN=$("$CLANG" --version 2>/dev/null | head -n 1)
GCC_MAJOR=$(basename "$GCC_DIR")
LIBSTDCXX="g++ $(g++-"$GCC_MAJOR" -dumpfullversion 2>/dev/null || echo "$GCC_MAJOR") headers ($GCC_DIR)"
echo "toolchain: $TOOLCHAIN"
echo "libstdc++: $LIBSTDCXX"

# -ffp-contract=off is required: clang contracts a*b+c into FMA by default, which
# changes the NNUE's float bake (the d5/d7/d9 fingerprint catches it). -fno-pie
# matches the -no-pie link. The -Wno flags only silence GCC-specific pragmas.
COMMON="-std=gnu++17 -O3 -march=haswell -mtune=haswell -ffp-contract=off -pthread -fno-pie \
-fno-plt -fno-semantic-interposition -ffunction-sections -fdata-sections -flto=thin \
-Wno-unknown-pragmas -Wno-ignored-attributes -Wno-ignored-pragmas -Wno-gcc-compat"
LINK="-fuse-ld=lld -no-pie -Wl,--gc-sections -Wl,-O2 -Wl,--as-needed -Wl,--hash-style=gnu -Wl,--icf=all -s"

# The translation unit: native_main.cpp includes sc_body.cpp, which is
# cg_selfcheck.cpp with its main() renamed (and that file includes
# codingame_nnue.cpp with the bot's main() renamed).
mkdir -p "$B/src"
sed 's/^int main(int argc, char \*\*argv) {/static int selfcheck_main(int argc, char **argv) {/' \
    "$CPP/cg_selfcheck.cpp" > "$B/src/sc_body.cpp"
grep -q 'static int selfcheck_main' "$B/src/sc_body.cpp" || die "could not rename cg_selfcheck.cpp's main()"
cp "$ROOT/tools/cg_native/native_main.cpp" "$B/src/native_main.cpp"

echo "compiling (thin LTO)"
t0=$(date +%s)
(cd "$B" && $RUN "$CLANG" --gcc-install-dir="$GCC_DIR" $COMMON -I"$CPP" \
   -o native.bin src/native_main.cpp $LINK)
echo "compiled in $(( $(date +%s) - t0 )) s: $(stat -c %s "$B/native.bin") B, sha256 $(sha256sum < "$B/native.bin" | cut -c1-64)"

# CodinGame's runtime (measured 2026-10-05): Python 3.11.5, glibc 2.36,
# libstdc++.so.6.0.30 (GLIBCXX_3.4.30, CXXABI_1.3.13).
vers=$("$OBJDUMP" -T "$B/native.bin" | grep -o 'GLIBCXX_[0-9.]*\|GLIBC_[0-9.]*\|CXXABI_[0-9.]*' | sort -uV)
vmax() { echo "$vers" | grep "^$1_[0-9]" | sed "s/^$1_//" | sort -V | tail -n 1; }
vle() { [ -z "$1" ] || [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$1" ]; }
GLIBC_MAX=$(vmax GLIBC); GLIBCXX_MAX=$(vmax GLIBCXX); CXXABI_MAX=$(vmax CXXABI)
echo "needs GLIBC_$GLIBC_MAX GLIBCXX_$GLIBCXX_MAX CXXABI_$CXXABI_MAX"
vle "$GLIBC_MAX" 2.36 || die "needs GLIBC_$GLIBC_MAX; CodinGame has 2.36. Build on an older distribution (Ubuntu 22.04 / Debian 12)."
vle "$GLIBCXX_MAX" 3.4.30 || die "needs GLIBCXX_$GLIBCXX_MAX; CodinGame has 3.4.30. Use g++ 11 or 12 headers (CF_GCC_INSTALL_DIR)."
vle "$CXXABI_MAX" 1.3.13 || die "needs CXXABI_$CXXABI_MAX; CodinGame has 1.3.13."
NEEDED=$("$OBJDUMP" -p "$B/native.bin" | awk '$1 == "NEEDED" {print $2}' | sort | tr '\n' ' ')
echo "NEEDED: $NEEDED"
for lib in $NEEDED; do
  case $lib in libstdc++.so.6|libm.so.6|libgcc_s.so.1|libc.so.6) ;;
    *) die "links $lib, which the launcher cannot rely on at CodinGame" ;;
  esac
done

SOURCES="$CPP/codingame_nnue.cpp $CPP/d16_helpers.hpp $CPP/macro_eval.hpp $CPP/nnue_b64.hpp \
$CPP/nnue_b64_net.hpp $CPP/play_book.hpp $CPP/play_book_data.hpp $CPP/cg_selfcheck.cpp \
$ROOT/tools/cg_native/native_main.cpp"
src_args=(); for s in $SOURCES; do src_args+=(--source "$s"); done
"$PY" "$ROOT/tools/cg_native/pack.py" "$B/native.bin" -o "$OUT" --root "$ROOT" \
  --manifest "$ROOT/tools/cg_native/manifest.json" "${src_args[@]}" \
  --note "toolchain=$TOOLCHAIN" --note "libstdcxx=$LIBSTDCXX" \
  --note "compile_flags=$COMMON" --note "link_flags=$LINK" \
  --note "symbol_versions=GLIBC_$GLIBC_MAX GLIBCXX_$GLIBCXX_MAX CXXABI_$CXXABI_MAX" \
  --note "needed=${NEEDED% }"
echo "next: make cg-native-check"
