#!/usr/bin/env bash
# Checks for the native CodinGame submission (documentation/native_build.md).
# Run through `make cg-native-check`, which first builds bin/cg_selfcheck (the
# readable C++ build) and bin/play_book_text_dump. Linux x86-64 only.
#   1. static: tools/test_cg_native.py (Python 3 parses it, size under the cap,
#      no surrogates, payload decodes to the manifest's binary, sources unchanged
#      since the build);
#   2. identity: the launcher's `selfcheck 120 d` must equal bin/cg_selfcheck's
#      at depths 5, 7 and 9 (nodes, checksum, book=ok and the book table checksum),
#      through Python, the U15 decode, xz and memfd exec;
#   3. protocol: CF_PROTOCOL_GAMES (default 40) real CodinGame-protocol games
#      through the launcher, with the exact book check.
# Environment: CF_RUN (command prefix, e.g. "taskset -c 0-3 nice -n 10"),
# CF_NATIVE_OUT (default cpp_impl/cg_input_native.py), PYTHON.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/cpp_impl
BIN=$CPP/bin
B=$BIN/native
F=${CF_NATIVE_OUT:-$CPP/cg_input_native.py}
RUN=${CF_RUN:-}
PY=${PYTHON:-python3}
mkdir -p "$B"
fail=0

echo "== 1. static checks ($F)"
if out=$(cd "$ROOT" && "$PY" -m unittest tools.test_cg_native 2>&1); then echo "$out" | tail -n 3
else echo "$out"; fail=1; fi

echo "== 2. identity: launcher selfcheck vs bin/cg_selfcheck (the readable C++ build)"
norm() { grep -v seconds | sed 's/ book_ms=.*//'; }
for d in 5 7 9; do
  a=$($RUN "$PY" "$F" selfcheck 120 $d | norm)
  b=$($RUN "$BIN/cg_selfcheck" 120 $d | norm)
  if [ "$a" = "$b" ] && echo "$a" | grep -q 'book=ok'; then echo "depth $d: IDENTICAL ($a)"
  else echo "depth $d: DIFFERENT"; echo "  launcher:     $a"; echo "  cg_selfcheck: $b"; fail=1; fi
done

echo "== 3. protocol: ${CF_PROTOCOL_GAMES:-40} CodinGame-protocol games through the launcher, exact book check"
"$BIN/play_book_text_dump" > "$B/play_book.txt"
printf '#!/bin/sh\nexec %s %s "$@"\n' "$PY" "$F" > "$B/run_launcher.sh"
chmod +x "$B/run_launcher.sh"
(cd "$ROOT" && $RUN "$PY" tools/play_book_protocol_check.py "$B/run_launcher.sh" "${CF_PROTOCOL_GAMES:-40}" 24 \
   --book "$B/play_book.txt" | tail -n 3) || fail=1

if [ $fail = 0 ]; then echo "cg-native-check: OK"; else echo "cg-native-check: FAILED"; exit 1; fi
