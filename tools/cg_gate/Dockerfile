# CodinGame's C++ toolchain for tools/cg_perf_gate.py: Linux, g++ 11.2.
# CodinGame compiles with `g++ -std=gnu++17 -Werror=return-type -g -pthread`
# (no -O), so the submission's speed depends on this exact compiler honouring
# its `#pragma GCC optimize/target` lines and always_inline attributes.
#
# Build and run from the repository root:
#   docker build -t crossfish-cg-gate tools/cg_gate
#   docker run --rm -v "$PWD:/repo" crossfish-cg-gate --base origin/main
FROM gcc:11.2

# gcc:11.2 is Debian bullseye, past the end of its support: the security
# suite now 404s on packages it has superseded. This throwaway build image
# only needs Python and numpy, which the main suite still serves.
RUN sed -i '/security.debian.org/d' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-numpy \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

WORKDIR /repo
ENTRYPOINT ["python3", "tools/cg_perf_gate.py", "--cxx", "g++"]
