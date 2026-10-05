// Native CodinGame bot: the shipped codingame_nnue.cpp bot plus the repo's
// cg_selfcheck identity driver, in one binary (documentation/native_build.md).
//
//   <bin>                       CodinGame protocol (the shipped main())
//   <bin> match                 the repo's NEW/APPLY/GO match protocol
//   <bin> selfcheck <pos> <d>   cg_selfcheck: fixed-depth fingerprint + book check
//
// sc_body.cpp is cg_selfcheck.cpp with its main() renamed selfcheck_main
// (tools/cg_native/build.sh generates it); cg_selfcheck.cpp includes
// codingame_nnue.cpp with that file's main() renamed cg_shipped_main_unused.
#include <cstring>
#include "sc_body.cpp"

int main(int argc, char **argv) {
    if (argc >= 2 && std::strcmp(argv[1], "selfcheck") == 0)
        return selfcheck_main(argc - 1, argv + 1);
    return cg_shipped_main_unused(argc, argv);
}
