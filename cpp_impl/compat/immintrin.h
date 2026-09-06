#pragma once
// arm64 build shim: the engine sources target x86 AVX2. On non-x86 hosts this
// directory is put ahead of the system includes so <immintrin.h> resolves here
// and simde maps the intrinsics onto NEON. x86 builds never see this file.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include_next <immintrin.h>
#else
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/x86/avx2.h>
#endif
