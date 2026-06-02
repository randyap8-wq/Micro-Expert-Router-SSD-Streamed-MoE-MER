#pragma once
/*
 * glibc 2.41 <-> CUDA host-compiler compatibility shim.
 *
 * glibc 2.41's <bits/mathcalls.h> adds the C23 functions
 *   sinpi / cospi / sinpif / cospif (and friends)
 * gated on __GLIBC_USE (IEC_60559_FUNCS_EXT_C23). CUDA's
 * crt/math_functions.h already declares those same symbols with a
 * different exception specification, so when nvcc's host pass sees both
 * it errors with:
 *
 *   exception specification is incompatible with that of previous
 *   function "cospi"
 *
 * __GLIBC_USE(opt) expands to (__GLIBC_USE_##opt), i.e. it reads the
 * object-like macro __GLIBC_USE_IEC_60559_FUNCS_EXT_C23. That macro is
 * (re)defined by <features.h> every time the header is processed, so a
 * plain -D on the command line gets overwritten and cannot suppress it.
 *
 * The trick: <features.h> has an include guard (_FEATURES_H). If we
 * include it ourselves *first* and then force the option macro to 0,
 * every later #include <features.h> (pulled in transitively by math.h /
 * the CUDA headers) is a no-op and our override survives. With the
 * option forced off glibc never declares the C23 *pi functions, leaving
 * CUDA's declarations as the only ones -> no conflict.
 *
 * This is force-included ahead of every translation unit (see the
 * vendored candle-kernels build.rs, which passes `-include <this file>`
 * to nvcc). It is a strict no-op on:
 *   - glibc < 2.41 (guarded by __GLIBC_PREREQ(2, 41)),
 *   - macOS / Windows / musl (where __GLIBC__ is not defined).
 * It touches no system headers and needs no CUDA upgrade.
 */

#if defined(__GLIBC__) || defined(__gnu_linux__) || defined(__linux__)
#  include <features.h>
#endif

#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#  if __GLIBC_PREREQ(2, 41)
#    undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#    define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#  endif
#endif
