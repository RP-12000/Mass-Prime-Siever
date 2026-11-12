#pragma once
// Portable wrappers for compiler-specific intrinsics and C runtime functions.
// MSVC keeps its native intrinsics (_umul128, _udiv128, _addcarry_u64, fopen_s);
// GCC/Clang use unsigned __int128 and standard C equivalents.

#include <cstdint>
#include <cstdio>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace siever {

// 64x64 -> 128 multiply. Returns the low half, stores the high half in *hi.
inline uint64_t mul_64x64_128(uint64_t a, uint64_t b, uint64_t* hi) {
#if defined(_MSC_VER) && !defined(__clang__)
    return _umul128(a, b, hi);
#else
    const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
    *hi = static_cast<uint64_t>(product >> 64);
    return static_cast<uint64_t>(product);
#endif
}

// (hi:lo) mod d. Requires hi < d (as _udiv128 does).
inline uint64_t mod_128_64(uint64_t hi, uint64_t lo, uint64_t d) {
#if defined(_MSC_VER) && !defined(__clang__)
    uint64_t rem = 0;
    _udiv128(hi, lo, d, &rem);
    return rem;
#elif defined(__x86_64__)
    uint64_t quotient = 0;
    uint64_t rem = 0;
    __asm__("divq %4" : "=a"(quotient), "=d"(rem) : "a"(lo), "d"(hi), "rm"(d) : "cc");
    (void)quotient;
    return rem;
#else
    const unsigned __int128 n = (static_cast<unsigned __int128>(hi) << 64) | lo;
    return static_cast<uint64_t>(n % d);
#endif
}

// a + b + carry_in. Stores the sum in *out and returns the carry out.
inline unsigned char add_carry_64(unsigned char carry_in, uint64_t a, uint64_t b, uint64_t* out) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long long sum = 0;
    const unsigned char carry = _addcarry_u64(carry_in, a, b, &sum);
    *out = sum;
    return carry;
#else
    const unsigned __int128 sum = static_cast<unsigned __int128>(a) + b + carry_in;
    *out = static_cast<uint64_t>(sum);
    return static_cast<unsigned char>(sum >> 64);
#endif
}

// fopen that returns nullptr on failure on every platform.
inline FILE* open_file(const char* path, const char* mode) {
#if defined(_MSC_VER)
    FILE* file = nullptr;
    if (fopen_s(&file, path, mode) != 0) {
        return nullptr;
    }
    return file;
#else
    return std::fopen(path, mode);
#endif
}

}  // namespace siever
