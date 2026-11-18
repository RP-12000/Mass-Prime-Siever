#pragma once

#include "Math.hpp"
#include "Types.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace siever {

inline std::vector<uint32_t> sieve_small_primes(uint64_t limit) {
    if (limit < 2) {
        return {};
    }
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = false;
    if (limit >= 1) {
        is_prime[1] = false;
    }

    for (uint64_t i = 2; i * i <= limit; ++i) {
        if (!is_prime[i]) {
            continue;
        }
        for (uint64_t j = i * i; j <= limit; j += i) {
            is_prime[j] = false;
        }
    }

    std::vector<uint32_t> primes;
    for (uint64_t i = 2; i <= limit; ++i) {
        if (is_prime[i]) {
            primes.push_back(static_cast<uint32_t>(i));
        }
    }
    return primes;
}

inline uint64_t sieve_limit_for(const WorkItem& item) {
    const uint64_t limit = to_uint64(isqrt_ceil(item.P1));
    return std::max<uint64_t>(limit, 2);
}

// 13.0: cache-blocked odd-only sieve of [lo0, hi), hi < 2^63, lo0 > 3.  Small primes are marked per
// 128 KB segment from a saved offset; primes larger than a segment go through a bucket sieve, so every
// write lands in the segment that is in L2.  Output is identical to the 12.0 loop (ascending primes).
inline void segmented_sieve_blocked(uint64_t lo0, uint64_t hi, const std::vector<uint32_t>& small_primes,
                                    std::vector<uint64_t>& primes_out) {
    const uint64_t lo = lo0 | 1ULL;
    if (lo >= hi) return;
    const uint64_t count = (hi - lo + 1) / 2;  // odd numbers lo, lo+2, ... < hi
    constexpr uint32_t kSeg = 1u << 17;        // odd numbers per segment (128 KB, fits L2)
    const uint64_t nseg = (count + kSeg - 1) / kSeg;
    if (count >= (1ULL << 32)) return;         // chunks are far smaller; keep 32-bit offsets valid

    struct Hit { uint32_t q; uint32_t pos; };
    thread_local std::vector<uint32_t> sq, snext;
    thread_local std::vector<std::vector<Hit>> buckets;
    thread_local std::vector<uint8_t> seg;
    sq.clear(); snext.clear();
    if (buckets.size() < nseg) buckets.resize(nseg);
    for (uint64_t s = 0; s < nseg; ++s) buckets[s].clear();
    seg.assign(kSeg, 0);

    for (const uint32_t p32 : small_primes) {
        const uint64_t p = p32;
        if (p == 2) continue;
        if (p * p >= hi) break;
        uint64_t start = std::max<uint64_t>(p * p, ((lo + p - 1) / p) * p);
        if ((start & 1) == 0) start += p;
        const uint64_t idx = (start - lo) / 2;
        if (idx >= count) continue;
        if (p < kSeg) {
            sq.push_back(p32);
            snext.push_back(static_cast<uint32_t>(idx));
        } else {
            buckets[idx / kSeg].push_back(Hit{p32, static_cast<uint32_t>(idx)});
        }
    }

    for (uint64_t s = 0; s < nseg; ++s) {
        const uint64_t base = s * kSeg;
        const uint32_t len = static_cast<uint32_t>(std::min<uint64_t>(kSeg, count - base));
        std::memset(seg.data(), 0, len);
        uint8_t* const z = seg.data();
        const uint64_t end = base + len;
        for (std::size_t k = 0; k < sq.size(); ++k) {
            const uint32_t q = sq[k];
            uint64_t v = snext[k];
            for (; v < end; v += q) z[v - base] = 1;
            snext[k] = static_cast<uint32_t>(std::min<uint64_t>(v, 0xFFFFFFFFULL));
        }
        std::vector<Hit>& bk = buckets[s];
        for (const Hit h : bk) {
            z[h.pos - base] = 1;
            const uint64_t nx = uint64_t(h.pos) + h.q;
            if (nx < count) buckets[nx / kSeg].push_back(Hit{h.q, static_cast<uint32_t>(nx)});
        }
        bk.clear();
        const uint64_t vbase = lo + 2 * base;
        for (uint32_t i = 0; i < len; ++i) {
            if (!z[i]) primes_out.push_back(vbase + 2ULL * i);
        }
    }
}


inline void segmented_sieve(UInt128 low, UInt128 high, const std::vector<uint32_t>& small_primes,
                            std::vector<uint64_t>& primes_out) {
    if (low >= high) {
        return;
    }

    // 12.0 fast path: native 64-bit, odd numbers only (the old loop used boost uint128 for every step).
    if (high < (UInt128(1) << 63) && low > 3) {
        const uint64_t lo = static_cast<uint64_t>(low) | 1ULL;  // first odd >= low
        const uint64_t hi = static_cast<uint64_t>(high);
        if (lo >= hi) {
            return;
        }
        if ((hi - lo + 1) / 2 < (1ULL << 32)) {
            segmented_sieve_blocked(lo, hi, small_primes, primes_out);  // 13.0
            return;
        }
        const uint64_t count = (hi - lo + 1) / 2;  // odd numbers lo, lo+2, ... < hi
        std::vector<uint8_t> composite(static_cast<std::size_t>(count), 0);
        for (const uint32_t p32 : small_primes) {
            const uint64_t p = p32;
            if (p == 2) continue;
            if (p * p >= hi) break;
            uint64_t start = std::max<uint64_t>(p * p, ((lo + p - 1) / p) * p);
            if ((start & 1) == 0) start += p;
            for (uint64_t v = (start - lo) / 2; v < count; v += p) {
                composite[static_cast<std::size_t>(v)] = 1;
            }
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (!composite[static_cast<std::size_t>(i)]) {
                primes_out.push_back(lo + 2 * i);
            }
        }
        return;
    }

    const std::size_t segment_size = static_cast<std::size_t>(high - low);
    std::vector<uint8_t> composite(segment_size, 0);

    if (low <= 1 && high > 1) {
        composite[static_cast<std::size_t>(1 - low)] = 1;
    }

    for (const uint32_t prime : small_primes) {
        const UInt128 prime128 = prime;
        if (prime128 * prime128 >= high) {
            break;
        }

        UInt128 start = low;
        if (start <= prime128) {
            start = prime128 * 2;
        } else {
            const UInt128 rem = start % prime128;
            if (rem != 0) {
                start += prime128 - rem;
            }
        }

        for (UInt128 value = start; value < high; value += prime128) {
            composite[static_cast<std::size_t>(value - low)] = 1;
        }
    }

    for (std::size_t offset = 0; offset < segment_size; ++offset) {
        if (composite[offset]) {
            continue;
        }
        const UInt128 candidate = low + offset;
        if (candidate < 2) {
            continue;
        }
        primes_out.push_back(static_cast<uint64_t>(candidate));
    }
}

}  // namespace siever
