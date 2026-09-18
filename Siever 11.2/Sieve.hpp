#pragma once

#include "Math.hpp"
#include "Types.hpp"

#include <algorithm>
#include <cstdint>
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

inline void segmented_sieve(UInt128 low, UInt128 high, const std::vector<uint32_t>& small_primes,
                            std::vector<uint64_t>& primes_out) {
    if (low >= high) {
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
