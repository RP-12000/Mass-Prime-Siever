#pragma once

#include "Types.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace siever {

inline uint64_t to_uint64(Int128 value) {
    return static_cast<uint64_t>(value);
}

inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t mod) {
    UInt128 product = UInt128(a) * UInt128(b);
    return static_cast<uint64_t>(product % mod);
}

inline uint64_t powmod_uint64(uint64_t base, UInt128 exponent, uint64_t mod) {
    if (mod == 1) {
        return 0;
    }
    uint64_t result = 1;
    base %= mod;
    while (exponent > 0) {
        if ((exponent & 1) != 0) {
            result = mulmod(result, base, mod);
        }
        base = mulmod(base, base, mod);
        exponent >>= 1;
    }
    return result;
}

inline uint64_t mod_inverse(uint64_t value, uint64_t mod) {
    return powmod_uint64(value % mod, mod - 2, mod);
}

inline Int128 isqrt(Int128 n) {
    if (n <= 0) {
        return 0;
    }
    Int128 lo = 1;
    Int128 hi = n;
    Int128 ans = 1;
    while (lo <= hi) {
        const Int128 mid = (lo + hi) / 2;
        if (mid * mid <= n) {
            ans = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return ans;
}

inline Int128 isqrt_ceil(Int128 n) {
    if (n <= 0) {
        return 0;
    }
    const Int128 root = isqrt(n);
    if (root * root < n) {
        return root + 1;
    }
    return root;
}

inline Int128 ceil_div(Int128 a, Int128 b) {
    if (b <= 0) {
        throw std::invalid_argument("ceil_div requires positive divisor");
    }
    if (a >= 0) {
        return (a + b - 1) / b;
    }
    return a / b;
}

inline Int128 floor_div(Int128 a, Int128 b) {
    if (b <= 0) {
        throw std::invalid_argument("floor_div requires positive divisor");
    }
    if (a >= 0) {
        return a / b;
    }
    return (a - b + 1) / b;
}

inline uint64_t mod128(Int128 value, uint64_t mod) {
    const Int128 reduced = value % Int128(mod);
    if (reduced < 0) {
        return static_cast<uint64_t>(reduced + Int128(mod));
    }
    return static_cast<uint64_t>(reduced);
}

inline uint64_t mod_neg(Int128 value, uint64_t mod) {
    const uint64_t positive = mod128(value, mod);
    return positive == 0 ? 0 : mod - positive;
}

inline bool is_prime64(uint64_t n) {
    if (n < 2) {
        return false;
    }
    static const uint64_t small_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (const uint64_t p : small_primes) {
        if (n == p) {
            return true;
        }
        if (n % p == 0) {
            return false;
        }
    }

    auto check = [&](uint64_t a) {
        uint64_t d = n - 1;
        int s = 0;
        while ((d & 1) == 0) {
            d >>= 1;
            ++s;
        }
        uint64_t x = powmod_uint64(a % n, d, n);
        if (x == 1 || x == n - 1) {
            return true;
        }
        for (int r = 1; r < s; ++r) {
            x = mulmod(x, x, n);
            if (x == n - 1) {
                return true;
            }
        }
        return false;
    };

    return check(2) && check(3) && check(5) && check(7) && check(11);
}

inline uint64_t pollard_rho(uint64_t n, std::mt19937_64& rng) {
    if ((n & 1ULL) == 0) {
        return 2;
    }
    while (true) {
        const uint64_t c = std::uniform_int_distribution<uint64_t>(1, n - 1)(rng);
        uint64_t x = std::uniform_int_distribution<uint64_t>(0, n - 1)(rng);
        uint64_t y = x;
        uint64_t d = 1;
        auto step = [&](uint64_t v) { return (mulmod(v, v, n) + c) % n; };
        while (d == 1) {
            x = step(x);
            y = step(step(y));
            d = std::gcd(x > y ? x - y : y - x, n);
        }
        if (d != n) {
            return d;
        }
    }
}

inline void factor_integer(uint64_t n, std::vector<uint64_t>& prime_factors, std::mt19937_64& rng) {
    std::vector<uint64_t> stack;
    stack.push_back(n);
    while (!stack.empty()) {
        const uint64_t current = stack.back();
        stack.pop_back();
        if (current == 1) {
            continue;
        }
        if (is_prime64(current)) {
            prime_factors.push_back(current);
            continue;
        }
        const uint64_t divisor = pollard_rho(current, rng);
        stack.push_back(divisor);
        stack.push_back(current / divisor);
    }
    std::sort(prime_factors.begin(), prime_factors.end());
}

inline uint64_t multiplicative_order(uint64_t base, uint64_t prime, std::mt19937_64& rng) {
    uint64_t order = prime - 1;
    std::vector<uint64_t> factors;
    factor_integer(prime - 1, factors, rng);

    uint64_t previous = 0;
    for (const uint64_t factor : factors) {
        if (factor == previous) {
            continue;
        }
        previous = factor;
        while (order % factor == 0 && powmod_uint64(base % prime, order / factor, prime) == 1) {
            order /= factor;
        }
    }
    return order;
}

struct Relation {
    uint64_t p = 0;
    uint64_t M = 0;
    uint64_t q = 0;
};

inline bool run_bsgs(uint64_t prime, const WorkItem& item, Relation& out, std::mt19937_64& rng) {
    if (prime < 2) {
        return false;
    }

    const uint64_t b_mod = mod128(item.b, prime);
    const uint64_t neg_c_mod = mod_neg(item.c, prime);

    if (b_mod == 0) {
        if (neg_c_mod == 0) {
            out.p = prime;
            out.M = to_uint64(item.N0);
            out.q = 1;
            return true;
        }
        return false;
    }

    const Int128 m128 = (Int128(prime) > item.N1) ? isqrt_ceil(item.N1) : isqrt_ceil(Int128(prime));
    if (m128 <= 0) {
        return false;
    }
    const uint64_t m = to_uint64(m128);
    if (m == 0) {
        return false;
    }

    std::unordered_map<uint64_t, uint64_t> baby;
    baby.reserve(static_cast<std::size_t>(m * 2));

    uint64_t power = 1;
    for (uint64_t i = 0; i < m; ++i) {
        if (baby.find(power) == baby.end()) {
            baby.emplace(power, i);
        }
        power = mulmod(power, b_mod, prime);
    }

    const uint64_t step = powmod_uint64(b_mod, m, prime);
    const uint64_t step_inverse = mod_inverse(step, prime);

    uint64_t giant = neg_c_mod;
    for (uint64_t j = 0; j < m; ++j) {
        const auto it = baby.find(giant);
        if (it != baby.end()) {
            out.p = prime;
            out.M = it->second + j * m;
            out.q = multiplicative_order(b_mod, prime, rng);
            return true;
        }
        giant = mulmod(giant, step_inverse, prime);
    }

    return false;
}

}  // namespace siever
