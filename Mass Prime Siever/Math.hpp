#pragma once

#include "Types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <intrin.h>

namespace siever {

inline uint64_t to_uint64(Int128 value) {
    return static_cast<uint64_t>(value);
}

// 12.0: hardware 64x64->128 multiply + 128/64 divide (was boost::multiprecision software division).
inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t hi = 0;
    const uint64_t lo = _umul128(a % mod, b % mod, &hi);
    uint64_t rem = 0;
    _udiv128(hi, lo, mod, &rem);
    return rem;
}

// 12.0: Montgomery arithmetic for odd moduli < 2^63 (hot path of BSGS).
struct Montgomery {
    uint64_t n = 0;
    uint64_t ninv = 0;  // -n^{-1} mod 2^64
    uint64_t r2 = 0;    // R^2 mod n
    uint64_t one = 0;   // R mod n

    explicit Montgomery(uint64_t modulus) : n(modulus) {
        uint64_t inv = modulus;  // correct to 3 bits for odd n
        for (int i = 0; i < 5; ++i) {
            inv *= 2 - modulus * inv;
        }
        ninv = 0 - inv;
        one = (0 - modulus) % modulus;
        uint64_t hi = 0;
        const uint64_t lo = _umul128(one, one, &hi);
        _udiv128(hi, lo, modulus, &r2);
    }

    uint64_t redc(uint64_t hi, uint64_t lo) const {
        const uint64_t m = lo * ninv;
        uint64_t mhi = 0;
        const uint64_t mlo = _umul128(m, n, &mhi);
        uint64_t discard = 0;
        unsigned char carry = _addcarry_u64(0, lo, mlo, &discard);
        uint64_t t = 0;
        _addcarry_u64(carry, hi, mhi, &t);  // n < 2^63 => no overflow
        return t >= n ? t - n : t;
    }
    uint64_t mul(uint64_t a, uint64_t b) const {
        uint64_t hi = 0;
        const uint64_t lo = _umul128(a, b, &hi);
        return redc(hi, lo);
    }
    uint64_t to_mont(uint64_t a) const { return mul(a % n, r2); }
    uint64_t from_mont(uint64_t a) const { return redc(0, a); }
    uint64_t pow(uint64_t base_m, uint64_t e) const {
        uint64_t result = one;
        while (e) {
            if (e & 1) result = mul(result, base_m);
            base_m = mul(base_m, base_m);
            e >>= 1;
        }
        return result;
    }
    // Euler criterion: +1 residue, -1 non-residue, 0 if a == 0.
    int legendre(uint64_t a_m) const {
        if (a_m == 0) return 0;
        const uint64_t r = pow(a_m, (n - 1) >> 1);
        return r == one ? 1 : -1;
    }
};

// 12.0: flat open-addressing hash table, reused per thread (replaces per-prime std::unordered_map).
class BabyTable {
public:
    static constexpr uint64_t kEmpty = ~0ULL;
    void reset(uint64_t entries) {
        unsigned bits = 4;
        while ((1ULL << bits) < entries * 2) ++bits;
        bits_ = bits;
        mask_ = (1ULL << bits) - 1;
        if (keys_.size() < (1ULL << bits)) {
            keys_.resize(1ULL << bits);
            vals_.resize(1ULL << bits);
        }
        std::fill(keys_.begin(), keys_.begin() + static_cast<std::ptrdiff_t>(mask_ + 1), kEmpty);
    }
    // inserts if absent; keeps the first (smallest) index
    void insert(uint64_t key, uint32_t value) {
        uint64_t h = slot(key);
        while (keys_[h] != kEmpty) {
            if (keys_[h] == key) return;
            h = (h + 1) & mask_;
        }
        keys_[h] = key;
        vals_[h] = value;
    }
    bool find(uint64_t key, uint32_t& value) const {
        uint64_t h = slot(key);
        while (keys_[h] != kEmpty) {
            if (keys_[h] == key) {
                value = vals_[h];
                return true;
            }
            h = (h + 1) & mask_;
        }
        return false;
    }

private:
    uint64_t slot(uint64_t key) const { return (key * 0x9E3779B97F4A7C15ULL) >> (64 - bits_); }
    std::vector<uint64_t> keys_;
    std::vector<uint32_t> vals_;
    uint64_t mask_ = 0;
    unsigned bits_ = 4;
};

inline uint64_t powmod_uint64(uint64_t base, uint64_t exponent, uint64_t mod) {
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

inline bool run_bsgs_reference(uint64_t prime, const WorkItem& item, Relation& out, std::mt19937_64& rng);

// 11.4 fast path.
//  * Searches only n in [N0, N1) with n == cls_res (mod cls_mod): n = n_start + cls_mod*k,
//    i.e. B^k == -c * b^{-n_start} with B = b^cls_mod.  BSGS cost ~ sqrt((N1-N0)/cls_mod).
//  * Power-residue filter: B is a g-th power with g = gcd(cls_mod, p-1) (g >= 2 whenever cls_mod
//    is even), so the target must be a g-th power residue, otherwise p never divides an in-class
//    candidate.  For cls_mod = 1 the 12.0 QR filter / parity split is applied instead.
//  * Montgomery arithmetic + flat hash table (12.0).
//  Output: p, M = smallest in-class solution n >= N0, q = ord(b).  Every n == M (mod q) is
//  divisible by p and every in-class solution in [N0, N1) is of that form.
inline bool run_bsgs(uint64_t prime, const WorkItem& item, Relation& out, std::mt19937_64& rng) {
    if (prime < 1000 || prime >= (1ULL << 62) || (prime & 1) == 0) {
        return run_bsgs_reference(prime, item, out, rng);
    }
    const uint64_t b_mod = mod128(item.b, prime);
    const uint64_t neg_c_mod = mod_neg(item.c, prime);
    if (b_mod == 0) {
        return run_bsgs_reference(prime, item, out, rng);
    }
    if (neg_c_mod == 0) {
        return false;  // b^n == 0 (mod p) impossible when p does not divide b
    }

    const uint64_t N0 = item.N0 < 0 ? 0 : to_uint64(item.N0);
    const uint64_t N1 = to_uint64(item.N1);
    const uint64_t cm = item.cls_mod == 0 ? 1 : item.cls_mod;
    const uint64_t n_start = N0 + ((item.cls_res % cm) + cm - (N0 % cm)) % cm;
    if (n_start >= N1) {
        return false;
    }
    uint64_t kcount = (N1 - n_start + cm - 1) / cm;  // k in [0, kcount)
    if (kcount > prime) {
        kcount = prime;  // ord(B) <= p - 1
    }

    const Montgomery mt(prime);
    const uint64_t b_m = mt.to_mont(b_mod);
    uint64_t B = mt.pow(b_m, cm);
    uint64_t target = mt.to_mont(neg_c_mod);
    if (n_start != 0) {
        target = mt.mul(target, mt.pow(mt.pow(b_m, n_start), prime - 2));
    }

    uint64_t stride = 1;
    uint64_t parity = 0;
    const uint64_t g = std::gcd(cm, prime - 1);
    if (g >= 2) {
        // B is a g-th power residue => target must be one too.
        if (mt.pow(target, (prime - 1) / g) != mt.one) {
            return false;
        }
    } else {
        const int leg_B = mt.legendre(B);
        const int leg_t = mt.legendre(target);
        if (leg_B == 1) {
            if (leg_t == -1) {
                return false;
            }
        } else {
            stride = 2;
            if (leg_t == -1) {
                parity = 1;  // k odd: (B^2)^j == target * B^{-1}
                target = mt.mul(target, mt.pow(B, prime - 2));
            }
            B = mt.mul(B, B);
        }
    }
    if (kcount <= parity) {
        return false;
    }
    const uint64_t jcount = (kcount - parity + stride - 1) / stride;  // j in [0, jcount)

    uint64_t m = static_cast<uint64_t>(std::sqrt(static_cast<double>(jcount)));
    while (m * m < jcount) ++m;
    if (m == 0) m = 1;

    thread_local BabyTable table;
    table.reset(m);

    uint64_t power = mt.one;
    uint64_t cycle = 0;  // order of B found during baby steps (0 = not found)
    for (uint64_t i = 0; i < m; ++i) {
        if (i > 0 && power == mt.one) {
            cycle = i;
            break;
        }
        table.insert(power, static_cast<uint32_t>(i));
        power = mt.mul(power, B);
    }

    uint64_t j = 0;
    bool found = false;
    uint32_t idx = 0;
    if (cycle != 0) {
        if (table.find(target, idx)) {
            j = idx;
            found = true;
        }
    } else {
        const uint64_t step_inv = mt.pow(power, prime - 2);  // power == B^m
        uint64_t giant = target;
        for (uint64_t gstep = 0; gstep * m < jcount; ++gstep) {
            if (table.find(giant, idx)) {
                j = idx + gstep * m;
                found = true;
                break;
            }
            giant = mt.mul(giant, step_inv);
        }
    }
    if (!found || j >= jcount) {
        return false;
    }

    const uint64_t k = j * stride + parity;
    out.p = prime;
    out.M = n_start + k * cm;
    out.q = multiplicative_order(b_mod, prime, rng);
    return true;
}

inline bool run_bsgs_reference(uint64_t prime, const WorkItem& item, Relation& out, std::mt19937_64& rng) {
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
