#pragma once
#include <cstdint>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <iostream>

/* ============================================================
   64¡Á64 ¡ú 128 ³Ë·¨£¨´¿ 32-bit ²ð·Ö£¬ÎÞ±àÒëÆ÷À©Õ¹£©
   ============================================================ */
static inline void mul64(uint64_t a, uint64_t b,
    uint64_t& hi, uint64_t& lo)
{
    uint64_t a0 = (uint32_t)a;
    uint64_t a1 = a >> 32;
    uint64_t b0 = (uint32_t)b;
    uint64_t b1 = b >> 32;

    uint64_t p0 = a0 * b0;
    uint64_t p1 = a0 * b1;
    uint64_t p2 = a1 * b0;
    uint64_t p3 = a1 * b1;

    uint64_t mid = (p0 >> 32) + (uint32_t)p1 + (uint32_t)p2;
    lo = (p0 & 0xffffffffULL) | (mid << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}

/* ============================================================
   clz64£¨MSVC / GCC / fallback£©
   ============================================================ */
static inline int clz64(uint64_t x) {
#if defined(_MSC_VER)
    unsigned long idx;
    if (_BitScanReverse64(&idx, x))
        return 63 - (int)idx;
    return 64;
#elif defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clzll(x) : 64;
#else
    int n = 0;
    while (x && (x & (1ULL << 63)) == 0) { x <<= 1; ++n; }
    return x ? n : 64;
#endif
}

/* ============================================================
   Int128
   ============================================================ */
class Int128 {
public:
    int64_t  high;
    uint64_t low;

    /* ---------- constructors ---------- */

    Int128() : high(0), low(0) {}
    Int128(long long v) : high(v < 0 ? -1 : 0), low((uint64_t)v) {}
    explicit Int128(unsigned long long v) : high(0), low(v) {}
    Int128(int64_t h, uint64_t l) : high(h), low(l) {}

    /* ---------- helpers ---------- */

    bool isZero() const { return high == 0 && low == 0; }
    bool isNegative() const { return high < 0; }

    static Int128 abs(const Int128& v) {
        return v.isNegative() ? -v : v;
    }

    /* ---------- unary ---------- */

    Int128 operator-() const {
        Int128 r(~high, ~low + 1);
        if (r.low == 0) r.high++;
        return r;
    }

    /* ---------- comparison ---------- */

    friend bool operator==(const Int128& a, const Int128& b) {
        return a.high == b.high && a.low == b.low;
    }
    friend bool operator!=(const Int128& a, const Int128& b) {
        return !(a == b);
    }
    friend bool operator<(const Int128& a, const Int128& b) {
        return (a.high != b.high) ? (a.high < b.high) : (a.low < b.low);
    }
    friend bool operator>(const Int128& a, const Int128& b) { return b < a; }
    friend bool operator<=(const Int128& a, const Int128& b) { return !(b < a); }
    friend bool operator>=(const Int128& a, const Int128& b) { return !(a < b); }

    /* ---------- addition / subtraction ---------- */

    friend Int128 operator+(const Int128& a, const Int128& b) {
        Int128 r;
        r.low = a.low + b.low;
        r.high = a.high + b.high + (r.low < a.low);
        return r;
    }

    friend Int128 operator-(const Int128& a, const Int128& b) {
        return a + (-b);
    }

    /* ---------- multiplication ---------- */

    friend Int128 operator*(const Int128& x, const Int128& y) {
        bool neg = x.isNegative() ^ y.isNegative();
        Int128 a = abs(x), b = abs(y);

        uint64_t hi0, lo0;
        mul64(a.low, b.low, hi0, lo0);

        uint64_t hi1, lo1;
        mul64(a.low, (uint64_t)b.high, hi1, lo1);

        uint64_t hi2, lo2;
        mul64((uint64_t)a.high, b.low, hi2, lo2);

        Int128 r;
        r.low = lo0;
        r.high = (int64_t)(hi0 + lo1 + lo2);

        return neg ? -r : r;
    }

    /* ---------- division (Knuth Algorithm D, base 2^64) ---------- */

    friend Int128 operator/(const Int128& x, const Int128& y) {
        if (y.isZero())
            throw std::runtime_error("Int128: division by zero");

        bool neg = x.isNegative() ^ y.isNegative();
        Int128 u = abs(x);
        Int128 v = abs(y);

        /* single-limb divisor */
        if (v.high == 0) {
            uint64_t rem;
            Int128 q = div64(u, v.low, rem);
            return neg ? -q : q;
        }

        /* normalize */
        int s = clz64((uint64_t)v.high);
        uint64_t v1 = ((uint64_t)v.high << s) | (v.low >> (64 - s));
        uint64_t v0 = v.low << s;

        uint64_t u2 = (uint64_t)u.high >> (64 - s);
        uint64_t u1 = ((uint64_t)u.high << s) | (u.low >> (64 - s));
        uint64_t u0 = u.low << s;

        /* quotient estimate */
        uint64_t qhat = u2 / v1;
        uint64_t rhat = u2 % v1;

        /* correction */
        while (true) {
            uint64_t hi, lo;
            mul64(qhat, v0, hi, lo);
            if (hi < rhat || (hi == rhat && lo <= u1))
                break;
            --qhat;
            rhat += v1;
            if (rhat < v1) break;
        }

        Int128 q((int64_t)qhat, 0);
        return neg ? -q : q;
    }

    friend Int128 operator%(const Int128& a, const Int128& b) {
        return a - (a / b) * b;
    }

    /* ---------- power ---------- */

    Int128 pow(uint64_t e) const {
        Int128 base = *this;
        Int128 r = 1;
        while (e) {
            if (e & 1) r = r * base;
            base = base * base;
            e >>= 1;
        }
        return r;
    }

    /* ---------- checked arithmetic ---------- */

    static bool add_checked(const Int128& a, const Int128& b, Int128& out) {
        out = a + b;
        if ((a.high > 0 && b.high > 0 && out.high < 0) ||
            (a.high < 0 && b.high < 0 && out.high >= 0))
            return false;
        return true;
    }

    static bool sub_checked(const Int128& a, const Int128& b, Int128& out) {
        return add_checked(a, -b, out);
    }

    static bool mul_checked(const Int128& a, const Int128& b, Int128& out) {
        out = a * b;
        if (!a.isZero() && out / a != b) return false;
        return true;
    }

    static bool pow_checked(Int128 base, uint64_t exp, Int128& out) {
        out = 1;
        while (exp) {
            if (exp & 1) {
                Int128 tmp;
                if (!mul_checked(out, base, tmp)) return false;
                out = tmp;
            }
            exp >>= 1;
            if (exp) {
                Int128 tmp;
                if (!mul_checked(base, base, tmp)) return false;
                base = tmp;
            }
        }
        return true;
    }

    /* ---------- decimal I/O ---------- */

    static Int128 fromString(const std::string& s) {
        Int128 r = 0;
        bool neg = false;
        size_t i = 0;
        if (s[0] == '-') { neg = true; i = 1; }
        for (; i < s.size(); ++i)
            r = r * 10 + (s[i] - '0');
        return neg ? -r : r;
    }

    std::string toString() const {
        if (isZero()) return "0";
        Int128 v = abs(*this);
        std::string s;
        while (!v.isZero()) {
            uint64_t r;
            v = div64(v, 10, r);
            s.push_back(char('0' + r));
        }
        if (isNegative()) s.push_back('-');
        std::reverse(s.begin(), s.end());
        return s;
    }

    friend std::ostream& operator<<(std::ostream& os, const Int128& v) {
        return os << v.toString();
    }

    friend std::istream& operator>>(std::istream& is, Int128& v) {
        std::string s;
        is >> s;
        v = fromString(s);
        return is;
    }

private:
    static Int128 div64(const Int128& a, uint64_t d, uint64_t& rem) {
        uint64_t qh = (uint64_t)a.high / d;
        uint64_t r = (uint64_t)a.high % d;

        uint64_t hi, lo;
        mul64(r, std::numeric_limits<uint64_t>::max(), hi, lo);
        uint64_t n = lo + a.low;

        uint64_t ql = n / d;
        rem = n % d;

        return Int128((int64_t)qh, ql);
    }
};

/* ============================================================
   User-defined literal
   ============================================================ */
inline Int128 operator"" _i128(unsigned long long v) {
    return Int128(v);
}