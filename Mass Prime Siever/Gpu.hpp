#pragma once
// 11.4 GPU pre-filter (OpenCL).
// The GPU runs the same BSGS as run_bsgs() for a batch of primes, but with 32-bit fingerprints in
// the baby-step table.  It only reports "p may divide an in-class candidate"; every flagged prime
// is then re-checked exactly on the CPU with run_bsgs(), so fingerprint collisions can never
// produce a wrong relation, and a real solution is never missed (a real match always matches its
// fingerprint).

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "Cuda.hpp"
#include "Logger.hpp"
#include "Math.hpp"
#include "Types.hpp"

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace siever {

// Kernel source shared by the OpenCL and CUDA (NVRTC) backends. Each backend prepends its own
// prelude that maps the macros below (KERNEL, GLOBAL, LOCAL, BARRIER, ...) to its dialect.
static const char* kOpenClPrelude = R"CLC(
typedef ulong u64;
typedef uint u32;
typedef long i64;
#define KERNEL __kernel
#define GLOBAL __global
#define LOCAL __local
#define DEVICE_FN inline
#define DEVICE_BIG_FN
#define DYN_LOCAL_PARAM(name) , __local u32* name
#define DYN_LOCAL_DECL(name)
#define ATOMIC_INC(ptr) atomic_inc(ptr)
#define ATOMIC_CAS(ptr, cmp, val) atomic_cmpxchg(ptr, cmp, val)
#define BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#define GLOBAL_ID() get_global_id(0)
#define LOCAL_ID() get_local_id(0)
#define GROUP_ID() get_group_id(0)
#define MUL_HI(a, b) mul_hi(a, b)
#define POPCNT(x) popcount(x)
#define ATOMIC_OR(ptr, v) atomic_or(ptr, v)
#define ATOMIC_ADD(ptr, v) atomic_add(ptr, v)
)CLC";

static const char* kCudaPrelude = R"CLC(
typedef unsigned long long u64;
typedef unsigned int u32;
typedef long long i64;
typedef unsigned char uchar;
typedef unsigned int uint;
#define KERNEL extern "C" __global__
#define GLOBAL
#define LOCAL __shared__
#define DEVICE_FN __device__ __forceinline__
#define DEVICE_BIG_FN __device__
#define DYN_LOCAL_PARAM(name)
#define DYN_LOCAL_DECL(name) extern __shared__ u32 name[];
#define ATOMIC_INC(ptr) atomicAdd((unsigned int*)(ptr), 1u)
#define ATOMIC_CAS(ptr, cmp, val) atomicCAS(ptr, cmp, val)
#define BARRIER() __syncthreads()
#define GLOBAL_ID() (blockIdx.x * blockDim.x + threadIdx.x)
#define LOCAL_ID() threadIdx.x
#define GROUP_ID() blockIdx.x
#define MUL_HI(a, b) __umul64hi(a, b)
#define POPCNT(x) __popc(x)
#define ATOMIC_OR(ptr, v) atomicOr((unsigned int*)(ptr), v)
#define ATOMIC_ADD(ptr, v) atomicAdd((unsigned int*)(ptr), v)
)CLC";

static const char* kGpuKernelBody = R"CLC(

DEVICE_FN u64 mmul(u64 a, u64 b, u64 n, u64 ninv) {
    u64 lo = a * b;
    u64 hi = MUL_HI(a, b);
    u64 m = lo * ninv;
    u64 mlo = m * n;
    u64 mhi = MUL_HI(m, n);
    u64 s = lo + mlo;
    u64 t = hi + mhi + (s < lo ? 1UL : 0UL);
    return t >= n ? t - n : t;
}

DEVICE_FN u64 mpow(u64 base, u64 e, u64 one, u64 n, u64 ninv) {
    u64 r = one;
    while (e) {
        if (e & 1UL) r = mmul(r, base, n, ninv);
        base = mmul(base, base, n, ninv);
        e >>= 1;
    }
    return r;
}

DEVICE_FN u64 gcd64(u64 a, u64 b) {
    while (b) { u64 t = a % b; a = b; b = t; }
    return a;
}

DEVICE_FN u32 fp32(u64 x) {
    u32 f = (u32)(x ^ (x >> 29));
    return f == 0xFFFFFFFFu ? 0xFFFFFFFEu : f;
}

// flags: 0 = no in-class solution, 1 = maybe (verify on CPU), 2 = handle on CPU
KERNEL void bsgs(GLOBAL const u64* primes, GLOBAL uchar* flags, GLOBAL u32* tables,
                   const u64 b, const i64 c, const u64 N0, const u64 N1,
                   const u64 cm, const u64 cres, const u32 tbits, const u32 count) {
    const u32 gid = GLOBAL_ID();
    if (gid >= count) return;
    const u64 p = primes[gid];
    if (p < 1000UL || (p & 1UL) == 0) { flags[gid] = 2; return; }

    const u64 b_mod = b % p;
    u64 t;
    if (c >= 0) { u64 cm_ = ((u64)c) % p; t = cm_ == 0 ? 0 : p - cm_; }
    else { t = ((u64)(-c)) % p; }
    if (b_mod == 0) { flags[gid] = 2; return; }
    if (t == 0) { flags[gid] = 0; return; }

    const u64 n_start = N0 + ((cres % cm) + cm - (N0 % cm)) % cm;
    if (n_start >= N1) { flags[gid] = 0; return; }
    u64 kcount = (N1 - n_start + cm - 1) / cm;
    if (kcount > p) kcount = p;

    // Montgomery setup
    u64 inv = p;
    for (int i = 0; i < 5; ++i) inv *= 2UL - p * inv;
    const u64 ninv = 0UL - inv;
    const u64 one = (0UL - p) % p;
    u64 r2 = one;                         // R^2 mod p by 64 modular doublings
    for (int i = 0; i < 64; ++i) { r2 = (r2 >= p - r2) ? r2 - (p - r2) : r2 + r2; }

    const u64 b_m = mmul(b_mod, r2, p, ninv);
    u64 B = mpow(b_m, cm, one, p, ninv);
    u64 target = mmul(t, r2, p, ninv);
    if (n_start != 0) {
        u64 bs = mpow(b_m, n_start, one, p, ninv);
        target = mmul(target, mpow(bs, p - 2UL, one, p, ninv), p, ninv);
    }

    u64 stride = 1, parity = 0;
    const u64 g = gcd64(cm, p - 1UL);
    if (g >= 2UL) {
        if (mpow(target, (p - 1UL) / g, one, p, ninv) != one) { flags[gid] = 0; return; }
    } else {
        const u64 e = (p - 1UL) >> 1;
        const int lB = mpow(B, e, one, p, ninv) == one;
        const int lt = mpow(target, e, one, p, ninv) == one;
        if (lB) {
            if (!lt) { flags[gid] = 0; return; }
        } else {
            stride = 2;
            if (!lt) { parity = 1; target = mmul(target, mpow(B, p - 2UL, one, p, ninv), p, ninv); }
            B = mmul(B, B, p, ninv);
        }
    }
    if (kcount <= parity) { flags[gid] = 0; return; }
    const u64 jcount = (kcount - parity + stride - 1) / stride;

    u64 m = (u64)sqrt((double)jcount);
    while (m * m < jcount) ++m;
    if (m == 0) m = 1;
    const u32 tsize = 1u << tbits;
    const u32 mask = tsize - 1u;
    // the host sized the table for the largest m of the batch
    GLOBAL u32* tab = tables + (u64)gid * tsize;
    for (u32 i = 0; i < tsize; ++i) tab[i] = 0xFFFFFFFFu;

    u64 power = one;
    int cycle = 0;
    for (u64 i = 0; i < m; ++i) {
        if (i > 0 && power == one) { cycle = 1; break; }
        u32 f = fp32(power);
        u32 h = (u32)((power * 0x9E3779B97F4A7C15UL) >> (64 - tbits));
        while (tab[h] != 0xFFFFFFFFu && tab[h] != f) h = (h + 1u) & mask;
        tab[h] = f;
        power = mmul(power, B, p, ninv);
    }

    uchar result = 0;
    if (cycle) {
        u32 f = fp32(target);
        u32 h = (u32)((target * 0x9E3779B97F4A7C15UL) >> (64 - tbits));
        while (tab[h] != 0xFFFFFFFFu) { if (tab[h] == f) { result = 1; break; } h = (h + 1u) & mask; }
    } else {
        const u64 step_inv = mpow(power, p - 2UL, one, p, ninv);
        u64 giant = target;
        for (u64 gs = 0; gs * m < jcount && !result; ++gs) {
            u32 f = fp32(giant);
            u32 h = (u32)((giant * 0x9E3779B97F4A7C15UL) >> (64 - tbits));
            while (tab[h] != 0xFFFFFFFFu) { if (tab[h] == f) { result = 1; break; } h = (h + 1u) & mask; }
            giant = mmul(giant, step_inv, p, ninv);
        }
    }
    flags[gid] = result;
}
// ---------------- 13.0: prep (filter + setup) then one workgroup per prime ----------------
// The equation is G0 * B^j == t (mod p), j in [0, jcount), with t = -c and G0 = b^n_start (times B for
// the odd-k parity class).  Writing j = g*m - i (0 <= i < m, 0 <= g <= G) gives G0 * (B^m)^g == t * B^i,
// so neither the setup nor the BSGS needs a modular inverse.  Everything shared by the lanes of a
// workgroup (B^m, B^WG, (B^m)^WG) is computed here, one thread per prime, so bsgs_wg has no
// single-lane sections.
#ifndef WG
#define WG 64
#endif
#define ENT 11
// entry layout (ENT x u64): idx, p, ninv, one, B, t, G0, jcount, Bm, B^WG, Bm^WG

// Jacobi symbol (a/n) for odd n, binary algorithm (no division).
DEVICE_FN int jacobi64(u64 a, u64 n) {
    int s = 1;
    while (a != 0UL) {
        const int r8 = ((n & 7UL) == 3UL || (n & 7UL) == 5UL);
        while ((a & 1UL) == 0UL) { a >>= 1; if (r8) s = -s; }
        if (a < n) {
            const u64 x = a; a = n; n = x;
            if ((a & 3UL) == 3UL && (n & 3UL) == 3UL) s = -s;
        }
        a -= n;
    }
    return n == 1UL ? s : 0;
}

// One prime (or, with need_prime, one sieve survivor that may be composite): residue filters, then
// the setup for bsgs_wg.  flags[gid]: 0 = no in-class solution (or not prime), 2 = handle on the CPU,
// 1 is set later by bsgs_wg.  An entry that does not fit (slot >= ent_cap) is also left to the CPU.
DEVICE_BIG_FN void prep_core(const u64 p, const u32 gid, const int need_prime, GLOBAL uchar* flags,
                         GLOBAL u64* ent, GLOBAL volatile uint* ent_count, const uint ent_cap,
                         const u64 b, const i64 c, const u64 N0, const u64 N1, const u64 cm, const u64 cres) {
    flags[gid] = 0;
    if (p < 1000UL || (p & 1UL) == 0) { flags[gid] = 2; return; }
    const u64 b_mod = b % p;
    u64 t;
    if (c >= 0) { u64 cm_ = ((u64)c) % p; t = cm_ == 0 ? 0 : p - cm_; }
    else { t = ((u64)(-c)) % p; }
    if (b_mod == 0) { flags[gid] = 2; return; }
    if (t == 0) return;
    const u64 n_start = N0 + ((cres % cm) + cm - (N0 % cm)) % cm;
    if (n_start >= N1) return;
    u64 kcount = (N1 - n_start + cm - 1) / cm;
    if (kcount > p) kcount = p;

    // Residue filters first: they need no Montgomery setup when gcd(cm, p-1) <= 2.
    // T = t * b^-n_start must lie in <B>, B = b^cm.
    u64 stride = 1, parity = 0;
    const u64 g = gcd64(cm, p - 1UL);
    const int jb = jacobi64(b_mod, p);
    const int jt = jacobi64(t, p);
    const int lT = jt * (((n_start & 1UL) != 0UL) ? jb : 1);   // Legendre(T)
    if (g == 2UL) {
        if (lT != 1) return;                 // B is a square, so T must be one
    } else if (g < 2UL) {                    // cm odd: Legendre(B) = Legendre(b)
        if (jb == 1) {
            if (lT != 1) return;
        } else {
            stride = 2;
            if (lT != 1) parity = 1;         // k odd
        }
    }

    u64 inv = p;
    for (int i = 0; i < 5; ++i) inv *= 2UL - p * inv;
    const u64 ninv = 0UL - inv;
    const u64 one = (0UL - p) % p;
    if (need_prime) {
        // 13.1: base-2 Fermat test for sieve survivors.  A prime always passes; the few
        // pseudoprimes that also pass are rejected by the exact CPU check of flagged numbers.
        const u64 e = p - 1UL;
        int k = 63;
        while (k > 0 && ((e >> k) & 1UL) == 0UL) --k;
        u64 x = one;
        for (; k >= 0; --k) {
            x = mmul(x, x, p, ninv);
            if ((e >> k) & 1UL) { x += x; if (x >= p) x -= p; }
        }
        if (x != one) return;
    }
    u64 r2 = one;
    for (int i = 0; i < 64; ++i) { r2 = (r2 >= p - r2) ? r2 - (p - r2) : r2 + r2; }
    const u64 b_m = mmul(b_mod, r2, p, ninv);
    u64 B = mpow(b_m, cm, one, p, ninv);
    const u64 tm = mmul(t, r2, p, ninv);
    u64 G0 = mpow(b_m, n_start, one, p, ninv);
    if (g > 2UL) {
        // B is a g-th power residue, so T^((p-1)/g) == 1, i.e. t^e == (b^n_start)^e.
        const u64 e = (p - 1UL) / g;
        if (mpow(tm, e, one, p, ninv) != mpow(G0, e, one, p, ninv)) return;
    }
    if (stride == 2UL) {
        if (parity) G0 = mmul(G0, B, p, ninv);
        B = mmul(B, B, p, ninv);
    }
    if (kcount <= parity) return;
    const u64 jcount = (kcount - parity + stride - 1) / stride;
    u64 m = (u64)sqrt((double)jcount);
    while (m * m < jcount) ++m;
    if (m == 0) m = 1;
    const u64 Bm = mpow(B, m, one, p, ninv);
    const uint slot = ATOMIC_INC(ent_count);
    if (slot >= ent_cap) { flags[gid] = 2; return; }
    GLOBAL u64* e = ent + (u64)slot * ENT;
    e[0] = gid; e[1] = p; e[2] = ninv; e[3] = one; e[4] = B; e[5] = tm; e[6] = G0; e[7] = jcount;
    e[8] = Bm; e[9] = mpow(B, WG, one, p, ninv); e[10] = mpow(Bm, WG, one, p, ninv);
}

KERNEL void prep(GLOBAL const u64* primes, GLOBAL uchar* flags, GLOBAL u64* ent,
                   GLOBAL volatile uint* ent_count,
                   const u64 b, const i64 c, const u64 N0, const u64 N1,
                   const u64 cm, const u64 cres, const u32 count) {
    const u32 gid = GLOBAL_ID();
    if (gid >= count) return;
    prep_core(primes[gid], gid, 0, flags, ent, ent_count, 0xFFFFFFFFu, b, c, N0, N1, cm, cres);
}

// 13.1: same as prep, for the survivors of psieve (Fermat test included).
KERNEL void prep_sv(GLOBAL const u64* surv, GLOBAL uchar* flags, GLOBAL u64* ent,
                      GLOBAL volatile uint* ent_count, const uint ent_cap,
                      const u64 b, const i64 c, const u64 N0, const u64 N1,
                      const u64 cm, const u64 cres, const u32 count) {
    const u32 gid = GLOBAL_ID();
    if (gid >= count) return;
    prep_core(surv[gid], gid, 1, flags, ent, ent_count, ent_cap, b, c, N0, N1, cm, cres);
}

KERNEL void bsgs_wg(GLOBAL const u64* ent, GLOBAL uchar* flags, const u32 tbits
                      DYN_LOCAL_PARAM(tab)) {
    DYN_LOCAL_DECL(tab)
    LOCAL volatile int hit;
    const u32 lid = LOCAL_ID();
    GLOBAL const u64* e = ent + (u64)GROUP_ID() * ENT;
    const u64 idx = e[0], p = e[1], ninv = e[2], one = e[3], B = e[4], tm = e[5], G0 = e[6], jcount = e[7];
    const u64 Bm = e[8], BL = e[9], BmL = e[10];
    u64 m = (u64)sqrt((double)jcount);
    while (m * m < jcount) ++m;
    if (m == 0) m = 1;
    const u32 tsize = 1u << tbits, mask = tsize - 1u;
    for (u32 i = lid; i < tsize; i += WG) tab[i] = 0xFFFFFFFFu;
    if (lid == 0) hit = 0;
    BARRIER();
    // baby steps: t * B^i, i = lid, lid + WG, ...
    u64 v = mmul(tm, mpow(B, lid, one, p, ninv), p, ninv);
    for (u64 i = lid; i < m; i += WG) {
        u32 f = fp32(v);
        u32 h = (u32)((v * 0x9E3779B97F4A7C15UL) >> (64 - tbits));
        for (;;) {
            u32 prev = ATOMIC_CAS(&tab[h], 0xFFFFFFFFu, f);
            if (prev == 0xFFFFFFFFu || prev == f) break;
            h = (h + 1u) & mask;
        }
        v = mmul(v, BL, p, ninv);
    }
    BARRIER();
    // giant steps: G0 * (B^m)^g, g = lid, lid + WG, ..., g <= G
    const u64 G = (jcount + m - 1) / m;
    v = mmul(G0, mpow(Bm, lid, one, p, ninv), p, ninv);
    for (u64 gs = lid; gs <= G; gs += WG) {
        if (hit) break;
        u32 f = fp32(v);
        u32 h = (u32)((v * 0x9E3779B97F4A7C15UL) >> (64 - tbits));
        while (tab[h] != 0xFFFFFFFFu) { if (tab[h] == f) { hit = 1; break; } h = (h + 1u) & mask; }
        v = mmul(v, BmL, p, ninv);
    }
    BARRIER();
    if (lid == 0 && hit) flags[idx] = 1;
}

// ---------------- 13.1: prime generation on the GPU ----------------
// psieve: one workgroup per segment of SV_ODDS odd numbers v0, v0+2, ... (bitmap in local memory),
// crossed off with the odd primes in sp[] (3 .. 65521).  The survivors (no factor <= 65521) go to
// surv[]; prep_sv then Fermat-tests them.  A prime is never crossed off: marking starts at
// max(q^2, first odd multiple of q >= v0).
#define SV_WORDS 7680u   // 30 KB: fits the 32 KB local memory limit of AMD OpenCL
#define SV_ODDS (SV_WORDS * 32u)
#define SV_WG 256

// Index j of the first odd multiple of q that is >= max(v0, q^2), as v0 + 2j (v0 odd).
DEVICE_FN uint sv_start(const u64 v0, const uint q) {
    const u64 r = v0 % (u64)q;
    u64 j = (((u64)q - r) % (u64)q) * (u64)((q + 1u) >> 1) % (u64)q;   // v0 + 2j == 0 (mod q)
    const u64 qq = (u64)q * (u64)q;
    if (v0 + 2 * j < qq) j = (qq - v0) >> 1;
    return j > (u64)0xFFFFFFFFu ? 0xFFFFFFFFu : (uint)j;
}

KERNEL void psieve(const u64 lo, const u64 count, GLOBAL const uint* sp, const uint nsp, const uint nsmall,
                     GLOBAL u64* surv, GLOBAL volatile uint* surv_count, const uint cap) {
    LOCAL uint bits[SV_WORDS];
    LOCAL uint st[SV_WG];
    LOCAL uint lcount;
    LOCAL uint gbase;
    const uint lid = LOCAL_ID();
    const u64 base = (u64)GROUP_ID() * (u64)SV_ODDS;
    const u64 rest = count - base;                       // the host launches ceil(count / SV_ODDS) groups
    const uint len = rest < (u64)SV_ODDS ? (uint)rest : SV_ODDS;
    const u64 v0 = lo + 2 * base;
    for (uint i = lid; i < SV_WORDS; i += SV_WG) bits[i] = 0u;
    if (lid < nsmall) st[lid] = sv_start(v0, sp[lid]);
    if (lid == 0) lcount = 0u;
    BARRIER();
    // primes below 1024 (nsmall of them): all lanes share each prime
    for (uint k = 0; k < nsmall; ++k) {
        const uint q = sp[k];
        const uint s = st[k];
        if (s >= len) continue;
        for (uint i = s + lid * q; i < len; i += SV_WG * q) ATOMIC_OR(&bits[i >> 5], 1u << (i & 31u));
    }
    // larger primes: one lane per prime
    for (uint k = nsmall + lid; k < nsp; k += SV_WG) {
        const uint q = sp[k];
        for (uint i = sv_start(v0, q); i < len; i += q) ATOMIC_OR(&bits[i >> 5], 1u << (i & 31u));
    }
    BARRIER();
    const uint words = (len + 31u) >> 5;
    uint mine = 0u;
    for (uint w = lid; w < words; w += SV_WG) {
        uint x = ~bits[w];
        const uint valid = len - (w << 5);
        if (valid < 32u) x &= (1u << valid) - 1u;
        mine += POPCNT(x);
    }
    const uint off = ATOMIC_ADD(&lcount, mine);
    BARRIER();
    if (lid == 0) gbase = ATOMIC_ADD(surv_count, lcount);
    BARRIER();
    uint pos = gbase + off;
    for (uint w = lid; w < words; w += SV_WG) {
        uint x = ~bits[w];
        const uint valid = len - (w << 5);
        if (valid < 32u) x &= (1u << valid) - 1u;
        while (x != 0u) {
            const uint j = POPCNT((x & (0u - x)) - 1u);
            if (pos < cap) surv[pos] = v0 + 2 * ((u64)(w << 5) + (u64)j);
            ++pos;
            x &= x - 1u;
        }
    }
}

// 13.1: the numbers the GPU could not rule out (flags 1 or 2) -> out[].
KERNEL void collect(GLOBAL const u64* list, GLOBAL const uchar* flags, const u32 count,
                      GLOBAL u64* out, GLOBAL volatile uint* out_count) {
    const u32 gid = GLOBAL_ID();
    if (gid >= count) return;
    if (flags[gid] != 0) out[ATOMIC_INC(out_count)] = list[gid];
}

)CLC";

class OpenClSiever {
public:
    bool init(int device_index) {
        cl_uint np = 0;
        if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return fail("no OpenCL platform");
        std::vector<cl_platform_id> plats(np);
        clGetPlatformIDs(np, plats.data(), nullptr);
        std::vector<cl_device_id> devs;
        // SIEVER_OPENCL_ANY_DEVICE=1 also accepts CPU OpenCL devices (e.g. PoCL), for testing
        // the GPU code path on machines without a GPU.
        const cl_device_type dev_type =
            get_env("SIEVER_OPENCL_ANY_DEVICE") == "1" ? CL_DEVICE_TYPE_ALL : CL_DEVICE_TYPE_GPU;
        for (auto pl : plats) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(pl, dev_type, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
            std::vector<cl_device_id> d(nd);
            clGetDeviceIDs(pl, dev_type, nd, d.data(), nullptr);
            devs.insert(devs.end(), d.begin(), d.end());
        }
        if (devs.empty()) return fail("no OpenCL GPU device");
        if (device_index < 0 || device_index >= static_cast<int>(devs.size())) device_index = 0;
        dev_ = devs[device_index];
        char name[256] = {};
        clGetDeviceInfo(dev_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        cl_ulong gmem = 0, maxalloc = 0;
        clGetDeviceInfo(dev_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
        clGetDeviceInfo(dev_, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxalloc), &maxalloc, nullptr);
        max_alloc_ = maxalloc;
        cl_int err = 0;
        ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return fail("clCreateContext " + std::to_string(err));
        queue_ = clCreateCommandQueue(ctx_, dev_, 0, &err);
        if (err != CL_SUCCESS) return fail("clCreateCommandQueue " + std::to_string(err));
        const char* srcs[2] = {kOpenClPrelude, kGpuKernelBody};
        prog_ = clCreateProgramWithSource(ctx_, 2, srcs, nullptr, &err);
        if (err != CL_SUCCESS) return fail("clCreateProgramWithSource " + std::to_string(err));
        err = clBuildProgram(prog_, 1, &dev_, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t len = 0;
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            std::string log(len, '\0');
            clGetProgramBuildInfo(prog_, dev_, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
            return fail("kernel build failed: " + log);
        }
        kernel_ = clCreateKernel(prog_, "bsgs", &err);
        if (err != CL_SUCCESS) return fail("clCreateKernel " + std::to_string(err));
        k_prep_ = clCreateKernel(prog_, "prep", &err);
        if (err != CL_SUCCESS) return fail("clCreateKernel prep " + std::to_string(err));
        k_wg_ = clCreateKernel(prog_, "bsgs_wg", &err);
        if (err != CL_SUCCESS) return fail("clCreateKernel bsgs_wg " + std::to_string(err));
        // 13.1: GPU prime generation; without these kernels sieve_filter() declines and the CPU sieves.
        k_psieve_ = clCreateKernel(prog_, "psieve", &err);
        if (err == CL_SUCCESS) k_prepsv_ = clCreateKernel(prog_, "prep_sv", &err);
        if (err == CL_SUCCESS) k_collect_ = clCreateKernel(prog_, "collect", &err);
        if (err != CL_SUCCESS) {
            log_info("OpenCL: GPU sieve kernels unavailable (" + std::to_string(err) + "), sieving on the CPU");
            k_psieve_ = k_prepsv_ = k_collect_ = nullptr;
        }
        cl_ulong lmem = 0;
        clGetDeviceInfo(dev_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        local_mem_ = lmem;
        log_info(std::string("GPU enabled (OpenCL): ") + name + ", global mem " + std::to_string(gmem >> 20) +
                 " MB, max alloc " + std::to_string(maxalloc >> 20) + " MB");
        ok_ = true;
        return true;
    }

    bool ok() const { return ok_; }

    // 12.0 v2 path: prep kernel (filter + compaction) + workgroup-per-prime BSGS in LDS.
    bool filter(const std::vector<uint64_t>& primes, const WorkItem& item, std::vector<std::size_t>& todo) {
        if (primes.empty()) return true;
        const uint64_t N0 = item.N0 < 0 ? 0 : to_uint64(item.N0);
        const uint64_t N1 = to_uint64(item.N1);
        const uint64_t cm = item.cls_mod == 0 ? 1 : item.cls_mod;
        const uint64_t kmax = (N1 - N0) / cm + 2;
        const uint64_t m = static_cast<uint64_t>(std::sqrt(static_cast<double>(kmax))) + 2;
        unsigned tbits = 4;
        while ((1ULL << tbits) < m + m / 4) ++tbits;  // load <= 80%: keeps the LDS table at 16 KB for typical ranges
        const size_t lbytes = (size_t(1) << tbits) * sizeof(uint32_t);
        if (lbytes + 1024 > local_mem_ || tbits > 14) {
            return filter_v1(primes, item, todo);
        }
        const size_t batch = 262144;
        cl_int err = CL_SUCCESS;
        if (ent_cap_ < batch) {
            d_ent_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, batch * 11 * sizeof(uint64_t), nullptr, &err);
            if (err != CL_SUCCESS) return fail("ent alloc");
            d_cnt_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
            if (err != CL_SUCCESS) return fail("cnt alloc");
            d_p2_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY, batch * sizeof(uint64_t), nullptr, &err);
            if (err != CL_SUCCESS) return fail("p2 alloc");
            d_f2_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, batch, nullptr, &err);
            if (err != CL_SUCCESS) return fail("f2 alloc");
            ent_cap_ = batch;
        }
        std::vector<uint8_t> flags(batch);
        const cl_ulong b = static_cast<cl_ulong>(to_uint64(item.b));
        const cl_long c = static_cast<cl_long>(item.c);
        const cl_ulong n0 = N0, n1 = N1, cmod = cm, cres = item.cls_res;
        const cl_uint tb = tbits;
        for (size_t off = 0; off < primes.size(); off += batch) {
            const cl_uint count = static_cast<cl_uint>(std::min(batch, primes.size() - off));
            const cl_uint zero = 0;
            err = clEnqueueWriteBuffer(queue_, d_p2_, CL_FALSE, 0, count * sizeof(uint64_t), primes.data() + off, 0,
                                       nullptr, nullptr);
            err |= clEnqueueWriteBuffer(queue_, d_cnt_, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return fail("v2 write");
            cl_uint a = 0;
            clSetKernelArg(k_prep_, a++, sizeof(cl_mem), &d_p2_);
            clSetKernelArg(k_prep_, a++, sizeof(cl_mem), &d_f2_);
            clSetKernelArg(k_prep_, a++, sizeof(cl_mem), &d_ent_);
            clSetKernelArg(k_prep_, a++, sizeof(cl_mem), &d_cnt_);
            clSetKernelArg(k_prep_, a++, sizeof(cl_ulong), &b);
            clSetKernelArg(k_prep_, a++, sizeof(cl_long), &c);
            clSetKernelArg(k_prep_, a++, sizeof(cl_ulong), &n0);
            clSetKernelArg(k_prep_, a++, sizeof(cl_ulong), &n1);
            clSetKernelArg(k_prep_, a++, sizeof(cl_ulong), &cmod);
            clSetKernelArg(k_prep_, a++, sizeof(cl_ulong), &cres);
            clSetKernelArg(k_prep_, a++, sizeof(cl_uint), &count);
            size_t local = 64, global = ((count + 63) / 64) * 64;
            err = clEnqueueNDRangeKernel(queue_, k_prep_, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return fail("prep launch " + std::to_string(err));
            cl_uint n_ent = 0;
            err = clEnqueueReadBuffer(queue_, d_cnt_, CL_TRUE, 0, sizeof(cl_uint), &n_ent, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return fail("cnt read");
            if (n_ent > 0) {
                clSetKernelArg(k_wg_, 0, sizeof(cl_mem), &d_ent_);
                clSetKernelArg(k_wg_, 1, sizeof(cl_mem), &d_f2_);
                clSetKernelArg(k_wg_, 2, sizeof(cl_uint), &tb);
                clSetKernelArg(k_wg_, 3, lbytes, nullptr);
                size_t l2 = 64, g2 = size_t(n_ent) * 64;
                err = clEnqueueNDRangeKernel(queue_, k_wg_, 1, nullptr, &g2, &l2, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) return fail("wg launch " + std::to_string(err));
            }
            err = clEnqueueReadBuffer(queue_, d_f2_, CL_TRUE, 0, count, flags.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return fail("flags read " + std::to_string(err));
            for (cl_uint i = 0; i < count; ++i) {
                if (flags[i] != 0) todo.push_back(off + i);
            }
        }
        return true;
    }

    // v1 path (11.4 prototype): one work-item per prime, table in global memory.
    bool filter_v1(const std::vector<uint64_t>& primes, const WorkItem& item, std::vector<std::size_t>& todo) {
        if (primes.empty()) return true;
        const uint64_t N0 = item.N0 < 0 ? 0 : to_uint64(item.N0);
        const uint64_t N1 = to_uint64(item.N1);
        const uint64_t cm = item.cls_mod == 0 ? 1 : item.cls_mod;
        const uint64_t kmax = (N1 - N0) / cm + 2;
        uint64_t m = static_cast<uint64_t>(std::sqrt(static_cast<double>(kmax))) + 2;
        unsigned tbits = 4;
        while ((1ULL << tbits) < 2 * m) ++tbits;
        const size_t table_bytes = (size_t(1) << tbits) * sizeof(uint32_t);
        size_t budget = std::min<size_t>(static_cast<size_t>(max_alloc_), size_t(256) << 20);
        size_t batch = std::max<size_t>(256, budget / table_bytes);
        batch = std::min<size_t>(batch, 32768);

        cl_int err = 0;
        ensure_buffers(batch, table_bytes, err);
        if (err != CL_SUCCESS) return fail("buffer alloc " + std::to_string(err));

        std::vector<uint8_t> flags(batch);
        const cl_ulong b = static_cast<cl_ulong>(to_uint64(item.b));
        const cl_long c = static_cast<cl_long>(item.c);
        const cl_ulong n0 = N0, n1 = N1, cmod = cm, cres = item.cls_res;
        const cl_uint tb = tbits;
        for (size_t off = 0; off < primes.size(); off += batch) {
            const cl_uint count = static_cast<cl_uint>(std::min(batch, primes.size() - off));
            err = clEnqueueWriteBuffer(queue_, d_primes_, CL_TRUE, 0, count * sizeof(uint64_t), primes.data() + off,
                                       0, nullptr, nullptr);
            if (err != CL_SUCCESS) return fail("write " + std::to_string(err));
            clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_primes_);
            clSetKernelArg(kernel_, 1, sizeof(cl_mem), &d_flags_);
            clSetKernelArg(kernel_, 2, sizeof(cl_mem), &d_tables_);
            clSetKernelArg(kernel_, 3, sizeof(cl_ulong), &b);
            clSetKernelArg(kernel_, 4, sizeof(cl_long), &c);
            clSetKernelArg(kernel_, 5, sizeof(cl_ulong), &n0);
            clSetKernelArg(kernel_, 6, sizeof(cl_ulong), &n1);
            clSetKernelArg(kernel_, 7, sizeof(cl_ulong), &cmod);
            clSetKernelArg(kernel_, 8, sizeof(cl_ulong), &cres);
            clSetKernelArg(kernel_, 9, sizeof(cl_uint), &tb);
            clSetKernelArg(kernel_, 10, sizeof(cl_uint), &count);
            size_t local = 64;
            size_t global = ((count + local - 1) / local) * local;
            err = clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return fail("launch " + std::to_string(err));
            err = clEnqueueReadBuffer(queue_, d_flags_, CL_TRUE, 0, count, flags.data(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return fail("read " + std::to_string(err));
            for (cl_uint i = 0; i < count; ++i) {
                if (flags[i] != 0) todo.push_back(off + i);
            }
        }
        return true;
    }

    // 13.1: the primes of [lo0, hi) are generated on the GPU (psieve, then a Fermat test in prep_sv) and
    // filtered like filter().  The numbers the GPU could not rule out are appended to `flagged`; they
    // can include a few composites, so the caller checks primality.  Returns false when this chunk has
    // to take the CPU path (no sieve kernels, BSGS table too large, buffer overflow, OpenCL error).
    bool sieve_filter(uint64_t lo0, uint64_t hi, const WorkItem& item, std::vector<uint64_t>& flagged) {
        if (!ok_ || sv_broken_ || k_psieve_ == nullptr) return false;
        const uint64_t lo = lo0 | 1ULL;
        if (lo >= hi) return true;
        const uint64_t count = (hi - lo + 1) / 2;
        const uint64_t N0 = item.N0 < 0 ? 0 : to_uint64(item.N0);
        const uint64_t N1 = to_uint64(item.N1);
        const uint64_t cm = item.cls_mod == 0 ? 1 : item.cls_mod;
        const uint64_t kmax = (N1 - N0) / cm + 2;
        const uint64_t m = static_cast<uint64_t>(std::sqrt(static_cast<double>(kmax))) + 2;
        unsigned tbits = 4;
        while ((1ULL << tbits) < m + m / 4) ++tbits;
        const size_t lbytes = (size_t(1) << tbits) * sizeof(uint32_t);
        if (lbytes + 1024 > local_mem_ || tbits > 14) return false;
        cl_int err = CL_SUCCESS;
        const cl_uint cap = kGpuSieveSurvCap, ent_cap = kGpuSieveEntCap;
        const std::vector<uint32_t>& sp = gpu_sieve_primes();
        if (!sv_ready_) {
            d_sp_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY, sp.size() * sizeof(uint32_t), nullptr, &err);
            if (err == CL_SUCCESS) d_surv_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, size_t(cap) * sizeof(uint64_t), nullptr, &err);
            if (err == CL_SUCCESS) d_sflags_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, cap, nullptr, &err);
            if (err == CL_SUCCESS) d_sent_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, size_t(ent_cap) * 11 * sizeof(uint64_t), nullptr, &err);
            if (err == CL_SUCCESS) d_out_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, size_t(cap) * sizeof(uint64_t), nullptr, &err);
            if (err == CL_SUCCESS) d_scnt_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
            if (err == CL_SUCCESS) d_ecnt_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
            if (err == CL_SUCCESS) d_ocnt_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
            if (err == CL_SUCCESS) {
                err = clEnqueueWriteBuffer(queue_, d_sp_, CL_TRUE, 0, sp.size() * sizeof(uint32_t), sp.data(), 0,
                                           nullptr, nullptr);
            }
            if (err != CL_SUCCESS) return sv_fail("buffer setup", err);
            sv_ready_ = true;
        }
        const cl_uint nsp = static_cast<cl_uint>(sp.size()), nsmall = gpu_sieve_nsmall();
        const cl_ulong b = static_cast<cl_ulong>(to_uint64(item.b));
        const cl_long c = static_cast<cl_long>(item.c);
        const cl_ulong n0 = N0, n1 = N1, cmod = cm, cres = item.cls_res;
        const cl_uint tb = tbits;
        const cl_uint zero = 0;
        std::vector<uint64_t> buf;
        for (uint64_t pb = 0; pb < count; pb += kGpuSievePassOdds) {
            const cl_ulong pc = std::min<uint64_t>(kGpuSievePassOdds, count - pb);
            const cl_ulong plo = lo + 2 * pb;
            err = clEnqueueWriteBuffer(queue_, d_scnt_, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
            err |= clEnqueueWriteBuffer(queue_, d_ecnt_, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
            err |= clEnqueueWriteBuffer(queue_, d_ocnt_, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return sv_fail("counter reset", err);
            cl_uint a = 0;
            clSetKernelArg(k_psieve_, a++, sizeof(cl_ulong), &plo);
            clSetKernelArg(k_psieve_, a++, sizeof(cl_ulong), &pc);
            clSetKernelArg(k_psieve_, a++, sizeof(cl_mem), &d_sp_);
            clSetKernelArg(k_psieve_, a++, sizeof(cl_uint), &nsp);
            clSetKernelArg(k_psieve_, a++, sizeof(cl_uint), &nsmall);
            clSetKernelArg(k_psieve_, a++, sizeof(cl_mem), &d_surv_);
            clSetKernelArg(k_psieve_, a++, sizeof(cl_mem), &d_scnt_);
            clSetKernelArg(k_psieve_, a++, sizeof(cl_uint), &cap);
            size_t l = kGpuSieveWg, g = ((pc + kGpuSieveSegOdds - 1) / kGpuSieveSegOdds) * kGpuSieveWg;
            err = clEnqueueNDRangeKernel(queue_, k_psieve_, 1, nullptr, &g, &l, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return sv_fail("psieve launch", err);
            cl_uint n = 0;
            err = clEnqueueReadBuffer(queue_, d_scnt_, CL_TRUE, 0, sizeof(cl_uint), &n, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return sv_fail("survivor count", err);
            if (n > cap) return false;  // more survivors than the buffer holds (only for tiny p): CPU path
            if (n == 0) continue;
            a = 0;
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_mem), &d_surv_);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_mem), &d_sflags_);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_mem), &d_sent_);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_mem), &d_ecnt_);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_uint), &ent_cap);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_ulong), &b);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_long), &c);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_ulong), &n0);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_ulong), &n1);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_ulong), &cmod);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_ulong), &cres);
            clSetKernelArg(k_prepsv_, a++, sizeof(cl_uint), &n);
            size_t local = 64, global = ((size_t(n) + 63) / 64) * 64;
            err = clEnqueueNDRangeKernel(queue_, k_prepsv_, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return sv_fail("prep_sv launch", err);
            cl_uint n_ent = 0;
            err = clEnqueueReadBuffer(queue_, d_ecnt_, CL_TRUE, 0, sizeof(cl_uint), &n_ent, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return sv_fail("entry count", err);
            n_ent = std::min(n_ent, ent_cap);  // entries past ent_cap were flagged 2 (CPU) by prep_sv
            if (n_ent > 0) {
                clSetKernelArg(k_wg_, 0, sizeof(cl_mem), &d_sent_);
                clSetKernelArg(k_wg_, 1, sizeof(cl_mem), &d_sflags_);
                clSetKernelArg(k_wg_, 2, sizeof(cl_uint), &tb);
                clSetKernelArg(k_wg_, 3, lbytes, nullptr);
                size_t l2 = 64, g2 = size_t(n_ent) * 64;
                err = clEnqueueNDRangeKernel(queue_, k_wg_, 1, nullptr, &g2, &l2, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) return sv_fail("bsgs_wg launch", err);
            }
            a = 0;
            clSetKernelArg(k_collect_, a++, sizeof(cl_mem), &d_surv_);
            clSetKernelArg(k_collect_, a++, sizeof(cl_mem), &d_sflags_);
            clSetKernelArg(k_collect_, a++, sizeof(cl_uint), &n);
            clSetKernelArg(k_collect_, a++, sizeof(cl_mem), &d_out_);
            clSetKernelArg(k_collect_, a++, sizeof(cl_mem), &d_ocnt_);
            err = clEnqueueNDRangeKernel(queue_, k_collect_, 1, nullptr, &global, &local, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return sv_fail("collect launch", err);
            cl_uint n_out = 0;
            err = clEnqueueReadBuffer(queue_, d_ocnt_, CL_TRUE, 0, sizeof(cl_uint), &n_out, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return sv_fail("flag count", err);
            if (n_out > 0) {
                buf.resize(n_out);
                err = clEnqueueReadBuffer(queue_, d_out_, CL_TRUE, 0, buf.size() * sizeof(uint64_t), buf.data(), 0,
                                          nullptr, nullptr);
                if (err != CL_SUCCESS) return sv_fail("flag read", err);
                flagged.insert(flagged.end(), buf.begin(), buf.end());
            }
        }
        return true;
    }

private:
    bool sv_fail(const char* what, cl_int err) {
        log_info(std::string("OpenCL GPU sieve disabled (") + what + " " + std::to_string(err) +
                 "), sieving on the CPU");
        sv_broken_ = true;
        return false;
    }
    void ensure_buffers(size_t batch, size_t table_bytes, cl_int& err) {
        err = CL_SUCCESS;
        if (batch == batch_ && table_bytes == table_bytes_) return;
        release_buffers();
        d_primes_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY, batch * sizeof(uint64_t), nullptr, &err);
        if (err != CL_SUCCESS) return;
        d_flags_ = clCreateBuffer(ctx_, CL_MEM_WRITE_ONLY, batch, nullptr, &err);
        if (err != CL_SUCCESS) return;
        d_tables_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, batch * table_bytes, nullptr, &err);
        if (err != CL_SUCCESS) return;
        batch_ = batch;
        table_bytes_ = table_bytes;
    }
    void release_buffers() {
        if (d_primes_) clReleaseMemObject(d_primes_);
        if (d_flags_) clReleaseMemObject(d_flags_);
        if (d_tables_) clReleaseMemObject(d_tables_);
        d_primes_ = d_flags_ = d_tables_ = nullptr;
        batch_ = table_bytes_ = 0;
    }
    bool fail(const std::string& msg) {
        log_info("OpenCL unavailable: " + msg);
        ok_ = false;
        return false;
    }

    bool ok_ = false;
    cl_device_id dev_ = nullptr;
    cl_context ctx_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program prog_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_kernel k_prep_ = nullptr;
    cl_kernel k_wg_ = nullptr;
    cl_ulong local_mem_ = 0;
    cl_mem d_ent_ = nullptr, d_cnt_ = nullptr, d_p2_ = nullptr, d_f2_ = nullptr;
    size_t ent_cap_ = 0;
    cl_mem d_primes_ = nullptr, d_flags_ = nullptr, d_tables_ = nullptr;
    size_t batch_ = 0, table_bytes_ = 0;
    cl_ulong max_alloc_ = 0;
    // 13.1 GPU sieve
    cl_kernel k_psieve_ = nullptr, k_prepsv_ = nullptr, k_collect_ = nullptr;
    cl_mem d_sp_ = nullptr, d_surv_ = nullptr, d_sflags_ = nullptr, d_sent_ = nullptr, d_out_ = nullptr;
    cl_mem d_scnt_ = nullptr, d_ecnt_ = nullptr, d_ocnt_ = nullptr;
    bool sv_ready_ = false, sv_broken_ = false;
};

// Picks the GPU backend. GPU_backend in prime.ini: "auto" (default: CUDA if an NVIDIA driver and
// NVRTC are present, otherwise OpenCL), "cuda" or "opencl".
class GpuSiever {
public:
    bool init(int device_index, const std::string& backend = "auto") {
        if (backend == "cuda" || backend == "auto") {
            if (cuda_.init(device_index, std::string(kCudaPrelude) + kGpuKernelBody)) {
                use_cuda_ = true;
                return true;
            }
            if (backend == "cuda") {
                log_info("GPU disabled: GPU_backend=cuda but CUDA is unavailable");
                return false;
            }
        }
        if (opencl_.init(device_index)) return true;
        log_info("GPU disabled: no usable GPU backend");
        return false;
    }
    bool ok() const { return use_cuda_ ? cuda_.ok() : opencl_.ok(); }
    bool filter(const std::vector<uint64_t>& primes, const WorkItem& item, std::vector<std::size_t>& todo) {
        std::lock_guard<std::mutex> lock(mutex_);  // 13.0: several feeder threads share one GPU
        return use_cuda_ ? cuda_.filter(primes, item, todo) : opencl_.filter(primes, item, todo);
    }
    // 13.1: prime generation + filter on the GPU (see OpenClSiever::sieve_filter).
    bool sieve_filter(uint64_t lo, uint64_t hi, const WorkItem& item, std::vector<uint64_t>& flagged) {
        std::lock_guard<std::mutex> lock(mutex_);
        return use_cuda_ ? cuda_.sieve_filter(lo, hi, item, flagged) : opencl_.sieve_filter(lo, hi, item, flagged);
    }

private:
    std::mutex mutex_;
    CudaSiever cuda_;
    OpenClSiever opencl_;
    bool use_cuda_ = false;
};

}  // namespace siever
