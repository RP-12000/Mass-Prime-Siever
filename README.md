# Mass Prime Siever

A sieve for exponential sequences `b^n + c` (for example `10^n + 111`, `10^n + 9`, `2^n - 25`).
For every prime `p` in `[P0, P1)` it finds all `n` in `[N0, N1)` with `p | b^n + c`, using a
baby-step giant-step discrete log, and writes the surviving `n` (PRP candidates) to `results.json`.

Current version: **12.0** (successor of Siever 11.2; output format unchanged).

## What's new in 12.0

- **Montgomery arithmetic** and a flat, per-thread hash table in BSGS (11.2 used boost 128-bit
  division and a new `std::unordered_map` per prime). About 15-18x faster per thread.
- **Quadratic / g-th power residue filter**: primes that cannot divide any candidate are skipped
  before BSGS.
- **Congruence classes**: sieve only `n == res (mod mod)` (e.g. odd `n` for `2^n - 25`, or the
  `4 mod 6` class of `10^n + 111`). Search cost drops by `sqrt(mod)`.
- **Search starts at N0** instead of `n = 0`.
- **OpenCL GPU sieving** (tested on AMD Radeon 780M): a prefilter kernel plus one workgroup per
  prime with the BSGS table in local memory. Every prime the GPU flags is re-checked exactly on the
  CPU, so results are identical to the CPU path.
- `b^n + c == p` is no longer reported as "factored by p" (e.g. `10^2 + 111 = 211`).
- `prime.ini` is no longer rewritten (and extra keys lost) on every start.

## Input

`worktodo.txt`, one item per line:

```
Factor=b,c,P0,P1,N0,N1[,mod,res]
```

| field | meaning |
|---|---|
| `b, c` | the sequence `b^n + c` (`c` may be negative) |
| `P0, P1` | sieve primes `P0 <= p < P1` (`P1 < 2^62`) |
| `N0, N1` | exponent range `N0 <= n < N1` (`N1 - N0 <= 10,000,000`) |
| `mod, res` | optional: only `n == res (mod mod)` |

Examples:

```
Factor=2,-25,1000000000000,2000000000000,3600000,5000000,2,1
Factor=10,111,1000000000000,2000000000000,0,10000000,6,4
```

`prime.ini`:

```
CPU_count=15      # CPU worker threads (0 = GPU only, needs UseGPU=1)
MemoryAllowed=64  # MB, sets the chunk size of the prime range
UseGPU=1          # 1 = add an OpenCL GPU worker
GPU_device=0      # index of the OpenCL GPU
```

## Output

- `relations.p`: `p, M, q` per line - `p` divides `b^n + c` for every `n == M (mod q)`.
- `results.json`: per work item, the list of surviving `n` (`prp`) and the factored `n` with their
  factors. Only in-class `n` are listed when a class is given.
- `siever.log`: progress, ETA and timing. Closing the console window checkpoints into `core_*.p`;
  the next start resumes automatically.

## Benchmarks (Ryzen 7 8845H + Radeon 780M, 2^n-25, odd n, n = 3.6M-10M, p in 950G-951G)

| configuration | time |
|---|---|
| Siever 11.2, 16 threads (full n range) | ~2090 s (extrapolated) |
| 12.0, 16 CPU threads | 39.6 s |
| 12.0, GPU only | 27.5 s |
| 12.0, GPU + 15 CPU threads | 18.1 s |

## Building

Visual Studio 2026 (v145 toolset), C++20, vcpkg with `boost` (multiprecision, json, date_time) and
`opencl`. Open `Mass Prime Siever.slnx` and build `Release | x64`.

## License

Apache License 2.0, see `LICENSE.txt`.
