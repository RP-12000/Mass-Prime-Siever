# Mass Prime Siever

A sieve for exponential sequences `b^n + c` (for example `10^n + 111`, `10^n + 9`, `2^n - 25`).
For every prime `p` in `[P0, P1)` it finds all `n` in `[N0, N1)` with `p | b^n + c`, using a
baby-step giant-step discrete log, and writes the surviving `n` (PRP candidates) to `results.json`.

Current version: **13.1** (successor of 13.0; same file formats, see `relations.p` below).

## What's new in 13.1

- **Prime generation on the GPU** (`GPU_sieve=1`): each GPU workgroup sieves a 30 KB segment of odd
  numbers with the primes below 65536 in local memory, the survivors get a base-2 Fermat test in the
  `prep` stage, and only the numbers the GPU cannot rule out come back to the CPU, which checks them
  exactly (deterministic Miller-Rabin, then the usual BSGS). The CPU work of a GPU worker drops about
  5x, so a GPU is no longer held back by a slow CPU (Kaggle: 4 vCPUs for 2 T4s). The GPU spends about
  3% more time per chunk.
- **Residue classes per prime** (`GPU_classes=1`, default; GPU sieve path, even `mod`): with
  g = gcd(lcm(mod, 24), p - 1) the in-class n are split by lcm(mod, g). b^lcm(mod, g) is a g-th power
  residue, so a class can only hold a solution if t * b^-n is one too; this one test rejects about
  70% of the primes (the quadratic-residue test alone: 50%), and the remaining primes run one BSGS over
  their surviving classes with a shared baby-step table. Every class is covered, so the relations do
  not change. About 1.5x less GPU time on 2^n-25 at p = 95T.
- CUDA waits for the GPU with a blocking sync instead of spinning, so a GPU worker leaves the CPU idle.
- `is_prime64` uses the 12 bases 2..37 and is exact for all 64-bit numbers (bases 2..11 are exact
  only below 2.15e12).
- `kaggle/kaggle_run.py`: `--gpu-sieve` (default 1); a state saved by an older version is resumed
  with the current `CPU_count` / `GPU_threads` / `GPU_sieve`; all `core_*.p` files of a resumed
  checkpoint are merged, whatever thread layout wrote them.
- Relations are identical to 13.0 and to the CPU path (checked on `2^n-25` with classes 2/1, 6/1 and
  none, `10^n+111` with 6/4 and none, and on p from 2 and from 10^6 where the sieve primes reach
  past sqrt(p)).

## What's new in 13.0

- **GPU kernel without single-lane work**: the BSGS is written as `G0 * (B^m)^g == t * B^i`
  (`j = g*m - i`), so neither the setup nor the search needs a modular inverse, and everything the
  lanes of a workgroup share is computed in the per-prime `prep` kernel. The quadratic-residue test
  uses the Jacobi symbol instead of a modular power. GPU time about 2.6x lower.
- **Cache-blocked prime sieve**: 128 KB segments plus a bucket sieve for the large primes, about
  2.4x faster than the 12.0 loop, identical output.
- **Several feeder threads per GPU** (`GPU_threads`, default 2): one thread sieves its next chunk on
  the CPU while another one's chunk is on the GPU. `kaggle/kaggle_run.py --gpu-threads` (default 2)
  gives the cores to the feeders (`CPU_count=0` on a 4-core, 2-GPU Kaggle machine).
- Relations are identical to 12.0 and to the CPU path (checked on `2^n-25` with classes 2/1, 6/1 and
  none, and on `10^n+111`).

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
GPU_threads=2     # 13.0: feeder threads sharing the GPU
GPU_sieve=0       # 13.1: 1 = generate the primes on the GPU too (for weak CPUs, e.g. Kaggle)
GPU_classes=1     # 13.1: residue classes per prime on the GPU sieve path (0 = off)
```

## Output

- `relations.p`: `p, M, q` per line - `p` divides `b^n + c` for every `n == M (mod q)`.
  Since 12.0 only relations that hit at least one `n` in `[N0, N1)` are written, and `M` may be
  the first such `n` rather than the smallest residue mod `q` (use `M mod q` when comparing with
  11.2 output). Skipping relations that miss the range keeps the file and the merge step smaller.
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

13.0 against 12.0 (same laptop, `2^n-25`, odd n = 3.6M-5M, p in [95T, 95T + 2e9)):

| configuration | 12.0 | 13.0 |
|---|---|---|
| Radeon 780M only (13.0: `GPU_threads=2`) | 34.1 s | 9.9 s |
| GPU + CPU threads (12.0: 15, 13.0: 14) | 21.3 s | 13.7 s |

On the laptop 13.0 is fastest with the GPU alone (`CPU_count=0`): the integrated GPU shares power
and memory bandwidth with the CPU cores. On Kaggle (2 x T4), p = 95T-100T took 1.8 h with 13.0
against about 9 h with 12.0 for the same 5T at 90T-95T.

13.1 with `GPU_sieve=1` on the same laptop range (GPU only, `GPU_threads=2`):

| configuration | 13.0 | 13.1 `GPU_sieve=1` |
|---|---|---|
| all 16 logical CPUs available | 10.1 s, 4.7 s CPU time | 10.3 s, 0.9 s CPU time |
| process limited to 1 logical CPU | 15.0 s | 10.4 s |

With `GPU_classes=1` (later measurement, laptop warmer: 13.0 took 13.3 s here) the same range takes
8.5 s against 12.7 s for 13.1 without classes.

The second row is the situation of a fast GPU next to a slow CPU, as on Kaggle, where 13.0 spent
about 0.18 s per 64 MB chunk mostly waiting for the CPU sieve.

## Building

The same sources build with MSVC on Windows and with GCC/Clang on Linux. Compiler-specific
intrinsics (`_umul128`, `_udiv128`, `_addcarry_u64`, `fopen_s`) are wrapped in `Portable.hpp`;
MSVC still uses the native intrinsics.

**Windows (Visual Studio).** Visual Studio 2026 (v145 toolset), C++20, vcpkg with `boost`
(multiprecision, json, date_time) and `opencl`. Open `Mass Prime Siever.slnx` and build
`Release | x64`. CMake also works:
`cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake && cmake --build build --config Release`.

**Linux.**

```
sudo apt-get install cmake g++ libboost-dev libboost-json-dev libboost-date-time-dev ocl-icd-opencl-dev opencl-headers
cmake -B build -DSIEVER_NATIVE=ON && cmake --build build -j
```

Ctrl+C (SIGINT), SIGTERM and SIGHUP checkpoint into `core_*.p` like closing the console window
on Windows. `SIEVER_OPENCL_ANY_DEVICE=1` lets the GPU worker use a CPU OpenCL device such as PoCL
(for testing the GPU code path without a GPU).

**GPU backends.** `GPU_backend=auto|cuda|opencl` in `prime.ini` (default `auto`: CUDA when an
NVIDIA driver and NVRTC are present, otherwise OpenCL). Both backends run the same kernel source
(`kGpuKernelBody` in `Gpu.hpp`, with a small prelude per dialect). The CUDA backend loads the
driver and NVRTC at run time, so building needs no CUDA toolkit; set `SIEVER_NVRTC` to the
library path if NVRTC is not on the default search path.

**Kaggle.** See `kaggle/`: `mass_prime_siever_kaggle.ipynb` builds the siever, runs one worker per
GPU, stops and checkpoints before the 12 h session limit, and resumes from the previous
session's output.

## License

Apache License 2.0, see `LICENSE.txt`.
