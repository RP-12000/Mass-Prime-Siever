#pragma once
// CUDA backend for the GPU pre-filter (NVIDIA cards).
//
// It runs the same kernels as the OpenCL backend (kGpuKernelBody in Gpu.hpp), compiled at run
// time with NVRTC. The CUDA driver (libcuda / nvcuda.dll) and NVRTC are loaded dynamically, so
// building the siever needs no CUDA toolkit and the binary still runs on machines without CUDA:
// init() then simply fails and the caller falls back to OpenCL or the CPU.
//
// NVRTC is searched for in this order: $SIEVER_NVRTC (full path), the usual sonames, then
// /usr/local/cuda/lib64.

#include "Logger.hpp"
#include "Math.hpp"
#include "Types.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace siever {

namespace cudadyn {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
using CUcontext = struct CUctx_st*;
using CUmodule = struct CUmod_st*;
using CUfunction = struct CUfunc_st*;
using CUstream = struct CUstream_st*;
using nvrtcResult = int;
using nvrtcProgram = struct _nvrtcProgram*;

constexpr int kAttrMaxSharedPerBlock = 8;    // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
constexpr int kAttrCcMajor = 75;             // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
constexpr int kAttrCcMinor = 76;             // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

#ifdef _WIN32
#define SIEVER_CUDAAPI __stdcall
#else
#define SIEVER_CUDAAPI
#endif

struct Api {
    CUresult(SIEVER_CUDAAPI* cuInit)(unsigned int) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuDeviceGetCount)(int*) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuDeviceTotalMem)(size_t*, CUdevice) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuCtxSynchronize)() = nullptr;
    CUresult(SIEVER_CUDAAPI* cuModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuMemFree)(CUdeviceptr) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuMemsetD32)(CUdeviceptr, unsigned int, size_t) = nullptr;
    CUresult(SIEVER_CUDAAPI* cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                             unsigned, unsigned, CUstream, void**, void**) = nullptr;

    nvrtcResult (*nvrtcVersion)(int*, int*) = nullptr;
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*,
                                      const char* const*) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram, int, const char* const*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram, size_t*) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
    nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram, size_t*) = nullptr;  // CUDA >= 11.1, optional
    nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram, char*) = nullptr;        // CUDA >= 11.1, optional
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;
};

inline void* open_library(const std::vector<std::string>& names) {
    for (const std::string& name : names) {
        if (name.empty()) continue;
#ifdef _WIN32
        if (HMODULE h = LoadLibraryA(name.c_str())) return reinterpret_cast<void*>(h);
#else
        if (void* h = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL)) return h;
#endif
    }
    return nullptr;
}

inline void* find_symbol(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

template <typename F>
inline bool bind(void* lib, F& fn, const char* name) {
    fn = reinterpret_cast<F>(find_symbol(lib, name));
    return fn != nullptr;
}

// Loads the driver and NVRTC. Returns an empty string on success, otherwise the reason.
inline std::string load(Api& api) {
#ifdef _WIN32
    void* cuda = open_library({"nvcuda.dll"});
#else
    void* cuda = open_library({"libcuda.so.1", "libcuda.so"});
#endif
    if (cuda == nullptr) return "CUDA driver not found";
    bool ok = bind(cuda, api.cuInit, "cuInit") && bind(cuda, api.cuDeviceGetCount, "cuDeviceGetCount") &&
              bind(cuda, api.cuDeviceGet, "cuDeviceGet") && bind(cuda, api.cuDeviceGetName, "cuDeviceGetName") &&
              bind(cuda, api.cuDeviceGetAttribute, "cuDeviceGetAttribute") &&
              bind(cuda, api.cuDeviceTotalMem, "cuDeviceTotalMem_v2") &&
              bind(cuda, api.cuCtxCreate, "cuCtxCreate_v2") && bind(cuda, api.cuCtxSetCurrent, "cuCtxSetCurrent") &&
              bind(cuda, api.cuCtxSynchronize, "cuCtxSynchronize") &&
              bind(cuda, api.cuModuleLoadData, "cuModuleLoadData") &&
              bind(cuda, api.cuModuleGetFunction, "cuModuleGetFunction") &&
              bind(cuda, api.cuMemAlloc, "cuMemAlloc_v2") && bind(cuda, api.cuMemFree, "cuMemFree_v2") &&
              bind(cuda, api.cuMemcpyHtoD, "cuMemcpyHtoD_v2") && bind(cuda, api.cuMemcpyDtoH, "cuMemcpyDtoH_v2") &&
              bind(cuda, api.cuMemsetD32, "cuMemsetD32_v2") && bind(cuda, api.cuLaunchKernel, "cuLaunchKernel");
    if (!ok) return "CUDA driver is missing required functions";

    const char* env = std::getenv("SIEVER_NVRTC");
#ifdef _WIN32
    void* nvrtc = open_library({env ? env : "", "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"});
#else
    void* nvrtc = open_library({env ? env : "", "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12",
                                "libnvrtc.so.11.2", "/usr/local/cuda/lib64/libnvrtc.so"});
#endif
    if (nvrtc == nullptr) return "NVRTC not found (set SIEVER_NVRTC to the path of libnvrtc)";
    ok = bind(nvrtc, api.nvrtcVersion, "nvrtcVersion") &&
         bind(nvrtc, api.nvrtcCreateProgram, "nvrtcCreateProgram") &&
         bind(nvrtc, api.nvrtcCompileProgram, "nvrtcCompileProgram") &&
         bind(nvrtc, api.nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize") &&
         bind(nvrtc, api.nvrtcGetProgramLog, "nvrtcGetProgramLog") &&
         bind(nvrtc, api.nvrtcGetPTXSize, "nvrtcGetPTXSize") && bind(nvrtc, api.nvrtcGetPTX, "nvrtcGetPTX") &&
         bind(nvrtc, api.nvrtcDestroyProgram, "nvrtcDestroyProgram");
    if (!ok) return "NVRTC is missing required functions";
    bind(nvrtc, api.nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
    bind(nvrtc, api.nvrtcGetCUBIN, "nvrtcGetCUBIN");
    return {};
}

// Compiles `source` with NVRTC. With `cubin` it asks for SASS for the exact GPU (sm_XY), which the
// driver loads without JIT; otherwise PTX (compute_XY). Returns false and fills `error` on failure.
inline bool compile(const Api& api, const std::string& source, int cc_major, int cc_minor, bool cubin,
                    std::vector<char>& image, std::string& error) {
    nvrtcProgram prog = nullptr;
    if (api.nvrtcCreateProgram(&prog, source.c_str(), "siever.cu", 0, nullptr, nullptr) != 0) {
        error = "nvrtcCreateProgram failed";
        return false;
    }
    const std::string arch = std::string(cubin ? "--gpu-architecture=sm_" : "--gpu-architecture=compute_") +
                             std::to_string(cc_major) + std::to_string(cc_minor);
    const char* options[] = {arch.c_str(), "--std=c++17"};
    const int rc = api.nvrtcCompileProgram(prog, 2, options);
    if (rc != 0) {
        size_t len = 0;
        api.nvrtcGetProgramLogSize(prog, &len);
        std::string log(len, '\0');
        if (len > 0) api.nvrtcGetProgramLog(prog, log.data());
        error = "NVRTC compile failed (" + arch + "): " + log;
        api.nvrtcDestroyProgram(&prog);
        return false;
    }
    size_t size = 0;
    if (cubin) {
        api.nvrtcGetCUBINSize(prog, &size);
        image.assign(size, '\0');
        api.nvrtcGetCUBIN(prog, image.data());
    } else {
        api.nvrtcGetPTXSize(prog, &size);
        image.assign(size, '\0');
        api.nvrtcGetPTX(prog, image.data());
    }
    api.nvrtcDestroyProgram(&prog);
    return size > 0;
}

}  // namespace cudadyn

class CudaSiever {
public:
    // kernel_source: the CUDA prelude followed by the shared kernel body.
    bool init(int device_index, const std::string& kernel_source) {
        const std::string why = cudadyn::load(api_);
        if (!why.empty()) return fail(why);
        if (api_.cuInit(0) != 0) return fail("cuInit failed (no NVIDIA GPU?)");
        int count = 0;
        if (api_.cuDeviceGetCount(&count) != 0 || count == 0) return fail("no CUDA device");
        if (device_index < 0 || device_index >= count) device_index = 0;
        if (api_.cuDeviceGet(&dev_, device_index) != 0) return fail("cuDeviceGet failed");
        char name[256] = {};
        api_.cuDeviceGetName(name, sizeof(name), dev_);
        int major = 0, minor = 0, shared = 0;
        api_.cuDeviceGetAttribute(&major, cudadyn::kAttrCcMajor, dev_);
        api_.cuDeviceGetAttribute(&minor, cudadyn::kAttrCcMinor, dev_);
        api_.cuDeviceGetAttribute(&shared, cudadyn::kAttrMaxSharedPerBlock, dev_);
        shared_mem_ = static_cast<size_t>(shared);
        size_t total = 0;
        api_.cuDeviceTotalMem(&total, dev_);
        if (api_.cuCtxCreate(&ctx_, 0, dev_) != 0) return fail("cuCtxCreate failed");

        // Prefer a cubin for this exact GPU; fall back to PTX that the driver JIT-compiles.
        std::vector<char> image;
        std::string error;
        bool loaded = false;
        if (api_.nvrtcGetCUBIN != nullptr &&
            cudadyn::compile(api_, kernel_source, major, minor, true, image, error)) {
            loaded = api_.cuModuleLoadData(&module_, image.data()) == 0;
        }
        if (!loaded) {
            if (!cudadyn::compile(api_, kernel_source, major, minor, false, image, error)) return fail(error);
            const int rc = api_.cuModuleLoadData(&module_, image.data());
            if (rc != 0) {
                return fail("cuModuleLoadData failed (" + std::to_string(rc) +
                            "); NVRTC may be newer than the driver");
            }
        }
        if (api_.cuModuleGetFunction(&k_prep_, module_, "prep") != 0 ||
            api_.cuModuleGetFunction(&k_wg_, module_, "bsgs_wg") != 0) {
            return fail("kernel lookup failed");
        }
        int nv_major = 0, nv_minor = 0;
        api_.nvrtcVersion(&nv_major, &nv_minor);
        log_info(std::string("GPU enabled (CUDA): ") + name + ", compute " + std::to_string(major) + "." +
                 std::to_string(minor) + ", " + std::to_string(total >> 20) + " MB, NVRTC " +
                 std::to_string(nv_major) + "." + std::to_string(nv_minor));
        ok_ = true;
        return true;
    }

    bool ok() const { return ok_; }

    // Same contract as the OpenCL filter: indices of primes that may divide an in-class candidate
    // are appended to `todo`. Returns false if the GPU cannot handle this work item (the caller then
    // checks every prime on the CPU).
    bool filter(const std::vector<uint64_t>& primes, const WorkItem& item, std::vector<std::size_t>& todo) {
        if (primes.empty()) return true;
        if (!ok_) return false;
        api_.cuCtxSetCurrent(ctx_);  // the context is per thread; filter() runs on the GPU worker thread
        const uint64_t N0 = item.N0 < 0 ? 0 : to_uint64(item.N0);
        const uint64_t N1 = to_uint64(item.N1);
        const uint64_t cm = item.cls_mod == 0 ? 1 : item.cls_mod;
        const uint64_t kmax = (N1 - N0) / cm + 2;
        const uint64_t m = static_cast<uint64_t>(std::sqrt(static_cast<double>(kmax))) + 2;
        unsigned tbits = 4;
        while ((1ULL << tbits) < m + m / 4) ++tbits;  // load <= 80%, same sizing as the OpenCL path
        const size_t lbytes = (size_t(1) << tbits) * sizeof(uint32_t);
        if (lbytes + 1024 > shared_mem_ || tbits > 14) {
            if (!warned_table_) {
                log_info("CUDA: BSGS table does not fit in shared memory for this range, using CPU");
                warned_table_ = true;
            }
            return false;
        }

        const size_t batch = 262144;
        if (cap_ < batch) {
            release();
            if (api_.cuMemAlloc(&d_ent_, batch * 7 * sizeof(uint64_t)) != 0 ||
                api_.cuMemAlloc(&d_cnt_, sizeof(uint32_t)) != 0 ||
                api_.cuMemAlloc(&d_primes_, batch * sizeof(uint64_t)) != 0 ||
                api_.cuMemAlloc(&d_flags_, batch) != 0) {
                return fail("cuMemAlloc failed");
            }
            cap_ = batch;
        }

        std::vector<uint8_t> flags(batch);
        unsigned long long b = to_uint64(item.b);
        long long c = static_cast<long long>(item.c);
        unsigned long long n0 = N0, n1 = N1, cmod = cm, cres = item.cls_res;
        unsigned int tb = tbits;
        for (size_t off = 0; off < primes.size(); off += batch) {
            unsigned int count = static_cast<unsigned int>(std::min(batch, primes.size() - off));
            if (api_.cuMemcpyHtoD(d_primes_, primes.data() + off, count * sizeof(uint64_t)) != 0 ||
                api_.cuMemsetD32(d_cnt_, 0, 1) != 0) {
                return fail("upload failed");
            }
            void* prep_args[] = {&d_primes_, &d_flags_, &d_ent_, &d_cnt_, &b, &c, &n0, &n1, &cmod, &cres, &count};
            if (api_.cuLaunchKernel(k_prep_, (count + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr, prep_args, nullptr) != 0) {
                return fail("prep launch failed");
            }
            uint32_t n_ent = 0;
            if (api_.cuMemcpyDtoH(&n_ent, d_cnt_, sizeof(n_ent)) != 0) return fail("count read failed");
            if (n_ent > 0) {
                void* wg_args[] = {&d_ent_, &d_flags_, &tb};
                if (api_.cuLaunchKernel(k_wg_, n_ent, 1, 1, 64, 1, 1, static_cast<unsigned>(lbytes), nullptr,
                                        wg_args, nullptr) != 0) {
                    return fail("bsgs_wg launch failed");
                }
            }
            if (api_.cuMemcpyDtoH(flags.data(), d_flags_, count) != 0) return fail("flags read failed");
            for (unsigned int i = 0; i < count; ++i) {
                if (flags[i] != 0) todo.push_back(off + i);
            }
        }
        return true;
    }

private:
    void release() {
        if (d_ent_) api_.cuMemFree(d_ent_);
        if (d_cnt_) api_.cuMemFree(d_cnt_);
        if (d_primes_) api_.cuMemFree(d_primes_);
        if (d_flags_) api_.cuMemFree(d_flags_);
        d_ent_ = d_cnt_ = d_primes_ = d_flags_ = 0;
        cap_ = 0;
    }
    bool fail(const std::string& msg) {
        log_info("CUDA unavailable: " + msg);
        ok_ = false;
        return false;
    }

    cudadyn::Api api_;
    bool ok_ = false;
    bool warned_table_ = false;
    cudadyn::CUdevice dev_ = 0;
    cudadyn::CUcontext ctx_ = nullptr;
    cudadyn::CUmodule module_ = nullptr;
    cudadyn::CUfunction k_prep_ = nullptr;
    cudadyn::CUfunction k_wg_ = nullptr;
    size_t shared_mem_ = 0;
    cudadyn::CUdeviceptr d_ent_ = 0, d_cnt_ = 0, d_primes_ = 0, d_flags_ = 0;
    size_t cap_ = 0;
};

}  // namespace siever
