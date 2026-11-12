#pragma once

#include "Logger.hpp"
#include "Types.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace siever {

struct PrimeConfig {
    int cpu_count = 0;
    int memory_allowed_mb = 10240;
    int use_gpu = 0;      // 11.4: UseGPU=1 adds an OpenCL GPU worker
    int gpu_device = 0;   // 11.4: GPU_device=index
};

inline std::string trim(std::string value) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

inline int detect_cpu_count() {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    return hw > 0 ? hw : 1;
}

inline bool parse_key_value(const std::string& line, std::string& key, std::string& value) {
    const auto pos = line.find('=');
    if (pos == std::string::npos) {
        return false;
    }
    key = trim(line.substr(0, pos));
    value = trim(line.substr(pos + 1));
    return !key.empty();
}

inline PrimeConfig load_prime_config(const std::string& path) {
    PrimeConfig config;
    config.cpu_count = detect_cpu_count();

    std::ifstream in(path);
    bool has_cpu = false;
    bool has_memory = false;

    if (in) {
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::string key;
            std::string value;
            if (!parse_key_value(line, key, value)) {
                continue;
            }

            if (key == "CPU_count") {
                config.cpu_count = std::stoi(value);
                has_cpu = true;
            } else if (key == "UseGPU") {
                config.use_gpu = std::stoi(value);
            } else if (key == "GPU_device") {
                config.gpu_device = std::stoi(value);
            } else if (key == "MemoryAllowed") {
                config.memory_allowed_mb = std::stoi(value);
                has_memory = true;
            }
        }
    }

    if (config.cpu_count == 0 && config.use_gpu != 0) {
        // 11.4: GPU-only sieving (leave the CPU cores to PRP testing)
    } else if (config.cpu_count <= 0) {
        config.cpu_count = detect_cpu_count();
        has_cpu = false;
    }
    if (config.memory_allowed_mb <= 0) {
        config.memory_allowed_mb = 10240;
        has_memory = false;
    }

    const bool file_existed = static_cast<bool>(std::ifstream(path));
    if (!has_cpu || !has_memory || !file_existed) {
        std::ofstream out(path, std::ios::trunc);
        if (!out) {
            log_fatal("Failed to write prime.ini at " + path);
        }
        out << "CPU_count=" << config.cpu_count << "\n";
        out << "MemoryAllowed=" << config.memory_allowed_mb << "\n";
        out << "UseGPU=" << config.use_gpu << "\n";
        out << "GPU_device=" << config.gpu_device << "\n";
        log_info("Wrote prime.ini with CPU_count=" + std::to_string(config.cpu_count) +
                 " MemoryAllowed=" + std::to_string(config.memory_allowed_mb) + " MB");
    }

    return config;
}

inline std::vector<WorkItem> load_worktodo(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        log_fatal("Failed to open worktodo file: " + path);
    }

    std::vector<WorkItem> items;
    std::string line;
    std::size_t line_no = 0;

    while (std::getline(in, line)) {
        ++line_no;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        constexpr const char* prefix = "Factor=";
        if (line.rfind(prefix, 0) == 0) {
            line = line.substr(std::char_traits<char>::length(prefix));
        }

        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (!token.empty()) {
                parts.push_back(token);
            }
        }

        if (parts.size() != 6 && parts.size() != 8) {
            log_fatal("worktodo line " + std::to_string(line_no) +
                      " must have 6 or 8 comma-separated values (b,c,P0,P1,N0,N1[,mod,res])");
        }

        WorkItem item;
        try {
            item.b = Int128(parts[0]);
            item.c = Int128(parts[1]);
            item.P0 = Int128(parts[2]);
            item.P0_original = item.P0;
            item.P1 = Int128(parts[3]);
            item.N0 = Int128(parts[4]);
            item.N1 = Int128(parts[5]);
            if (parts.size() == 8) {
                item.cls_mod = std::stoull(parts[6]);
                item.cls_res = std::stoull(parts[7]);
                if (item.cls_mod == 0 || item.cls_res >= item.cls_mod) {
                    throw std::invalid_argument("class must satisfy 0 <= res < mod");
                }
            }
        } catch (const std::exception& ex) {
            log_fatal("worktodo line " + std::to_string(line_no) + " parse error: " + ex.what());
        }

        if (item.P0 >= item.P1) {
            log_fatal("worktodo line " + std::to_string(line_no) + " requires P0 < P1");
        }
        if (item.N0 >= item.N1) {
            log_fatal("worktodo line " + std::to_string(line_no) + " requires N0 < N1");
        }
        const Int128 range = item.N1 - item.N0;
        if (range > Int128(10'000'000)) {
            log_fatal("worktodo line " + std::to_string(line_no) + " range N1-N0 exceeds 10 million");
        }

        items.push_back(item);
    }

    if (items.empty()) {
        log_fatal("No work items found in " + path);
    }

    return items;
}

}  // namespace siever
