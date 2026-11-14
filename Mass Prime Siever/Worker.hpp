#pragma once

#include "Gpu.hpp"
#include "Logger.hpp"
#include "Math.hpp"
#include "Sieve.hpp"
#include "Types.hpp"

#include <atomic>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace siever {

// Global shutdown flag - set by the console control handler to ask workers
// to finish their current chunk and exit cleanly.
inline std::atomic<bool> shutdown_requested{false};

struct PrimeChunk {
    UInt128 low{};
    UInt128 high{};
};

class ChunkQueue {
public:
    void push(PrimeChunk chunk) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            chunks_.push(std::move(chunk));
        }
        cv_.notify_one();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_all();
    }

    std::optional<PrimeChunk> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        // Use a timed wait so we wake up periodically to check shutdown_requested.
        while (!done_ && chunks_.empty() && !shutdown_requested.load(std::memory_order_relaxed)) {
            cv_.wait_for(lock, std::chrono::milliseconds(50));
        }
        if (chunks_.empty() || shutdown_requested.load(std::memory_order_relaxed)) {
            return std::nullopt;
        }
        PrimeChunk chunk = chunks_.front();
        chunks_.pop();
        return chunk;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<PrimeChunk> chunks_;
    bool done_ = false;
};

class WorkerProgress {
public:
    WorkerProgress(std::size_t total_chunks, std::uint64_t initial_relations = 0)
        : total_chunks_(total_chunks),
          global_relations_(initial_relations),
          start_time_(boost::posix_time::microsec_clock::universal_time()) {}

    void on_chunk_complete(std::size_t relations_in_chunk) {
        const std::uint64_t global_chunks = global_chunks_.fetch_add(1, std::memory_order_relaxed) + 1;
        const std::uint64_t global_relations =
            global_relations_.fetch_add(relations_in_chunk, std::memory_order_relaxed) +
            relations_in_chunk;

        std::ostringstream message;
        message << "global " << global_chunks << "/" << total_chunks_ << " chunks ("
                << format_percent(global_chunks, total_chunks_) << "%), +" << relations_in_chunk
                << " relations, " << global_relations << " total relations, ETA "
                << estimate_finish_time(global_chunks);

        log_info(message.str());
    }

    [[nodiscard]] std::uint64_t total_relations() const {
        return global_relations_.load(std::memory_order_relaxed);
    }

private:
    static std::string format_utc_datetime(const boost::posix_time::ptime& timestamp) {
        const auto& date = timestamp.date();
        const auto time = timestamp.time_of_day();
        std::ostringstream out;
        out << std::setfill('0') << date.year() << '-' << std::setw(2) << date.month().as_number() << '-'
            << std::setw(2) << date.day() << ' ' << std::setw(2) << time.hours() << '-' << std::setw(2)
            << time.minutes() << '-' << std::setw(2) << time.seconds() << " UTC+0";
        return out.str();
    }

    static std::string format_percent(std::uint64_t completed, std::size_t total) {
        if (total == 0) {
            return "100.00";
        }
        std::ostringstream out;
        out << std::fixed << std::setprecision(2)
            << (100.0 * static_cast<double>(completed) / static_cast<double>(total));
        return out.str();
    }

    std::string estimate_finish_time(std::uint64_t completed) const {
        if (completed == 0) {
            return "unknown";
        }
        if (completed >= total_chunks_) {
            return "complete";
        }

        const auto now = boost::posix_time::microsec_clock::universal_time();
        const double elapsed_seconds =
            static_cast<double>((now - start_time_).total_microseconds()) / 1'000'000.0;
        if (elapsed_seconds <= 0.0) {
            return "unknown";
        }

        const double seconds_per_chunk = elapsed_seconds / static_cast<double>(completed);
        const double remaining_seconds =
            seconds_per_chunk * static_cast<double>(total_chunks_ - completed);
        const auto eta = now + boost::posix_time::microsec(static_cast<long long>(remaining_seconds * 1'000'000.0));
        return format_utc_datetime(eta);
    }

    std::size_t total_chunks_;
    std::atomic<std::uint64_t> global_chunks_{0};
    std::atomic<std::uint64_t> global_relations_{0};
    boost::posix_time::ptime start_time_;
};

inline std::size_t worker_chunk_size_bytes(int memory_allowed_mb, int cpu_count) {
    const unsigned long long total_bytes =
        static_cast<unsigned long long>(memory_allowed_mb) * 1024ULL * 1024ULL;
    const unsigned long long per_worker = std::max(1ULL, total_bytes / static_cast<unsigned long long>(std::max(1, cpu_count)));
    return static_cast<std::size_t>(per_worker);
}

inline std::size_t count_prime_chunks(const WorkItem& item, std::size_t chunk_size) {
    std::size_t count = 0;
    UInt128 cursor = static_cast<UInt128>(item.P0);
    const UInt128 end = static_cast<UInt128>(item.P1);

    while (cursor < end) {
        UInt128 remaining = end - cursor;
        UInt128 step = remaining;
        if (static_cast<UInt128>(chunk_size) < remaining) {
            step = static_cast<UInt128>(chunk_size);
        }
        ++count;
        cursor += step;
    }
    return count;
}

inline std::size_t enqueue_prime_chunks(const WorkItem& item, std::size_t chunk_size, ChunkQueue& queue) {
    std::size_t count = 0;
    UInt128 cursor = static_cast<UInt128>(item.P0);
    const UInt128 end = static_cast<UInt128>(item.P1);

    while (cursor < end) {
        UInt128 remaining = end - cursor;
        UInt128 step = remaining;
        if (static_cast<UInt128>(chunk_size) < remaining) {
            step = static_cast<UInt128>(chunk_size);
        }
        queue.push(PrimeChunk{cursor, cursor + step});
        ++count;
        cursor += step;
    }
    queue.finish();
    return count;
}

inline std::size_t process_chunk(const PrimeChunk& chunk, int thread_id, const WorkItem& item,
                                 const std::vector<uint32_t>& small_primes, std::atomic<bool>& failed,
                                 GpuSiever* gpu = nullptr) {
    if (failed.load()) {
        return 0;
    }

    const std::string output_path = "core_" + std::to_string(thread_id) + ".p";
    std::vector<Relation> relations;
    std::vector<uint64_t> primes;
    primes.reserve(1'000'000);

    try {
        segmented_sieve(chunk.low, chunk.high, small_primes, primes);

        std::mt19937_64 rng(static_cast<uint64_t>(thread_id + 1) ^
                            static_cast<uint64_t>(chunk.low & UInt128(0xFFFFFFFFFFFFFFFFULL)));

        if (gpu != nullptr) {
            // 11.4: GPU pre-filter, exact CPU re-check of the (few) flagged primes.
            std::vector<std::size_t> todo;
            if (gpu->filter(primes, item, todo)) {
                std::vector<uint64_t> flagged;
                flagged.reserve(todo.size());
                for (const std::size_t i : todo) flagged.push_back(primes[i]);
                primes.swap(flagged);
            }
        }

        for (const uint64_t prime : primes) {
            // Check for shutdown between each prime so we exit quickly
            // and flush whatever relations have already been found.
            if (shutdown_requested.load(std::memory_order_relaxed)) {
                break;
            }
            Relation relation;
            if (run_bsgs(prime, item, relation, rng)) {
                relations.push_back(relation);
            }
        }
    } catch (const std::exception& ex) {
        log_fatal(std::string("Worker ") + std::to_string(thread_id) + " failed: " + ex.what());
        failed.store(true);
        return 0;
    }

    if (!relations.empty()) {
        FILE* file = open_file(output_path.c_str(), "ab");
        if (file == nullptr) {
            log_fatal("Failed to append to " + output_path);
        }

        for (const Relation& relation : relations) {
            std::fprintf(file, "%llu, %llu, %llu\n",
                         static_cast<unsigned long long>(relation.p),
                         static_cast<unsigned long long>(relation.M),
                         static_cast<unsigned long long>(relation.q));
        }
        std::fclose(file);
    }

    return relations.size();
}

inline void run_worker_pool(const WorkItem& item, int cpu_count, int memory_allowed_mb,
                            const std::vector<uint32_t>& small_primes, GpuSiever* gpu = nullptr) {
    ChunkQueue queue;
    const std::size_t chunk_size = worker_chunk_size_bytes(memory_allowed_mb, cpu_count);
    const std::size_t total_chunks = count_prime_chunks(item, chunk_size);
    log_info("Chunk size per task: " + std::to_string(chunk_size) + " integers, " +
             std::to_string(total_chunks) + " total chunks queued");

    WorkerProgress progress(total_chunks, item.relations_restored);

    std::thread producer([&] { enqueue_prime_chunks(item, chunk_size, queue); });

    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(cpu_count));

    for (int thread_id = 0; thread_id < cpu_count; ++thread_id) {
        workers.emplace_back([&, thread_id]() {
            while (!failed.load() && !shutdown_requested.load()) {
                const auto chunk = queue.pop();
                if (!chunk) {
                    break;
                }
                const std::size_t relations_found =
                    process_chunk(*chunk, thread_id, item, small_primes, failed);
                if (!failed.load()) {
                    progress.on_chunk_complete(relations_found);
                }
            }
        });
    }

    if (gpu != nullptr && gpu->ok()) {
        const int gpu_id = cpu_count;  // core_<cpu_count>.p
        workers.emplace_back([&, gpu_id]() {
            while (!failed.load() && !shutdown_requested.load()) {
                const auto chunk = queue.pop();
                if (!chunk) {
                    break;
                }
                const std::size_t relations_found =
                    process_chunk(*chunk, gpu_id, item, small_primes, failed, gpu);
                if (!failed.load()) {
                    progress.on_chunk_complete(relations_found);
                }
            }
        });
    }

    producer.join();
    for (std::thread& worker : workers) {
        worker.join();
    }

    if (failed.load()) {
        log_fatal("Worker pool failed");
    }

    log_info("Worker pool complete, " + std::to_string(progress.total_relations()) + " total relations");
}

inline void read_checkpoint(WorkItem& item) {
    const Int128 max_p1 = item.P1;
    Int128 new_p0 = max_p1;
    bool any_valid = false;

    for (int id = 0; ; ++id) {
        std::string core_path = "core_" + std::to_string(id) + ".p";
        std::ifstream in(core_path);
        if (!in) {
            std::string next_path = "core_" + std::to_string(id + 1) + ".p";
            std::ifstream next_in(next_path);
            if (!next_in) break;
            continue;
        }

        std::string line;
        std::string last_line;
        while (std::getline(in, line)) {
            if (!line.empty()) {
                last_line = line;
            }
        }
        in.close();

        if (last_line.empty()) {
            continue;
        }

        std::stringstream ss(last_line);
        unsigned long long p_val = 0;
        char comma = 0;
        if (ss >> p_val && ss >> comma && comma == ',') {
            Int128 p128 = static_cast<Int128>(p_val);
            if (p128 < new_p0) {
                new_p0 = p128;
                any_valid = true;
            }
        }
    }

    if (!any_valid) {
        log_info("No checkpoint found, starting from original P0.");
        return;
    }

    if (new_p0 <= item.P0) {
        log_info("Checkpoint not ahead of current P0, starting from original P0.");
        return;
    }

    item.P0 = new_p0;
    std::uint64_t total_relations = 0;

    for (int id = 0; ; ++id) {
        std::string core_path = "core_" + std::to_string(id) + ".p";
        std::ifstream in(core_path);
        if (!in) {
            std::string next_path = "core_" + std::to_string(id + 1) + ".p";
            std::ifstream next_in(next_path);
            if (!next_in) break;
            continue;
        }

        std::vector<std::string> kept_lines;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            unsigned long long p_val = 0;
            char comma = 0;
            if (ss >> p_val && ss >> comma && comma == ',') {
                if (static_cast<Int128>(p_val) < new_p0) {
                    kept_lines.push_back(line);
                }
                // else drop this relation
            }
        }
        in.close();

        std::ofstream out(core_path, std::ios::trunc);
        if (!out) {
            log_fatal("Failed to trim checkpoint file: " + core_path);
        }
        for (const auto& l : kept_lines) {
            out << l << '\n';
        }
        total_relations += kept_lines.size();
        out.close();
    }

    item.relations_restored = total_relations;

    log_info("Checkpoint recovered: new P0 = " + to_string(item.P0) +
        ", relations restored = " + std::to_string(total_relations));
}

inline void remove_file_if_exists(const std::string& path) {
    std::remove(path.c_str());
}

inline void merge_core_files(int cpu_count) {
    std::ofstream merged("relations.p", std::ios::binary | std::ios::trunc);
    if (!merged) {
        log_fatal("Failed to create relations.p");
    }

    for (int id = 0; id < cpu_count; ++id) {
        const std::string core_path = "core_" + std::to_string(id) + ".p";
        std::ifstream core(core_path, std::ios::binary);
        if (!core) {
            continue;
        }
        merged << core.rdbuf();
    }

    merged.flush();
    if (!merged) {
        log_fatal("Failed while writing relations.p");
    }
    merged.close();

    for (int id = 0; id < cpu_count; ++id) {
        remove_file_if_exists("core_" + std::to_string(id) + ".p");
    }
}

inline std::vector<Relation> load_relations(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }

    std::vector<Relation> relations;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        Relation relation;
        std::stringstream parser(line);
        char comma = 0;
        unsigned long long p = 0;
        unsigned long long M = 0;
        unsigned long long q = 0;
        if (!(parser >> p) || !(parser >> comma) || comma != ',' || !(parser >> M) || !(parser >> comma) ||
            comma != ',' || !(parser >> q)) {
            log_fatal("Malformed relation line: " + line);
        }
        relation.p = static_cast<uint64_t>(p);
        relation.M = static_cast<uint64_t>(M);
        relation.q = static_cast<uint64_t>(q);
        relations.push_back(relation);
    }
    return relations;
}

inline void apply_relations(const WorkItem& item, const std::vector<Relation>& relations,
                            std::vector<std::vector<uint64_t>>& candidates) {
    const std::size_t count = static_cast<std::size_t>(item.N1 - item.N0);
    candidates.assign(count, {});

    for (const Relation& relation : relations) {
        const Int128 q = Int128(relation.q);
        const Int128 M = Int128(relation.M);
        const Int128 x0 = ceil_div(item.N0 - M, q);
        const Int128 x1 = floor_div(item.N1 - 1 - M, q);

        for (Int128 x = x0; x <= x1; ++x) {
            const Int128 n = q * x + M;
            const Int128 index = n - item.N0;
            if (index < 0 || index >= Int128(count)) {
                continue;
            }
            // 12.0: do not mark b^n+c as factored when it IS the prime p (e.g. 10^2+111 = 211).
            {
                Int128 value = 1;
                bool fits = true;
                for (Int128 e = 0; e < n && fits; ++e) {
                    value *= item.b;
                    if (value > Int128(relation.p) * 2) fits = false;
                }
                if (fits && value + item.c == Int128(relation.p)) {
                    continue;
                }
            }
            candidates[static_cast<std::size_t>(index)].push_back(relation.p);
        }
    }
}

}  // namespace siever
