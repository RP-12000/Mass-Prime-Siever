#include "Config.hpp"
#include "Logger.hpp"
#include "Results.hpp"
#include "Sieve.hpp"
#include "Worker.hpp"
#include "Gpu.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#define NOMINMAX
#include <windows.h>

// ---------------------------------------------------------------------------
// Globals used by the Windows console control handler
// ---------------------------------------------------------------------------
namespace {
    using Clock = std::chrono::steady_clock;

    // Written by main(), read by the close handler.
    Clock::time_point g_program_start;
    std::atomic<bool> g_start_valid{false};

    // Previous-session total read from the log, set before workers start.
    double g_previous_total_seconds = 0.0;

    // Signals the close handler to return after log lines are written.
    std::atomic<bool> g_shutdown_logged{false};
}

// Writes the two elapsed-time lines to the log.
static void log_elapsed_times()
{
    const double session_seconds =
        std::chrono::duration<double>(Clock::now() - g_program_start).count();
    const double total_seconds = g_previous_total_seconds + session_seconds;
    siever::Logger::instance().write_elapsed(session_seconds, total_seconds);
    g_shutdown_logged.store(true);
}

BOOL WINAPI ConsoleCtrlHandler(DWORD ctrl_type)
{
    if (ctrl_type == CTRL_CLOSE_EVENT) {
        // Ask workers to stop after their current chunk.
        siever::shutdown_requested.store(true);

        // Wait up to ~10 seconds for main() to write the log lines.
        for (int i = 0; i < 100 && !g_shutdown_logged.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        return TRUE;
    }
    return FALSE;
}

int main() {

    using namespace siever;

    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

    try {

        g_program_start = Clock::now();
        g_start_valid.store(true);

        log_info("Mass Prime Siever 12.0 starting (November 11, 2026)");

        Logger::instance().init("siever.log");

        // Read previous total elapsed from last session before writing anything.
        g_previous_total_seconds = Logger::instance().read_previous_total_elapsed();
        if (g_previous_total_seconds > 0.0) {
            log_info("Resuming session. Previous total elapsed: " +
                     std::to_string(static_cast<long long>(g_previous_total_seconds)) + " s");
        }

        const PrimeConfig config = load_prime_config("prime.ini");

        std::vector<WorkItem> work_items = load_worktodo("worktodo.txt");

        log_info("Loaded " + std::to_string(work_items.size()) + " work item(s)");
        log_info("CPU_count=" + std::to_string(config.cpu_count) +
                 " MemoryAllowed=" + std::to_string(config.memory_allowed_mb) + " MB");

        siever::GpuSiever gpu;
        siever::GpuSiever* gpu_ptr = nullptr;
        if (config.use_gpu != 0 && gpu.init(config.gpu_device)) {
            gpu_ptr = &gpu;
        }

        for (std::size_t work_index = 0; work_index < work_items.size(); ++work_index) {

            if (shutdown_requested.load()) {
                log_info("Shutdown requested, stopping before next work item.");
                break;
            }

            WorkItem& item = work_items[work_index];

            log_info("Processing work item " + std::to_string(work_index + 1) + "/" +
                     std::to_string(work_items.size()) + " b=" + to_string(item.b) + " c=" + to_string(item.c));

            remove_file_if_exists("relations.p");

            siever::read_checkpoint(item);

            const uint64_t sieve_limit = sieve_limit_for(item);
            log_info("Sieving small primes up to " + std::to_string(sieve_limit));
            const std::vector<uint32_t> small_primes = sieve_small_primes(sieve_limit);

            run_worker_pool(item, config.cpu_count, config.memory_allowed_mb, small_primes, gpu_ptr);

            if (shutdown_requested.load()) {
                log_info("Shutdown: checkpoint preserved in core_*.p files");
                break;
            }

            log_info("Merging core files");
            merge_core_files(config.cpu_count + 1);

            const std::vector<Relation> relations = load_relations("relations.p");
            log_info("Loaded " + std::to_string(relations.size()) + " relation(s)");

            std::vector<std::vector<uint64_t>> candidates;
            apply_relations(item, relations, candidates);
            append_results_json(item, candidates);

            const double item_elapsed =
                std::chrono::duration<double>(Clock::now() - g_program_start).count();
            log_final_status(item, candidates, item_elapsed);
            log_info("Finished work item " + std::to_string(work_index + 1));

            for (int id = 0; id <= config.cpu_count; ++id) {
                remove_file_if_exists("core_" + std::to_string(id) + ".p");
            }
        }

        if (!shutdown_requested.load()) {
            const double total_elapsed =
                std::chrono::duration<double>(Clock::now() - g_program_start).count();
            log_info("Siever completed successfully, total elapsed " + format_elapsed(total_elapsed));
        }

        // Always write session/total elapsed lines at exit.
        log_elapsed_times();

        if (!shutdown_requested.load()) {
            std::cout << "\nPress Enter to exit..." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin.get();
        }

        return 0;

    } catch (const std::exception& ex) {
        log_fatal(std::string("Unhandled exception: ") + ex.what());
    }
}


