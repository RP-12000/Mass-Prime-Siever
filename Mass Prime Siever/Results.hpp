#pragma once

#include "Logger.hpp"
#include "Types.hpp"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/json.hpp>

#include <limits>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace siever {

inline std::string format_elapsed(double seconds) {
    const auto total_seconds = static_cast<std::int64_t>(seconds);
    const auto hours = total_seconds / 3600;
    const auto minutes = (total_seconds % 3600) / 60;
    const auto secs = total_seconds % 60;
    std::ostringstream out;
    out << std::setfill('0') << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2) << secs;
    return out.str();
}

inline std::size_t count_survivors(const std::vector<std::vector<uint64_t>>& candidates) {
    std::size_t survivors = 0;
    for (const auto& factors : candidates) {
        if (factors.empty()) {
            ++survivors;
        }
    }
    return survivors;
}

inline void log_final_status(const WorkItem& item, const std::vector<std::vector<uint64_t>>& candidates,
                             double elapsed_seconds) {
    std::size_t total = 0;
    std::size_t survivors = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!in_class(item, item.N0 + Int128(i))) continue;
        ++total;
        if (candidates[i].empty()) ++survivors;
    }
    const double survival_rate = total == 0 ? 0.0 : 100.0 * static_cast<double>(survivors) / static_cast<double>(total);

    std::ostringstream message;
    message << "Final status: " << survivors << '/' << total << " candidates survived ("
            << std::fixed << std::setprecision(2) << survival_rate
            << "% survival rate), elapsed " << format_elapsed(elapsed_seconds);
    log_info(message.str());
}

inline boost::json::value json_int(Int128 value) {
    if (value > Int128((std::numeric_limits<std::int64_t>::max)()) ||
        value < Int128((std::numeric_limits<std::int64_t>::min)())) {
        return boost::json::value(to_string(value));
    }
    return boost::json::value(static_cast<std::int64_t>(value));
}

inline boost::json::value json_uint(uint64_t value) {
    if (value > static_cast<uint64_t>((std::numeric_limits<std::int64_t>::max)())) {
        return boost::json::value(std::to_string(value));
    }
    return boost::json::value(static_cast<std::int64_t>(value));
}

inline void append_results_json(const WorkItem& item, const std::vector<std::vector<uint64_t>>& candidates) {
    boost::json::object entry;
    const auto now = boost::posix_time::second_clock::universal_time();
    entry["Timestamp"] = format_utc_timestamp(now);
    entry["b"] = json_int(item.b);
    entry["c"] = json_int(item.c);
    entry["P0"] = json_int(item.P0_original);
    entry["P1"] = json_int(item.P1);
    entry["N0"] = json_int(item.N0);
    entry["N1"] = json_int(item.N1);

    boost::json::array prp;
    boost::json::array factored;

    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const Int128 n = item.N0 + Int128(index);
        if (!in_class(item, n)) {
            continue;
        }
        if (candidates[index].empty()) {
            prp.push_back(json_int(n));
            continue;
        }

        boost::json::object factored_entry;
        factored_entry["n"] = json_int(n);
        boost::json::array factors;
        for (const uint64_t factor : candidates[index]) {
            factors.push_back(json_uint(factor));
        }
        factored_entry["factors"] = std::move(factors);
        factored.push_back(std::move(factored_entry));
    }

    entry["prp"] = std::move(prp);
    entry["factored"] = std::move(factored);

    boost::json::array all_results;
    {
        std::ifstream in("results.json");
        if (in) {
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (!content.empty()) {
                try {
                    const boost::json::value parsed = boost::json::parse(content);
                    if (parsed.is_array()) {
                        all_results = parsed.as_array();
                    } else {
                        log_fatal("results.json must contain a JSON array");
                    }
                } catch (const std::exception& ex) {
                    log_fatal(std::string("Failed to parse results.json: ") + ex.what());
                }
            }
        }
    }

    all_results.push_back(std::move(entry));

    std::ofstream out("results.json", std::ios::trunc);
    if (!out) {
        log_fatal("Failed to write results.json");
    }
    out << boost::json::serialize(all_results) << '\n';
    out.flush();
    if (!out) {
        log_fatal("Failed while saving results.json");
    }
}

}  // namespace siever
