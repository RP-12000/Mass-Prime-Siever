#pragma once

#include <boost/date_time/posix_time/posix_time.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace siever {

// Format a boost ptime as "YYYY-MM-DD HH:MM:SS UTC"
inline std::string format_utc_timestamp(const boost::posix_time::ptime& t) {
    const auto& d = t.date();
    const auto& tod = t.time_of_day();
    std::ostringstream out;
    out << std::setfill('0')
        << d.year() << '-'
        << std::setw(2) << static_cast<int>(d.month()) << '-'
        << std::setw(2) << static_cast<int>(d.day()) << ' '
        << std::setw(2) << tod.hours() << ':'
        << std::setw(2) << tod.minutes() << ':'
        << std::setw(2) << tod.seconds()
        << " UTC+0";
    return out.str();
}

class Logger {
public:
    static Logger& instance() {
        static Logger log;
        return log;
    }

    void init(const std::string& log_path = "siever.log") { 
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            return;
        }
        log_path_ = log_path;
        log_file_.open(log_path, std::ios::app);
        if (!log_file_) {
            throw std::runtime_error("Failed to open log file: " + log_path);
        }
        initialized_ = true;
    }

    void info(const std::string& message) { write("INFO", message); }
    void warn(const std::string& message) { write("WARN", message); }

    [[noreturn]] void fatal(const std::string& message) {
        write("FATAL", message);
        std::exit(1);
    }

    // Write the session/total elapsed time lines, persist total to checkpoint file, and flush.
    void write_elapsed(double session_seconds, double total_seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = boost::posix_time::second_clock::universal_time();
        const std::string ts = format_utc_timestamp(now);

        auto make_line = [&](const std::string& label, double secs) -> std::string {
            std::ostringstream line;
            line << ts << " [INFO] " << label << std::fixed << std::setprecision(0) << secs << " s";
            return line.str();
        };

        const std::string session_line = make_line("Current Session Elapsed: ", session_seconds);
        const std::string total_line   = make_line("Total elapsed time: ", total_seconds);

        std::cerr << session_line << std::endl;
        std::cerr << total_line   << std::endl;
        if (initialized_) {
            log_file_ << session_line << '\n';
            log_file_ << total_line   << '\n';
            log_file_.flush();
        }

        write_elapsed_checkpoint(total_seconds);
    }

    // Read persisted total elapsed seconds from the checkpoint file.
    // Falls back to scanning the log once for runs that predate the checkpoint file.
    double read_previous_total_elapsed() const {
        if (const double from_checkpoint = read_elapsed_checkpoint(); from_checkpoint > 0.0) {
            return from_checkpoint;
        }
        return read_elapsed_from_log();
    }

private:
    Logger() = default;

    static constexpr const char* elapsed_checkpoint_path_ = "elapsed.checkpoint";

    static void write_elapsed_checkpoint(double total_seconds) {
        std::ofstream out(elapsed_checkpoint_path_, std::ios::trunc);
        if (!out) {
            return;
        }
        out << std::fixed << std::setprecision(0) << total_seconds << '\n';
    }

    static double read_elapsed_checkpoint() {
        std::ifstream in(elapsed_checkpoint_path_);
        if (!in) {
            return 0.0;
        }
        double value = 0.0;
        if (!(in >> value) || value < 0.0) {
            return 0.0;
        }
        return value;
    }

    double read_elapsed_from_log() const {
        if (log_path_.empty()) {
            return 0.0;
        }
        std::ifstream in(log_path_);
        if (!in) {
            return 0.0;
        }

        // Pattern: ... [INFO] Total elapsed time: <number>[ s]
        const std::regex pattern(R"(\[INFO\] Total elapsed time:\s+([\d.]+)(?:\s+s)?)");
        double last_found = 0.0;
        std::string line;
        while (std::getline(in, line)) {
            std::smatch m;
            if (std::regex_search(line, m, pattern)) {
                try {
                    last_found = std::stod(m[1].str());
                } catch (...) {}
            }
        }
        return last_found;
    }

    void write(const char* level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = boost::posix_time::second_clock::universal_time();
        std::ostringstream line;
        line << format_utc_timestamp(now) << " [" << level << "] " << message;

        std::cerr << line.str() << std::endl;
        if (initialized_) {
            log_file_ << line.str() << std::endl;
            log_file_.flush();
        }
    }

    std::mutex mutex_;
    std::ofstream log_file_;
    std::string log_path_;
    bool initialized_ = false;
};

inline void log_info(const std::string& message) { Logger::instance().info(message); }
inline void log_warn(const std::string& message) { Logger::instance().warn(message); }
[[noreturn]] inline void log_fatal(const std::string& message) { Logger::instance().fatal(message); }

}  // namespace siever
