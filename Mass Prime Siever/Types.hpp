#pragma once

#include <boost/lexical_cast.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace siever {

using Int128 = boost::multiprecision::int128_t;
using UInt128 = boost::multiprecision::uint128_t;

struct WorkItem {
    Int128 b{};
    Int128 c{};
    Int128 P0{};
    Int128 P0_original{};  // original value from worktodo.txt (unchanged by checkpoint recovery)
    Int128 P1{};
    Int128 N0{};
    Int128 N1{};
    std::uint64_t relations_restored{};  // relations recovered from checkpoint at session start
    // 11.4: optional congruence class: only n == cls_res (mod cls_mod) are sieved/reported.
    std::uint64_t cls_mod{1};
    std::uint64_t cls_res{0};
};

inline bool in_class(const WorkItem& item, Int128 n) {
    if (item.cls_mod <= 1) return true;
    return static_cast<std::uint64_t>(n % Int128(item.cls_mod)) == item.cls_res;
}

inline std::string to_string(Int128 value) {
    return boost::lexical_cast<std::string>(value);
}

inline std::string to_string(UInt128 value) {
    return boost::lexical_cast<std::string>(value);
}

}  // namespace siever
