cmake_minimum_required(VERSION 3.20)
project(MassPrimeSiever LANGUAGES CXX)

# Builds on Windows (MSVC + vcpkg) and Linux (GCC/Clang). The Visual Studio solution
# (Mass Prime Siever.slnx) is kept and still works on its own.
#   Linux:   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#   Windows: cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#            cmake --build build --config Release
option(SIEVER_OPENCL "Build the OpenCL GPU worker" ON)
option(SIEVER_NATIVE "Optimize for the build machine's CPU (-march=native)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS json)

add_executable(mass-prime-siever "Mass Prime Siever/Siever.cpp")
target_link_libraries(mass-prime-siever PRIVATE Threads::Threads Boost::boost Boost::json)

if(SIEVER_OPENCL)
    find_package(OpenCL REQUIRED)
    target_link_libraries(mass-prime-siever PRIVATE OpenCL::OpenCL)
else()
    target_compile_definitions(mass-prime-siever PRIVATE SIEVER_NO_OPENCL)
endif()

if(MSVC)
    target_compile_options(mass-prime-siever PRIVATE /utf-8 /O2)
else()
    target_compile_options(mass-prime-siever PRIVATE -O3 -Wall)
    if(SIEVER_NATIVE)
        target_compile_options(mass-prime-siever PRIVATE -march=native)
    endif()
endif()
