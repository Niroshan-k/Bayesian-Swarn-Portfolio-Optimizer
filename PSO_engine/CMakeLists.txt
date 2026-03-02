cmake_minimum_required(VERSION 3.15)
project(PSO_Engine)

set(CMAKE_CXX_STANDARD 14)

# Tell CMake to find your Mac's Python environment first
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 1. Try to find an existing pybind11 installation first
find_package(pybind11 CONFIG)

# 2. If not found, download it automatically
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found globally. Downloading via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1 
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Create the Python module
pybind11_add_module(pso_engine main.cpp)

# Apply max speed optimizations
if(MSVC)
    target_compile_options(pso_engine PRIVATE /O2)
else()
    target_compile_options(pso_engine PRIVATE -O3)
endif()