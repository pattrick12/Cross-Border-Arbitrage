cmake_minimum_required(VERSION 3.15)
project(GraphEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the Python extension module
pybind11_add_module(graph_core src/graph.cpp)

# Critical: Include Python headers so compilation doesn't fail
target_include_directories(graph_core PRIVATE ${Python3_INCLUDE_DIRS})

# Enable detailed error messages for Pybind11 (useful for debugging)
target_compile_definitions(graph_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

# Optimization flags for HFT (O3 for speed, march=native for CPU specific instructions)
if(MSVC)
    target_compile_options(graph_core PRIVATE /O2)
else()
    target_compile_options(graph_core PRIVATE -O3 -march=native)
endif()