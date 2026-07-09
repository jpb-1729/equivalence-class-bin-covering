cmake_minimum_required(VERSION 3.18)
project(equivalence_class_bin_covering CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ortools CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(bin_covering src/main.cpp src/csp_solver.cpp)
target_include_directories(bin_covering PRIVATE src)
target_link_libraries(bin_covering PRIVATE ortools::ortools nlohmann_json::nlohmann_json)
