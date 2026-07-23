cmake_minimum_required(VERSION 3.18)
project(equivalence_class_bin_covering CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Benchmarks should not silently use an unoptimized build. Multi-config
# generators select their configuration at build time, so this only applies to
# single-config generators such as Unix Makefiles and Ninja.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(ortools CONFIG REQUIRED)

add_executable(bin_covering
  src/main.cpp
  src/csp_solver.cpp
  src/monte_carlo.cpp
)
target_include_directories(bin_covering PRIVATE src)
target_link_libraries(bin_covering PRIVATE ortools::ortools)
