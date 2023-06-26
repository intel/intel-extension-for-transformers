cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(XeTLA)

include(CTest)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/tools/cmake")
find_package(MKL REQUIRED)

# debug option
set(DEBUG "off" CACHE STRING "set debug version")
message(STATUS "debug is set to " ${DEBUG})
if (${DEBUG} STREQUAL "on")
    add_compile_options(-debug=minimal -Rno-debug-disables-optimization)
endif ()

add_compile_options(-fsycl)
add_link_options(-fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm)
link_libraries(-lgtest -lgtest_main)

add_subdirectory(tests)
add_subdirectory(examples)
