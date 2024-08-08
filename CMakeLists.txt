cmake_minimum_required(VERSION 3.17)

include(FindCUDAToolkit)
include(FetchContent)

set(CMAKE_SUPPRESS_REGENERATION ON CACHE BOOL "" FORCE)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX "")
endif()

# MandelbrotSetExplorer
project(MandelbrotSetExplorer LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# glfw-external
FetchContent_Declare(glfw-external
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip
  URL_HASH MD5=725C211378D00F3EEE5DF5ACC8108742
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/GLFW
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

# rapidjson-external
set(RAPIDJSON_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RAPIDJSON_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(rapidjson-external
  GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
  GIT_TAG v1.1.0
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/rapidjson
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

# googlebenchmark-external
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googlebenchmark-external
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.5
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

# googletest-external
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(googletest-external
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

# lodepng-external
FetchContent_Declare(lodepng-external
  GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
  GIT_TAG ed3c5f14b136c936d615ee3b38aaa7e7e642f12c
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_MakeAvailable(
  glfw-external
  googlebenchmark-external
  googletest-external
  rapidjson-external
  lodepng-external
)

FetchContent_GetProperties(
  rapidjson-external
)

# Google benchmark
# benchmark and benchmark_main targets are available
set_target_properties(benchmark PROPERTIES FOLDER "External")
set_target_properties(benchmark_main PROPERTIES FOLDER "External")

# Google test
# gtest and gtest_main targets are available
set_target_properties(gtest PROPERTIES FOLDER "External")
set_target_properties(gtest_main PROPERTIES FOLDER "External")

# OpenGL
add_library(OpenGL STATIC IMPORTED)
set_target_properties(OpenGL PROPERTIES
  IMPORTED_LOCATION opengl32.lib
)

# GLFW
add_library(GLFW STATIC IMPORTED)
add_dependencies(GLFW glfw-external)
target_link_libraries(GLFW INTERFACE OpenGL)
set_target_properties(GLFW PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/GLFW/lib-vc2022/glfw3.lib
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/GLFW/include
)

# RapidJSON
add_library(RapidJSON INTERFACE)
target_include_directories(RapidJSON INTERFACE "$<BUILD_INTERFACE:${rapidjson-external_SOURCE_DIR}/include>")
set_target_properties(travis_doc PROPERTIES FOLDER "External")

# LodePNG
add_library(LodePNG STATIC
  "${lodepng-external_SOURCE_DIR}/lodepng.cpp"
  "${lodepng-external_SOURCE_DIR}/lodepng.h"
)
target_include_directories(LodePNG PUBLIC
  "${lodepng-external_SOURCE_DIR}"
)
set_target_properties(LodePNG PROPERTIES FOLDER "External")

# Core library
set(CORE_INCLUDE_FOLDER "include/core")
set(CORE_SOURCE_FOLDER "src/core")
file(GLOB CORE_HEADERS ${CORE_INCLUDE_FOLDER}/*.h ${CORE_INCLUDE_FOLDER}/*.cuh)
file(GLOB CORE_SOURCES ${CORE_SOURCE_FOLDER}/*.cpp ${CORE_SOURCE_FOLDER}/*.cu)
add_library(MandelbrotSetCore STATIC ${CORE_HEADERS} ${CORE_SOURCES})
target_include_directories(MandelbrotSetCore PUBLIC ${CORE_INCLUDE_FOLDER})
target_link_libraries(MandelbrotSetCore PRIVATE CUDA::cudart_static LodePNG)

# Benchmarks
set(BENCHMARKS_SOURCE_FOLDER "src/benchmarks")
file(GLOB BENCHMARKS_SOURCES ${BENCHMARKS_SOURCE_FOLDER}/*.cpp)
add_executable(Benchmarks ${BENCHMARKS_SOURCES})
target_link_libraries(Benchmarks PRIVATE benchmark benchmark_main MandelbrotSetCore)

# Tests
enable_testing()
set(TESTS_SOURCE_FOLDER "src/tests")
file(GLOB TESTS_SOURCE_FOLDER ${TESTS_SOURCE_FOLDER}/*.cpp)
add_executable(Tests ${TESTS_SOURCE_FOLDER})
target_link_libraries(Tests PRIVATE gtest gtest_main)
target_link_libraries(Tests PUBLIC MandelbrotSetCore)
include(GoogleTest)
gtest_discover_tests(Tests)

# Application executable
set(APP_INCLUDE_FOLDER "include/application")
set(APP_SOURCE_FOLDER "src/application")
file(GLOB APP_HEADERS ${APP_INCLUDE_FOLDER}/*.h ${APP_INCLUDE_FOLDER}/*.cuh)
file(GLOB APP_SOURCES ${APP_SOURCE_FOLDER}/*.cpp ${APP_SOURCE_FOLDER}/*.cu)
add_executable(MandelbrotSetExplorer ${APP_HEADERS} ${APP_SOURCES})
target_include_directories(MandelbrotSetExplorer PUBLIC ${APP_INCLUDE_FOLDER})
target_link_libraries(MandelbrotSetExplorer PUBLIC MandelbrotSetCore)
target_link_libraries(MandelbrotSetExplorer PRIVATE GLFW RapidJSON)
set_property(TARGET MandelbrotSetExplorer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_custom_command(
    TARGET MandelbrotSetExplorer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/data/config.json
    $<TARGET_FILE_DIR:MandelbrotSetExplorer>/config.json
)

# Set starup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MandelbrotSetExplorer)