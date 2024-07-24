cmake_minimum_required(VERSION 3.17)

include(FindCUDAToolkit)
include(FetchContent)

set(CMAKE_SUPPRESS_REGENERATION true)
set_property( GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# glfw-external
FetchContent_Declare(glfw-external
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip
  URL_HASH MD5=725C211378D00F3EEE5DF5ACC8108742
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/GLFW
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

# googlebenchmark-external
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_Declare(googlebenchmark-external
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.5
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/benchmark
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

FetchContent_MakeAvailable(
  glfw-external
  googlebenchmark-external
)

# Google benchmark
# benchmark and benchmark_main targets are available
set_target_properties(benchmark PROPERTIES FOLDER "External")
set_target_properties(benchmark_main PROPERTIES FOLDER "External")

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

# MandelbrotSetExplorer
project(MandelbrotSetExplorer LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

# Core library
set(CORE_INCLUDE_FOLDER "include/core")
set(CORE_SOURCE_FOLDER "src/core")
file(GLOB CORE_HEADERS ${CORE_INCLUDE_FOLDER}/*.h ${CORE_INCLUDE_FOLDER}/*.cuh)
file(GLOB CORE_SOURCES ${CORE_SOURCE_FOLDER}/*.cpp ${CORE_SOURCE_FOLDER}/*.cu)
add_library(MandelbrotSetCore STATIC ${CORE_HEADERS} ${CORE_SOURCES})
target_include_directories(MandelbrotSetCore PUBLIC ${CORE_INCLUDE_FOLDER})
target_link_libraries(MandelbrotSetCore CUDA::cudart_static)

# Benchmarks
set(BENCHMARKS_SOURCE_FOLDER "src/benchmarks")
file(GLOB BENCHMARKS_SOURCES ${BENCHMARKS_SOURCE_FOLDER}/*.cpp)
add_executable(Benchmarks ${BENCHMARKS_SOURCES})
target_link_libraries(Benchmarks benchmark benchmark_main MandelbrotSetCore)

# Application executable
set(APP_INCLUDE_FOLDER "include/application")
set(APP_SOURCE_FOLDER "src/application")
file(GLOB APP_HEADERS ${APP_INCLUDE_FOLDER}/*.h ${APP_INCLUDE_FOLDER}/*.cuh)
file(GLOB APP_SOURCES ${APP_SOURCE_FOLDER}/*.cpp ${APP_SOURCE_FOLDER}/*.cu)
add_executable(MandelbrotSetExplorer ${APP_HEADERS} ${APP_SOURCES})
target_include_directories(MandelbrotSetExplorer PUBLIC ${APP_INCLUDE_FOLDER})
target_link_libraries(MandelbrotSetExplorer MandelbrotSetCore GLFW)

# Set starup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MandelbrotSetExplorer)