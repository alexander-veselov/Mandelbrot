cmake_minimum_required(VERSION 3.17)

include(FindCUDAToolkit)
include(FetchContent)

set(CMAKE_SUPPRESS_REGENERATION true)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# glfw-external
FetchContent_Declare(glfw-external
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.bin.WIN64.zip
  URL_HASH MD5=725C211378D00F3EEE5DF5ACC8108742
  DOWNLOAD_EXTRACT_TIMESTAMP true
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/GLFW
)
FetchContent_MakeAvailable(glfw-external)

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

# MandelbrotSetGPU
project(MandelbrotSetGPU LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

# Core library
set(CORE_INCLUDE_FOLDER "include/core")
set(CORE_SOURCE_FOLDER "src/core")
file(GLOB CORE_HEADERS ${CORE_INCLUDE_FOLDER}/*.h ${CORE_INCLUDE_FOLDER}/*.cuh)
file(GLOB CORE_SOURCES ${CORE_SOURCE_FOLDER}/*.cpp ${CORE_SOURCE_FOLDER}/*.cu)
add_library(MandelbrotSetCore STATIC ${CORE_HEADERS} ${CORE_SOURCES})
target_include_directories(MandelbrotSetCore PUBLIC ${CORE_INCLUDE_FOLDER})
target_link_libraries(MandelbrotSetCore CUDA::cudart_static)

# Application executable
set(APP_INCLUDE_FOLDER "include/application")
set(APP_SOURCE_FOLDER "src/application")
file(GLOB APP_HEADERS ${APP_INCLUDE_FOLDER}/*.h ${APP_INCLUDE_FOLDER}/*.cuh)
file(GLOB APP_SOURCES ${APP_SOURCE_FOLDER}/*.cpp ${APP_SOURCE_FOLDER}/*.cu)
add_executable(MandelbrotSetGPU ${APP_HEADERS} ${APP_SOURCES})
target_include_directories(MandelbrotSetGPU PUBLIC ${APP_INCLUDE_FOLDER})
target_link_libraries(MandelbrotSetGPU MandelbrotSetCore GLFW)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MandelbrotSetGPU)