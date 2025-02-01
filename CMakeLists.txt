cmake_minimum_required(VERSION 3.24)

project(exclamation)

option(EXCLAMATION_USE_CCACHE "Enable ccache" ON)
option(EXCLAMATION_BUILD_PROFILING "Build for profiling purpose" OFF)

if(EXCLAMATION_USE_CCACHE)
  include(cmake/ccache.cmake)
  find_program(CCACHE_PATH ccache)
  configure_ccache(${CCACHE_PATH})
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  # set(CMAKE_CUDA_ARCHITECTURES "80-real;90-real;90-virtual")
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit)
enable_language(CUDA)

if (EXCLAMATION_BUILD_PROFILING)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo >")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--dopt on >")
endif()

include(cmake/third_party/cutlass_repo.cmake)

add_subdirectory(exclamation)
add_subdirectory(python)
