cmake_minimum_required(VERSION 3.24)

project(k3_rtx5080_probe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(K3_CUDA_ARCH "120" CACHE STRING "CUDA compute capability without a decimal; RTX 5080 = 120")

add_executable(k3_rtx5080_probe rtx5080_probe.cu)
set_target_properties(k3_rtx5080_probe PROPERTIES
  CUDA_ARCHITECTURES "${K3_CUDA_ARCH}"
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)

if(MSVC)
  target_compile_options(k3_rtx5080_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
endif()

message(STATUS "K3 RTX probe CUDA architecture: sm_${K3_CUDA_ARCH}")
