#pragma once

#include <cstdint>
#include "exclamation/cuda/cuda_common.h"
#include "exclamation/cuda/platform_ext.cuh"

#define SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)

namespace exclamation {

__forceinline__ __device__ auto
lane_id() {
  return threadIdx.x % constant::WarpSize;
}

__forceinline__ __device__ auto
warp_id() {
  return threadIdx.x / constant::WarpSize;
}

#if !defined(__CUDACC__)
__forceinline__
#endif
__host__ __device__ constexpr uint32_t
ceil_log2(uint32_t n) {
  return n <= 1 ? 0 : 1 + ceil_log2((n + 1) / 2);
}

__forceinline__ __host__ __device__ constexpr uint32_t
next_power_of_two(uint32_t n) {
  return uint32_t(1) << ceil_log2(n);
}

}  // namespace exclamation
