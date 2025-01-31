#pragma once

#include <cstdio>

#include <cuda_runtime.h>
#include "exclamation/cuda/config.h"

#define CUDA_CHECK(expr)                                                                               \
  do {                                                                                                 \
    cudaError_t err = (expr);                                                                          \
    if (err != cudaSuccess) {                                                                          \
      fprintf(stderr, "CUDA Error on %s:%d\n", __FILE__, __LINE__);                                    \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", err, cudaGetErrorString(err)); \
      exit(err);                                                                                       \
    }                                                                                                  \
  } while (0)

namespace exclamation {

template <typename... Ts>
constexpr bool always_false = false;

using stream_t = cudaStream_t;
using dev_props_ptr = const cudaDeviceProp*;

}  // namespace exclamation
