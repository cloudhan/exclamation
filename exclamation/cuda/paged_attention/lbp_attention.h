#pragma once

#include "exclamation/cuda/cuda_common.h"

namespace exclamation {

struct DataParallelOutOfPlace {};
struct DataParallelInPlace {};
struct WorkStealing {};

template <typename TIO_, typename TKV_, typename TSB_, typename Sch>
struct LBPAttentionKernel {
  using TIO = TIO_;
  using TKV = TKV_;
  using TSB = TSB_;

  static void launch(
      stream_t stream,
      dev_props_ptr dev_props,
      void* workspace,
      TIO* out_ptr,
      const TIO* q_ptr,
      const TKV* k_cache_ptr,
      const TKV* v_cache_ptr,
      const TSB* scalebias_ptr,
      const int* page_table_ptr,
      const int* context_lens_ptr,
      const float* alibi_slopes_ptr,
      const float scale,
      const int num_seqs,
      const int num_heads,
      const int num_kv_heads,
      const int head_size,
      const int page_size,
      const int max_num_pages_per_seq,
      const int q_stride,
      const int max_context_len
  );

  static void create_workspace(stream_t stream, void** workspace, size_t* size, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len);
  static void destroy_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len);
  static void init_workspace(stream_t stream, void* workspace, int num_seqs, int num_heads, int num_kv_heads, int head_size, int max_context_len);
};

}  // namespace exclamation
