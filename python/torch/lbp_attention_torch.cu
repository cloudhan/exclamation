
#include "torch/extension.h"
#include "ATen/cuda/CUDAContext.h"

#include "cute/config.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "exclamation/cuda/paged_attention/lbp_attention.h"

namespace exclamation {
namespace python {

torch::Tensor create_workspace_tensor(const torch::Device& dev, size_t workspace_size) {
  return torch::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().dtype(torch::kU8).device(dev)
  );
}

void lbp_attention(
    torch::Tensor& out,                                // [num_seqs, num_heads, head_size]
    const torch::Tensor& query,                        // [num_seqs, num_heads, head_size]
    const torch::Tensor& k_cache,                      // [num_pages, num_heads, head_size/x, page_size, x]
    const torch::Tensor& v_cache,                      // [num_pages, num_heads, head_size, page_size]
    const c10::optional<torch::Tensor>& scalebias,     // [num_pages, num_heads, head_size/chunk_size, page_size]
    const torch::Tensor& page_table,                   // [num_seqs, max_num_pages_per_seq]
    const torch::Tensor& context_lens,                 // [num_seqs], map to num_tokens in context
    const c10::optional<torch::Tensor>& alibi_slopes,  // [num_heads]
    float scale,
    int num_kv_heads,
    int page_size,
    int max_context_len,
    int sch
) {
  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
  TORCH_CHECK(page_table.is_contiguous(), "page_table must be contiguous");
  TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");

  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_pages_per_seq = page_table.size(1);
  int q_stride = query.stride(0);

  const float* alibi_slopes_ptr = nullptr;
  if (alibi_slopes) {
    TORCH_CHECK(alibi_slopes->is_contiguous(), "context_lens must be contiguous");
    alibi_slopes_ptr = alibi_slopes->data_ptr<float>();
  }

  const stream_t stream = at::cuda::getCurrentCUDAStream();
  const auto dev_props = at::cuda::getDeviceProperties(query.device().index());
  size_t workspace_size;

#define LAUNCH_LBP_ATTENTION_KERNEL()                                                                                                    \
  LBPAttentionKernel::create_workspace(stream, nullptr, &workspace_size, num_seqs, num_heads, num_kv_heads, head_size, max_context_len); \
  auto workspace = create_workspace_tensor(query.device(), workspace_size);                                                              \
  LBPAttentionKernel::init_workspace(stream, workspace.data_ptr(), num_seqs, num_heads, num_kv_heads, head_size, max_context_len);       \
  LBPAttentionKernel::launch(                                                                                                            \
      stream,                                                                                                                            \
      dev_props,                                                                                                                         \
      workspace.data_ptr(),                                                                                                              \
      reinterpret_cast<LBPAttentionKernel::TIO*>(out.data_ptr()),                                                                        \
      reinterpret_cast<const LBPAttentionKernel::TIO*>(query.data_ptr()),                                                                \
      reinterpret_cast<const LBPAttentionKernel::TKV*>(k_cache.data_ptr()),                                                              \
      reinterpret_cast<const LBPAttentionKernel::TKV*>(v_cache.data_ptr()),                                                              \
      reinterpret_cast<const LBPAttentionKernel::TSB*>(scalebias.has_value() ? scalebias->data_ptr() : nullptr),                         \
      page_table.data_ptr<int>(),                                                                                                        \
      context_lens.data_ptr<int>(),                                                                                                      \
      alibi_slopes_ptr,                                                                                                                  \
      scale, num_seqs, num_heads, num_kv_heads, head_size, page_size,                                                                    \
      max_num_pages_per_seq, q_stride, max_context_len                                                                                   \
  );

  if (sch == 0) {
    if (query.dtype() == at::ScalarType::Half && !scalebias.has_value()) {
      using LBPAttentionKernel = exclamation::LBPAttentionKernel<half, half, void, DataParallelOutOfPlace>;
      LAUNCH_LBP_ATTENTION_KERNEL();
    } else if (query.dtype() == at::ScalarType::Half && scalebias.has_value()) {
      using LBPAttentionKernel = exclamation::LBPAttentionKernel<half, cute::float_e4m3_t, half, DataParallelOutOfPlace>;
      LAUNCH_LBP_ATTENTION_KERNEL();
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else if (sch == 1) {
    if (query.dtype() == at::ScalarType::Half && !scalebias.has_value()) {
      using LBPAttentionKernel = exclamation::LBPAttentionKernel<half, half, void, DataParallelInPlace>;
      LAUNCH_LBP_ATTENTION_KERNEL();
    } else if (query.dtype() == at::ScalarType::Half && scalebias.has_value()) {
      using LBPAttentionKernel = exclamation::LBPAttentionKernel<half, cute::float_e4m3_t, half, DataParallelInPlace>;
      LAUNCH_LBP_ATTENTION_KERNEL();
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else if (sch == 2) {
    if (query.dtype() == at::ScalarType::Half && !scalebias.has_value()) {
      using LBPAttentionKernel = exclamation::LBPAttentionKernel<half, half, void, WorkStealing>;
      LAUNCH_LBP_ATTENTION_KERNEL();
    } else if (query.dtype() == at::ScalarType::Half && scalebias.has_value()) {
      using LBPAttentionKernel = exclamation::LBPAttentionKernel<half, cute::float_e4m3_t, half, WorkStealing>;
      LAUNCH_LBP_ATTENTION_KERNEL();
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else {
    TORCH_CHECK(false, "Unsupported sch: 0:DataParallelOutOfPlace, 1:DataParallelInPlace, 2:WorkStealing");
  }
  CUDA_CHECK(cudaGetLastError());
  return;
}

}  // namespace python
}  // namespace exclamation
