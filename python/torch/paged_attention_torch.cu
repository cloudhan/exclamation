#include "torch/extension.h"
#include "ATen/cuda/CUDAContext.h"
#include "cute/config.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "exclamation/cuda/paged_attention/paged_attention.h"

namespace exclamation {
namespace python {

void paged_attention(
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
    int max_context_len
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
  const dev_props_ptr dev_props = at::cuda::getDeviceProperties(query.device().index());
  const void* no_kv_scalebias = nullptr;

  if (query.dtype() == at::ScalarType::Float) {
    launch_paged_attention_kernel(
        stream, dev_props,
        reinterpret_cast<float*>(out.data_ptr()),
        reinterpret_cast<const float*>(query.data_ptr()),
        reinterpret_cast<const float*>(k_cache.data_ptr()),
        reinterpret_cast<const float*>(v_cache.data_ptr()),
        no_kv_scalebias,
        page_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        alibi_slopes_ptr,
        scale,
        num_seqs, num_heads, num_kv_heads, head_size, page_size, max_num_pages_per_seq, q_stride, max_context_len
    );

  } else if (query.dtype() == at::ScalarType::Half && !scalebias.has_value()) {
    launch_paged_attention_kernel(
        stream, dev_props,
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<const half*>(query.data_ptr()),
        reinterpret_cast<const half*>(k_cache.data_ptr()),
        reinterpret_cast<const half*>(v_cache.data_ptr()),
        no_kv_scalebias,
        page_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        alibi_slopes_ptr,
        scale,
        num_seqs, num_heads, num_kv_heads, head_size, page_size, max_num_pages_per_seq, q_stride, max_context_len
    );
  } else if (query.dtype() == at::ScalarType::Half && scalebias.has_value()) {
    launch_paged_attention_kernel<half, cute::float_e4m3_t, half>(
        stream, dev_props,
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<const half*>(query.data_ptr()),
        reinterpret_cast<const cute::float_e4m3_t*>(k_cache.data_ptr()),
        reinterpret_cast<const cute::float_e4m3_t*>(v_cache.data_ptr()),
        reinterpret_cast<const half*>(scalebias->data_ptr()),
        page_table.data_ptr<int>(),
        context_lens.data_ptr<int>(),
        alibi_slopes_ptr,
        scale,
        num_seqs, num_heads, num_kv_heads, head_size, page_size, max_num_pages_per_seq, q_stride, max_context_len
    );
  } else {
    TORCH_CHECK(false, "unsupported dtype: ", query.dtype());
  }
  CUDA_CHECK(cudaGetLastError());

  return;
}

}  // namespace python
}  // namespace exclamation
