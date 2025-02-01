#pragma once

#include "torch/extension.h"

namespace exclamation {
namespace python {

void paged_attention(
    torch::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const torch::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const torch::Tensor& k_cache,                   // [num_pages, num_heads, head_size/x, page_size, x]
    const torch::Tensor& v_cache,                   // [num_pages, num_heads, head_size, page_size]
    const c10::optional<torch::Tensor>& scalebias,  // [num_pages, 2, num_heads, 2, head_size/chunk_size, page_size]
    const torch::Tensor& page_table,                // [num_seqs, max_num_pages_per_seq]
    const torch::Tensor& context_lens,              // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    float scale,
    int num_kv_heads,
    int page_size,
    int max_context_len
);

}  // namespace python
}  // namespace exclamation
