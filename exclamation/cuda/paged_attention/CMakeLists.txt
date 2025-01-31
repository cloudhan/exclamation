set(paged_attention_srcs
  paged_attention.cu
  lbp_attention.cu
  reshape_and_cache_fp8.cu
)

include(paged_attention.cmake)
list(APPEND paged_attention_srcs ${paged_attention_generated_srcs})

add_library(paged_attention OBJECT ${paged_attention_srcs})
target_link_libraries(paged_attention PUBLIC nvidia::cutlass::cutlass)
target_compile_options(paged_attention PUBLIC --expt-relaxed-constexpr)

target_link_libraries(paged_attention
  PUBLIC
    exclamation_base
    exclamation_split_compilation
)
