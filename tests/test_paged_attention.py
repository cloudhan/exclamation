from __future__ import annotations

import functools
import pytest
from paged_attention_common import *


@pytest.mark.parametrize("use_alibi", [False, True])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("seq_len", [None])
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("page_size", PAGE_SIZES)
@pytest.mark.parametrize("kv_dtype", DTYPES)
def test_paged_attention(
    kv_cache_factory,
    num_seqs: int,
    seq_len: int | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    kv_dtype: str,
    use_alibi: bool,
):
    ops_paged_attention = lambda func_args: ops.paged_attention(*func_args.values())
    functools.partial(test_exclamation_ops, ops_paged_attention)(
        kv_cache_factory,
        num_seqs,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_size,
        page_size,
        kv_dtype,
        use_alibi,
    )
