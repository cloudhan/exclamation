from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python/torch"))

import random
from collections import OrderedDict

import torch

import exclamation_ops as ops
import reference

DTYPES = ["float16", "float32", "float8_e4m3fn"]
NUM_GEN_SEQS = [1, 7]
NUM_HEADS = [
    (19, 19),  # mha
    (16, 4),  # gqa4
    (32, 2),  # gqa8 with 2x broadcasting
]
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
PAGE_SIZES = [8, 16, 32]


def generate_page_mapping(
    max_num_pages: int,
    num_seqs: int,
    max_context_len: int,
    page_size: int,
):
    unique_page_mapping = [i for i in range(max_num_pages)]
    max_num_pages_per_seq = (max_context_len + page_size - 1) // page_size
    random.shuffle(unique_page_mapping)
    page_table = []
    for i in range(num_seqs):
        page_table.append(unique_page_mapping[i * max_num_pages_per_seq : (i + 1) * max_num_pages_per_seq])
    assert len(page_table[-1]) == max_num_pages_per_seq, "alloc more pages to allow generating unique page mapping"
    return page_table


def get_useful_slots(
    fill_non_used: float | None,
    page_size: int,
    page_table: list[list[int]],
    context_lens: list[int],
):
    useful_slots = None
    if fill_non_used is not None:
        useful_slots = []
        for seq, end in enumerate(context_lens):
            seq_num_pages = (end - 1) // page_size + 1
            seq_useful_slots = []
            for logical_pid in range(seq_num_pages):
                physical_pid = page_table[seq][logical_pid]
                seq_useful_slots.extend(list(range(physical_pid * page_size, physical_pid * page_size + page_size)))
            useful_slots.extend(seq_useful_slots[:end])
    return useful_slots


def get_dtypes(kv_dtype) -> tuple:
    if isinstance(kv_dtype, str):
        kv_dtype = getattr(torch, kv_dtype)
    mapping = {
        None: (torch.float16, torch.float16, None),
        torch.float16: (torch.float16, torch.float16, None),
        torch.float32: (torch.float32, torch.float32, None),
        torch.float8_e4m3fn: (torch.float16, torch.float8_e4m3fn, torch.float16),
    }
    return mapping.get(kv_dtype)


@torch.inference_mode()
def test_exclamation_ops(
    func,
    kv_cache_factory,
    num_seqs: int,
    seq_len: int | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    kv_dtype: str | torch.dtype | None,
    use_alibi: bool,
    seed: int = int(os.environ.get("SEED", "0")),
    max_seq_len: int = 4096,
    max_num_pages: int = 5000,
    fill_non_used: float | None = None,
):
    print()
    dtype, kv_dtype, kv_scalebias_dtype = get_dtypes(kv_dtype)
    print(dtype, kv_dtype, kv_scalebias_dtype)

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    q = torch.empty(num_seqs, num_q_heads, head_size, dtype=dtype, device="cuda")
    q.uniform_(-scale, scale)
    # q = torch.randn((num_seqs, num_q_heads, head_size), dtype=dtype, device="cuda")

    assert num_q_heads % num_kv_heads == 0

    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_q_heads, dtype=torch.float, device="cuda") * scale * 4

    if seq_len is None:
        seq_len = os.getenv("SEQ_LEN", None)
        seq_len = int(seq_len) if seq_len is not None else None
    if seq_len is None:
        context_lens = [random.randint(1, max_seq_len) for _ in range(num_seqs)]
    elif isinstance(seq_len, int):
        context_lens = [seq_len for _ in range(num_seqs)]
    else:
        context_lens = seq_len
        assert len(context_lens) == num_seqs
    max_context_len = max(context_lens)
    context_lens_py = context_lens
    context_lens = torch.tensor(context_lens_py, dtype=torch.int, device="cuda")
    print("context_lens:", context_lens_py)

    page_table_py = generate_page_mapping(max_num_pages, num_seqs, max_context_len, page_size)
    page_table = torch.tensor(page_table_py, dtype=torch.int, device="cuda")
    print("page_table:", page_table_py)

    useful_slots = get_useful_slots(fill_non_used, page_size, page_table_py, context_lens_py)

    k_caches, v_caches, kv_scalebiases = kv_cache_factory(
        max_num_pages,
        page_size,
        1,
        num_kv_heads,
        head_size,
        kv_dtype=kv_dtype,
        kv_scalebias_dtype=kv_scalebias_dtype,
        useful_slots=useful_slots,
        fill_value=fill_non_used,
    )
    k_cache, v_cache, kv_scalebias = k_caches[0], v_caches[0], kv_scalebiases[0]

    if kv_dtype == torch.float8_e4m3fn:
        assert kv_scalebias is not None
    else:
        assert kv_scalebias_dtype is None
        assert kv_scalebias is None

    output = torch.empty_like(q)

    func_args = OrderedDict(
        output=output,
        query=q,
        k_cache=k_cache,
        v_cache=v_cache,
        kv_scalebias=kv_scalebias,
        page_table=page_table,
        context_lens=context_lens,
        alibi_slopes=alibi_slopes,
        scale=scale,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        max_context_len=max_context_len,
    )
    func(func_args)

    ref_output = torch.empty_like(q)
    reference.reconstruct_kv_then_mha(
        ref_output,
        q,
        k_cache,
        v_cache,
        kv_scalebias,
        page_table,
        context_lens,
        scale,
        alibi_slopes,
    )
    torch.cuda.synchronize()

    if bool(int(os.environ.get("DUMP_RESULTS", "0"))):
        print(ref_output)
        diff = output - ref_output
        print(diff.abs().max())
        print(diff)

        import numpy as np

        np.save("ref.npy", ref_output.cpu().numpy())
        np.save("our.npy", output.cpu().numpy())

    if dtype == torch.float32:
        assert torch.allclose(output, ref_output, atol=1e-5, rtol=1e-6)
    else:
        assert torch.allclose(output, ref_output, atol=2e-4, rtol=1e-5)
