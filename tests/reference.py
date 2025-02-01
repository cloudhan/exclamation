from __future__ import annotations

import torch


def masked_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", q, k).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(v.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, v)
    return out


def reconstruct_kv(
    page_mapping_of_seq: torch.Tensor,
    context_len: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_scalebias: torch.Tensor | None = None,
    chunksize=32,
):
    #  (num_pages, num_heads, head_size//chunksize, page_size)
    k_padded = k_cache.view(torch.int8).index_select(0, page_mapping_of_seq).view(k_cache.dtype).clone()
    v_padded = v_cache.view(torch.int8).index_select(0, page_mapping_of_seq).view(v_cache.dtype).clone()
    #  (num_pages, 2, num_heads, 2, head_size//chunksize, page_size)
    if kv_scalebias is not None:
        sb_padded = kv_scalebias.index_select(0, page_mapping_of_seq).clone()

    k_padded = k_padded.moveaxis(0, -3)  #  (num_heads, head_size//x, num_pages, page_size, x)
    k_padded = k_padded.moveaxis(-1, -3)  # (num_heads, head_size//x, x, num_pages, page_size)
    v_padded = v_padded.moveaxis(0, -2)  #  (num_heads, head_size,    num_pages, page_size)
    # print(k_padded.shape)
    # print(v_padded.shape)
    if kv_scalebias is not None:
        sb_padded = sb_padded.moveaxis(0, -2)  #  (2, num_heads, 2, head_size//chunksize, num_pages, page_size)
        # print(sb_padded.shape)

    k_padded = k_padded.reshape(
        (k_padded.shape[0], k_padded.shape[1] * k_padded.shape[2], k_padded.shape[3] * k_padded.shape[4])
    )
    v_padded = v_padded.reshape(v_padded.shape[:2] + (v_padded.shape[2] * v_padded.shape[3],))
    # print(k_padded.shape)
    # print(v_padded.shape)
    if kv_scalebias is not None:
        sb_padded = sb_padded.reshape(sb_padded.shape[:4] + (sb_padded.shape[4] * sb_padded.shape[5],))
        sb_padded = sb_padded.repeat_interleave(chunksize, dim=3)
        # print(sb_padded.shape)

    # (num_heads, head_size, context_len)
    k = k_padded[:, :, :context_len]
    v = v_padded[:, :, :context_len]
    # print(k.shape, k.dtype)
    # print(v.shape, v.dtype)
    if kv_scalebias is not None:
        valid_head_size = k.shape[1]
        sb = sb_padded[:, :, :, :valid_head_size, :context_len]
        k = sb[0, :, 0] * k.to(sb.dtype) + sb[0, :, 1]
        v = sb[1, :, 0] * v.to(sb.dtype) + sb[1, :, 1]
        # print("k scale", sb[0, :, 0].max())
        # print("k bias ", sb[0, :, 1].max())
        # print("v scale", sb[1, :, 0].max())
        # print("v bias ", sb[1, :, 1].max())

    # (context_len, num_heads, head_size)
    original_k = k.moveaxis(-1, 0).contiguous()
    original_v = v.moveaxis(-1, 0).contiguous()
    if kv_scalebias is None:
        # print(original_k.shape, original_v.shape)
        return original_k, original_v
    else:
        sb_chunked = sb[:, :, :, ::chunksize].moveaxis(-1, 0).contiguous()
        # print(original_k.shape, original_v.shape, sb_chunked.shape)
        return original_k, original_v, sb_chunked


def reconstruct_kv_then_mha(
    output: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_scalebias: torch.Tensor | None,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: torch.Tensor | None,
) -> None:
    num_q_heads = q.shape[1]
    num_seqs = q.shape[0]
    _, num_kv_heads, head_size, page_size = v_cache.shape
    context_lens = context_lens.cpu().tolist()
    if kv_scalebias is not None:
        num_chunks = kv_scalebias.shape[-2]
        chunk_size = head_size // num_chunks

        assert k_cache.dtype == torch.float8_e4m3fn
        assert v_cache.dtype == torch.float8_e4m3fn

    for i in range(num_seqs):
        context_len = int(context_lens[i])
        num_pages = (context_len - 1) // page_size + 1
        page_mapping_of_seq = page_table[i].to(torch.int64)[:num_pages]
        reconstructed_kv = reconstruct_kv(page_mapping_of_seq, context_len, k_cache, v_cache, kv_scalebias)
        k = reconstructed_kv[0]
        v = reconstructed_kv[1]

        num_query_heads_per_kv_head = num_q_heads // num_kv_heads
        assert num_q_heads % num_kv_heads == 0
        if num_query_heads_per_kv_head > 1:
            k = torch.repeat_interleave(k, num_query_heads_per_kv_head, dim=1)
            v = torch.repeat_interleave(v, num_query_heads_per_kv_head, dim=1)

        attn_bias = None
        if alibi_slopes is not None:
            position_ids = torch.arange(context_len, device="cuda").int()
            attn_bias = (position_ids - context_len + 1).float()
            attn_bias = alibi_slopes.view(-1, 1, 1) * attn_bias.view(1, 1, -1)

        out = masked_mha(q[i].unsqueeze(0), k, v, scale, attn_bias)
        out = out.view(num_q_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    torch.cuda.synchronize()
