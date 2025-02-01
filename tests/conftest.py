from __future__ import annotations

import gc
import pytest

import torch
import torch.distributions as dist


def gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=False):
    if multidist:
        num_dist = num_pages * num_heads * page_size

        b = (torch.rand(num_dist).cuda().abs() * 16 + 0.0005) * scale
        a = torch.zeros_like(b)
        a = a.uniform_(-4, 4) * scale

        uniform_dist = dist.Normal(a, b)  # a batch of Uniform distributions
        samples = uniform_dist.sample((head_size,))  # (head_size, num_dist)

        kv_data = samples.reshape(
            (head_size, num_heads, num_pages, page_size)
        )  # (head_size, num_heads, num_pages, page_size)
        kv_data = kv_data.swapaxes(0, 2)  # (num_pages, num_heads, head_size, page_size)
    else:
        kv_data = torch.empty(size=(num_pages, num_heads, head_size, page_size), dtype=torch.float32, device="cuda")
        kv_data = kv_data.uniform_(-scale, scale)
    return kv_data.contiguous()


def div_x(tensor, x):
    shape = tensor.shape  # (num_pages, num_heads, head_size, page_size)
    assert shape[-2] % x == 0
    tensor = tensor.reshape(
        shape[:-2] + (shape[-2] // x, x) + shape[-1:]
    )  # (num_pages, num_heads, head_size//x, x, page_size)
    return torch.swapaxes(tensor, -1, -2).contiguous()  # (num_pages, num_heads, head_size//x, page_size, x)


def fill_unused_slots(kv_data, useful_slots, fill_value):
    # kv_data.shape (num_pages, num_heads, head_size, page_size)
    kv_data = kv_data.swapaxes(0, 2)  # (head_size, num_heads, num_pages, page_size)
    old_shape = kv_data.shape
    kv_data = kv_data.reshape(kv_data.shape[:2] + (-1,))
    kv_data_new = torch.full_like(kv_data, fill_value=fill_value)
    kv_data_new[:, :, useful_slots] = kv_data[:, :, useful_slots]
    kv_data_new = kv_data_new.reshape(old_shape)
    return kv_data_new.swapaxes(0, 2).contiguous()


def create_kv_caches(
    num_pages: int,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    kv_dtype: torch.dtype,
    useful_slots=None,
    fill_value=None,
    multidist_key=True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    gc.collect()

    dtype = kv_dtype
    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()

    key_caches = []
    for _ in range(num_layers):
        key_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=multidist_key).to(kv_dtype)
        if useful_slots is not None:
            key_cache = fill_unused_slots(key_cache, useful_slots, fill_value)
        key_cache = div_x(key_cache, x)
        key_caches.append(key_cache)

    value_caches = []
    for _ in range(num_layers):
        value_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=False).to(kv_dtype)
        if useful_slots is not None:
            value_cache = fill_unused_slots(value_cache, useful_slots, fill_value)
        value_caches.append(value_cache)

    return key_caches, value_caches


def pad_to_chunk_size(tensor, chunk_size) -> tuple[bool, torch.Tensor]:
    def ceil_div(x, y):
        return (x - 1) // y + 1

    shape = tensor.shape  # (num_pages, num_heads, head_size, page_size)
    needs_pad = shape[2] % chunk_size != 0
    if needs_pad:
        padder_shape = [shape[0], shape[1], ceil_div(shape[2], chunk_size) * chunk_size - shape[2], shape[3]]
        tensor_padder = torch.ones(padder_shape, dtype=tensor.dtype, device=tensor.device) * float("nan")
        tensor = torch.cat([tensor, tensor_padder], dim=2)
    return needs_pad, tensor


def get_scaled_and_scalebias(tensor, chunk_size, scaled_dtype, scalebias_dtype):
    dummy = False
    # dummy = True  # FIXME: dummy values
    target_max = torch.finfo(scaled_dtype).max - 32
    if dummy:
        target_max = 400
    print("[get_scaled_and_scalebias] target_max =", target_max)

    original_head_size = tensor.shape[2]
    is_padded, tensor = pad_to_chunk_size(tensor, chunk_size)
    shape = tensor.shape  # (num_pages, num_heads, head_size, page_size)
    # (num_pages, num_heads, ceil_div(head_size,chunk_size), chunk_size, page_size)
    chunked_shape = [shape[0], shape[1], shape[2] // chunk_size, chunk_size, shape[3]]
    tensor = tensor.clone().to(torch.float32)
    tensor = tensor.reshape(chunked_shape)
    masked_inf = tensor.isinf()
    tensor[masked_inf] = float("nan")
    mean = torch.nanmean(tensor, -2, True)
    if dummy:
        mean = torch.zeros_like(mean)

    shifted = tensor - mean
    if is_padded:
        shifted = torch.nan_to_num(shifted, 0)

    amax, _ = torch.max(shifted.abs(), -2, True)
    if dummy:
        amax = 0.1 * torch.ones_like(amax)

    scaled = (shifted / amax) * target_max
    scaled[masked_inf] = float("inf")
    scaled = scaled.reshape(shape)
    if is_padded:
        scaled = scaled[:, :, :original_head_size, :]
    scaled = scaled.to(scaled_dtype).contiguous()

    # original = a * scaled + b
    b = mean
    a = amax / target_max

    # (num_pages, num_heads, ceil_div(head_size, chunk_size), 1, page_size) -> (num_pages, num_heads, ceil_div(head_size, chunk_size), page_size)
    a = torch.squeeze(a, -2)
    b = torch.squeeze(b, -2)

    scalebias = torch.stack([a, b], dim=2).unsqueeze(1).to(scalebias_dtype)

    return scaled, scalebias


def create_fp8_kv_caches(
    num_pages: int,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    kv_dtype: torch.dtype,
    kv_scalebias_dtype=torch.float16,
    kv_scalebias_chunk_size: int = 32,
    useful_slots=None,
    fill_value=None,
    multidist_key=True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    gc.collect()
    assert kv_dtype in (torch.float8_e4m3fn,)
    # assert head_size % kv_scalebias_chunk_size == 0

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=kv_dtype).element_size()

    key_caches = []
    value_caches = []
    kv_scalebias = []
    for _ in range(num_layers):
        k_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=multidist_key)
        v_cache = gen_kv(num_pages, num_heads, head_size, page_size, scale, multidist=False)

        if useful_slots is not None:
            k_cache = fill_unused_slots(k_cache, useful_slots, fill_value)
            v_cache = fill_unused_slots(v_cache, useful_slots, fill_value)

        k_cache, k_scalebias = get_scaled_and_scalebias(k_cache, kv_scalebias_chunk_size, kv_dtype, kv_scalebias_dtype)
        v_cache, v_scalebias = get_scaled_and_scalebias(v_cache, kv_scalebias_chunk_size, kv_dtype, kv_scalebias_dtype)

        k_cache = div_x(k_cache, x)

        key_caches.append(k_cache)
        value_caches.append(v_cache)

        scalebias = torch.cat([k_scalebias, v_scalebias], dim=1)
        kv_scalebias.append(scalebias)

    return key_caches, value_caches, kv_scalebias


def create_kv_caches_wrapper(*args, **kwargs):
    if kwargs.get("kv_dtype", None) == torch.float8_e4m3fn:
        return create_fp8_kv_caches(*args, **kwargs)
    else:
        kwargs.pop("kv_scalebias_dtype", None)
        key_caches, value_caches = create_kv_caches(*args, **kwargs)
        dummy_scalebias = [None] * len(key_caches[0])
        return key_caches, value_caches, dummy_scalebias


@pytest.fixture()
def kv_cache_factory():
    return create_kv_caches_wrapper
