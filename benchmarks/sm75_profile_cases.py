import argparse
import os

import torch

from flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_last_flash_attn_backend,
    reset_last_flash_attn_backend,
)


def _rand(*shape, heads_k=None, dtype=torch.float16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def dense_hd64_causal_mha():
    q = _rand(1, 512, 8, 64)
    k = _rand(1, 512, 8, 64)
    v = _rand(1, 512, 8, 64)
    return flash_attn_func(q, k, v, causal=True)


def dense_hd128_causal_gqa():
    q = _rand(1, 512, 16, 128)
    k = _rand(1, 512, 4, 128)
    v = _rand(1, 512, 4, 128)
    return flash_attn_func(q, k, v, causal=True)


def generic_hd160():
    q = _rand(1, 128, 8, 160)
    k = _rand(1, 128, 4, 160)
    v = _rand(1, 128, 4, 160)
    return flash_attn_func(q, k, v, causal=True)


def varlen_hd128():
    q = _rand(1024, 8, 128)
    k = _rand(1024, 4, 128)
    v = _rand(1024, 4, 128)
    cu = torch.tensor([0, 512, 1024], device="cuda", dtype=torch.int32)
    return flash_attn_varlen_func(q, k, v, cu, cu, 512, 512, causal=True)


def kvcache_contiguous_hd128_gqa():
    q = _rand(2, 1, 8, 128)
    k_cache = _rand(2, 512, 2, 128)
    v_cache = _rand(2, 512, 2, 128)
    cache_seqlens = torch.tensor([512, 384], device="cuda", dtype=torch.int32)
    return flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True)


def kvcache_paged_hd128_gqa():
    q = _rand(2, 1, 8, 128)
    k_cache = _rand(4, 256, 2, 128)
    v_cache = _rand(4, 256, 2, 128)
    block_table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([512, 384], device="cuda", dtype=torch.int32)
    return flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )


def local_mask_path():
    q = _rand(1, 256, 8, 128)
    k = _rand(1, 256, 4, 128)
    v = _rand(1, 256, 4, 128)
    return flash_attn_func(q, k, v, causal=False, window_size=(128, 0))


def dropout_path():
    q = _rand(1, 128, 8, 128).requires_grad_(True)
    k = _rand(1, 128, 4, 128).requires_grad_(True)
    v = _rand(1, 128, 4, 128).requires_grad_(True)
    out = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)
    out.sum().backward()
    return out


CASES = {
    "dense_hd64_causal_mha": dense_hd64_causal_mha,
    "dense_hd128_causal_gqa": dense_hd128_causal_gqa,
    "generic_hd160": generic_hd160,
    "varlen_hd128": varlen_hd128,
    "kvcache_contiguous_hd128_gqa": kvcache_contiguous_hd128_gqa,
    "kvcache_paged_hd128_gqa": kvcache_paged_hd128_gqa,
    "local_mask_path": local_mask_path,
    "dropout_path": dropout_path,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    os.environ.setdefault("FLASH_ATTN_SM75_NATIVE_ONLY", "1")
    os.environ.setdefault("FLASH_ATTN_SM75_BAN_FALLBACKS", "1")
    fn = CASES[args.case]
    torch.manual_seed(0)
    reset_last_flash_attn_backend()
    fn()
    torch.cuda.synchronize()
    reset_last_flash_attn_backend()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    backend = get_last_flash_attn_backend()
    print(f"case {args.case}")
    print(f"backend {backend}")
    if not backend or not backend.startswith("sm75_native"):
        raise SystemExit(f"non-SM75 backend: {backend}")


if __name__ == "__main__":
    main()
