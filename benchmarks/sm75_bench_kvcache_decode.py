import argparse
import json
import time

import torch
import torch.nn.functional as F

from flash_attn import flash_attn_with_kvcache, get_last_flash_attn_backend


def _event_time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _sdpa_decode(q, k_cache, v_cache, causal):
    q_t = q.transpose(1, 2)
    k_t = k_cache.transpose(1, 2)
    v_t = v_cache.transpose(1, 2)
    if q_t.shape[1] != k_t.shape[1]:
        repeat = q_t.shape[1] // k_t.shape[1]
        k_t = k_t.repeat_interleave(repeat, dim=1)
        v_t = v_t.repeat_interleave(repeat, dim=1)
    attn_mask = None
    if causal:
        seqlen_q, seqlen_k = q.shape[1], k_cache.shape[1]
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        attn_mask = col <= row + seqlen_k - seqlen_q
    return F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask).transpose(1, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-len", type=int, default=4096)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", default=f"sm75_kvcache_decode_{int(time.time())}.json")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    device = torch.device("cuda")
    q = torch.randn(1, args.q_len, 16, 64, device=device, dtype=torch.float16)
    k_cache = torch.randn(1, args.cache_len, 4, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.full((1,), args.cache_len, device=device, dtype=torch.int32)
    native_ms = _event_time_ms(
        lambda: flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True),
        iters=args.iters,
    )
    backend = get_last_flash_attn_backend()
    sdpa_ms = _event_time_ms(lambda: _sdpa_decode(q, k_cache, v_cache, causal=True), iters=args.iters)
    result = {
        "name": f"decode_q{args.q_len}_cache{args.cache_len}_hd64",
        "native_ms": native_ms,
        "sdpa_ms": sdpa_ms,
        "speedup_vs_sdpa": sdpa_ms / native_ms,
        "backend": backend,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
