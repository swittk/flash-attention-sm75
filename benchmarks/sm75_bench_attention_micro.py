import argparse
import json
import time

import torch
import torch.nn.functional as F

from flash_attn import flash_attn_func, get_last_flash_attn_backend


DEFAULT_SHAPES = [
    dict(name="llama_prefill_1k_hd64", batch=1, seqlen_q=1024, seqlen_k=1024, nheads=16, nheads_k=16, headdim=64, causal=True),
    dict(name="gqa_prefill_2k_hd128", batch=1, seqlen_q=2048, seqlen_k=2048, nheads=16, nheads_k=4, headdim=128, causal=True),
    dict(name="cross_attention_hd64", batch=2, seqlen_q=512, seqlen_k=1024, nheads=8, nheads_k=8, headdim=64, causal=False),
]


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


def _sdpa(q, k, v, causal):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    if q_t.shape[1] != k_t.shape[1]:
        repeat = q_t.shape[1] // k_t.shape[1]
        k_t = k_t.repeat_interleave(repeat, dim=1)
        v_t = v_t.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal).transpose(1, 2)


def run_shape(shape, quick=False):
    device = torch.device("cuda")
    q = torch.randn(shape["batch"], shape["seqlen_q"], shape["nheads"], shape["headdim"], device=device, dtype=torch.float16)
    k = torch.randn(shape["batch"], shape["seqlen_k"], shape["nheads_k"], shape["headdim"], device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    iters = 5 if quick else 20
    native_ms = _event_time_ms(lambda: flash_attn_func(q, k, v, causal=shape["causal"]), iters=iters)
    backend = get_last_flash_attn_backend()
    sdpa_ms = _event_time_ms(lambda: _sdpa(q, k, v, shape["causal"]), iters=iters)
    return {
        "name": shape["name"],
        "native_ms": native_ms,
        "sdpa_ms": sdpa_ms,
        "speedup_vs_sdpa": sdpa_ms / native_ms,
        "backend": backend,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=f"sm75_attention_micro_{int(time.time())}.json")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    results = [run_shape(shape, quick=args.quick) for shape in DEFAULT_SHAPES]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
