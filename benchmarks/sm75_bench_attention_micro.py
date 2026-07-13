import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USE_INSTALLED_FLASH_ATTN = os.getenv("FLASH_ATTN_BENCH_USE_INSTALLED", "0").upper() in {
    "1",
    "TRUE",
    "YES",
    "ON",
}
if not USE_INSTALLED_FLASH_ATTN and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    get_last_flash_attn_backend,
)


DEFAULT_SHAPES = [
    dict(api="dense", name="llama_prefill_1k_hd64", batch=1, seqlen_q=1024, seqlen_k=1024, nheads=16, nheads_k=16, headdim=64, causal=True),
    dict(api="dense", name="gqa_prefill_2k_hd128", batch=1, seqlen_q=2048, seqlen_k=2048, nheads=16, nheads_k=4, headdim=128, causal=True),
    dict(api="dense", name="cross_attention_hd64", batch=2, seqlen_q=512, seqlen_k=1024, nheads=8, nheads_k=8, headdim=64, causal=False),
]

PREFILL_AXES = {
    "batch": [1, 2, 4],
    "seqlen": [512, 1024, 2048, 4096, 8192, 16384],
    "heads_q_kv": [(32, 32), (32, 8), (16, 1)],
    "head_dim": [64, 80, 96, 128, 160, 192, 224, 256],
    "causal": [True, False],
    "layouts": ["dense", "qkvpacked", "kvpacked", "varlen", "varlen_qkvpacked", "varlen_kvpacked"],
}

VARLEN_AXES = {
    "total_tokens": [512, 1024, 2048, 4096, 8192, 16384],
    "num_sequences": [1, 2, 4],
    "length_distribution": ["uniform", "one_long_many_short", "random_zipf"],
    "head_dim": [64, 80, 96, 128, 160, 192, 224, 256],
    "heads_q_kv": [(32, 32), (32, 8), (16, 1)],
    "causal": [True, False],
}


def _classify_backend(backend):
    if not backend:
        return "unknown"
    if "generic" in backend or "sliced" in backend or "paged" in backend:
        return "generic native CUDA"
    return "specialized SM75 tiled/online-softmax kernel"


def _event_time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(1, iters), torch.cuda.max_memory_allocated() / 1024**2


def _sdpa(q, k, v, causal):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    if q_t.shape[1] != k_t.shape[1]:
        repeat = q_t.shape[1] // k_t.shape[1]
        k_t = k_t.repeat_interleave(repeat, dim=1)
        v_t = v_t.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal).transpose(1, 2)


def _varlen_sdpa(q, k, v, cu_q, cu_k, causal):
    q_offsets = cu_q.cpu().tolist()
    k_offsets = cu_k.cpu().tolist()
    outs = []
    for q0, q1, k0, k1 in zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]):
        out = _sdpa(q[q0:q1].unsqueeze(0), k[k0:k1].unsqueeze(0), v[k0:k1].unsqueeze(0), causal)
        outs.append(out.squeeze(0))
    return torch.cat(outs, dim=0)


def _max_abs_error(a, b):
    return float((a.float() - b.float()).abs().max().item())


def _make_dense_tensors(shape, device):
    torch.manual_seed(123)
    q = torch.randn(shape["batch"], shape["seqlen_q"], shape["nheads"], shape["headdim"], device=device, dtype=torch.float16)
    k = torch.randn(shape["batch"], shape["seqlen_k"], shape["nheads_k"], shape["headdim"], device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    return q, k, v


def _make_varlen_lengths(total_tokens, num_sequences, distribution):
    if num_sequences == 1:
        return [total_tokens]
    if distribution == "uniform":
        base = total_tokens // num_sequences
        lengths = [base] * num_sequences
        lengths[-1] += total_tokens - sum(lengths)
    elif distribution == "one_long_many_short":
        long_len = max(1, total_tokens - (num_sequences - 1) * 16)
        lengths = [long_len] + [16] * (num_sequences - 1)
    else:
        weights = [1.0 / (idx + 1) for idx in range(num_sequences)]
        denom = sum(weights)
        lengths = [max(1, int(total_tokens * w / denom)) for w in weights]
        lengths[-1] += total_tokens - sum(lengths)
    return lengths


def _make_cu(lengths, device):
    cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lengths, device=device, dtype=torch.int32), dim=0)
    return cu


def _make_varlen_tensors(shape, device):
    lengths = _make_varlen_lengths(shape["total_tokens"], shape["num_sequences"], shape["length_distribution"])
    cu = _make_cu(lengths, device)
    torch.manual_seed(456)
    q = torch.randn(shape["total_tokens"], shape["nheads"], shape["headdim"], device=device, dtype=torch.float16)
    k = torch.randn(shape["total_tokens"], shape["nheads_k"], shape["headdim"], device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    return q, k, v, cu, max(lengths)


def _run_dense(shape, warmup, iters):
    device = torch.device("cuda")
    q, k, v = _make_dense_tensors(shape, device)
    api = shape["api"]
    causal = shape["causal"]

    if api == "dense":
        native_fn = lambda: flash_attn_func(q, k, v, causal=causal)
    elif api == "qkvpacked":
        if q.shape[2] != k.shape[2] or q.shape[1] != k.shape[1]:
            raise ValueError("qkvpacked requires MHA and matching q/k lengths")
        qkv = torch.stack((q, k, v), dim=2)
        native_fn = lambda: flash_attn_qkvpacked_func(qkv, causal=causal)
    elif api == "kvpacked":
        kv = torch.stack((k, v), dim=2)
        native_fn = lambda: flash_attn_kvpacked_func(q, kv, causal=causal)
    else:
        raise ValueError(api)

    baseline_fn = lambda: _sdpa(q, k, v, causal)
    native_out = native_fn()
    torch.cuda.synchronize()
    backend = get_last_flash_attn_backend()
    baseline_out = baseline_fn()
    torch.cuda.synchronize()
    native_ms, native_peak = _event_time_ms(native_fn, warmup, iters)
    baseline_ms, baseline_peak = _event_time_ms(baseline_fn, warmup, iters)
    return {
        "name": shape["name"],
        "api": api,
        "layout": api,
        "family": "prefill",
        "batch": shape["batch"],
        "seqlen_q": shape["seqlen_q"],
        "seqlen_k": shape["seqlen_k"],
        "nheads": shape["nheads"],
        "nheads_k": shape["nheads_k"],
        "headdim": shape["headdim"],
        "causal": causal,
        "native_ms": native_ms,
        "baseline": "torch.scaled_dot_product_attention",
        "baseline_ms": baseline_ms,
        "sdpa_ms": baseline_ms,
        "speedup_vs_baseline": baseline_ms / native_ms,
        "speedup_vs_sdpa": baseline_ms / native_ms,
        "peak_vram_native_mb": native_peak,
        "peak_vram_baseline_mb": baseline_peak,
        "correctness_max_abs_error": _max_abs_error(native_out, baseline_out),
        "backend": backend,
        "backend_class": _classify_backend(backend),
    }


def _run_varlen(shape, warmup, iters):
    device = torch.device("cuda")
    q, k, v, cu, max_len = _make_varlen_tensors(shape, device)
    api = shape["api"]
    causal = shape["causal"]

    if api == "varlen":
        native_fn = lambda: flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, causal=causal)
    elif api == "varlen_qkvpacked":
        if q.shape[1] != k.shape[1]:
            raise ValueError("varlen_qkvpacked requires MHA")
        qkv = torch.stack((q, k, v), dim=1)
        native_fn = lambda: flash_attn_varlen_qkvpacked_func(qkv, cu, max_len, causal=causal)
    elif api == "varlen_kvpacked":
        kv = torch.stack((k, v), dim=1)
        native_fn = lambda: flash_attn_varlen_kvpacked_func(q, kv, cu, cu, max_len, max_len, causal=causal)
    else:
        raise ValueError(api)

    baseline_fn = lambda: _varlen_sdpa(q, k, v, cu, cu, causal)
    native_out = native_fn()
    torch.cuda.synchronize()
    backend = get_last_flash_attn_backend()
    baseline_out = baseline_fn()
    torch.cuda.synchronize()
    native_ms, native_peak = _event_time_ms(native_fn, warmup, iters)
    baseline_ms, baseline_peak = _event_time_ms(baseline_fn, warmup, iters)
    return {
        "name": shape["name"],
        "api": api,
        "layout": api,
        "family": "varlen",
        "total_tokens": shape["total_tokens"],
        "num_sequences": shape["num_sequences"],
        "length_distribution": shape["length_distribution"],
        "nheads": shape["nheads"],
        "nheads_k": shape["nheads_k"],
        "headdim": shape["headdim"],
        "causal": causal,
        "native_ms": native_ms,
        "baseline": "per-sequence torch.scaled_dot_product_attention",
        "baseline_ms": baseline_ms,
        "sdpa_ms": baseline_ms,
        "speedup_vs_baseline": baseline_ms / native_ms,
        "speedup_vs_sdpa": baseline_ms / native_ms,
        "peak_vram_native_mb": native_peak,
        "peak_vram_baseline_mb": baseline_peak,
        "correctness_max_abs_error": _max_abs_error(native_out, baseline_out),
        "backend": backend,
        "backend_class": _classify_backend(backend),
    }


def _prefill_exhaustive():
    rows = []
    for batch in PREFILL_AXES["batch"]:
        for seqlen in PREFILL_AXES["seqlen"]:
            for nheads, nheads_k in PREFILL_AXES["heads_q_kv"]:
                for headdim in PREFILL_AXES["head_dim"]:
                    for causal in PREFILL_AXES["causal"]:
                        rows.append(dict(api="dense", name=f"dense_b{batch}_s{seqlen}_h{nheads}_{nheads_k}_d{headdim}_{'causal' if causal else 'noncausal'}", batch=batch, seqlen_q=seqlen, seqlen_k=seqlen, nheads=nheads, nheads_k=nheads_k, headdim=headdim, causal=causal))
    return rows


def _broad_shapes():
    rows = []
    # Deterministic coverage sample: every requested axis value appears at least once.
    for idx, seqlen in enumerate(PREFILL_AXES["seqlen"]):
        nheads, nheads_k = PREFILL_AXES["heads_q_kv"][idx % len(PREFILL_AXES["heads_q_kv"])]
        headdim = PREFILL_AXES["head_dim"][idx % len(PREFILL_AXES["head_dim"])]
        batch = PREFILL_AXES["batch"][idx % len(PREFILL_AXES["batch"])]
        if seqlen >= 8192:
            batch = 1
            nheads, nheads_k = (16, 1)
            headdim = 64 if seqlen == 8192 else 128
        causal = PREFILL_AXES["causal"][idx % 2]
        rows.append(dict(api="dense", name=f"dense_cover_s{seqlen}_d{headdim}", batch=batch, seqlen_q=seqlen, seqlen_k=seqlen, nheads=nheads, nheads_k=nheads_k, headdim=headdim, causal=causal))
    for idx, headdim in enumerate(PREFILL_AXES["head_dim"]):
        nheads, nheads_k = PREFILL_AXES["heads_q_kv"][idx % len(PREFILL_AXES["heads_q_kv"])]
        api = "kvpacked" if nheads != nheads_k else "qkvpacked"
        rows.append(dict(api=api, name=f"{api}_cover_d{headdim}", batch=1 + (idx % 2), seqlen_q=512, seqlen_k=512, nheads=nheads, nheads_k=nheads_k, headdim=headdim, causal=bool(idx % 2)))
    for idx, total_tokens in enumerate(VARLEN_AXES["total_tokens"]):
        nheads, nheads_k = VARLEN_AXES["heads_q_kv"][idx % len(VARLEN_AXES["heads_q_kv"])]
        api = ["varlen", "varlen_qkvpacked" if nheads == nheads_k else "varlen_kvpacked"][idx % 2]
        rows.append(dict(api=api, name=f"{api}_tokens{total_tokens}", total_tokens=total_tokens, num_sequences=VARLEN_AXES["num_sequences"][idx % len(VARLEN_AXES["num_sequences"])], length_distribution=VARLEN_AXES["length_distribution"][idx % len(VARLEN_AXES["length_distribution"])], nheads=nheads, nheads_k=nheads_k, headdim=VARLEN_AXES["head_dim"][idx % len(VARLEN_AXES["head_dim"])], causal=bool(idx % 2)))
    # Explicit generic-path and cross-attention rows.
    rows.extend(
        [
            dict(api="dense", name="generic_hd160_localish_proxy", batch=1, seqlen_q=1024, seqlen_k=1024, nheads=32, nheads_k=8, headdim=160, causal=True),
            dict(api="dense", name="generic_hd256_cross", batch=1, seqlen_q=512, seqlen_k=2048, nheads=16, nheads_k=1, headdim=256, causal=False),
        ]
    )
    return rows


def _final_broad_shapes():
    rows = []
    for api in ("dense", "qkvpacked", "kvpacked"):
        for batch in PREFILL_AXES["batch"]:
            for seqlen in PREFILL_AXES["seqlen"]:
                for nheads, nheads_k in PREFILL_AXES["heads_q_kv"]:
                    if api == "qkvpacked" and nheads != nheads_k:
                        continue
                    for headdim in PREFILL_AXES["head_dim"]:
                        for causal in PREFILL_AXES["causal"]:
                            rows.append(
                                dict(
                                    api=api,
                                    name=(
                                        f"{api}_b{batch}_s{seqlen}_h{nheads}_{nheads_k}"
                                        f"_d{headdim}_{'causal' if causal else 'noncausal'}"
                                    ),
                                    batch=batch,
                                    seqlen_q=seqlen,
                                    seqlen_k=seqlen,
                                    nheads=nheads,
                                    nheads_k=nheads_k,
                                    headdim=headdim,
                                    causal=causal,
                                )
                            )
    for api in ("varlen", "varlen_qkvpacked", "varlen_kvpacked"):
        for batch in PREFILL_AXES["batch"]:
            for seqlen in PREFILL_AXES["seqlen"]:
                total_tokens = batch * seqlen
                for nheads, nheads_k in PREFILL_AXES["heads_q_kv"]:
                    if api == "varlen_qkvpacked" and nheads != nheads_k:
                        continue
                    for headdim in PREFILL_AXES["head_dim"]:
                        for causal in PREFILL_AXES["causal"]:
                            for distribution in VARLEN_AXES["length_distribution"]:
                                rows.append(
                                    dict(
                                        api=api,
                                        name=(
                                            f"{api}_b{batch}_s{seqlen}_h{nheads}_{nheads_k}"
                                            f"_d{headdim}_{distribution}_{'causal' if causal else 'noncausal'}"
                                        ),
                                        batch=batch,
                                        seqlen=seqlen,
                                        total_tokens=total_tokens,
                                        num_sequences=batch,
                                        length_distribution=distribution,
                                        nheads=nheads,
                                        nheads_k=nheads_k,
                                        headdim=headdim,
                                        causal=causal,
                                    )
                                )
    return rows


def _select_shapes(matrix):
    if matrix == "default":
        return DEFAULT_SHAPES
    if matrix == "broad":
        return _broad_shapes()
    if matrix == "final-broad":
        return _final_broad_shapes()
    if matrix == "exhaustive":
        return _prefill_exhaustive()
    raise ValueError(matrix)


def run_shape(shape, warmup, iters):
    if shape["api"].startswith("varlen"):
        return _run_varlen(shape, warmup, iters)
    return _run_dense(shape, warmup, iters)


def _write_payload(path, matrix, warmup, iters, results, shard_index, num_shards, planned_rows, complete):
    payload = {
        "matrix": matrix,
        "warmup": warmup,
        "iters": iters,
        "rows": results,
        "axis_spec": {"prefill": PREFILL_AXES, "varlen": VARLEN_AXES},
        "shard_index": shard_index,
        "num_shards": num_shards,
        "row_count": len(results),
        "planned_row_count": planned_rows,
        "complete": complete,
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--matrix", choices=("default", "broad", "final-broad", "exhaustive"), default="default")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", "--out", dest="output", default=f"sm75_attention_micro_{int(time.time())}.json")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    warmup = 2 if args.quick else args.warmup
    iters = 5 if args.quick else args.iters
    shapes = _select_shapes(args.matrix)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1:
        shapes = [shape for idx, shape in enumerate(shapes) if idx % args.num_shards == args.shard_index]
    if args.max_rows:
        shapes = shapes[: args.max_rows]
    results = []
    for shape in shapes:
        results.append(run_shape(shape, warmup=warmup, iters=iters))
        _write_payload(
            args.output,
            args.matrix,
            warmup,
            iters,
            results,
            args.shard_index,
            args.num_shards,
            len(shapes),
            False,
        )
    payload = _write_payload(
        args.output,
        args.matrix,
        warmup,
        iters,
        results,
        args.shard_index,
        args.num_shards,
        len(shapes),
        True,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
