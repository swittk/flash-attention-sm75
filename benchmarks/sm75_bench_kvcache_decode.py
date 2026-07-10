import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flash_attn import flash_attn_with_kvcache, get_last_flash_attn_backend


DECODE_AXES = {
    "batch": [1, 2, 4, 8, 16],
    "q_len": [1, 2, 4, 8],
    "cache_len": [512, 2048, 4096, 8192, 16384, 32768],
    "heads_q_kv": [(32, 8), (16, 1)],
    "head_dim": [64, 128],
    "cache_layout": ["contiguous", "paged"],
    "rotary": [False, True],
    "cache_batch_idx": [False, True],
    "cache_leftpad": [False, True],
}


def _classify_backend(backend):
    if not backend:
        return "unknown"
    if "paged_direct" in backend:
        return "direct native CUDA paged KV-cache kernel"
    if "paged_gather_dense" in backend:
        return "Python page gather plus specialized SM75 tiled kernel"
    if "generic" in backend or "varlen" in backend or "paged_repack" in backend:
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


def _rotate(x, cos, sin, positions, interleaved):
    rotary_dim = cos.shape[-1] * 2
    cos_pos = cos.index_select(0, positions.reshape(-1)).reshape(*positions.shape, cos.shape[-1]).unsqueeze(-2)
    sin_pos = sin.index_select(0, positions.reshape(-1)).reshape(*positions.shape, sin.shape[-1]).unsqueeze(-2)
    x_rot = x[..., :rotary_dim]
    if interleaved:
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        out_rot = torch.empty_like(x_rot)
        out_rot[..., 0::2] = x_even * cos_pos - x_odd * sin_pos
        out_rot[..., 1::2] = x_odd * cos_pos + x_even * sin_pos
    else:
        x_first, x_second = x_rot.chunk(2, dim=-1)
        out_rot = torch.cat((x_first * cos_pos - x_second * sin_pos, x_second * cos_pos + x_first * sin_pos), dim=-1)
    return torch.cat((out_rot, x[..., rotary_dim:]), dim=-1)


def _paged_to_dense(k_cache, v_cache, block_table, cache_len, batch):
    page_size = k_cache.shape[1]
    k_rows = []
    v_rows = []
    for b in range(batch):
        num_pages = math.ceil(cache_len / page_size)
        block_ids = block_table[b, :num_pages].to(device=k_cache.device, dtype=torch.long)
        k_rows.append(k_cache.index_select(0, block_ids).reshape(-1, k_cache.shape[2], k_cache.shape[3])[:cache_len])
        v_rows.append(v_cache.index_select(0, block_ids).reshape(-1, v_cache.shape[2], v_cache.shape[3])[:cache_len])
    return torch.stack(k_rows, dim=0), torch.stack(v_rows, dim=0)


def _make_case_tensors(case, device):
    torch.manual_seed(789)
    batch = case["batch"]
    q_len = case["q_len"]
    cache_len = case["cache_len"]
    nheads, nheads_k = case["nheads"], case["nheads_k"]
    headdim = case["headdim"]
    q = torch.randn(batch, q_len, nheads, headdim, device=device, dtype=torch.float16)
    k_new = torch.randn(batch, q_len, nheads_k, headdim, device=device, dtype=torch.float16)
    v_new = torch.randn_like(k_new)
    cache_seqlens = torch.full((batch,), cache_len, device=device, dtype=torch.int32)
    cache_batch_idx = None
    cache_leftpad = None
    cache_rows = batch
    if case["cache_batch_idx"]:
        cache_rows = batch + 2
        cache_batch_idx = torch.arange(batch, device=device, dtype=torch.int32).flip(0)
    if case.get("cache_leftpad", False):
        max_leftpad = max(1, min(128, cache_len // 16))
        cache_leftpad = (torch.arange(batch, device=device, dtype=torch.int32) % max_leftpad) + 1

    block_table = None
    if case["cache_layout"] == "paged":
        page_size = 256
        pages_per_seq = math.ceil((cache_len + q_len) / page_size)
        cache_rows_for_pages = cache_rows * pages_per_seq
        k_cache = torch.randn(cache_rows_for_pages, page_size, nheads_k, headdim, device=device, dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
        block_table = torch.arange(cache_rows_for_pages, device=device, dtype=torch.int32).reshape(cache_rows, pages_per_seq)
    else:
        k_cache = torch.randn(cache_rows, cache_len + q_len, nheads_k, headdim, device=device, dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)

    rotary_cos = rotary_sin = None
    if case["rotary"]:
        rotary_dim = min(64, headdim)
        if rotary_dim % 16 != 0:
            rotary_dim -= rotary_dim % 16
        positions = torch.arange(cache_len + q_len + 1, device=device, dtype=torch.float32)[:, None]
        dims = torch.arange(rotary_dim // 2, device=device, dtype=torch.float32)[None, :]
        freqs = positions / (10000 ** (2 * dims / rotary_dim))
        rotary_cos = torch.cos(freqs).to(torch.float16)
        rotary_sin = torch.sin(freqs).to(torch.float16)

    return q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, cache_leftpad, block_table, rotary_cos, rotary_sin


def _baseline(case, q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, cache_leftpad, block_table, rotary_cos, rotary_sin):
    batch, q_len = q.shape[0], q.shape[1]
    cache_len = int(cache_seqlens[0].item())
    if block_table is None:
        idx = torch.arange(batch, device=q.device) if cache_batch_idx is None else cache_batch_idx.to(torch.long)
        dense_k = k_cache.index_select(0, idx)[:, :cache_len].clone()
        dense_v = v_cache.index_select(0, idx)[:, :cache_len].clone()
    else:
        idx = torch.arange(batch, device=q.device) if cache_batch_idx is None else cache_batch_idx.to(torch.long)
        dense_k, dense_v = _paged_to_dense(k_cache, v_cache, block_table.index_select(0, idx), cache_len, batch)
    q_attn = q
    k_update = k_new
    if rotary_cos is not None:
        offsets = cache_seqlens.to(torch.long)[:, None] + torch.arange(q_len, device=q.device, dtype=torch.long)[None, :]
        q_attn = _rotate(q, rotary_cos, rotary_sin, offsets, interleaved=False)
        k_update = _rotate(k_new, rotary_cos, rotary_sin, offsets, interleaved=False)
    dense_k = torch.cat((dense_k, k_update), dim=1)
    dense_v = torch.cat((dense_v, v_new), dim=1)
    if cache_leftpad is None or bool(torch.all(cache_leftpad == 0)):
        return _sdpa_decode(q_attn, dense_k, dense_v, causal=True)
    outs = []
    end = cache_len + q_len
    for batch_idx, leftpad in enumerate(cache_leftpad.tolist()):
        outs.append(
            _sdpa_decode(
                q_attn[batch_idx : batch_idx + 1],
                dense_k[batch_idx : batch_idx + 1, leftpad:end],
                dense_v[batch_idx : batch_idx + 1, leftpad:end],
                causal=True,
            )
        )
    return torch.cat(outs, dim=0)


def run_case(case, warmup, iters):
    device = torch.device("cuda")
    (
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        rotary_cos,
        rotary_sin,
    ) = _make_case_tensors(case, device)
    # Benchmark update+decode on cloned caches so the measured function is repeatable.
    def native_fn():
        return flash_attn_with_kvcache(
            q,
            k_cache.clone(),
            v_cache.clone(),
            k=k_new,
            v=v_new,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            cache_leftpad=cache_leftpad,
            block_table=block_table,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            causal=True,
            rotary_interleaved=False,
        )

    def baseline_fn():
        return _baseline(case, q, k_cache, v_cache, k_new, v_new, cache_seqlens, cache_batch_idx, cache_leftpad, block_table, rotary_cos, rotary_sin)

    native_out = native_fn()
    torch.cuda.synchronize()
    backend = get_last_flash_attn_backend()
    baseline_out = baseline_fn()
    torch.cuda.synchronize()
    native_ms, native_peak = _event_time_ms(native_fn, warmup, iters)
    baseline_ms, baseline_peak = _event_time_ms(baseline_fn, warmup, iters)
    return {
        "name": case["name"],
        "api": "flash_attn_with_kvcache",
        "family": "kvcache_decode",
        "cache_layout": case["cache_layout"],
        "batch": case["batch"],
        "q_len": case["q_len"],
        "cache_len": case["cache_len"],
        "nheads": case["nheads"],
        "nheads_k": case["nheads_k"],
        "headdim": case["headdim"],
        "rotary": case["rotary"],
        "cache_batch_idx": case["cache_batch_idx"],
        "cache_leftpad": case.get("cache_leftpad", False),
        "native_ms": native_ms,
        "baseline": "torch.scaled_dot_product_attention",
        "baseline_ms": baseline_ms,
        "sdpa_ms": baseline_ms,
        "speedup_vs_baseline": baseline_ms / native_ms,
        "speedup_vs_sdpa": baseline_ms / native_ms,
        "peak_vram_native_mb": native_peak,
        "peak_vram_baseline_mb": baseline_peak,
        "correctness_max_abs_error": float((native_out.float() - baseline_out.float()).abs().max().item()),
        "backend": backend,
        "backend_class": _classify_backend(backend),
    }


def _default_cases(cache_len, q_len):
    return [
        dict(
            name=f"decode_q{q_len}_cache{cache_len}_hd64",
            batch=1,
            q_len=q_len,
            cache_len=cache_len,
            nheads=16,
            nheads_k=4,
            headdim=64,
            cache_layout="contiguous",
            rotary=False,
            cache_batch_idx=False,
            cache_leftpad=False,
        )
    ]


def _broad_cases():
    rows = []
    for idx, cache_len in enumerate(DECODE_AXES["cache_len"]):
        nheads, nheads_k = DECODE_AXES["heads_q_kv"][idx % len(DECODE_AXES["heads_q_kv"])]
        rows.append(
            dict(
                name=f"decode_cover_cache{cache_len}",
                batch=DECODE_AXES["batch"][idx % len(DECODE_AXES["batch"])],
                q_len=DECODE_AXES["q_len"][idx % len(DECODE_AXES["q_len"])],
                cache_len=cache_len,
                nheads=nheads,
                nheads_k=nheads_k,
                headdim=DECODE_AXES["head_dim"][idx % len(DECODE_AXES["head_dim"])],
                cache_layout=DECODE_AXES["cache_layout"][idx % len(DECODE_AXES["cache_layout"])],
                rotary=DECODE_AXES["rotary"][idx % len(DECODE_AXES["rotary"])],
                cache_batch_idx=DECODE_AXES["cache_batch_idx"][(idx // 2) % len(DECODE_AXES["cache_batch_idx"])],
                cache_leftpad=DECODE_AXES["cache_leftpad"][(idx // 3) % len(DECODE_AXES["cache_leftpad"])],
            )
        )
    for q_len in DECODE_AXES["q_len"]:
        rows.append(dict(name=f"decode_q{q_len}_paged_rotary", batch=2, q_len=q_len, cache_len=2048, nheads=32, nheads_k=8, headdim=128, cache_layout="paged", rotary=True, cache_batch_idx=False, cache_leftpad=False))
    rows.append(dict(name="decode_batch16_contiguous", batch=16, q_len=1, cache_len=4096, nheads=16, nheads_k=1, headdim=64, cache_layout="contiguous", rotary=False, cache_batch_idx=True, cache_leftpad=True))
    return rows


def _final_broad_cases():
    rows = []
    for batch in DECODE_AXES["batch"]:
        for q_len in DECODE_AXES["q_len"]:
            for cache_len in DECODE_AXES["cache_len"]:
                for nheads, nheads_k in DECODE_AXES["heads_q_kv"]:
                    for headdim in DECODE_AXES["head_dim"]:
                        for cache_layout in DECODE_AXES["cache_layout"]:
                            for rotary in DECODE_AXES["rotary"]:
                                for cache_batch_idx in DECODE_AXES["cache_batch_idx"]:
                                    for cache_leftpad in DECODE_AXES["cache_leftpad"]:
                                        rows.append(
                                            dict(
                                                name=(
                                                    f"decode_b{batch}_q{q_len}_c{cache_len}_h{nheads}_{nheads_k}"
                                                    f"_d{headdim}_{cache_layout}_rot{int(rotary)}"
                                                    f"_idx{int(cache_batch_idx)}_left{int(cache_leftpad)}"
                                                ),
                                                batch=batch,
                                                q_len=q_len,
                                                cache_len=cache_len,
                                                nheads=nheads,
                                                nheads_k=nheads_k,
                                                headdim=headdim,
                                                cache_layout=cache_layout,
                                                rotary=rotary,
                                                cache_batch_idx=cache_batch_idx,
                                                cache_leftpad=cache_leftpad,
                                            )
                                        )
    return rows


def _exhaustive_cases():
    rows = []
    for batch in DECODE_AXES["batch"]:
        for q_len in DECODE_AXES["q_len"]:
            for cache_len in DECODE_AXES["cache_len"]:
                for nheads, nheads_k in DECODE_AXES["heads_q_kv"]:
                    for headdim in DECODE_AXES["head_dim"]:
                        for cache_layout in DECODE_AXES["cache_layout"]:
                            for rotary in DECODE_AXES["rotary"]:
                                for cache_batch_idx in DECODE_AXES["cache_batch_idx"]:
                                    for cache_leftpad in DECODE_AXES["cache_leftpad"]:
                                        rows.append(dict(name=f"decode_b{batch}_q{q_len}_c{cache_len}_h{nheads}_{nheads_k}_d{headdim}_{cache_layout}_rot{int(rotary)}_idx{int(cache_batch_idx)}_left{int(cache_leftpad)}", batch=batch, q_len=q_len, cache_len=cache_len, nheads=nheads, nheads_k=nheads_k, headdim=headdim, cache_layout=cache_layout, rotary=rotary, cache_batch_idx=cache_batch_idx, cache_leftpad=cache_leftpad))
    return rows


def _write_payload(path, matrix, warmup, iters, rows, shard_index, num_shards, planned_rows, complete):
    payload = {
        "matrix": matrix,
        "warmup": warmup,
        "iters": iters,
        "rows": rows,
        "axis_spec": {"decode": DECODE_AXES},
        "shard_index": shard_index,
        "num_shards": num_shards,
        "row_count": len(rows),
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
    parser.add_argument("--cache-len", type=int, default=4096)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--matrix", choices=("default", "broad", "final-broad", "exhaustive"), default="default")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", "--out", dest="output", default=f"sm75_kvcache_decode_{int(time.time())}.json")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("SM75 CUDA device is required")
    if args.matrix == "default":
        cases = _default_cases(args.cache_len, args.q_len)
    elif args.matrix == "broad":
        cases = _broad_cases()
    elif args.matrix == "final-broad":
        cases = _final_broad_cases()
    else:
        cases = _exhaustive_cases()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1:
        cases = [case for idx, case in enumerate(cases) if idx % args.num_shards == args.shard_index]
    if args.max_rows:
        cases = cases[: args.max_rows]
    rows = []
    for case in cases:
        rows.append(run_case(case, args.warmup, args.iters))
        _write_payload(
            args.output,
            args.matrix,
            args.warmup,
            args.iters,
            rows,
            args.shard_index,
            args.num_shards,
            len(cases),
            False,
        )
    payload = _write_payload(
        args.output,
        args.matrix,
        args.warmup,
        args.iters,
        rows,
        args.shard_index,
        args.num_shards,
        len(cases),
        True,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
