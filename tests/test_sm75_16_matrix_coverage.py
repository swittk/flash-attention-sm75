import os

import pytest
import torch
import torch.nn.functional as F

from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

from sm75_test_utils import attention_ref, assert_sm75_backend, close_to_ref, enable_native_only, require_sm75, varlen_ref


FULL_MATRIX = os.getenv("FLASH_ATTN_SM75_FULL_MATRIX", "0").upper() in {"1", "TRUE", "YES", "ON"}

BATCH_AXIS = [1, 2, 4, 8]
SEQLEN_Q_AXIS = [1, 2, 8, 16, 128, 512, 1024, 2048, 4096, 8192, 16384]
SEQLEN_K_AXIS = [1, 128, 512, 1024, 2048, 4096, 8192, 16384]
HEADS_Q_KV_AXIS = [(32, 32), (32, 16), (32, 8), (32, 4), (16, 1)]
HEAD_DIM_AXIS = [32, 64, 80, 96, 112, 128, 160, 192, 224, 256]
CACHE_Q_LEN_AXIS = [1, 2, 4, 8]
CACHE_LEN_AXIS = [512, 2048, 4096, 8192, 16384, 32768]
CACHE_HEADS_Q_KV_AXIS = [(32, 8), (16, 1)]
CACHE_HEAD_DIM_AXIS = [64, 128]


def _dense_cases():
    cases = [
        (1, 1, 1, 32, 32, 32, True),
        (2, 2, 128, 32, 16, 64, False),
        (4, 8, 512, 32, 8, 80, True),
        (1, 16, 1024, 32, 4, 96, False),
        (2, 128, 128, 16, 1, 112, True),
        (1, 512, 1024, 32, 8, 128, True),
        (1, 128, 512, 32, 4, 160, False),
        (1, 128, 128, 32, 32, 192, True),
        (1, 64, 512, 16, 1, 224, False),
        (1, 64, 128, 32, 8, 256, True),
    ]
    if FULL_MATRIX:
        for idx, seqlen_q in enumerate(SEQLEN_Q_AXIS):
            batch = BATCH_AXIS[idx % len(BATCH_AXIS)]
            nheads, nheads_k = HEADS_Q_KV_AXIS[idx % len(HEADS_Q_KV_AXIS)]
            headdim = HEAD_DIM_AXIS[idx % len(HEAD_DIM_AXIS)]
            seqlen_k = seqlen_q if seqlen_q in SEQLEN_K_AXIS else SEQLEN_K_AXIS[idx % len(SEQLEN_K_AXIS)]
            if seqlen_q >= 4096:
                batch = 1
                nheads, nheads_k = (16, 1)
                headdim = 64 if seqlen_q == 16384 else headdim
            cases.append((batch, seqlen_q, seqlen_k, nheads, nheads_k, headdim, bool(idx % 2)))
        for idx, seqlen_k in enumerate(SEQLEN_K_AXIS):
            cases.append((1, min(128, seqlen_k), seqlen_k, 32, HEADS_Q_KV_AXIS[idx % len(HEADS_Q_KV_AXIS)][1], HEAD_DIM_AXIS[(idx + 3) % len(HEAD_DIM_AXIS)], bool((idx + 1) % 2)))
    return cases


def _sdpa_ref(q, k, v, causal):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    if q_t.shape[1] != k_t.shape[1]:
        repeat = q_t.shape[1] // k_t.shape[1]
        k_t = k_t.repeat_interleave(repeat, dim=1)
        v_t = v_t.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal).transpose(1, 2)


def _dense_ref(q, k, v, causal):
    score_cells = q.shape[0] * q.shape[1] * k.shape[1] * q.shape[2]
    if score_cells <= 64_000_000:
        return attention_ref(q, k, v, causal=causal)
    return _sdpa_ref(q, k, v, causal=causal)


@pytest.mark.parametrize("batch,seqlen_q,seqlen_k,nheads,nheads_k,headdim,causal", _dense_cases())
def test_dense_axis_matrix_forward_native(monkeypatch, batch, seqlen_q, seqlen_k, nheads, nheads_k, headdim, causal):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1600 + headdim + seqlen_q)
    q = torch.randn(batch, seqlen_q, nheads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch, seqlen_k, nheads_k, headdim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    out = flash_attn_func(q, k, v, causal=causal)
    ref = _dense_ref(q, k, v, causal=causal)
    close_to_ref(out, ref, atol=8e-2, rtol=8e-2)
    assert not torch.isnan(out).any()
    assert_sm75_backend()


@pytest.mark.parametrize(
    "seqlen_q,backend",
    [
        (128, "sm75_native_dense_hd128_fa2_shortq_tiled_kernel128"),
        (256, "sm75_native_fwd_hd128_kernel128"),
    ],
)
def test_dense_hd128_short_query_dispatch_boundary(monkeypatch, seqlen_q, backend):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1640 + seqlen_q)
    q = torch.randn(1, seqlen_q, 32, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 512, 8, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    out = flash_attn_func(q, k, v, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True), atol=8e-2, rtol=8e-2)
    assert_sm75_backend(backend)


@pytest.mark.parametrize("headdim", HEAD_DIM_AXIS if FULL_MATRIX else [64, 96, 160, 224, 256])
def test_packed_axis_matrix_forward_native(monkeypatch, headdim):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1650 + headdim)
    q = torch.randn(1, 64, 8, headdim, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qkv = torch.stack((q, k, v), dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True), atol=8e-2, rtol=8e-2)
    assert_sm75_backend()

    q_gqa = torch.randn(1, 64, 8, headdim, device=device, dtype=torch.float16)
    k_gqa = torch.randn(1, 64, 2, headdim, device=device, dtype=torch.float16)
    v_gqa = torch.randn_like(k_gqa)
    kv = torch.stack((k_gqa, v_gqa), dim=2)
    out = flash_attn_kvpacked_func(q_gqa, kv, causal=False)
    close_to_ref(out, attention_ref(q_gqa, k_gqa, v_gqa, causal=False), atol=8e-2, rtol=8e-2)
    assert_sm75_backend()


@pytest.mark.parametrize("headdim", [32, 64, 80, 96, 112, 128, 160, 192, 224, 256])
def test_dense_dropout_axis_matrix_rejected_on_sm75(monkeypatch, headdim):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1700 + headdim)
    q = torch.randn(1, 64, 8, headdim, device=device, dtype=torch.float16)
    k = torch.randn(1, 80, 4, headdim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    with pytest.raises(RuntimeError, match="SM75 attention dropout is disabled"):
        flash_attn_func(q, k, v, dropout_p=0.1, causal=True, window_size=(32, 0))


def test_qkvpacked_and_kvpacked_backward_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1800)
    qkv = torch.randn(2, 64, 3, 8, 64, device=device, dtype=torch.float16, requires_grad=True)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    out.float().square().mean().backward()
    assert qkv.grad is not None
    assert_sm75_backend()

    q = torch.randn(2, 64, 16, 96, device=device, dtype=torch.float16, requires_grad=True)
    kv = torch.randn(2, 64, 2, 4, 96, device=device, dtype=torch.float16, requires_grad=True)
    out = flash_attn_kvpacked_func(q, kv, causal=False)
    out.float().square().mean().backward()
    assert q.grad is not None and kv.grad is not None
    assert_sm75_backend()


@pytest.mark.parametrize("api", ["varlen", "varlen_qkvpacked", "varlen_kvpacked"])
def test_varlen_packed_backward_native(monkeypatch, api):
    enable_native_only(monkeypatch)
    device = require_sm75()
    lengths = [96, 64, 32, 16] if FULL_MATRIX else [64, 32]
    cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lengths, device=device, dtype=torch.int32), dim=0)
    total = int(cu[-1].item())
    max_len = max(lengths)
    torch.manual_seed(1900 + len(lengths))
    q = torch.randn(total, 8, 128, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(total, 4 if api != "varlen_qkvpacked" else 8, 128, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    if api == "varlen":
        out = flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, causal=True)
    elif api == "varlen_qkvpacked":
        qkv = torch.stack((q, k, v), dim=1)
        qkv.retain_grad()
        out = flash_attn_varlen_qkvpacked_func(qkv, cu, max_len, causal=True)
    else:
        kv = torch.stack((k, v), dim=1)
        kv.retain_grad()
        out = flash_attn_varlen_kvpacked_func(q, kv, cu, cu, max_len, max_len, causal=False)
    ref = varlen_ref(q.detach(), k.detach(), v.detach(), cu, cu, causal=(api != "varlen_kvpacked"))
    close_to_ref(out, ref, atol=8e-2, rtol=8e-2)
    out.float().square().mean().backward()
    assert not torch.isnan(out).any()
    assert_sm75_backend()


def _cache_axis_cases():
    cases = [
        dict(batch=1, q_len=1, cache_len=512, nheads=32, nheads_k=8, headdim=64, paged=False, rotary=False, cache_batch_idx=False, cache_leftpad=False),
        dict(batch=2, q_len=2, cache_len=2048, nheads=16, nheads_k=1, headdim=128, paged=True, rotary=True, cache_batch_idx=False, cache_leftpad=False),
        dict(batch=4, q_len=4, cache_len=4096, nheads=32, nheads_k=8, headdim=64, paged=False, rotary=True, cache_batch_idx=True, cache_leftpad=True),
        dict(batch=2, q_len=8, cache_len=8192, nheads=16, nheads_k=1, headdim=128, paged=True, rotary=False, cache_batch_idx=True, cache_leftpad=True),
    ]
    if FULL_MATRIX:
        for idx, cache_len in enumerate(CACHE_LEN_AXIS):
            nheads, nheads_k = CACHE_HEADS_Q_KV_AXIS[idx % len(CACHE_HEADS_Q_KV_AXIS)]
            cases.append(
                dict(
                    batch=BATCH_AXIS[idx % len(BATCH_AXIS)],
                    q_len=CACHE_Q_LEN_AXIS[idx % len(CACHE_Q_LEN_AXIS)],
                    cache_len=cache_len,
                    nheads=nheads,
                    nheads_k=nheads_k,
                    headdim=CACHE_HEAD_DIM_AXIS[idx % len(CACHE_HEAD_DIM_AXIS)],
                    paged=bool(idx % 2),
                    rotary=bool((idx // 2) % 2),
                    cache_batch_idx=bool((idx // 3) % 2),
                    cache_leftpad=bool((idx // 4) % 2),
                )
            )
    return cases


def _rotary_ref(x, cos, sin, positions):
    rotary_dim = cos.shape[-1] * 2
    cos_pos = cos.index_select(0, positions.reshape(-1)).reshape(*positions.shape, cos.shape[-1]).unsqueeze(-2)
    sin_pos = sin.index_select(0, positions.reshape(-1)).reshape(*positions.shape, sin.shape[-1]).unsqueeze(-2)
    x_first, x_second = x[..., :rotary_dim].chunk(2, dim=-1)
    out_rot = torch.cat((x_first * cos_pos - x_second * sin_pos, x_second * cos_pos + x_first * sin_pos), dim=-1)
    return torch.cat((out_rot, x[..., rotary_dim:]), dim=-1)


@pytest.mark.parametrize("case", _cache_axis_cases())
def test_kvcache_axis_matrix_native(monkeypatch, case):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(2000 + case["cache_len"] + case["q_len"])
    batch = case["batch"]
    q_len = case["q_len"]
    cache_len = case["cache_len"]
    nheads = case["nheads"]
    nheads_k = case["nheads_k"]
    headdim = case["headdim"]
    q = torch.randn(batch, q_len, nheads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch, q_len, nheads_k, headdim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    cache_rows = batch + 2 if case["cache_batch_idx"] else batch
    cache_batch_idx = torch.arange(batch, device=device, dtype=torch.int32).flip(0) if case["cache_batch_idx"] else None
    cache_seqlens = torch.full((batch,), cache_len, device=device, dtype=torch.int32)
    cache_leftpad = None
    if case["cache_leftpad"]:
        cache_leftpad = torch.arange(1, batch + 1, device=device, dtype=torch.int32).clamp_max(max(1, min(64, cache_len // 8)))
    block_table = None
    page_size = 256
    if case["paged"]:
        pages = (cache_len + q_len + page_size - 1) // page_size
        k_cache = torch.randn(cache_rows * pages, page_size, nheads_k, headdim, device=device, dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
        block_table = torch.arange(cache_rows * pages, device=device, dtype=torch.int32).reshape(cache_rows, pages)
    else:
        k_cache = torch.randn(cache_rows, cache_len + q_len, nheads_k, headdim, device=device, dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
    rotary_cos = rotary_sin = None
    q_ref = q
    k_ref_update = k
    if case["rotary"]:
        rotary_dim = min(64, headdim)
        positions = torch.arange(cache_len + q_len + 1, device=device, dtype=torch.float32)[:, None]
        dims = torch.arange(rotary_dim // 2, device=device, dtype=torch.float32)[None, :]
        freqs = positions / (10000 ** (2 * dims / rotary_dim))
        rotary_cos = torch.cos(freqs).to(torch.float16)
        rotary_sin = torch.sin(freqs).to(torch.float16)
        offsets = cache_seqlens.to(torch.long)[:, None] + torch.arange(q_len, device=device, dtype=torch.long)[None, :]
        q_ref = _rotary_ref(q, rotary_cos, rotary_sin, offsets)
        k_ref_update = _rotary_ref(k, rotary_cos, rotary_sin, offsets)

    k_before = k_cache.clone()
    v_before = v_cache.clone()
    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        block_table=block_table,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=False,
        causal=True,
    )

    expected = []
    batch_indices = cache_batch_idx.tolist() if cache_batch_idx is not None else list(range(batch))
    leftpads = cache_leftpad.tolist() if cache_leftpad is not None else [0] * batch
    for q_batch, cache_batch in enumerate(batch_indices):
        leftpad = leftpads[q_batch]
        end = cache_len + q_len
        if block_table is None:
            dense_k = k_before[cache_batch : cache_batch + 1, :end].clone()
            dense_v = v_before[cache_batch : cache_batch + 1, :end].clone()
        else:
            pages = (end + page_size - 1) // page_size
            blocks = block_table[cache_batch, :pages].to(torch.long)
            dense_k = k_before.index_select(0, blocks).reshape(1, -1, nheads_k, headdim)[:, :end].clone()
            dense_v = v_before.index_select(0, blocks).reshape(1, -1, nheads_k, headdim)[:, :end].clone()
        dense_k[:, cache_len:end].copy_(k_ref_update[q_batch : q_batch + 1])
        dense_v[:, cache_len:end].copy_(v[q_batch : q_batch + 1])
        expected.append(attention_ref(q_ref[q_batch : q_batch + 1], dense_k[:, leftpad:end], dense_v[:, leftpad:end], causal=True))
    close_to_ref(out, torch.cat(expected, dim=0), atol=8e-2, rtol=8e-2)
    assert not torch.isnan(out).any()
    assert_sm75_backend()
