import pytest
import torch

from flash_attn import flash_attn_with_kvcache, get_last_flash_attn_backend

from sm75_test_utils import attention_ref, close_to_ref, enable_native_only, make_qkv, require_sm75


def _rotary_ref(x, cos, sin, positions, interleaved=False):
    rotary_dim = cos.shape[-1] * 2
    cos_pos = cos.index_select(0, positions.reshape(-1)).reshape(*positions.shape, cos.shape[-1]).unsqueeze(-2)
    sin_pos = sin.index_select(0, positions.reshape(-1)).reshape(*positions.shape, sin.shape[-1]).unsqueeze(-2)
    x_rot = x[..., :rotary_dim]
    if interleaved:
        out_rot = torch.empty_like(x_rot)
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        out_rot[..., 0::2] = x_even * cos_pos - x_odd * sin_pos
        out_rot[..., 1::2] = x_odd * cos_pos + x_even * sin_pos
    else:
        x_first, x_second = x_rot.chunk(2, dim=-1)
        out_rot = torch.cat((x_first * cos_pos - x_second * sin_pos, x_second * cos_pos + x_first * sin_pos), dim=-1)
    return torch.cat((out_rot, x[..., rotary_dim:]), dim=-1)


@pytest.mark.parametrize("q_len", [1, 3])
def test_kvcache_update_and_decode_native(monkeypatch, q_len):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, k, v = make_qkv(batch=2, seqlen_q=q_len, seqlen_k=q_len, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(2, 16, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor([5, 7], device=device, dtype=torch.int32)
    k_before = k_cache.clone()
    v_before = v_cache.clone()

    out, lse = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        causal=True,
        return_softmax_lse=True,
    )

    expected = []
    for batch, start in enumerate(cache_seqlens.tolist()):
        expected_k = torch.cat([k_before[batch : batch + 1, :start], k[batch : batch + 1]], dim=1)
        expected_v = torch.cat([v_before[batch : batch + 1, :start], v[batch : batch + 1]], dim=1)
        expected.append(attention_ref(q[batch : batch + 1], expected_k, expected_v, causal=True))
        torch.testing.assert_close(k_cache[batch, start : start + q_len], k[batch])
        torch.testing.assert_close(v_cache[batch, start : start + q_len], v[batch])

    close_to_ref(out, torch.cat(expected, dim=0))
    assert lse.shape == (q.shape[0], q.shape[2], q.shape[1])
    assert get_last_flash_attn_backend().startswith("sm75_native_kvcache_fwd")


def test_kvcache_cache_batch_idx_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, k, v = make_qkv(batch=2, seqlen_q=1, seqlen_k=1, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(3, 12, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    cache_batch_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([4, 6], device=device, dtype=torch.int32)
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
        causal=True,
    )

    expected = []
    for q_batch, cache_batch in enumerate(cache_batch_idx.tolist()):
        start = int(cache_seqlens[q_batch].item())
        expected_k = torch.cat([k_before[cache_batch : cache_batch + 1, :start], k[q_batch : q_batch + 1]], dim=1)
        expected_v = torch.cat([v_before[cache_batch : cache_batch + 1, :start], v[q_batch : q_batch + 1]], dim=1)
        expected.append(attention_ref(q[q_batch : q_batch + 1], expected_k, expected_v, causal=True))
        torch.testing.assert_close(k_cache[cache_batch, start : start + 1], k[q_batch])
        torch.testing.assert_close(v_cache[cache_batch, start : start + 1], v[q_batch])

    close_to_ref(out, torch.cat(expected, dim=0))
    assert get_last_flash_attn_backend().startswith("sm75_native_kvcache_fwd")


def test_kvcache_softcap_local_alibi_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, _k, _v = make_qkv(batch=2, seqlen_q=3, seqlen_k=3, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(2, 10, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor([6, 6], device=device, dtype=torch.int32)
    alibi_slopes = torch.rand(2, 4, device=device, dtype=torch.float32) * 0.2

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        causal=False,
        window_size=(3, 1),
        softcap=18.0,
        alibi_slopes=alibi_slopes,
    )

    expected = attention_ref(
        q,
        k_cache[:, :6],
        v_cache[:, :6],
        causal=False,
        window_size=(3, 1),
        softcap=18.0,
        alibi_slopes=alibi_slopes,
    )
    close_to_ref(out, expected, atol=7e-2, rtol=7e-2)
    assert get_last_flash_attn_backend().startswith("sm75_native_kvcache_fwd_dense")


def test_kvcache_leftpad_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, _k, _v = make_qkv(batch=2, seqlen_q=2, seqlen_k=2, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(2, 8, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor([6, 7], device=device, dtype=torch.int32)
    cache_leftpad = torch.tensor([2, 3], device=device, dtype=torch.int32)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        cache_leftpad=cache_leftpad,
        causal=False,
        window_size=(2, 1),
        num_splits=3,
    )

    expected = []
    for batch, (leftpad, end) in enumerate(zip(cache_leftpad.tolist(), cache_seqlens.tolist())):
        expected.append(
            attention_ref(
                q[batch : batch + 1],
                k_cache[batch : batch + 1, leftpad:end],
                v_cache[batch : batch + 1, leftpad:end],
                causal=False,
                window_size=(2, 1),
            )
        )
    close_to_ref(out, torch.cat(expected, dim=0), atol=7e-2, rtol=7e-2)
    assert get_last_flash_attn_backend().startswith("sm75_native_kvcache_fwd_varlen")


@pytest.mark.parametrize("interleaved", [False, True])
def test_kvcache_rotary_update_native(monkeypatch, interleaved):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, k, v = make_qkv(batch=2, seqlen_q=2, seqlen_k=2, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(2, 10, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    k_before = k_cache.clone()
    v_before = v_cache.clone()
    cache_seqlens = torch.tensor([3, 4], device=device, dtype=torch.int32)
    rotary_dim = 32
    angle = torch.rand(10, rotary_dim // 2, device=device, dtype=torch.float32) * 6.283185307179586
    cos = torch.cos(angle).to(torch.float16)
    sin = torch.sin(angle).to(torch.float16)
    offsets = cache_seqlens.to(torch.long)[:, None] + torch.arange(q.shape[1], device=device, dtype=torch.long)[None, :]
    q_ro = _rotary_ref(q, cos, sin, offsets, interleaved=interleaved)
    k_ro = _rotary_ref(k, cos, sin, offsets, interleaved=interleaved)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        causal=True,
        rotary_interleaved=interleaved,
    )

    expected = []
    for batch, start in enumerate(cache_seqlens.tolist()):
        end = start + q.shape[1]
        expected_k = k_before[batch : batch + 1].clone()
        expected_v = v_before[batch : batch + 1].clone()
        expected_k[:, start:end].copy_(k_ro[batch : batch + 1])
        expected_v[:, start:end].copy_(v[batch : batch + 1])
        expected.append(attention_ref(q_ro[batch : batch + 1], expected_k[:, :end], expected_v[:, :end], causal=True))
        torch.testing.assert_close(k_cache[batch, start:end], k_ro[batch])
        torch.testing.assert_close(v_cache[batch, start:end], v[batch])

    close_to_ref(out, torch.cat(expected, dim=0), atol=7e-2, rtol=7e-2)
    assert get_last_flash_attn_backend().startswith("sm75_native_kvcache_fwd_varlen")
