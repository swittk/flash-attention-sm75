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


def test_paged_kvcache_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, k, v = make_qkv(batch=2, seqlen_q=2, seqlen_k=2, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(4, 256, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    v_cache.normal_()
    block_table = torch.tensor([[1], [3]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([3, 5], device=device, dtype=torch.int32)
    k_before = k_cache.clone()
    v_before = v_cache.clone()

    out = flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, block_table=block_table, causal=True
    )

    expected = []
    for batch, block_id in enumerate(block_table[:, 0].tolist()):
        start = int(cache_seqlens[batch].item())
        expected_k = torch.cat([k_before[block_id : block_id + 1, :start], k[batch : batch + 1]], dim=1)
        expected_v = torch.cat([v_before[block_id : block_id + 1, :start], v[batch : batch + 1]], dim=1)
        expected.append(attention_ref(q[batch : batch + 1], expected_k, expected_v, causal=True))
        torch.testing.assert_close(k_cache[block_id, start : start + 2], k[batch])
        torch.testing.assert_close(v_cache[block_id, start : start + 2], v[batch])

    close_to_ref(out, torch.cat(expected, dim=0))
    assert "_paged_direct_fused_update_fa2_tiled_splitkv_" in get_last_flash_attn_backend()


def test_paged_kvcache_shuffled_batch_idx_uses_direct_fa2(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1104)
    q = torch.randn(2, 8, 8, 128, device=device, dtype=torch.float16)
    k_cache = torch.randn(8, 256, 2, 128, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor(
        [[6, 1], [3, 7], [5, 0], [2, 4]],
        device=device,
        dtype=torch.int32,
    )
    cache_batch_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    cache_seqlens = torch.full((2,), 512, device=device, dtype=torch.int32)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        block_table=block_table,
        causal=True,
    )

    expected = []
    for q_batch, cache_batch in enumerate(cache_batch_idx.tolist()):
        blocks = block_table[cache_batch].to(torch.long)
        dense_k = k_cache.index_select(0, blocks).reshape(1, 512, 2, 128)
        dense_v = v_cache.index_select(0, blocks).reshape(1, 512, 2, 128)
        expected.append(attention_ref(q[q_batch : q_batch + 1], dense_k, dense_v, causal=True))
    close_to_ref(out, torch.cat(expected, dim=0))
    assert "_paged_direct_fa2_tiled_splitkv_" in get_last_flash_attn_backend()


def test_paged_kvcache_direct_batch_idx_leftpad_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, k, v = make_qkv(batch=2, seqlen_q=2, seqlen_k=2, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(8, 8, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor(
        [
            [4, 1],
            [7, 2],
            [3, 6],
            [0, 5],
        ],
        device=device,
        dtype=torch.int32,
    )
    cache_batch_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([9, 10], device=device, dtype=torch.int32)
    cache_leftpad = torch.tensor([2, 1], device=device, dtype=torch.int32)

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
        causal=True,
    )

    expected = []
    page_size = k_cache.shape[1]
    for q_batch, cache_batch in enumerate(cache_batch_idx.tolist()):
        start = int(cache_seqlens[q_batch].item())
        end = start + q.shape[1]
        leftpad = int(cache_leftpad[q_batch].item())
        blocks = block_table[cache_batch, : (end + page_size - 1) // page_size].to(torch.long)
        dense_k = k_cache.index_select(0, blocks).reshape(1, -1, k_cache.shape[2], k_cache.shape[3])
        dense_v = v_cache.index_select(0, blocks).reshape(1, -1, v_cache.shape[2], v_cache.shape[3])
        expected.append(attention_ref(q[q_batch : q_batch + 1], dense_k[:, leftpad:end], dense_v[:, leftpad:end], causal=True))
        update_block = int(block_table[cache_batch, start // page_size].item())
        update_offset = start % page_size
        torch.testing.assert_close(k_cache[update_block, update_offset : update_offset + q.shape[1]], k[q_batch])
        torch.testing.assert_close(v_cache[update_block, update_offset : update_offset + q.shape[1]], v[q_batch])

    close_to_ref(out, torch.cat(expected, dim=0))
    assert "_paged_direct_fused_update_wmma_splitkv_" in get_last_flash_attn_backend()


def test_paged_kvcache_direct_rotary_batch_idx_leftpad_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    q, k, v = make_qkv(batch=2, seqlen_q=4, seqlen_k=4, nheads=4, nheads_k=2, headdim=64)
    k_cache = torch.randn(12, 8, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor(
        [
            [4, 1, 9],
            [7, 2, 10],
            [3, 6, 11],
            [0, 5, 8],
        ],
        device=device,
        dtype=torch.int32,
    )
    cache_batch_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([6, 7], device=device, dtype=torch.int32)
    cache_leftpad = torch.tensor([1, 2], device=device, dtype=torch.int32)
    rotary_dim = 32
    angle = torch.rand(16, rotary_dim // 2, device=device, dtype=torch.float32) * 6.283185307179586
    cos = torch.cos(angle).to(torch.float16)
    sin = torch.sin(angle).to(torch.float16)
    offsets = cache_seqlens.to(torch.long)[:, None] + torch.arange(q.shape[1], device=device, dtype=torch.long)[None, :]
    q_ro = _rotary_ref(q, cos, sin, offsets)
    k_ro = _rotary_ref(k, cos, sin, offsets)

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
        rotary_cos=cos,
        rotary_sin=sin,
        rotary_interleaved=False,
        causal=True,
    )

    expected = []
    page_size = k_cache.shape[1]
    for q_batch, cache_batch in enumerate(cache_batch_idx.tolist()):
        start = int(cache_seqlens[q_batch].item())
        end = start + q.shape[1]
        leftpad = int(cache_leftpad[q_batch].item())
        blocks = block_table[cache_batch, : (end + page_size - 1) // page_size].to(torch.long)
        dense_k = k_cache.index_select(0, blocks).reshape(1, -1, k_cache.shape[2], k_cache.shape[3])
        dense_v = v_cache.index_select(0, blocks).reshape(1, -1, v_cache.shape[2], v_cache.shape[3])
        expected.append(attention_ref(q_ro[q_batch : q_batch + 1], dense_k[:, leftpad:end], dense_v[:, leftpad:end], causal=True))
        for token_offset, pos in enumerate(range(start, end)):
            update_block = int(block_table[cache_batch, pos // page_size].item())
            update_offset = pos % page_size
            torch.testing.assert_close(
                k_cache[update_block, update_offset],
                k_ro[q_batch, token_offset],
                atol=1e-3,
                rtol=1e-3,
            )
            torch.testing.assert_close(v_cache[update_block, update_offset], v[q_batch, token_offset])

    close_to_ref(out, torch.cat(expected, dim=0), atol=7e-2, rtol=7e-2)
    assert "_paged_direct_fused_update_wmma_splitkv_" in get_last_flash_attn_backend()
