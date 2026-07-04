import torch

from flash_attn import flash_attn_with_kvcache, get_last_flash_attn_backend

from sm75_test_utils import attention_ref, close_to_ref, enable_native_only, make_qkv, require_sm75


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
    assert get_last_flash_attn_backend().startswith("sm75_native_kvcache_fwd_paged")
