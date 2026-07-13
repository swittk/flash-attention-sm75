import pytest
import torch

from flash_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

from sm75_test_utils import (
    assert_sm75_backend,
    attention_ref,
    close_to_ref,
    enable_native_only,
    require_sm75,
    varlen_ref,
)


def _wait_for(stream):
    done = torch.cuda.Event()
    done.record(stream)
    done.synchronize()


@pytest.mark.parametrize("headdim", [64, 80, 256])
def test_dense_forward_uses_current_stream(monkeypatch, headdim):
    enable_native_only(monkeypatch)
    device = require_sm75()
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        torch.manual_seed(7500 + headdim)
        q = torch.randn(1, 33, 8, headdim, device=device, dtype=torch.float16)
        k = torch.randn(1, 47, 2, headdim, device=device, dtype=torch.float16)
        v = torch.randn_like(k)
        out = flash_attn_func(q, k, v, causal=True)
        observed = out.clone()

    _wait_for(stream)
    close_to_ref(observed, attention_ref(q, k, v, causal=True), atol=7e-2, rtol=7e-2)
    assert_sm75_backend()


def test_dense_backward_uses_current_stream(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        torch.manual_seed(7564)
        q = torch.randn(1, 32, 4, 64, device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 32, 4, 64, device=device, dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 32, 4, 64, device=device, dtype=torch.float16, requires_grad=True)
        out = flash_attn_func(q, k, v, causal=True)
        out.float().square().mean().backward()
        grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    _wait_for(stream)
    assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0 for grad in grads)
    assert_sm75_backend()


def test_packed_and_varlen_use_current_stream(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        torch.manual_seed(7580)
        q = torch.randn(1, 31, 4, 80, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        packed_out = flash_attn_qkvpacked_func(torch.stack((q, k, v), dim=2), causal=True)

        q_var = torch.randn(17, 4, 96, device=device, dtype=torch.float16)
        k_var = torch.randn(23, 2, 96, device=device, dtype=torch.float16)
        v_var = torch.randn_like(k_var)
        cu_q = torch.tensor([0, 17], device=device, dtype=torch.int32)
        cu_k = torch.tensor([0, 23], device=device, dtype=torch.int32)
        varlen_out = flash_attn_varlen_func(
            q_var, k_var, v_var, cu_q, cu_k, 17, 23, causal=False
        )
        packed_observed = packed_out.clone()
        varlen_observed = varlen_out.clone()

    _wait_for(stream)
    close_to_ref(packed_observed, attention_ref(q, k, v, causal=True), atol=7e-2, rtol=7e-2)
    close_to_ref(
        varlen_observed,
        varlen_ref(q_var, k_var, v_var, cu_q, cu_k, causal=False),
        atol=7e-2,
        rtol=7e-2,
    )
    assert_sm75_backend()


def test_paged_kvcache_uses_current_stream(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        torch.manual_seed(7600)
        q = torch.randn(2, 1, 4, 64, device=device, dtype=torch.float16)
        k_cache = torch.randn(4, 16, 2, 64, device=device, dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
        block_table = torch.tensor([[1], [3]], device=device, dtype=torch.int32)
        cache_seqlens = torch.tensor([8, 8], device=device, dtype=torch.int32)
        out = flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        )
        observed = out.clone()

    _wait_for(stream)
    expected = []
    for batch, block in enumerate(block_table[:, 0].tolist()):
        expected.append(
            attention_ref(
                q[batch : batch + 1],
                k_cache[block : block + 1, :8],
                v_cache[block : block + 1, :8],
                causal=True,
            )
        )
    close_to_ref(observed, torch.cat(expected, dim=0), atol=7e-2, rtol=7e-2)
    assert_sm75_backend("sm75_native_kvcache_fwd_paged")
