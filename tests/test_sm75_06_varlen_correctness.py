import pytest
import torch

from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    get_last_flash_attn_backend,
)
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

from sm75_test_utils import attention_ref, assert_sm75_backend, close_to_ref, enable_native_only, make_varlen, require_sm75, varlen_ref


def test_varlen_forward_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v, cu_q, cu_k, max_q, max_k = make_varlen(headdim=64)
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=True)
    close_to_ref(out, varlen_ref(q, k, v, cu_q, cu_k, causal=True))
    assert_sm75_backend()


def test_varlen_kvpacked_forward_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v, cu_q, cu_k, max_q, max_k = make_varlen(headdim=64)
    kv = torch.stack([k, v], dim=1)
    out = flash_attn_varlen_kvpacked_func(q, kv, cu_q, cu_k, max_q, max_k, causal=False)
    close_to_ref(out, varlen_ref(q, k, v, cu_q, cu_k, causal=False))
    assert_sm75_backend()


def test_varlen_qkvpacked_forward_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v, cu_q, _cu_k, max_q, _max_k = make_varlen(batch=3, max_q=16, max_k=16, nheads=4, nheads_k=4, headdim=64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qkv = torch.stack([q, k, v], dim=1)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu_q, max_q, causal=True)
    close_to_ref(out, varlen_ref(q, k, v, cu_q, cu_q, causal=True))
    assert_sm75_backend()


def test_varlen_generic_head_dim_features_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v, cu_q, cu_k, max_q, max_k = make_varlen(batch=2, max_q=9, max_k=11, nheads=4, nheads_k=2, headdim=160)
    alibi_slopes = torch.rand(2, 4, device=q.device, dtype=torch.float32) * 0.2
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal=False,
        window_size=(3, 2),
        softcap=18.0,
        alibi_slopes=alibi_slopes,
    )
    ref = varlen_ref(
        q,
        k,
        v,
        cu_q,
        cu_k,
        causal=False,
        window_size=(3, 2),
        softcap=18.0,
        alibi_slopes=alibi_slopes,
    )
    close_to_ref(out, ref, atol=7e-2, rtol=7e-2)
    assert get_last_flash_attn_backend().startswith("sm75_native_varlen_generic_fwd")


def test_varlen_generic_backward_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v, cu_q, cu_k, max_q, max_k = make_varlen(batch=2, max_q=7, max_k=9, nheads=4, nheads_k=2, headdim=160)
    q, k, v = [x.detach().requires_grad_(True) for x in (q, k, v)]
    alibi_slopes = torch.rand(2, 4, device=q.device, dtype=torch.float32) * 0.15
    dout = torch.randn_like(q)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal=False,
        window_size=(2, 3),
        softcap=16.0,
        alibi_slopes=alibi_slopes,
    )
    (out * dout).sum().backward()

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    ref = varlen_ref(
        q_ref,
        k_ref,
        v_ref,
        cu_q,
        cu_k,
        causal=False,
        window_size=(2, 3),
        softcap=16.0,
        alibi_slopes=alibi_slopes,
    )
    (ref * dout.float()).sum().backward()

    close_to_ref(q.grad, q_ref.grad, atol=9e-2, rtol=9e-2)
    close_to_ref(k.grad, k_ref.grad, atol=9e-2, rtol=9e-2)
    close_to_ref(v.grad, v_ref.grad, atol=9e-2, rtol=9e-2)
    assert get_last_flash_attn_backend().startswith("sm75_native_varlen_generic_bwd")


def test_varlen_return_attn_probs_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v, cu_q, cu_k, max_q, max_k = make_varlen(batch=2, max_q=5, max_k=7, nheads=2, nheads_k=2, headdim=64)
    out, lse, probs, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        0.0,
        q.shape[-1] ** -0.5,
        False,
        return_softmax=True,
    )
    ref, ref_probs = varlen_ref(q, k, v, cu_q, cu_k, return_probs=True, max_seqlen_q=max_q, max_seqlen_k=max_k)
    close_to_ref(out, ref)
    close_to_ref(probs, ref_probs, atol=2e-3, rtol=2e-3)
    assert lse.shape == (cu_q.numel() - 1, q.shape[1], max_q)
    assert probs.shape == (cu_q.numel() - 1, q.shape[1], max_q, max_k)
    assert get_last_flash_attn_backend().startswith("sm75_native_varlen_generic_fwd")


def test_varlen_zero_length_query_backward_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    cu_q = torch.tensor([0, 0, 5, 7], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 10, 13], device=device, dtype=torch.int32)
    torch.manual_seed(2)
    q = torch.randn(7, 4, 160, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(13, 2, 160, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(13, 2, 160, device=device, dtype=torch.float16, requires_grad=True)

    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 5, 6, causal=False)
    close_to_ref(out, varlen_ref(q, k, v, cu_q, cu_k, causal=False), atol=7e-2, rtol=7e-2)
    out.sum().backward()

    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()
    assert k.grad[:4].abs().max() == 0
    assert v.grad[:4].abs().max() == 0
    assert get_last_flash_attn_backend().startswith("sm75_native_varlen_generic_bwd")


def test_varlen_qkvpacked_generic_head_dim_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, _k, _v, cu_q, _cu_k, max_q, _max_k = make_varlen(
        batch=2, max_q=8, max_k=8, nheads=2, nheads_k=2, headdim=160
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qkv = torch.stack([q, k, v], dim=1)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu_q, max_q, causal=True)
    close_to_ref(out, varlen_ref(q, k, v, cu_q, cu_q, causal=True), atol=7e-2, rtol=7e-2)
    assert get_last_flash_attn_backend().startswith("sm75_native_varlen_generic_fwd")


def test_varlen_leftpad_seqused_forward_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    cu_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 6, 11], device=device, dtype=torch.int32)
    leftpad_k = torch.tensor([2, 1], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([5, 4], device=device, dtype=torch.int32)
    torch.manual_seed(3)
    q = torch.randn(5, 2, 64, device=device, dtype=torch.float16)
    k = torch.randn(11, 2, 64, device=device, dtype=torch.float16)
    v = torch.randn(11, 2, 64, device=device, dtype=torch.float16)

    out, _lse, _probs, _rng = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        3,
        6,
        0.0,
        q.shape[-1] ** -0.5,
        False,
        leftpad_k=leftpad_k,
        seqused_k=seqused_k,
    )

    expected = []
    for batch in range(2):
        q_i = q[cu_q[batch].item() : cu_q[batch + 1].item()].unsqueeze(0)
        k_start = cu_k[batch].item()
        left = leftpad_k[batch].item()
        used = seqused_k[batch].item()
        k_i = k[k_start + left : k_start + used].unsqueeze(0)
        v_i = v[k_start + left : k_start + used].unsqueeze(0)
        expected.append(attention_ref(q_i, k_i, v_i).squeeze(0))
    close_to_ref(out, torch.cat(expected, dim=0))
    assert get_last_flash_attn_backend().startswith("sm75_native_varlen_sliced_fwd")


def test_varlen_paged_block_table_forward_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(4)
    q = torch.randn(2, 4, 64, device=device, dtype=torch.float16)
    k_cache = torch.randn(4, 4, 2, 64, device=device, dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    cu_q = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([5, 7], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 12], device=device, dtype=torch.int32)
    block_table = torch.tensor([[1, 0], [2, 3]], device=device, dtype=torch.int32)

    out, _lse, _probs, _rng = _flash_attn_varlen_forward(
        q,
        k_cache,
        v_cache,
        cu_q,
        cu_k,
        1,
        7,
        0.0,
        q.shape[-1] ** -0.5,
        False,
        block_table=block_table,
        seqused_k=seqused_k,
        num_splits=2,
    )

    expected = []
    for batch in range(2):
        block_ids = block_table[batch].to(torch.long)
        k_i = k_cache.index_select(0, block_ids).reshape(-1, k_cache.shape[2], k_cache.shape[3])[: seqused_k[batch].item()]
        v_i = v_cache.index_select(0, block_ids).reshape(-1, v_cache.shape[2], v_cache.shape[3])[: seqused_k[batch].item()]
        expected.append(attention_ref(q[batch : batch + 1].unsqueeze(0), k_i.unsqueeze(0), v_i.unsqueeze(0)).squeeze(0))
    close_to_ref(out, torch.cat(expected, dim=0))
    assert get_last_flash_attn_backend().startswith("sm75_native_varlen_paged_fwd")
