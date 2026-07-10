import pytest
import torch

from flash_attn import flash_attn_func

from sm75_test_utils import assert_sm75_backend, attention_ref, close_to_ref, enable_native_only, make_qkv


@pytest.mark.parametrize("headdim", [64, 96, 160, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_dense_backward_correctness(monkeypatch, causal, headdim):
    enable_native_only(monkeypatch)
    q, k, v = [x.detach().requires_grad_(True) for x in make_qkv(seqlen_q=13, seqlen_k=15, headdim=headdim)]
    dout = torch.randn_like(q)
    out = flash_attn_func(q, k, v, causal=causal)
    (out * dout).sum().backward()

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    out_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal)
    (out_ref * dout.float()).sum().backward()

    close_to_ref(q.grad, q_ref.grad, atol=7e-2, rtol=7e-2)
    close_to_ref(k.grad, k_ref.grad, atol=7e-2, rtol=7e-2)
    close_to_ref(v.grad, v_ref.grad, atol=7e-2, rtol=7e-2)


def test_dense_backward_softcap_alibi_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = [x.detach().requires_grad_(True) for x in make_qkv(seqlen_q=9, seqlen_k=11, headdim=64)]
    alibi_slopes = torch.rand(q.shape[0], q.shape[2], device=q.device, dtype=torch.float32) * 0.3
    dout = torch.randn_like(q)
    out = flash_attn_func(q, k, v, causal=True, softcap=20.0, alibi_slopes=alibi_slopes)
    assert_sm75_backend("sm75_native_generic_fwd")
    (out * dout).sum().backward()

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    out_ref = attention_ref(q_ref, k_ref, v_ref, causal=True, softcap=20.0, alibi_slopes=alibi_slopes)
    (out_ref * dout.float()).sum().backward()

    close_to_ref(q.grad, q_ref.grad, atol=7e-2, rtol=7e-2)
    close_to_ref(k.grad, k_ref.grad, atol=7e-2, rtol=7e-2)
    close_to_ref(v.grad, v_ref.grad, atol=7e-2, rtol=7e-2)
