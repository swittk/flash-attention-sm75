import pytest
import torch

from flash_attn import flash_attn_func, get_last_flash_attn_backend

from sm75_test_utils import assert_sm75_backend, attention_ref, close_to_ref, enable_native_only, make_qkv


@pytest.mark.parametrize("headdim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nheads,nheads_k", [(4, 4), (4, 2)])
def test_dense_forward_correctness(monkeypatch, headdim, causal, nheads, nheads_k):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(seqlen_q=17, seqlen_k=19, nheads=nheads, nheads_k=nheads_k, headdim=headdim)
    out = flash_attn_func(q, k, v, causal=causal)
    ref = attention_ref(q, k, v, causal=causal)
    close_to_ref(out, ref)
    assert_sm75_backend()


@pytest.mark.parametrize(
    "headdim",
    [
        32,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
    ],
)
def test_required_head_dims_are_native(monkeypatch, headdim):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(batch=1, seqlen_q=8, seqlen_k=8, nheads=2, nheads_k=2, headdim=headdim)
    out = flash_attn_func(q, k, v, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend()


def test_dense_d256_uses_fa2_tiled_kernel(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(
        batch=1,
        seqlen_q=64,
        seqlen_k=128,
        nheads=8,
        nheads_k=2,
        headdim=256,
    )
    out = flash_attn_func(q, k, v, causal=False)
    close_to_ref(out, attention_ref(q, k, v, causal=False))
    assert_sm75_backend("sm75_native_dense_hd256_fa2_tiled_kernel256")


def test_dense_d224_uses_uneven_fa2_tiled_kernel(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(
        batch=1,
        seqlen_q=32,
        seqlen_k=64,
        nheads=4,
        nheads_k=2,
        headdim=224,
    )
    out = flash_attn_func(q, k, v, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend("sm75_native_dense_hd224_fa2_tiled_kernel256")


@pytest.mark.parametrize("use_batch_alibi", [False, True])
def test_dense_softcap_alibi_native(monkeypatch, use_batch_alibi):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(batch=2, seqlen_q=11, seqlen_k=13, nheads=4, nheads_k=4, headdim=64)
    alibi_shape = (2, 4) if use_batch_alibi else (4,)
    alibi_slopes = torch.rand(alibi_shape, device=q.device, dtype=torch.float32) * 0.3
    out = flash_attn_func(q, k, v, causal=True, softcap=20.0, alibi_slopes=alibi_slopes)
    ref = attention_ref(q, k, v, causal=True, softcap=20.0, alibi_slopes=alibi_slopes)
    close_to_ref(out, ref)
    assert get_last_flash_attn_backend().startswith("sm75_native_generic_fwd")


def test_dense_return_attn_probs_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(batch=1, seqlen_q=7, seqlen_k=9, nheads=2, nheads_k=2, headdim=64)
    out, lse, probs = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    ref, ref_probs = attention_ref(q, k, v, causal=True, return_probs=True)
    close_to_ref(out, ref)
    close_to_ref(probs, ref_probs, atol=2e-3, rtol=2e-3)
    assert lse.shape == (q.shape[0], q.shape[2], q.shape[1])
    assert probs.shape == (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
    assert get_last_flash_attn_backend().startswith("sm75_native_generic_fwd")
