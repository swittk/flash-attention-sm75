import torch

from flash_attn import flash_attn_kvpacked_func, flash_attn_qkvpacked_func

from sm75_test_utils import assert_sm75_backend, attention_ref, close_to_ref, enable_native_only, make_qkv


def test_qkvpacked_forward_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(seqlen_q=16, seqlen_k=16, nheads=4, nheads_k=4, headdim=64)
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend("sm75_native_qkvpacked_direct")


def test_qkvpacked_d64_uses_single_packed_tensor_launch(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(
        batch=1,
        seqlen_q=128,
        seqlen_k=128,
        nheads=8,
        nheads_k=8,
        headdim=64,
    )
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=False)
    close_to_ref(out, attention_ref(q, k, v, causal=False))
    assert_sm75_backend(
        "sm75_native_qkvpacked_direct_packedtensor_"
        "dense_hd64_fa2_tiled_headmajor_kernel64"
    )


def test_kvpacked_forward_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(seqlen_q=15, seqlen_k=17, nheads=4, nheads_k=2, headdim=64)
    kv = torch.stack([k, v], dim=2)
    out = flash_attn_kvpacked_func(q, kv, causal=False)
    close_to_ref(out, attention_ref(q, k, v, causal=False))
    assert_sm75_backend("sm75_native_kvpacked_direct")


def test_kvpacked_d96_uses_direct_fa2_tiled_route(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(batch=1, seqlen_q=32, seqlen_k=32, nheads=8, nheads_k=2, headdim=96)
    kv = torch.stack([k, v], dim=2)
    out = flash_attn_kvpacked_func(q, kv, causal=False)
    close_to_ref(out, attention_ref(q, k, v, causal=False))
    assert_sm75_backend(
        "sm75_native_kvpacked_direct_strided_dense_hd96_fa2_tiled_kernel96"
    )


def test_qkvpacked_d80_uses_direct_uneven_fa2_tiled_route(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(batch=1, seqlen_q=33, seqlen_k=33, nheads=8, nheads_k=8, headdim=80)
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend(
        "sm75_native_qkvpacked_direct_strided_dense_hd80_fa2_tiled_kernel96"
    )


def test_qkvpacked_d80_aligned_vector_tail_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(
        batch=1,
        seqlen_q=128,
        seqlen_k=128,
        nheads=8,
        nheads_k=8,
        headdim=80,
    )
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend(
        "sm75_native_qkvpacked_direct_strided_dense_hd80_fa2_tiled_kernel96"
    )


def test_qkvpacked_d256_uses_direct_fa2_tiled_route(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(
        batch=1,
        seqlen_q=64,
        seqlen_k=64,
        nheads=8,
        nheads_k=8,
        headdim=256,
    )
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend(
        "sm75_native_qkvpacked_direct_strided_dense_hd256_fa2_tiled_kernel256"
    )


def test_qkvpacked_d224_uses_uneven_fa2_tiled_route(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(
        batch=1,
        seqlen_q=64,
        seqlen_k=64,
        nheads=8,
        nheads_k=8,
        headdim=224,
    )
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=False)
    close_to_ref(out, attention_ref(q, k, v, causal=False))
    assert_sm75_backend(
        "sm75_native_qkvpacked_direct_strided_dense_hd224_fa2_tiled_kernel256"
    )


def test_kvpacked_d256_uses_direct_fa2_tiled_route(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(
        batch=1,
        seqlen_q=64,
        seqlen_k=128,
        nheads=8,
        nheads_k=2,
        headdim=256,
    )
    kv = torch.stack([k, v], dim=2)
    out = flash_attn_kvpacked_func(q, kv, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend(
        "sm75_native_kvpacked_direct_strided_dense_hd256_fa2_tiled_kernel256"
    )
