import torch

from flash_attn import flash_attn_kvpacked_func, flash_attn_qkvpacked_func

from sm75_test_utils import assert_sm75_backend, attention_ref, close_to_ref, enable_native_only, make_qkv


def test_qkvpacked_forward_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(seqlen_q=16, seqlen_k=16, nheads=4, nheads_k=4, headdim=64)
    qkv = torch.stack([q, k, v], dim=2)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    close_to_ref(out, attention_ref(q, k, v, causal=True))
    assert_sm75_backend()


def test_kvpacked_forward_correctness(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(seqlen_q=15, seqlen_k=17, nheads=4, nheads_k=2, headdim=64)
    kv = torch.stack([k, v], dim=2)
    out = flash_attn_kvpacked_func(q, kv, causal=False)
    close_to_ref(out, attention_ref(q, k, v, causal=False))
    assert_sm75_backend()
