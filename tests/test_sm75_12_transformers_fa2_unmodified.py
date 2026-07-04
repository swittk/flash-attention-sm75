import pytest
import torch

from flash_attn import flash_attn_func

from sm75_test_utils import assert_sm75_backend, enable_native_only, make_qkv


def test_transformers_flash_attention_2_import_path_can_call_native(monkeypatch):
    pytest.importorskip("transformers")
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(batch=1, seqlen_q=8, seqlen_k=8, nheads=2, nheads_k=2, headdim=64)
    with torch.no_grad():
        flash_attn_func(q, k, v, causal=True)
    assert_sm75_backend()
