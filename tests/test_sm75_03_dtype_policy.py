import torch

from flash_attn import flash_attn_func, get_last_flash_attn_backend

from sm75_test_utils import assert_sm75_backend, enable_native_only, make_qkv


def test_bf16_inputs_use_explicit_fp16_policy(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = make_qkv(headdim=64, dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v)
    assert out.dtype == torch.bfloat16
    assert_sm75_backend()
    assert get_last_flash_attn_backend().endswith("_bf16_as_fp16")
