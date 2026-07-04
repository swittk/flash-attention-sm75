import torch

from flash_attn import flash_attn_func, get_last_flash_attn_backend

from sm75_test_utils import enable_native_only, make_qkv


def test_dropout_forward_backward_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = [x.detach().requires_grad_(True) for x in make_qkv(headdim=64)]
    out = flash_attn_func(q, k, v, dropout_p=0.1)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    assert get_last_flash_attn_backend().startswith("sm75_native_generic_fwd")
    out.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()
