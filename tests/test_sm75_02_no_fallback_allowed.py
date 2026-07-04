import torch.nn.functional as F

from flash_attn import flash_attn_func

from sm75_test_utils import assert_sm75_backend, enable_native_only, make_qkv


def test_native_only_rejects_sdpa_and_uses_sm75(monkeypatch):
    enable_native_only(monkeypatch)

    def _sdpa_forbidden(*args, **kwargs):
        raise AssertionError("SDPA fallback was called")

    monkeypatch.setattr(F, "scaled_dot_product_attention", _sdpa_forbidden)
    q, k, v = make_qkv(headdim=64)
    flash_attn_func(q, k, v, causal=True)
    assert_sm75_backend()


def test_native_only_rejects_xformers_if_present(monkeypatch):
    enable_native_only(monkeypatch)
    try:
        import xformers.ops as xops
    except ImportError:
        xops = None
    if xops is not None and hasattr(xops, "memory_efficient_attention"):
        monkeypatch.setattr(
            xops,
            "memory_efficient_attention",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("xFormers fallback was called")),
        )
    q, k, v = make_qkv(headdim=64)
    flash_attn_func(q, k, v)
    assert_sm75_backend()
