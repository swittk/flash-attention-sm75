from flash_attn import flash_attn_func

from sm75_test_utils import attention_ref, close_to_ref, enable_native_only, make_qkv


def test_local_mask_native(monkeypatch):
    enable_native_only(monkeypatch)
    q, k, v = [x.detach().requires_grad_(True) for x in make_qkv(headdim=64)]
    out = flash_attn_func(q, k, v, window_size=(4, 4))
    ref_q = q.detach().float().requires_grad_(True)
    ref_k = k.detach().float().requires_grad_(True)
    ref_v = v.detach().float().requires_grad_(True)
    ref = attention_ref(ref_q, ref_k, ref_v, window_size=(4, 4))
    close_to_ref(out, ref)
    dout = out.detach().clone().normal_()
    (out * dout).sum().backward()
    (ref * dout.float()).sum().backward()
    close_to_ref(q.grad, ref_q.grad, atol=8e-2, rtol=8e-2)
    close_to_ref(k.grad, ref_k.grad, atol=8e-2, rtol=8e-2)
    close_to_ref(v.grad, ref_v.grad, atol=8e-2, rtol=8e-2)
