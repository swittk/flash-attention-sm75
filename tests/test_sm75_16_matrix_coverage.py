import os

import pytest
import torch

from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

from sm75_test_utils import attention_ref, assert_sm75_backend, close_to_ref, enable_native_only, require_sm75, varlen_ref


FULL_MATRIX = os.getenv("FLASH_ATTN_SM75_FULL_MATRIX", "0").upper() in {"1", "TRUE", "YES", "ON"}


def _dense_cases():
    cases = [
        (1, 1, 1, 32, 32, 32, True),
        (2, 2, 128, 32, 16, 64, False),
        (4, 8, 512, 32, 8, 80, True),
        (1, 16, 1024, 32, 4, 96, False),
        (2, 128, 128, 16, 1, 112, True),
        (1, 512, 1024, 32, 8, 128, True),
        (1, 128, 512, 32, 4, 160, False),
        (1, 128, 128, 32, 32, 192, True),
        (1, 64, 512, 16, 1, 224, False),
        (1, 64, 128, 32, 8, 256, True),
    ]
    if FULL_MATRIX:
        cases.extend(
            [
                (1, 1024, 2048, 32, 8, 64, True),
                (2, 512, 4096, 32, 4, 128, False),
                (1, 2048, 2048, 16, 1, 160, True),
                (1, 4096, 4096, 32, 8, 96, True),
            ]
        )
    return cases


@pytest.mark.parametrize("batch,seqlen_q,seqlen_k,nheads,nheads_k,headdim,causal", _dense_cases())
def test_dense_axis_matrix_forward_native(monkeypatch, batch, seqlen_q, seqlen_k, nheads, nheads_k, headdim, causal):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1600 + headdim + seqlen_q)
    q = torch.randn(batch, seqlen_q, nheads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(batch, seqlen_k, nheads_k, headdim, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    out = flash_attn_func(q, k, v, causal=causal)
    ref = attention_ref(q, k, v, causal=causal)
    close_to_ref(out, ref, atol=8e-2, rtol=8e-2)
    assert not torch.isnan(out).any()
    assert_sm75_backend()


@pytest.mark.parametrize("headdim", [32, 64, 80, 96, 112, 128, 160, 192, 224, 256])
def test_dense_dropout_local_axis_matrix_native(monkeypatch, headdim):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1700 + headdim)
    q = torch.randn(1, 64, 8, headdim, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 80, 4, headdim, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    out = flash_attn_func(q, k, v, dropout_p=0.1, causal=True, window_size=(32, 0))
    loss = out.float().square().mean()
    loss.backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert not torch.isnan(out).any()
    assert_sm75_backend()


def test_qkvpacked_and_kvpacked_backward_native(monkeypatch):
    enable_native_only(monkeypatch)
    device = require_sm75()
    torch.manual_seed(1800)
    qkv = torch.randn(2, 64, 3, 8, 64, device=device, dtype=torch.float16, requires_grad=True)
    out = flash_attn_qkvpacked_func(qkv, causal=True)
    out.float().square().mean().backward()
    assert qkv.grad is not None
    assert_sm75_backend()

    q = torch.randn(2, 64, 16, 96, device=device, dtype=torch.float16, requires_grad=True)
    kv = torch.randn(2, 64, 2, 4, 96, device=device, dtype=torch.float16, requires_grad=True)
    out = flash_attn_kvpacked_func(q, kv, causal=False)
    out.float().square().mean().backward()
    assert q.grad is not None and kv.grad is not None
    assert_sm75_backend()


@pytest.mark.parametrize("api", ["varlen", "varlen_qkvpacked", "varlen_kvpacked"])
def test_varlen_packed_backward_native(monkeypatch, api):
    enable_native_only(monkeypatch)
    device = require_sm75()
    lengths = [96, 64, 32, 16] if FULL_MATRIX else [64, 32]
    cu = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lengths, device=device, dtype=torch.int32), dim=0)
    total = int(cu[-1].item())
    max_len = max(lengths)
    torch.manual_seed(1900 + len(lengths))
    q = torch.randn(total, 8, 128, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(total, 4 if api != "varlen_qkvpacked" else 8, 128, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    if api == "varlen":
        out = flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, causal=True)
    elif api == "varlen_qkvpacked":
        qkv = torch.stack((q, k, v), dim=1)
        qkv.retain_grad()
        out = flash_attn_varlen_qkvpacked_func(qkv, cu, max_len, causal=True)
    else:
        kv = torch.stack((k, v), dim=1)
        kv.retain_grad()
        out = flash_attn_varlen_kvpacked_func(q, kv, cu, cu, max_len, max_len, causal=False)
    ref = varlen_ref(q.detach(), k.detach(), v.detach(), cu, cu, causal=(api != "varlen_kvpacked"))
    close_to_ref(out, ref, atol=8e-2, rtol=8e-2)
    out.float().square().mean().backward()
    assert not torch.isnan(out).any()
    assert_sm75_backend()
