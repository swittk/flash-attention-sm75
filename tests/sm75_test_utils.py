import math

import pytest
import torch


def require_sm75():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SM75 tests")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (7, 5):
        pytest.skip("SM75 tests require a Turing GPU")
    try:
        import flash_attn_sm75_cuda  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"flash_attn_sm75_cuda is not installed: {exc}")
    return device


def enable_native_only(monkeypatch):
    monkeypatch.setenv("FLASH_ATTN_SM75_NATIVE_ONLY", "1")
    monkeypatch.setenv("FLASH_ATTN_SM75_BAN_FALLBACKS", "1")


def make_qkv(batch=2, seqlen_q=16, seqlen_k=16, nheads=4, nheads_k=4, headdim=64, dtype=torch.float16):
    device = require_sm75()
    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    return q, k, v


def attention_ref(
    q,
    k,
    v,
    causal=False,
    softmax_scale=None,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    return_probs=False,
):
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    qf = q.float()
    kf = k.float()
    vf = v.float()
    if qf.shape[2] != kf.shape[2]:
        repeat = qf.shape[2] // kf.shape[2]
        kf = kf.repeat_interleave(repeat, dim=2)
        vf = vf.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    if softcap > 0.0:
        scores = torch.tanh(scores / softcap) * softcap
    if alibi_slopes is not None:
        slopes = alibi_slopes.float()
        if slopes.dim() == 1:
            slopes = slopes.unsqueeze(0).expand(q.shape[0], -1)
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        rel = (row + seqlen_k - seqlen_q - col).abs().float()
        scores = scores - slopes[:, :, None, None] * rel[None, None]
    if causal:
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        keep = col <= row + seqlen_k - seqlen_q
        scores = scores.masked_fill(~keep[None, None], float("-inf"))
    if window_size != (-1, -1):
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        row = torch.arange(seqlen_q, device=q.device)[:, None]
        col = torch.arange(seqlen_k, device=q.device)[None, :]
        aligned_row = row + seqlen_k - seqlen_q
        left = seqlen_k if window_size[0] < 0 else window_size[0]
        right = seqlen_k if window_size[1] < 0 else window_size[1]
        keep = (col >= aligned_row - left) & (col <= aligned_row + right)
        scores = scores.masked_fill(~keep[None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, vf)
    return (out, probs) if return_probs else out


def assert_sm75_backend(prefix="sm75_native"):
    from flash_attn import get_last_flash_attn_backend

    backend = get_last_flash_attn_backend()
    assert backend is not None
    assert backend.startswith(prefix), backend


def make_varlen(batch=3, max_q=16, max_k=18, nheads=4, nheads_k=2, headdim=64):
    device = require_sm75()
    q_lens = torch.tensor([max_q, max(1, max_q // 2), max(1, max_q - 3)], dtype=torch.int32)
    k_lens = torch.tensor([max_k, max(1, max_k // 2), max(1, max_k - 5)], dtype=torch.int32)
    q_lens = q_lens[:batch]
    k_lens = k_lens[:batch]
    cu_q = torch.zeros(batch + 1, dtype=torch.int32)
    cu_k = torch.zeros(batch + 1, dtype=torch.int32)
    cu_q[1:] = torch.cumsum(q_lens, dim=0)
    cu_k[1:] = torch.cumsum(k_lens, dim=0)
    total_q = int(cu_q[-1])
    total_k = int(cu_k[-1])
    torch.manual_seed(1)
    q = torch.randn(total_q, nheads, headdim, device=device, dtype=torch.float16)
    k = torch.randn(total_k, nheads_k, headdim, device=device, dtype=torch.float16)
    v = torch.randn(total_k, nheads_k, headdim, device=device, dtype=torch.float16)
    return q, k, v, cu_q.to(device), cu_k.to(device), int(q_lens.max()), int(k_lens.max())


def varlen_ref(
    q,
    k,
    v,
    cu_q,
    cu_k,
    causal=False,
    softmax_scale=None,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    return_probs=False,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    outs = []
    probs = []
    cu_q_cpu = cu_q.cpu().tolist()
    cu_k_cpu = cu_k.cpu().tolist()
    batch = len(cu_q_cpu) - 1
    max_seqlen_q = max(cu_q_cpu[i + 1] - cu_q_cpu[i] for i in range(batch)) if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = max(cu_k_cpu[i + 1] - cu_k_cpu[i] for i in range(batch)) if max_seqlen_k is None else max_seqlen_k
    for idx in range(len(cu_q_cpu) - 1):
        q_i = q[cu_q_cpu[idx] : cu_q_cpu[idx + 1]].unsqueeze(0)
        k_i = k[cu_k_cpu[idx] : cu_k_cpu[idx + 1]].unsqueeze(0)
        v_i = v[cu_k_cpu[idx] : cu_k_cpu[idx + 1]].unsqueeze(0)
        alibi_i = None if alibi_slopes is None else alibi_slopes if alibi_slopes.dim() == 1 else alibi_slopes[idx : idx + 1]
        ref = attention_ref(
            q_i,
            k_i,
            v_i,
            causal=causal,
            softmax_scale=softmax_scale,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_i,
            return_probs=return_probs,
        )
        if return_probs:
            out_i, probs_i = ref
            padded = torch.zeros(
                (q.shape[1], max_seqlen_q, max_seqlen_k),
                dtype=probs_i.dtype,
                device=q.device,
            )
            padded[:, : q_i.shape[1], : k_i.shape[1]].copy_(probs_i.squeeze(0))
            probs.append(padded)
            outs.append(out_i.squeeze(0))
        else:
            outs.append(ref.squeeze(0))
    out = torch.cat(outs, dim=0) if outs else torch.empty_like(q)
    if return_probs:
        return out, torch.stack(probs, dim=0)
    return out


def close_to_ref(actual, expected, atol=5e-2, rtol=5e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
