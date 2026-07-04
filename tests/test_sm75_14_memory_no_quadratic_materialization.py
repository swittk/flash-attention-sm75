import torch

from flash_attn import flash_attn_func

from sm75_test_utils import assert_sm75_backend, enable_native_only, make_qkv


def test_long_context_sm75_path_does_not_call_eager_attention(monkeypatch):
    enable_native_only(monkeypatch)
    warm_q, warm_k, warm_v = make_qkv(batch=1, seqlen_q=128, seqlen_k=128, nheads=2, nheads_k=2, headdim=64)
    flash_attn_func(warm_q, warm_k, warm_v, causal=True)
    torch.cuda.synchronize(warm_q.device)
    torch.cuda.empty_cache()

    q, k, v = make_qkv(batch=1, seqlen_q=2048, seqlen_k=2048, nheads=2, nheads_k=2, headdim=64)
    torch.cuda.reset_peak_memory_stats(q.device)
    baseline = torch.cuda.memory_allocated(q.device)
    out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize(q.device)
    assert out.shape == q.shape
    assert_sm75_backend()
    peak_delta = torch.cuda.max_memory_allocated(q.device) - baseline
    full_scores_fp32 = q.shape[0] * q.shape[2] * q.shape[1] * k.shape[1] * 4
    assert peak_delta < full_scores_fp32
