# Copyright (c) 2026, FlashAttention contributors.

"""SM75/Turing native backend dispatch helpers."""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional, Tuple

import torch

try:
    import flash_attn_sm75_cuda as _sm75_cuda
except ImportError as exc:  # pragma: no cover - exercised in source-only installs
    _sm75_cuda = None
    _SM75_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _SM75_IMPORT_ERROR = None


_SUPPORTED_HEAD_DIMS = {32, 64, 80, 96, 112, 128, 160, 192, 224, 256}
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_STATE = threading.local()
_STATE_LOCK = threading.Lock()
_LAST_BACKEND: Optional[str] = None
_SM75_DEVICE_CACHE: Dict[int, bool] = {}
_DIRECT_FWD_BACKEND_CACHE: Dict[Tuple[int, bool, bool], str] = {}


def _env_true(name: str) -> bool:
    return os.getenv(name, "0").upper() in {"1", "TRUE", "YES", "ON"}


def native_only_enabled() -> bool:
    return _env_true("FLASH_ATTN_SM75_NATIVE_ONLY")


def ban_fallbacks_enabled() -> bool:
    return native_only_enabled() or _env_true("FLASH_ATTN_SM75_BAN_FALLBACKS")


def is_available() -> bool:
    return _sm75_cuda is not None


def import_error() -> Optional[ImportError]:
    return _SM75_IMPORT_ERROR


def get_last_backend() -> Optional[str]:
    return _LAST_BACKEND


def reset_last_backend() -> None:
    global _LAST_BACKEND
    _STATE.last_backend = None
    with _STATE_LOCK:
        _LAST_BACKEND = None


def _set_last_backend(value: str) -> None:
    global _LAST_BACKEND
    if getattr(_STATE, "last_backend", None) == value and _LAST_BACKEND == value:
        return
    _STATE.last_backend = value
    with _STATE_LOCK:
        if _LAST_BACKEND != value:
            _LAST_BACKEND = value


def _is_sm75_tensor(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    device_idx = x.get_device()
    cached = _SM75_DEVICE_CACHE.get(device_idx)
    if cached is None:
        cached = torch.cuda.get_device_capability(device_idx) == (7, 5)
        _SM75_DEVICE_CACHE[device_idx] = cached
    return cached


def is_sm75_call(q: torch.Tensor) -> bool:
    return _is_sm75_tensor(q)


def _dtype_suffix(dtype: torch.dtype) -> str:
    return "bf16_as_fp16" if dtype == torch.bfloat16 else "fp16"


def _head_group_suffix(q: torch.Tensor, k: torch.Tensor) -> str:
    return "gqa" if q.shape[-2] != k.shape[-2] else "mha"


def _direct_fwd_backend(head_dim: int, causal: bool, is_gqa: bool) -> str:
    key = (head_dim, causal, is_gqa)
    cached = _DIRECT_FWD_BACKEND_CACHE.get(key)
    if cached is not None:
        return cached
    if head_dim == 192:
        backend_kind = "dense_hd192_tiled_kernel192"
    else:
        backend_kind = f"fwd_hd{head_dim}_kernel{head_dim}"
    cached = (
        f"sm75_native_{backend_kind}_{'causal' if causal else 'noncausal'}_"
        f"{'gqa' if is_gqa else 'mha'}_fp16"
    )
    _DIRECT_FWD_BACKEND_CACHE[key] = cached
    return cached


def _kernel_head_dim(head_dim: int) -> Optional[int]:
    if head_dim <= 64:
        return 64
    if head_dim <= 96:
        return 96
    if head_dim <= 128:
        return 128
    if head_dim <= 192:
        return 192
    if head_dim <= 256:
        return 256
    return None


def _backward_kernel_head_dim(head_dim: int) -> Optional[int]:
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    return None


def _common_rejection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
) -> Optional[str]:
    if _sm75_cuda is None:
        return "the flash_attn_sm75_cuda extension is not importable"
    if not _is_sm75_tensor(q):
        return "the input tensor is not on an SM75 CUDA device"
    if not (k.is_cuda and v.is_cuda):
        return "q, k, and v must all be CUDA tensors"
    if q.device != k.device or q.device != v.device:
        return "q, k, and v must be on the same CUDA device"
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v must be FP16 or BF16 tensors with matching dtype"
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return "q, k, and v must have matching head_dim"
    if _kernel_head_dim(q.shape[-1]) is None:
        return f"head_dim {q.shape[-1]} has no native SM75 kernel in this backend slice"
    if dropout_p != 0.0:
        return "dropout needs a native SM75 dropout kernel"
    if window_size_left != -1 or window_size_right != -1:
        return "local attention needs a native SM75 local-mask kernel"
    if alibi_slopes is not None:
        return "ALiBi needs a native SM75 bias kernel"
    if q.shape[-2] % k.shape[-2] != 0:
        return "query heads must be divisible by key/value heads"
    return None


def _forward_rejection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
) -> Optional[str]:
    if _sm75_cuda is None:
        return "the flash_attn_sm75_cuda extension is not importable"
    if not _is_sm75_tensor(q):
        return "the input tensor is not on an SM75 CUDA device"
    if not (k.is_cuda and v.is_cuda):
        return "q, k, and v must all be CUDA tensors"
    if q.device != k.device or q.device != v.device:
        return "q, k, and v must be on the same CUDA device"
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v must be FP16 or BF16 tensors with matching dtype"
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return "q, k, and v must have matching head_dim"
    if q.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        return f"head_dim {q.shape[-1]} has no native SM75 forward kernel"
    if dropout_p < 0.0 or dropout_p >= 1.0:
        return "dropout_p must be in [0, 1)"
    if alibi_slopes is not None:
        if not (alibi_slopes.is_cuda and alibi_slopes.device == q.device):
            return "ALiBi slopes must be on the same CUDA device as q"
        if alibi_slopes.dtype != torch.float32:
            return "ALiBi slopes must be fp32"
        if alibi_slopes.dim() not in (1, 2):
            return "ALiBi slopes must have shape [nheads] or [batch, nheads]"
        if alibi_slopes.shape[-1] != q.shape[-2] or (alibi_slopes.dim() == 2 and alibi_slopes.shape[0] != q.shape[0]):
            return "ALiBi slopes shape must match q batch and heads"
    if q.shape[-2] % k.shape[-2] != 0:
        return "query heads must be divisible by key/value heads"
    return None


def _backward_rejection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
) -> Optional[str]:
    reason = _forward_rejection(
        q,
        k,
        v,
        dropout_p,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        return_softmax,
    )
    return reason


def _varlen_rejection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    alibi_slopes: Optional[torch.Tensor],
) -> Optional[str]:
    if _sm75_cuda is None:
        return "the flash_attn_sm75_cuda extension is not importable"
    if not _is_sm75_tensor(q):
        return "the input tensor is not on an SM75 CUDA device"
    if not (k.is_cuda and v.is_cuda):
        return "q, k, and v must all be CUDA tensors"
    if q.device != k.device or q.device != v.device:
        return "q, k, and v must be on the same CUDA device"
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v must be FP16 or BF16 tensors with matching dtype"
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        return "varlen q, k, and v must be rank-3 packed tensors"
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return "q, k, and v must have matching head_dim"
    if q.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        return f"head_dim {q.shape[-1]} has no native SM75 varlen kernel"
    if k.shape[0] != v.shape[0] or k.shape[1] != v.shape[1]:
        return "k and v must have matching total tokens and head count"
    if q.shape[-2] % k.shape[-2] != 0:
        return "query heads must be divisible by key/value heads"
    if dropout_p < 0.0 or dropout_p >= 1.0:
        return "dropout_p must be in [0, 1)"
    if not (cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda):
        return "cu_seqlens_q and cu_seqlens_k must be CUDA tensors"
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        return "cu_seqlens_q and cu_seqlens_k must be int32 tensors"
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        return "cu_seqlens_q and cu_seqlens_k must be rank-1 tensors"
    if cu_seqlens_q.numel() < 2 or cu_seqlens_k.numel() != cu_seqlens_q.numel():
        return "cu_seqlens_q/cu_seqlens_k must have matching [batch + 1] shape"
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        return "max_seqlen_q and max_seqlen_k must be non-negative"
    if alibi_slopes is not None:
        if not (alibi_slopes.is_cuda and alibi_slopes.device == q.device):
            return "ALiBi slopes must be on the same CUDA device as q"
        if alibi_slopes.dtype != torch.float32:
            return "ALiBi slopes must be fp32"
        if alibi_slopes.dim() not in (1, 2):
            return "ALiBi slopes must have shape [nheads] or [batch, nheads]"
        batch_size = cu_seqlens_q.numel() - 1
        if alibi_slopes.shape[-1] != q.shape[-2] or (alibi_slopes.dim() == 2 and alibi_slopes.shape[0] != batch_size):
            return "ALiBi slopes shape must match varlen batch and query heads"
    return None


def _varlen_offsets(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[Optional[str], list[int], list[int]]:
    q_offsets = _tensor_to_int_list(cu_seqlens_q)
    k_offsets = _tensor_to_int_list(cu_seqlens_k)
    if q_offsets[0] != 0 or k_offsets[0] != 0:
        return "cu_seqlens_q and cu_seqlens_k must start at zero", [], []
    if q_offsets[-1] != q.shape[0] or k_offsets[-1] != k.shape[0]:
        return "cu_seqlens totals must match packed q/k lengths", [], []
    for start, end in zip(q_offsets, q_offsets[1:]):
        if end < start or end - start > max_seqlen_q:
            return "cu_seqlens_q must be monotonic and bounded by max_seqlen_q", [], []
    for start, end in zip(k_offsets, k_offsets[1:]):
        if end < start or end - start > max_seqlen_k:
            return "cu_seqlens_k must be monotonic and bounded by max_seqlen_k", [], []
    return None, q_offsets, k_offsets


def _check_optional_i32_batch(x: Optional[torch.Tensor], name: str, batch_size: int, device: torch.device) -> Optional[str]:
    if x is None:
        return None
    if not (x.is_cuda and x.device == device):
        return f"{name} must be a CUDA tensor on the query device"
    if x.dtype != torch.int32:
        return f"{name} must be int32"
    if x.numel() != batch_size:
        return f"{name} must have one entry per sequence"
    return None


def _validate_varlen_aux_lengths(
    q: torch.Tensor,
    k_offsets: list[int],
    max_seqlen_k: int,
    leftpad_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
) -> tuple[Optional[str], list[int], list[int]]:
    batch_size = len(k_offsets) - 1
    leftpads = [0] * batch_size if leftpad_k is None else _tensor_to_int_list(leftpad_k)
    raw_lengths = [end - start for start, end in zip(k_offsets, k_offsets[1:])]
    used_lengths = raw_lengths if seqused_k is None else _tensor_to_int_list(seqused_k)
    effective_lengths = []
    for raw_length, used_length, leftpad in zip(raw_lengths, used_lengths, leftpads):
        if used_length < 0 or used_length > raw_length or used_length > max_seqlen_k:
            return "seqused_k must be non-negative and bounded by cu_seqlens_k/max_seqlen_k", [], []
        if leftpad < 0 or leftpad > used_length:
            return "leftpad_k must be between zero and seqused_k", [], []
        effective = used_length - leftpad
        if effective <= 0:
            return "SM75 varlen attention requires at least one unpadded key per sequence", [], []
        effective_lengths.append(effective)
    return None, leftpads, effective_lengths


def _select_varlen_alibi(alibi_slopes: Optional[torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
    if alibi_slopes is None:
        return None
    if alibi_slopes.dim() == 1:
        return alibi_slopes
    return alibi_slopes[batch_idx : batch_idx + 1].contiguous()


def _maybe_raise_for_banned_sm75(q: torch.Tensor, reason: Optional[str]) -> None:
    if reason is None:
        return
    # On Turing this fork treats the SM75 backend as authoritative: an
    # unsupported SM75 call must fail visibly instead of silently running a
    # different production backend.
    if _is_sm75_tensor(q):
        raise RuntimeError(
            "No SM75 native kernel is available for this call; "
            f"upstream fallback is disabled on SM75: {reason}"
        )


def _to_fp16_for_sm75(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16:
        return x.to(torch.float16).contiguous()
    return x.contiguous() if not x.is_contiguous() else x


def _restore_sm75_output_dtype(out: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.dtype == torch.bfloat16:
        return out.to(torch.bfloat16)
    return out


def _apply_rotary_sm75(
    x: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_interleaved: bool,
) -> torch.Tensor:
    x_sm75 = _to_fp16_for_sm75(x)
    cos_sm75 = _to_fp16_for_sm75(rotary_cos)
    sin_sm75 = _to_fp16_for_sm75(rotary_sin)
    rotary_dim = cos_sm75.shape[-1] * 2
    pos = positions.to(device=x.device, dtype=torch.long)
    cos = cos_sm75.index_select(0, pos.reshape(-1)).reshape(*pos.shape, cos_sm75.shape[-1]).unsqueeze(-2)
    sin = sin_sm75.index_select(0, pos.reshape(-1)).reshape(*pos.shape, sin_sm75.shape[-1]).unsqueeze(-2)
    x_rot = x_sm75[..., :rotary_dim]
    if rotary_interleaved:
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        out_rot = torch.empty_like(x_rot)
        out_rot[..., 0::2] = x_even * cos - x_odd * sin
        out_rot[..., 1::2] = x_odd * cos + x_even * sin
    else:
        x_first, x_second = x_rot.chunk(2, dim=-1)
        out_rot = torch.cat((x_first * cos - x_second * sin, x_second * cos + x_first * sin), dim=-1)
    return torch.cat((out_rot, x_sm75[..., rotary_dim:]), dim=-1)


def _prepare_for_sm75_kernel(
    x: torch.Tensor, kernel_head_dim: int, preserve_strides: bool = True
) -> torch.Tensor:
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float16)
    if kernel_head_dim in (96, 256) and (
        (64 < x.shape[-1] <= 96) or (192 < x.shape[-1] <= 256)
    ):
        kernel_head_dim = x.shape[-1]
    if x.shape[-1] == kernel_head_dim:
        if x.stride(-1) != 1 or not preserve_strides:
            return x.contiguous()
        return x
    return torch.nn.functional.pad(x, [0, kernel_head_dim - x.shape[-1]]).contiguous()


def _prepare_qkv_for_sm75_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_head_dim: int,
    use_native_pad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        use_native_pad
        and q.dtype == torch.float16
        and k.dtype == torch.float16
        and v.dtype == torch.float16
        and q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.shape[-1] == 96
        and q.shape[-1] < kernel_head_dim
        and hasattr(_sm75_cuda, "pad_qkv_head_dim")
    ):
        return tuple(_sm75_cuda.pad_qkv_head_dim(q, k, v, int(kernel_head_dim)))
    return tuple(_prepare_for_sm75_kernel(x, kernel_head_dim) for x in (q, k, v))


def try_qkvpacked_forward_fast(
    qkv: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_attn_probs: bool,
) -> Optional[torch.Tensor]:
    """Low-overhead direct-strided launch for packed inference fast paths."""
    if (
        _sm75_cuda is None
        or qkv.dim() != 5
        or qkv.shape[2] != 3
        or qkv.shape[-1] not in (64, 80, 96, 128, 192, 224, 256)
        or qkv.dtype != torch.float16
        or qkv.stride(-1) != 1
        or dropout_p != 0.0
        or window_size_left != -1
        or window_size_right != -1
        or softcap != 0.0
        or alibi_slopes is not None
        or return_attn_probs
        or not _is_sm75_tensor(qkv)
    ):
        return None
    head_dim = qkv.shape[-1]
    mask = "causal" if causal else "noncausal"
    if head_dim == 64:
        if not hasattr(_sm75_cuda, "fwd_qkvpacked_hd64_fa2"):
            return None
        out, _ = _sm75_cuda.fwd_qkvpacked_hd64_fa2(
            qkv, float(softmax_scale), bool(causal)
        )
        backend = (
            "sm75_native_qkvpacked_direct_packedtensor_"
            f"dense_hd64_fa2_tiled_headmajor_kernel64_{mask}_mha_fp16"
        )
    else:
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if head_dim in (80, 96):
            out, _ = _sm75_cuda.fwd_hd96_fa2(
                q, k, v, float(softmax_scale), bool(causal)
            )
            backend_kind = f"dense_hd{head_dim}_fa2_tiled_kernel96"
        elif head_dim in (224, 256):
            out, _ = _sm75_cuda.fwd_hd256_fa2(
                q, k, v, float(softmax_scale), bool(causal)
            )
            backend_kind = f"dense_hd{head_dim}_fa2_tiled_kernel256"
        else:
            out, _ = _sm75_cuda.fwd(
                q, k, v, float(softmax_scale), bool(causal)
            )
            backend_kind = (
                "fwd_hd128_kernel128"
                if head_dim == 128
                else "dense_hd192_tiled_kernel192"
            )
        backend = (
            f"sm75_native_qkvpacked_direct_strided_{backend_kind}_"
            f"{mask}_mha_fp16"
        )
    _set_last_backend(backend)
    return out


def try_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    use_native_pad: bool = False,
    packed_layout: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    reason = _forward_rejection(
        q,
        k,
        v,
        dropout_p,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        return_softmax,
    )
    _maybe_raise_for_banned_sm75(q, reason)
    if reason is not None:
        return None

    kernel_head_dim = _kernel_head_dim(q.shape[-1])
    use_fast_kernel = (
        kernel_head_dim is not None
        and dropout_p == 0.0
        and window_size_left == -1
        and window_size_right == -1
        and softcap == 0.0
        and alibi_slopes is None
        and not return_softmax
    )
    if use_fast_kernel:
        q_sm75, k_sm75, v_sm75 = _prepare_qkv_for_sm75_kernel(
            q, k, v, kernel_head_dim, use_native_pad=use_native_pad
        )
        use_hd64_mha_fa2 = (
            kernel_head_dim == 64
            and q.shape[-1] == 64
            and q.shape[-2] == k.shape[-2]
            and not causal
        )
        if use_hd64_mha_fa2:
            out, softmax_lse = _sm75_cuda.fwd_hd64_fa2(
                q_sm75,
                k_sm75,
                v_sm75,
                float(softmax_scale),
                bool(causal),
                bool(packed_layout),
            )
        elif kernel_head_dim == 96:
            out, softmax_lse = _sm75_cuda.fwd_hd96_fa2(
                q_sm75, k_sm75, v_sm75, float(softmax_scale), bool(causal)
            )
        elif kernel_head_dim == 256:
            out, softmax_lse = _sm75_cuda.fwd_hd256_fa2(
                q_sm75, k_sm75, v_sm75, float(softmax_scale), bool(causal)
            )
        else:
            out, softmax_lse = _sm75_cuda.fwd(
                q_sm75, k_sm75, v_sm75, float(softmax_scale), bool(causal)
            )
        out = out[..., : q.shape[-1]]
        if use_hd64_mha_fa2:
            backend_kind = (
                "dense_hd64_fa2_tiled_headmajor_kernel64"
                if packed_layout
                else "dense_hd64_fa2_tiled_kernel64"
            )
        elif kernel_head_dim == 96:
            backend_kind = f"dense_hd{q.shape[-1]}_fa2_tiled_kernel96"
        elif kernel_head_dim == 256:
            backend_kind = f"dense_hd{q.shape[-1]}_fa2_tiled_kernel256"
        elif kernel_head_dim is not None and kernel_head_dim > 128 and q.shape[-1] > 128:
            backend_kind = f"dense_hd{q.shape[-1]}_tiled_kernel{kernel_head_dim}"
        else:
            backend_kind = f"fwd_hd{q.shape[-1]}_kernel{kernel_head_dim}"
    else:
        q_sm75, k_sm75, v_sm75 = (_to_fp16_for_sm75(x) for x in (q, k, v))
        alibi_sm75 = (
            torch.empty((0,), dtype=torch.float32, device=q.device)
            if alibi_slopes is None
            else alibi_slopes.contiguous()
        )
        out, softmax_lse, s_dmask = _sm75_cuda.fwd_generic(
            q_sm75,
            k_sm75,
            v_sm75,
            float(softmax_scale),
            bool(causal),
            int(window_size_left),
            int(window_size_right),
            float(dropout_p),
            alibi_sm75,
            float(softcap),
            bool(return_softmax),
        )
        warp_wide_bridge = (
            q.shape[-1] >= 224
            and q.shape[-1] <= 256
            and k.shape[-3] >= 128
            and dropout_p == 0.0
            and window_size_left == -1
            and window_size_right == -1
            and softcap == 0.0
            and alibi_slopes is None
            and not return_softmax
        )
        splitkv_splits = (k.shape[-3] + 255) // 256
        splitkv_partial_elements = q.shape[0] * q.shape[-3] * q.shape[-2] * splitkv_splits * q.shape[-1]
        splitkv_bridge = warp_wide_bridge and splitkv_partial_elements <= 80_000_000
        if warp_wide_bridge:
            backend_kind = f"generic_warp_fwd_hd{q.shape[-1]}"
        elif splitkv_bridge:
            backend_kind = f"generic_splitkv_fwd_hd{q.shape[-1]}"
        else:
            backend_kind = f"generic_fwd_hd{q.shape[-1]}"
    out = _restore_sm75_output_dtype(out, q)
    if not return_softmax:
        s_dmask = torch.empty((0,), dtype=out.dtype, device=out.device)
    rng_state = torch.empty((2,), dtype=torch.int64, device=out.device)
    _set_last_backend(
        "sm75_native_"
        f"{backend_kind}"
        f"_{'causal' if causal else 'noncausal'}"
        f"_{_head_group_suffix(q, k)}"
        f"_{_dtype_suffix(q.dtype)}"
        f"{'_dropout' if dropout_p > 0.0 else ''}"
        f"{'_local' if window_size_left != -1 or window_size_right != -1 else ''}"
    )
    return out, softmax_lse, s_dmask, rng_state


def try_forward_direct_no_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
) -> Optional[torch.Tensor]:
    """Minimal inference dispatch for exact-dimension dense kernels."""
    if (
        _sm75_cuda is None
        or dropout_p != 0.0
        or window_size_left != -1
        or window_size_right != -1
        or softcap != 0.0
        or alibi_slopes is not None
        or return_softmax
    ):
        return None
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        return None
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return None
    batch, _, q_heads, head_dim = q.shape
    k_batch, k_seqlen, kv_heads, k_head_dim = k.shape
    v_batch, v_seqlen, v_heads, v_head_dim = v.shape
    if (
        head_dim not in (64, 128, 192)
        or head_dim != k_head_dim
        or head_dim != v_head_dim
        or batch != k_batch
        or batch != v_batch
        or k_seqlen != v_seqlen
        or kv_heads != v_heads
        or q_heads % kv_heads != 0
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
        or q.device != k.device
        or q.device != v.device
        or not _is_sm75_tensor(q)
    ):
        return None
    out, _ = _sm75_cuda.fwd(q, k, v, float(softmax_scale), bool(causal))
    _set_last_backend(_direct_fwd_backend(head_dim, bool(causal), q_heads != kv_heads))
    return out


def try_packed_forward(
    packed_kind: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Inference fast path for public packed APIs.

    The SM75 kernels already consume Q/K/V strides, so packed tensors can be
    sliced into strided views and launched directly without the autograd
    Function wrapper used by the generic public API path.
    """
    if packed_kind not in {"qkvpacked", "kvpacked"}:
        raise ValueError(f"unknown packed_kind {packed_kind!r}")
    kernel_head_dim = _kernel_head_dim(q.shape[-1])
    needs_pad = (
        kernel_head_dim is not None
        and kernel_head_dim != q.shape[-1]
        and not (kernel_head_dim == 96 and q.shape[-1] > 64)
        and not (kernel_head_dim == 256 and q.shape[-1] > 192)
    )
    needs_dtype_cast = q.dtype == torch.bfloat16
    needs_last_dim_copy = any(x.stride(-1) != 1 for x in (q, k, v))
    uses_native_pad = (
        needs_pad
        and q.shape[-1] == 96
        and q.dtype == torch.float16
        and hasattr(_sm75_cuda, "pad_qkv_head_dim")
    )

    result = try_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        return_softmax,
        use_native_pad=True,
        packed_layout=True,
    )
    if result is None:
        return None

    base_backend = get_last_backend() or "sm75_native_unknown"
    backend_tail = (
        base_backend[len("sm75_native_") :]
        if base_backend.startswith("sm75_native_")
        else base_backend
    )
    layout_bits = []
    if needs_pad:
        layout_bits.append("nativepad" if uses_native_pad else "padded")
    if needs_dtype_cast:
        layout_bits.append("fp16_compute")
    if needs_last_dim_copy:
        layout_bits.append("lastdim_copy")
    layout_tag = "strided" if not layout_bits else "_".join(layout_bits)
    _set_last_backend(f"sm75_native_{packed_kind}_direct_{layout_tag}_{backend_tail}")
    return result


def try_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool,
    block_table: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    num_splits: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if block_table is not None:
        batch_size = cu_seqlens_q.numel() - 1 if cu_seqlens_q.dim() == 1 else -1
        reason = None
        if _sm75_cuda is None:
            reason = "the flash_attn_sm75_cuda extension is not importable"
        elif not _is_sm75_tensor(q):
            reason = "the input tensor is not on an SM75 CUDA device"
        elif not (k.is_cuda and v.is_cuda and block_table.is_cuda):
            reason = "q, paged k/v cache, and block_table must be CUDA tensors"
        elif q.device != k.device or q.device != v.device or block_table.device != q.device:
            reason = "q, paged k/v cache, and block_table must be on the same CUDA device"
        elif q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
            reason = "q, k, and v must be FP16 or BF16 tensors with matching dtype"
        elif q.dim() != 3 or k.dim() != 4 or v.dim() != 4:
            reason = "paged varlen q must be rank-3 and paged k/v cache must be rank-4"
        elif q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            reason = "q, k, and v must have matching head_dim"
        elif q.shape[-1] not in _SUPPORTED_HEAD_DIMS:
            reason = f"head_dim {q.shape[-1]} has no native SM75 varlen kernel"
        elif k.shape[0] != v.shape[0] or k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
            reason = "paged k and v cache shapes must match"
        elif q.shape[-2] % k.shape[-2] != 0:
            reason = "query heads must be divisible by key/value heads"
        elif dropout_p < 0.0 or dropout_p >= 1.0:
            reason = "dropout_p must be in [0, 1)"
        elif not (cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda):
            reason = "cu_seqlens_q and cu_seqlens_k must be CUDA tensors"
        elif cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            reason = "cu_seqlens_q and cu_seqlens_k must be int32 tensors"
        elif cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1 or cu_seqlens_k.numel() != cu_seqlens_q.numel():
            reason = "cu_seqlens_q/cu_seqlens_k must have matching [batch + 1] shape"
        elif block_table.dtype != torch.int32 or block_table.dim() != 2 or block_table.shape[0] != batch_size:
            reason = "block_table must be int32 with shape [batch, max_blocks]"
        elif num_splits < 0:
            reason = "num_splits must be non-negative"
        if reason is None:
            reason = _check_optional_i32_batch(leftpad_k, "leftpad_k", batch_size, q.device)
        if reason is None:
            reason = _check_optional_i32_batch(seqused_k, "seqused_k", batch_size, q.device)
        if reason is None and alibi_slopes is not None:
            if not (alibi_slopes.is_cuda and alibi_slopes.device == q.device):
                reason = "ALiBi slopes must be on the same CUDA device as q"
            elif alibi_slopes.dtype != torch.float32:
                reason = "ALiBi slopes must be fp32"
            elif alibi_slopes.dim() not in (1, 2):
                reason = "ALiBi slopes must have shape [nheads] or [batch, nheads]"
            elif alibi_slopes.shape[-1] != q.shape[-2] or (alibi_slopes.dim() == 2 and alibi_slopes.shape[0] != batch_size):
                reason = "ALiBi slopes shape must match varlen batch and query heads"
        _maybe_raise_for_banned_sm75(q, reason)
        if reason is not None:
            return None

        q_offsets = _tensor_to_int_list(cu_seqlens_q)
        k_offsets = _tensor_to_int_list(cu_seqlens_k)
        if q_offsets[0] != 0 or q_offsets[-1] != q.shape[0]:
            reason = "cu_seqlens_q must start at zero and end at total query tokens"
        elif k_offsets[0] != 0:
            reason = "cu_seqlens_k must start at zero"
        else:
            for start, end in zip(q_offsets, q_offsets[1:]):
                if end < start or end - start > max_seqlen_q:
                    reason = "cu_seqlens_q must be monotonic and bounded by max_seqlen_q"
                    break
            if reason is None:
                for start, end in zip(k_offsets, k_offsets[1:]):
                    if end < start or end - start > max_seqlen_k:
                        reason = "cu_seqlens_k must be monotonic and bounded by max_seqlen_k"
                        break
        if reason is None:
            reason, leftpads, effective_lengths = _validate_varlen_aux_lengths(
                q, k_offsets, max_seqlen_k, leftpad_k, seqused_k
            )
        _maybe_raise_for_banned_sm75(q, reason)
        if reason is not None:
            return None

        page_size = k.shape[1]
        used_lengths = [leftpad + effective for leftpad, effective in zip(leftpads, effective_lengths)]
        k_chunks = []
        v_chunks = []
        block_rows = block_table.to(device=q.device, dtype=torch.long)
        for batch_idx, (leftpad, used_length) in enumerate(zip(leftpads, used_lengths)):
            num_pages = (used_length + page_size - 1) // page_size
            block_ids = block_rows[batch_idx, :num_pages]
            k_chunks.append(k.index_select(0, block_ids).reshape(-1, k.shape[2], k.shape[3])[leftpad:used_length])
            v_chunks.append(v.index_select(0, block_ids).reshape(-1, v.shape[2], v.shape[3])[leftpad:used_length])
        k_packed = torch.cat(k_chunks, dim=0).contiguous()
        v_packed = torch.cat(v_chunks, dim=0).contiguous()
        cu_k_eff = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
        cu_k_eff[1:] = torch.cumsum(torch.tensor(effective_lengths, dtype=torch.int32, device=q.device), dim=0)
        result = try_varlen_forward(
            q,
            k_packed,
            v_packed,
            cu_seqlens_q,
            cu_k_eff,
            max_seqlen_q,
            max(effective_lengths),
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            alibi_slopes,
            return_softmax,
            None,
            None,
            None,
            0,
        )
        if result is not None:
            _set_last_backend(
                "sm75_native_varlen_paged_fwd"
                f"_hd{q.shape[-1]}"
                f"_{'causal' if causal else 'noncausal'}"
                f"_{_head_group_suffix(q, k)}"
                f"_{_dtype_suffix(q.dtype)}"
            )
        return result

    reason = _varlen_rejection(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        alibi_slopes,
    )
    batch_size = cu_seqlens_q.numel() - 1 if cu_seqlens_q.dim() == 1 else -1
    if reason is None:
        reason = _check_optional_i32_batch(leftpad_k, "leftpad_k", batch_size, q.device)
    if reason is None:
        reason = _check_optional_i32_batch(seqused_k, "seqused_k", batch_size, q.device)
    if reason is None and num_splits < 0:
        reason = "num_splits must be non-negative"
    _maybe_raise_for_banned_sm75(q, reason)
    if reason is not None:
        return None

    if leftpad_k is not None or seqused_k is not None:
        offsets_reason, q_offsets, k_offsets = _varlen_offsets(
            q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )
        _maybe_raise_for_banned_sm75(q, offsets_reason)
        if offsets_reason is not None:
            return None
        aux_reason, leftpads, effective_lengths = _validate_varlen_aux_lengths(
            q, k_offsets, max_seqlen_k, leftpad_k, seqused_k
        )
        _maybe_raise_for_banned_sm75(q, aux_reason)
        if aux_reason is not None:
            return None
        used_lengths = [leftpad + effective for leftpad, effective in zip(leftpads, effective_lengths)]
        k_chunks = [
            k[k_start + leftpad : k_start + used_length]
            for k_start, leftpad, used_length in zip(k_offsets, leftpads, used_lengths)
        ]
        v_chunks = [
            v[k_start + leftpad : k_start + used_length]
            for k_start, leftpad, used_length in zip(k_offsets, leftpads, used_lengths)
        ]
        k_packed = torch.cat(k_chunks, dim=0).contiguous()
        v_packed = torch.cat(v_chunks, dim=0).contiguous()
        cu_k_eff = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
        cu_k_eff[1:] = torch.cumsum(torch.tensor(effective_lengths, dtype=torch.int32, device=q.device), dim=0)
        result = try_varlen_forward(
            q,
            k_packed,
            v_packed,
            cu_seqlens_q,
            cu_k_eff,
            max_seqlen_q,
            max(effective_lengths),
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            alibi_slopes,
            return_softmax,
            None,
            None,
            None,
            0,
        )
        if result is not None:
            _set_last_backend(
                "sm75_native_varlen_sliced_fwd"
                f"_hd{q.shape[-1]}"
                f"_{'causal' if causal else 'noncausal'}"
                f"_{_head_group_suffix(q, k)}"
                f"_{_dtype_suffix(q.dtype)}"
            )
        return result

    kernel_head_dim = _kernel_head_dim(q.shape[-1])
    use_fast_kernel = (
        kernel_head_dim is not None
        and dropout_p == 0.0
        and window_size_left == -1
        and window_size_right == -1
        and softcap == 0.0
        and alibi_slopes is None
        and not return_softmax
    )
    if use_fast_kernel:
        q_sm75, k_sm75, v_sm75 = (_prepare_for_sm75_kernel(x, kernel_head_dim) for x in (q, k, v))
        if kernel_head_dim == 96:
            out, softmax_lse = _sm75_cuda.varlen_fwd_hd96_fa2(
                q_sm75,
                k_sm75,
                v_sm75,
                cu_seqlens_q,
                cu_seqlens_k,
                int(max_seqlen_q),
                int(max_seqlen_k),
                float(softmax_scale),
                bool(causal),
            )
        elif kernel_head_dim == 256:
            out, softmax_lse = _sm75_cuda.varlen_fwd_hd256_fa2(
                q_sm75,
                k_sm75,
                v_sm75,
                cu_seqlens_q,
                cu_seqlens_k,
                int(max_seqlen_q),
                int(max_seqlen_k),
                float(softmax_scale),
                bool(causal),
            )
        else:
            out, softmax_lse = _sm75_cuda.varlen_fwd(
                q_sm75,
                k_sm75,
                v_sm75,
                cu_seqlens_q,
                cu_seqlens_k,
                int(max_seqlen_q),
                int(max_seqlen_k),
                float(softmax_scale),
                bool(causal),
            )
        out = out[..., : q.shape[-1]]
        s_dmask = torch.empty((0,), dtype=out.dtype, device=out.device)
        if kernel_head_dim == 96:
            backend_kind = f"varlen_dense_hd{q.shape[-1]}_fa2_tiled_kernel96"
        elif kernel_head_dim == 256:
            backend_kind = f"varlen_dense_hd{q.shape[-1]}_fa2_tiled_kernel256"
        elif kernel_head_dim is not None and kernel_head_dim > 128 and q.shape[-1] > 128:
            backend_kind = f"varlen_dense_hd{q.shape[-1]}_tiled_kernel{kernel_head_dim}"
        else:
            backend_kind = f"varlen_fwd_hd{q.shape[-1]}_kernel{kernel_head_dim}"
    else:
        offsets_reason, q_offsets, k_offsets = _varlen_offsets(
            q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )
        _maybe_raise_for_banned_sm75(q, offsets_reason)
        if offsets_reason is not None:
            return None
        q_sm75, k_sm75, v_sm75 = (_to_fp16_for_sm75(x) for x in (q, k, v))
        out = torch.empty_like(q_sm75)
        softmax_lse = torch.zeros(
            (len(q_offsets) - 1, q.shape[-2], int(max_seqlen_q)),
            dtype=torch.float32,
            device=q.device,
        )
        s_dmask = (
            torch.zeros(
                (len(q_offsets) - 1, q.shape[-2], int(max_seqlen_q), int(max_seqlen_k)),
                dtype=torch.float16,
                device=q.device,
            )
            if return_softmax
            else torch.empty((0,), dtype=torch.float16, device=q.device)
        )
        for batch_idx, (q_start, q_end, k_start, k_end) in enumerate(
            zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:])
        ):
            q_len = q_end - q_start
            k_len = k_end - k_start
            if q_len == 0:
                continue
            if k_len == 0:
                out[q_start:q_end].zero_()
                softmax_lse[batch_idx, :, :q_len].fill_(float("inf"))
                continue
            alibi_i = _select_varlen_alibi(alibi_slopes, batch_idx)
            result = try_forward(
                q_sm75[q_start:q_end].unsqueeze(0),
                k_sm75[k_start:k_end].unsqueeze(0),
                v_sm75[k_start:k_end].unsqueeze(0),
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_i,
                return_softmax,
            )
            if result is None:
                return None
            out_i, lse_i, probs_i, _ = result
            out[q_start:q_end].copy_(out_i.squeeze(0))
            softmax_lse[batch_idx, :, :q_len].copy_(lse_i.squeeze(0))
            if return_softmax:
                s_dmask[batch_idx, :, :q_len, :k_len].copy_(probs_i.squeeze(0))
        backend_kind = f"varlen_generic_fwd_hd{q.shape[-1]}"
    out = _restore_sm75_output_dtype(out, q)
    rng_state = torch.empty((2,), dtype=torch.int64, device=out.device)
    _set_last_backend(
        "sm75_native_"
        f"{backend_kind}"
        f"_{'causal' if causal else 'noncausal'}"
        f"_{_head_group_suffix(q, k)}"
        f"_{_dtype_suffix(q.dtype)}"
        f"{'_dropout' if dropout_p > 0.0 else ''}"
        f"{'_local' if window_size_left != -1 or window_size_right != -1 else ''}"
    )
    return out, softmax_lse, s_dmask, rng_state


def try_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool = False,
) -> Optional[torch.Tensor]:
    reason = _backward_rejection(
        q,
        k,
        v,
        dropout_p,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        return_softmax,
    )
    if reason is None and (dq is None or dk is None or dv is None):
        reason = "gradient output buffers are required for the SM75 native backward path"
    _maybe_raise_for_banned_sm75(q, reason)
    if reason is not None:
        return None

    kernel_head_dim = _backward_kernel_head_dim(q.shape[-1])
    use_fast_kernel = (
        kernel_head_dim is not None
        and kernel_head_dim <= 128
        and dropout_p == 0.0
        and window_size_left == -1
        and window_size_right == -1
        and softcap == 0.0
        and alibi_slopes is None
    )
    if use_fast_kernel:
        q_sm75, k_sm75, v_sm75, out_sm75, dout_sm75 = (
            _prepare_for_sm75_kernel(x, kernel_head_dim, preserve_strides=False)
            for x in (q, k, v, out, dout)
        )
        dq_sm75, dk_sm75, dv_sm75 = _sm75_cuda.bwd(
            q_sm75,
            k_sm75,
            v_sm75,
            out_sm75,
            softmax_lse,
            dout_sm75,
            float(softmax_scale),
            bool(causal),
        )
        dq.copy_(dq_sm75[..., : q.shape[-1]])
        dk.copy_(dk_sm75[..., : k.shape[-1]])
        dv.copy_(dv_sm75[..., : v.shape[-1]])
        backend_kind = f"bwd_hd{q.shape[-1]}_kernel{kernel_head_dim}"
    else:
        q_sm75, k_sm75, v_sm75, dout_sm75 = (_to_fp16_for_sm75(x) for x in (q, k, v, dout))
        alibi_sm75 = (
            torch.empty((0,), dtype=torch.float32, device=q.device)
            if alibi_slopes is None
            else alibi_slopes.contiguous()
        )
        dq_sm75, dk_sm75, dv_sm75 = _sm75_cuda.bwd_generic(
            q_sm75,
            k_sm75,
            v_sm75,
            dout_sm75,
            float(softmax_scale),
            bool(causal),
            int(window_size_left),
            int(window_size_right),
            float(dropout_p),
            alibi_sm75,
            float(softcap),
        )
        dq.copy_(dq_sm75)
        dk.copy_(dk_sm75)
        dv.copy_(dv_sm75)
        backend_kind = f"generic_bwd_hd{q.shape[-1]}"
    _set_last_backend(
        "sm75_native_"
        f"{backend_kind}"
        f"_{'causal' if causal else 'noncausal'}"
        f"_{_head_group_suffix(q, k)}"
        f"_{_dtype_suffix(q.dtype)}"
        f"{'_dropout' if dropout_p > 0.0 else ''}"
        f"{'_local' if window_size_left != -1 or window_size_right != -1 else ''}"
    )
    return torch.empty((0,), dtype=torch.float32, device=q.device)


def try_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    zero_tensors: bool,
) -> Optional[torch.Tensor]:
    reason = _varlen_rejection(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        alibi_slopes,
    )
    if reason is None and (dq is None or dk is None or dv is None):
        reason = "gradient output buffers are required for the SM75 native varlen backward path"
    _maybe_raise_for_banned_sm75(q, reason)
    if reason is not None:
        return None

    kernel_head_dim = _backward_kernel_head_dim(q.shape[-1])
    use_fast_kernel = (
        kernel_head_dim is not None
        and kernel_head_dim <= 128
        and dropout_p == 0.0
        and window_size_left == -1
        and window_size_right == -1
        and softcap == 0.0
        and alibi_slopes is None
    )
    if use_fast_kernel:
        q_sm75, k_sm75, v_sm75, out_sm75, dout_sm75 = (
            _prepare_for_sm75_kernel(x, kernel_head_dim, preserve_strides=False)
            for x in (q, k, v, out, dout)
        )
        dq_sm75, dk_sm75, dv_sm75 = _sm75_cuda.varlen_bwd(
            q_sm75,
            k_sm75,
            v_sm75,
            out_sm75,
            softmax_lse,
            dout_sm75,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            float(softmax_scale),
            bool(causal),
        )
        dq.copy_(dq_sm75[..., : q.shape[-1]])
        dk.copy_(dk_sm75[..., : k.shape[-1]])
        dv.copy_(dv_sm75[..., : v.shape[-1]])
        backend_kind = f"varlen_bwd_hd{q.shape[-1]}_kernel{kernel_head_dim}"
    else:
        offsets_reason, q_offsets, k_offsets = _varlen_offsets(
            q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )
        _maybe_raise_for_banned_sm75(q, offsets_reason)
        if offsets_reason is not None:
            return None
        dq.zero_()
        dk.zero_()
        dv.zero_()
        q_sm75, k_sm75, v_sm75, dout_sm75 = (_to_fp16_for_sm75(x) for x in (q, k, v, dout))
        for batch_idx, (q_start, q_end, k_start, k_end) in enumerate(
            zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:])
        ):
            q_len = q_end - q_start
            k_len = k_end - k_start
            if q_len == 0 or k_len == 0:
                continue
            alibi_i = _select_varlen_alibi(alibi_slopes, batch_idx)
            result = try_backward(
                dout_sm75[q_start:q_end].unsqueeze(0),
                q_sm75[q_start:q_end].unsqueeze(0),
                k_sm75[k_start:k_end].unsqueeze(0),
                v_sm75[k_start:k_end].unsqueeze(0),
                out[q_start:q_end].unsqueeze(0),
                softmax_lse[batch_idx : batch_idx + 1, :, :q_len],
                dq[q_start:q_end].unsqueeze(0),
                dk[k_start:k_end].unsqueeze(0),
                dv[k_start:k_end].unsqueeze(0),
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_i,
            )
            if result is None:
                return None
        backend_kind = f"varlen_generic_bwd_hd{q.shape[-1]}"
    _set_last_backend(
        "sm75_native_"
        f"{backend_kind}"
        f"_{'causal' if causal else 'noncausal'}"
        f"_{_head_group_suffix(q, k)}"
        f"_{_dtype_suffix(q.dtype)}"
        f"{'_dropout' if dropout_p > 0.0 else ''}"
        f"{'_local' if window_size_left != -1 or window_size_right != -1 else ''}"
    )
    return torch.empty((0,), dtype=torch.float32, device=q.device)


def _tensor_to_int_list(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def try_kvcache_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    cache_batch_idx: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    rotary_interleaved: bool,
    alibi_slopes: Optional[torch.Tensor],
    num_splits: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    reason = _forward_rejection(
        q,
        k_cache,
        v_cache,
        0.0,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        False,
    )
    batch_size, seqlen_q = q.shape[0], q.shape[1]
    paged_cache = block_table is not None
    if reason is None and paged_cache and cache_seqlens is None:
        reason = "cache_seqlens is required for SM75 paged KV cache"
    if reason is None and (rotary_cos is None) != (rotary_sin is None):
        reason = "rotary_cos and rotary_sin must either both be provided or both be omitted"
    if reason is None and rotary_cos is not None:
        if k is None or v is None:
            reason = "rotary KV cache requires new k/v tensors"
        elif not (rotary_cos.is_cuda and rotary_sin.is_cuda and rotary_cos.device == q.device and rotary_sin.device == q.device):
            reason = "rotary_cos and rotary_sin must be CUDA tensors on the query device"
        elif rotary_cos.dtype != q.dtype or rotary_sin.dtype != q.dtype:
            reason = "rotary_cos and rotary_sin must match q dtype"
        elif rotary_cos.dim() != 2 or rotary_sin.shape != rotary_cos.shape:
            reason = "rotary_cos and rotary_sin must have shape [seqlen_ro, rotary_dim / 2]"
        elif rotary_cos.shape[1] * 2 > q.shape[-1]:
            reason = "rotary_dim must be <= head_dim"
        elif (rotary_cos.shape[1] * 2) % 16 != 0:
            reason = "rotary_dim must be divisible by 16"
    if reason is None and cache_leftpad is not None:
        if not (cache_leftpad.is_cuda and cache_leftpad.device == q.device):
            reason = "cache_leftpad must be a CUDA tensor on the query device"
        elif cache_leftpad.dtype != torch.int32:
            reason = "cache_leftpad must be int32"
        elif cache_leftpad.numel() != batch_size:
            reason = "cache_leftpad must have one entry per query batch"
    if reason is None and num_splits < 0:
        reason = "num_splits must be non-negative"
    if reason is None and (k is None) != (v is None):
        reason = "k and v must either both be provided or both be omitted"
    if reason is None and k is not None:
        if k.shape[0] != batch_size or v.shape[0] != batch_size:
            reason = "new k/v batch size must match q"
        elif k.shape[1] != v.shape[1]:
            reason = "new k/v sequence lengths must match"
        elif k.shape[2:] != k_cache.shape[2:] or v.shape[2:] != v_cache.shape[2:]:
            reason = "new k/v head and headdim shapes must match the cache"
        elif k.dtype != k_cache.dtype or v.dtype != v_cache.dtype:
            reason = "new k/v dtype must match the cache dtype"
    if reason is None and cache_batch_idx is not None and cache_batch_idx.numel() != batch_size:
        reason = "cache_batch_idx must have one entry per query batch"
    if reason is None and cache_seqlens is not None and cache_seqlens.numel() != batch_size:
        reason = "cache_seqlens must have one entry per query batch"
    if reason is None and cache_seqlens is None and k is not None:
        reason = "cache_seqlens is required when updating the KV cache"
    _maybe_raise_for_banned_sm75(q, reason)
    if reason is not None:
        return None

    if cache_batch_idx is None:
        batch_indices = list(range(batch_size))
        batch_index_tensor = None
    else:
        batch_indices = _tensor_to_int_list(cache_batch_idx)
        batch_index_tensor = cache_batch_idx.to(device=k_cache.device, dtype=torch.long)
    batch_index_limit = block_table.shape[0] if paged_cache else k_cache.shape[0]
    if any(idx < 0 or idx >= batch_index_limit for idx in batch_indices):
        reason = "cache_batch_idx contains an out-of-range cache batch"
        _maybe_raise_for_banned_sm75(q, reason)
        return None

    if cache_seqlens is None:
        seqlens_before = [int(k_cache.shape[1])] * batch_size
    else:
        seqlens_before = _tensor_to_int_list(cache_seqlens)
    if cache_leftpad is None:
        cache_leftpads = [0] * batch_size
    else:
        cache_leftpads = _tensor_to_int_list(cache_leftpad)
    cache_capacity = block_table.shape[1] * k_cache.shape[1] if paged_cache else k_cache.shape[1]
    if any(length < 0 or length > cache_capacity for length in seqlens_before):
        reason = "cache_seqlens contains an out-of-range sequence length"
        _maybe_raise_for_banned_sm75(q, reason)
        return None
    if any(leftpad < 0 or leftpad > length for leftpad, length in zip(cache_leftpads, seqlens_before)):
        reason = "cache_leftpad must be between zero and cache_seqlens"
        _maybe_raise_for_banned_sm75(q, reason)
        return None

    seqlen_new = 0 if k is None else int(k.shape[1])
    seqlens_after = [length + seqlen_new for length in seqlens_before]
    if any(length > cache_capacity for length in seqlens_after):
        reason = "updated cache sequence length exceeds cache capacity"
        _maybe_raise_for_banned_sm75(q, reason)
        return None
    effective_seqlens = [length - leftpad for length, leftpad in zip(seqlens_after, cache_leftpads)]
    if any(length <= 0 for length in effective_seqlens):
        reason = "SM75 KV-cache attention requires at least one unpadded key per sequence"
        _maybe_raise_for_banned_sm75(q, reason)
        return None

    q_attention = q
    k_update = k
    fused_paged_update = False
    cache_seqlens_after_tensor = None
    sequential_q_positions = causal or window_size_left != -1 or window_size_right != -1
    can_fused_paged_update = (
        paged_cache
        and k is not None
        and v is not None
        and q.dtype == torch.float16
        and k.dtype == torch.float16
        and v.dtype == torch.float16
        and k_cache.dtype == torch.float16
        and v_cache.dtype == torch.float16
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and block_table is not None
        and block_table.is_cuda
        and block_table.dtype == torch.int32
        and block_table.is_contiguous()
        and cache_seqlens is not None
        and cache_seqlens.is_cuda
        and cache_seqlens.dtype == torch.int32
        and cache_seqlens.is_contiguous()
        and seqlen_q <= 8
        and q.shape[-1] <= 128
        and softcap == 0.0
        and alibi_slopes is None
        and (cache_batch_idx is None or (
            cache_batch_idx.is_cuda and cache_batch_idx.dtype == torch.int32 and cache_batch_idx.is_contiguous()
        ))
        and (cache_leftpad is None or (
            cache_leftpad.is_cuda and cache_leftpad.dtype == torch.int32 and cache_leftpad.is_contiguous()
        ))
        and (rotary_cos is None or (
            rotary_cos.is_contiguous() and rotary_sin is not None and rotary_sin.is_contiguous()
        ))
    )
    if can_fused_paged_update and rotary_cos is not None:
        max_q_pos = max(
            base + (seqlen_q - 1 if sequential_q_positions else 0)
            for base in seqlens_before
        )
        max_k_pos = max(seqlens_after) - 1
        if max(max_q_pos, max_k_pos) >= rotary_cos.shape[0]:
            reason = "rotary cos/sin sequence length is too short for cache positions"
            _maybe_raise_for_banned_sm75(q, reason)
            return None
    if can_fused_paged_update:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=q.device)
        empty_rotary = torch.empty((0,), dtype=q.dtype, device=q.device)
        q_attention, cache_seqlens_after_tensor = _sm75_cuda.paged_kvcache_update(
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_table,
            cache_seqlens,
            empty_i32 if cache_batch_idx is None else cache_batch_idx,
            empty_rotary if rotary_cos is None else rotary_cos,
            empty_rotary if rotary_sin is None else rotary_sin,
            bool(rotary_interleaved),
            bool(sequential_q_positions),
        )
        k_update = None
        fused_paged_update = True

    if not fused_paged_update and rotary_cos is not None:
        base_positions = torch.tensor(seqlens_before, dtype=torch.long, device=q.device)
        q_offsets = (
            base_positions[:, None] + torch.arange(seqlen_q, dtype=torch.long, device=q.device)[None, :]
            if sequential_q_positions
            else base_positions[:, None].expand(batch_size, seqlen_q)
        )
        k_offsets = base_positions[:, None] + torch.arange(seqlen_new, dtype=torch.long, device=q.device)[None, :]
        max_rotary_pos = int(torch.maximum(q_offsets.max(), k_offsets.max()).item())
        if max_rotary_pos >= rotary_cos.shape[0]:
            reason = "rotary cos/sin sequence length is too short for cache positions"
            _maybe_raise_for_banned_sm75(q, reason)
            return None
        q_attention = _apply_rotary_sm75(q, rotary_cos, rotary_sin, q_offsets, rotary_interleaved)
        k_update = _apply_rotary_sm75(k, rotary_cos, rotary_sin, k_offsets, rotary_interleaved)

    if not fused_paged_update and k_update is not None:
        for q_batch, cache_batch in enumerate(batch_indices):
            start = seqlens_before[q_batch]
            end = seqlens_after[q_batch]
            if paged_cache:
                page_size = k_cache.shape[1]
                for token_offset, pos in enumerate(range(start, end)):
                    block_slot = pos // page_size
                    block_offset = pos % page_size
                    block_id = int(block_table[cache_batch, block_slot].item())
                    k_cache[block_id, block_offset].copy_(k_update[q_batch, token_offset])
                    v_cache[block_id, block_offset].copy_(v[q_batch, token_offset])
            else:
                k_cache[cache_batch, start:end].copy_(k_update[q_batch])
                v_cache[cache_batch, start:end].copy_(v[q_batch])

    max_seqlen_k = max(effective_seqlens)

    if paged_cache:
        page_size = k_cache.shape[1]
        result = None
        path = None
        can_direct_paged = (
            q.dtype == torch.float16
            and k_cache.dtype == torch.float16
            and v_cache.dtype == torch.float16
            and seqlen_q <= 8
            and q.shape[-1] <= 128
            and softcap == 0.0
            and alibi_slopes is None
            and q_attention.is_cuda
            and k_cache.is_contiguous()
            and v_cache.is_contiguous()
            and block_table.is_contiguous()
            and cache_seqlens is not None
            and cache_seqlens.is_cuda
            and cache_seqlens.dtype == torch.int32
            and cache_seqlens.is_contiguous()
            and (cache_batch_idx is None or (
                cache_batch_idx.is_cuda and cache_batch_idx.dtype == torch.int32 and cache_batch_idx.is_contiguous()
            ))
            and (cache_leftpad is None or (
                cache_leftpad.is_cuda and cache_leftpad.dtype == torch.int32 and cache_leftpad.is_contiguous()
            ))
        )
        if can_direct_paged:
            empty_i32 = torch.empty((0,), dtype=torch.int32, device=q.device)
            cache_seqlens_after = (
                cache_seqlens_after_tensor
                if cache_seqlens_after_tensor is not None
                else cache_seqlens if seqlen_new == 0 else cache_seqlens + seqlen_new
            )
            direct_prefix = "paged_direct_fused_update" if fused_paged_update else "paged_direct"
            can_fa2_paged = (
                hasattr(_sm75_cuda, "paged_kvcache_fwd_fa2")
                and q.shape[-1] in (64, 128)
                and page_size % 256 == 0
                and all(leftpad == 0 for leftpad in cache_leftpads)
                and window_size_left == -1
                and window_size_right == -1
            )
            if can_fa2_paged:
                out, softmax_lse = _sm75_cuda.paged_kvcache_fwd_fa2(
                    _to_fp16_for_sm75(q_attention),
                    k_cache,
                    v_cache,
                    block_table,
                    cache_seqlens_after,
                    empty_i32 if cache_batch_idx is None else cache_batch_idx,
                    empty_i32,
                    float(softmax_scale),
                    bool(causal),
                    int(window_size_left),
                    int(window_size_right),
                    int(max_seqlen_k),
                    0,
                )
                path = f"{direct_prefix}_fa2_tiled_splitkv"
            else:
                out, softmax_lse = _sm75_cuda.paged_kvcache_fwd(
                    _to_fp16_for_sm75(q_attention),
                    k_cache,
                    v_cache,
                    block_table,
                    cache_seqlens_after,
                    empty_i32 if cache_batch_idx is None else cache_batch_idx,
                    empty_i32 if cache_leftpad is None else cache_leftpad,
                    float(softmax_scale),
                    bool(causal),
                    int(window_size_left),
                    int(window_size_right),
                    int(max_seqlen_k),
                )
                path = f"{direct_prefix}_wmma_splitkv"
            result = (out, softmax_lse, None, None)
        if (
            result is None
            and cache_batch_idx is None
            and all(leftpad == 0 for leftpad in cache_leftpads)
            and all(length == max_seqlen_k for length in effective_seqlens)
        ):
            num_pages = (max_seqlen_k + page_size - 1) // page_size
            actual_blocks = block_table[:batch_size, :num_pages]
            first_block = int(actual_blocks[0, 0].item())
            expected_blocks = torch.arange(
                first_block,
                first_block + batch_size * num_pages,
                dtype=block_table.dtype,
                device=block_table.device,
            ).reshape(batch_size, num_pages)
            if bool(torch.equal(actual_blocks, expected_blocks)):
                k_dense = k_cache.narrow(0, first_block, batch_size * num_pages).reshape(
                    batch_size, num_pages * page_size, k_cache.shape[2], k_cache.shape[3]
                )[:, :max_seqlen_k]
                v_dense = v_cache.narrow(0, first_block, batch_size * num_pages).reshape(
                    batch_size, num_pages * page_size, v_cache.shape[2], v_cache.shape[3]
                )[:, :max_seqlen_k]
                if q_attention.dtype != k_dense.dtype:
                    k_dense = k_dense.to(q_attention.dtype)
                    v_dense = v_dense.to(q_attention.dtype)
                result = try_forward(
                    q_attention,
                    k_dense,
                    v_dense,
                    0.0,
                    softmax_scale,
                    causal,
                    window_size_left,
                    window_size_right,
                    softcap,
                    alibi_slopes,
                    False,
                )
                path = "paged_view_dense" if result is not None else None
        if (
            result is None
            and all(leftpad == 0 for leftpad in cache_leftpads)
            and all(length == max_seqlen_k for length in effective_seqlens)
        ):
            num_pages = (max_seqlen_k + page_size - 1) // page_size
            block_rows = block_table[:batch_size] if batch_index_tensor is None else block_table.index_select(0, batch_index_tensor)
            selected_blocks = block_rows[:, :num_pages]
            if bool(torch.all((selected_blocks >= 0) & (selected_blocks < k_cache.shape[0]))):
                flat_blocks = selected_blocks.reshape(-1).to(device=k_cache.device, dtype=torch.long)
                k_dense = k_cache.index_select(0, flat_blocks).reshape(
                    batch_size, num_pages * page_size, k_cache.shape[2], k_cache.shape[3]
                )[:, :max_seqlen_k]
                v_dense = v_cache.index_select(0, flat_blocks).reshape(
                    batch_size, num_pages * page_size, v_cache.shape[2], v_cache.shape[3]
                )[:, :max_seqlen_k]
                if q_attention.dtype != k_dense.dtype:
                    k_dense = k_dense.to(q_attention.dtype)
                    v_dense = v_dense.to(q_attention.dtype)
                result = try_forward(
                    q_attention,
                    k_dense,
                    v_dense,
                    0.0,
                    softmax_scale,
                    causal,
                    window_size_left,
                    window_size_right,
                    softcap,
                    alibi_slopes,
                    False,
                )
                path = "paged_gather_dense" if result is not None else None
        if result is None:
            k_chunks = []
            v_chunks = []
            for cache_batch, leftpad, length in zip(batch_indices, cache_leftpads, seqlens_after):
                num_pages = (length + page_size - 1) // page_size
                block_ids = block_table[cache_batch, :num_pages].to(device=k_cache.device, dtype=torch.long)
                k_chunks.append(k_cache.index_select(0, block_ids).reshape(-1, k_cache.shape[2], k_cache.shape[3])[leftpad:length])
                v_chunks.append(v_cache.index_select(0, block_ids).reshape(-1, v_cache.shape[2], v_cache.shape[3])[leftpad:length])
            q_packed = q_attention.reshape(batch_size * seqlen_q, q_attention.shape[2], q_attention.shape[3])
            k_packed = torch.cat(k_chunks, dim=0)
            v_packed = torch.cat(v_chunks, dim=0)
            if q_packed.dtype != k_packed.dtype:
                k_packed = k_packed.to(q_packed.dtype)
                v_packed = v_packed.to(q_packed.dtype)
            cu_q = torch.arange(
                0,
                (batch_size + 1) * seqlen_q,
                step=seqlen_q,
                dtype=torch.int32,
                device=q.device,
            )
            cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
            cu_k[1:] = torch.cumsum(torch.tensor(effective_seqlens, dtype=torch.int32, device=q.device), dim=0)
            result = try_varlen_forward(
                q_packed,
                k_packed,
                v_packed,
                cu_q,
                cu_k,
                seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_slopes,
                False,
                None,
                None,
                None,
                0,
            )
            path = "paged_repack_varlen"
        if result is None:
            return None
        out_packed, softmax_lse, _, _ = result
        out = out_packed.reshape_as(q)
    else:
        selected_k_cache = k_cache[:batch_size] if batch_index_tensor is None else k_cache.index_select(0, batch_index_tensor)
        selected_v_cache = v_cache[:batch_size] if batch_index_tensor is None else v_cache.index_select(0, batch_index_tensor)
        if q_attention.dtype != selected_k_cache.dtype:
            selected_k_cache = selected_k_cache.to(q_attention.dtype)
            selected_v_cache = selected_v_cache.to(q_attention.dtype)

        if all(leftpad == 0 for leftpad in cache_leftpads) and all(length == max_seqlen_k for length in effective_seqlens):
            result = try_forward(
                q_attention,
                selected_k_cache[:, :max_seqlen_k],
                selected_v_cache[:, :max_seqlen_k],
                0.0,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_slopes,
                False,
            )
            if result is None:
                return None
            out, softmax_lse, _, _ = result
            path = "dense"
        else:
            q_packed = q_attention.reshape(batch_size * seqlen_q, q_attention.shape[2], q_attention.shape[3])
            k_chunks = [
                selected_k_cache[i, leftpad:length]
                for i, (leftpad, length) in enumerate(zip(cache_leftpads, seqlens_after))
            ]
            v_chunks = [
                selected_v_cache[i, leftpad:length]
                for i, (leftpad, length) in enumerate(zip(cache_leftpads, seqlens_after))
            ]
            k_packed = torch.cat(k_chunks, dim=0)
            v_packed = torch.cat(v_chunks, dim=0)
            cu_q = torch.arange(
                0,
                (batch_size + 1) * seqlen_q,
                step=seqlen_q,
                dtype=torch.int32,
                device=q.device,
            )
            cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
            cu_k[1:] = torch.cumsum(torch.tensor(effective_seqlens, dtype=torch.int32, device=q.device), dim=0)
            result = try_varlen_forward(
                q_packed,
                k_packed,
                v_packed,
                cu_q,
                cu_k,
                seqlen_q,
                max_seqlen_k,
                0.0,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                alibi_slopes,
                False,
                None,
                None,
                None,
                0,
            )
            if result is None:
                return None
            out_packed, softmax_lse, _, _ = result
            out = out_packed.reshape_as(q)
            path = "varlen"

    out = _restore_sm75_output_dtype(out, q)
    _set_last_backend(
        "sm75_native_kvcache_fwd"
        f"_{path}"
        f"_hd{q.shape[-1]}"
        f"_kernel{_kernel_head_dim(q.shape[-1])}"
        f"_{'causal' if causal else 'noncausal'}"
        f"_{_head_group_suffix(q, k_cache)}"
        f"_{_dtype_suffix(q.dtype)}"
        f"_q{seqlen_q}"
    )
    return out, softmax_lse
