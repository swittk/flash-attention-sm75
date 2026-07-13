#define FLASHATTN_ENABLE_SM75 1

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "../../flash_attn/src/flash_fwd_launch_template.h"

namespace sm75_paged_fa2_detail {

using Hd64Traits = ::Flash_fwd_kernel_traits<
    64,
    64,
    128,
    4,
    false,
    false,
    cutlass::half_t>;

using Hd128Traits = ::Flash_fwd_kernel_traits<
    128,
    64,
    64,
    4,
    false,
    false,
    cutlass::half_t>;

template <typename KernelTraits, bool IsCausal, bool Split>
void launch_partial(flash::Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t kSmemSize = KernelTraits::kSmemSize;
    const int m_blocks = cute::ceil_div(params.seqlen_q, KernelTraits::kBlockM);
    dim3 grid(
        m_blocks,
        Split ? params.num_splits : params.b,
        Split ? params.b * params.h : params.h);
    auto kernel = &flash::flash_fwd_splitkv_kernel<
        KernelTraits,
        IsCausal,
        false,
        false,
        false,
        true,
        false,
        Split,
        false>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }
    kernel<<<grid, KernelTraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KernelTraits, int LogMaxSplits>
void launch_combine(flash::Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int kRowsPerBlock = KernelTraits::kHeadDim % 128 == 0 ? 4 : 8;
    const int blocks = cute::ceil_div(
        params.b * params.h * params.seqlen_q,
        kRowsPerBlock);
    flash::flash_fwd_splitkv_combine_kernel<
        KernelTraits,
        kRowsPerBlock,
        LogMaxSplits,
        true><<<blocks, KernelTraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename KernelTraits, bool IsCausal>
void launch(flash::Flash_fwd_params &params, cudaStream_t stream) {
    if (params.num_splits <= 1) {
        launch_partial<KernelTraits, IsCausal, false>(params, stream);
        return;
    }
    launch_partial<KernelTraits, IsCausal, true>(params, stream);
    if (params.num_splits <= 2) {
        launch_combine<KernelTraits, 1>(params, stream);
    } else if (params.num_splits <= 4) {
        launch_combine<KernelTraits, 2>(params, stream);
    } else if (params.num_splits <= 8) {
        launch_combine<KernelTraits, 3>(params, stream);
    } else if (params.num_splits <= 16) {
        launch_combine<KernelTraits, 4>(params, stream);
    } else if (params.num_splits <= 32) {
        launch_combine<KernelTraits, 5>(params, stream);
    } else if (params.num_splits <= 64) {
        launch_combine<KernelTraits, 6>(params, stream);
    } else {
        launch_combine<KernelTraits, 7>(params, stream);
    }
}

template <typename KernelTraits>
void dispatch_causal(flash::Flash_fwd_params &params, cudaStream_t stream) {
    if (params.is_causal) {
        launch<KernelTraits, true>(params, stream);
    } else {
        launch<KernelTraits, false>(params, stream);
    }
}

}  // namespace sm75_paged_fa2_detail

std::vector<at::Tensor> mha_paged_kvcache_fwd_fa2_sm75(
    at::Tensor q,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor block_table,
    at::Tensor cache_seqlens,
    at::Tensor cache_batch_idx,
    at::Tensor cache_leftpad,
    const float softmax_scale,
    bool causal,
    const int window_left,
    const int window_right,
    const int max_seqlen_k,
    int num_splits) {
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "q and paged K/V cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda() && cache_seqlens.is_cuda(),
                "block_table and cache_seqlens must be CUDA tensors");
    TORCH_CHECK(q.device() == k_cache.device() && q.device() == v_cache.device()
                && q.device() == block_table.device() && q.device() == cache_seqlens.device(),
                "q, cache tensors, block_table, and cache_seqlens must be on the same CUDA device");
    TORCH_CHECK(!cache_batch_idx.defined() || cache_batch_idx.numel() == 0 || cache_batch_idx.device() == q.device(),
                "cache_batch_idx must be on the same CUDA device as q");
    TORCH_CHECK(!cache_leftpad.defined() || cache_leftpad.numel() == 0 || cache_leftpad.device() == q.device(),
                "cache_leftpad must be on the same CUDA device as q");
    at::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type() == at::kHalf
                && k_cache.scalar_type() == at::kHalf
                && v_cache.scalar_type() == at::kHalf,
                "SM75 paged FA2 requires FP16 tensors");
    TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4 && v_cache.dim() == 4,
                "q must be [B,S,H,D] and cache must be [blocks,page,H,D]");
    TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128,
                "SM75 paged FA2 supports head_dim 64 or 128");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "K/V cache shapes must match");
    TORCH_CHECK(q.size(3) == k_cache.size(3), "q/cache head_dim must match");
    TORCH_CHECK(q.size(2) % k_cache.size(2) == 0,
                "query heads must be divisible by KV heads");
    TORCH_CHECK(q.stride(3) == 1 && k_cache.stride(3) == 1 && v_cache.stride(3) == 1,
                "head-dimension stride must be 1");
    TORCH_CHECK(block_table.dim() == 2 && block_table.scalar_type() == at::kInt,
                "block_table must be int32 [cache_batch,max_blocks]");
    TORCH_CHECK(block_table.stride(1) == 1, "block_table last dimension must be contiguous");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.size(0) == q.size(0)
                && cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous(),
                "cache_seqlens must be contiguous int32 [B]");
    TORCH_CHECK(!cache_batch_idx.defined() || cache_batch_idx.numel() == 0
                || (cache_batch_idx.is_cuda() && cache_batch_idx.scalar_type() == at::kInt
                    && cache_batch_idx.is_contiguous() && cache_batch_idx.numel() == q.size(0)),
                "cache_batch_idx must be empty or contiguous CUDA int32 [B]");
    TORCH_CHECK(!cache_leftpad.defined() || cache_leftpad.numel() == 0
                || (cache_leftpad.is_cuda() && cache_leftpad.scalar_type() == at::kInt
                    && cache_leftpad.is_contiguous() && cache_leftpad.numel() == q.size(0)),
                "cache_leftpad must be empty or contiguous CUDA int32 [B]");
    TORCH_CHECK(cache_leftpad.numel() == 0,
                "tiled paged FA2 requires page-aligned cache data without left padding");
    TORCH_CHECK(k_cache.size(1) % 256 == 0, "page size must be divisible by 256");
    TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k must be positive");

    const int64_t batch = q.size(0);
    const int64_t original_q_len = q.size(1);
    const int64_t original_heads = q.size(2);
    const int64_t heads_k = k_cache.size(2);
    const int64_t head_dim = q.size(3);
    const bool swap_groups = original_q_len == 1
        && original_heads > heads_k
        && window_left < 0
        && window_right < 0;
    if (original_q_len == 1) {
        causal = false;
    }

    at::Tensor q_kernel = q;
    if (swap_groups) {
        const int64_t groups = original_heads / heads_k;
        q_kernel = q.reshape({batch, heads_k, groups, head_dim}).transpose(1, 2);
    }
    auto out_kernel = torch::empty_like(q_kernel);
    auto lse_kernel = torch::empty(
        {batch, q_kernel.size(2), q_kernel.size(1)},
        q.options().dtype(at::kFloat));

    flash::Flash_fwd_params params{};
    params.q_ptr = q_kernel.data_ptr();
    params.k_ptr = k_cache.data_ptr();
    params.v_ptr = v_cache.data_ptr();
    params.o_ptr = out_kernel.data_ptr();
    params.softmax_lse_ptr = lse_kernel.data_ptr();
    params.q_batch_stride = q_kernel.stride(0);
    params.k_batch_stride = k_cache.stride(0);
    params.v_batch_stride = v_cache.stride(0);
    params.o_batch_stride = out_kernel.stride(0);
    params.q_row_stride = q_kernel.stride(1);
    params.k_row_stride = k_cache.stride(1);
    params.v_row_stride = v_cache.stride(1);
    params.o_row_stride = out_kernel.stride(1);
    params.q_head_stride = q_kernel.stride(2);
    params.k_head_stride = k_cache.stride(2);
    params.v_head_stride = v_cache.stride(2);
    params.o_head_stride = out_kernel.stride(2);
    params.cu_seqlens_k = reinterpret_cast<int *>(cache_seqlens.data_ptr());
    params.leftpad_k = cache_leftpad.numel() > 0
        ? reinterpret_cast<int *>(cache_leftpad.data_ptr())
        : nullptr;
    params.cache_batch_idx = cache_batch_idx.numel() > 0
        ? reinterpret_cast<int *>(cache_batch_idx.data_ptr())
        : nullptr;
    params.block_table = reinterpret_cast<int *>(block_table.data_ptr());
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size = k_cache.size(1);

    params.b = batch;
    params.h = q_kernel.size(2);
    params.h_k = heads_k;
    params.h_h_k_ratio = params.h / params.h_k;
    params.seqlen_q = q_kernel.size(1);
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = cute::ceil_div(params.seqlen_q, 128) * 128;
    params.seqlen_k_rounded = cute::ceil_div(params.seqlen_k, 128) * 128;
    params.d = head_dim;
    params.d_rounded = head_dim;
    params.total_q = params.b * params.seqlen_q;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    params.window_size_left = window_left;
    params.window_size_right = causal ? 0 : window_right;
    params.is_causal = causal;
    params.is_seqlens_k_cumulative = false;
    params.philox_args = at::PhiloxCudaState(0, 0);
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    const int block_n = head_dim == 64 ? 128 : 64;
    const int n_blocks = cute::ceil_div(max_seqlen_k, block_n);
    if (num_splits <= 0) {
        const int base_blocks = params.b * params.h
            * cute::ceil_div(params.seqlen_q, 64);
        num_splits = std::min(
            128,
            std::min(n_blocks, std::max(1, cute::ceil_div(64, base_blocks))));
    }
    TORCH_CHECK(num_splits >= 1 && num_splits <= 128, "num_splits must be in [1, 128]");
    params.num_splits = num_splits;

    at::Tensor lse_accum;
    at::Tensor out_accum;
    if (num_splits > 1) {
        lse_accum = torch::empty(
            {num_splits, batch, params.h, params.seqlen_q},
            q.options().dtype(at::kFloat));
        out_accum = torch::empty(
            {num_splits, batch, params.h, params.seqlen_q, head_dim},
            q.options().dtype(at::kFloat));
        params.softmax_lseaccum_ptr = lse_accum.data_ptr();
        params.oaccum_ptr = out_accum.data_ptr();
    }

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (head_dim == 64) {
        sm75_paged_fa2_detail::dispatch_causal<sm75_paged_fa2_detail::Hd64Traits>(
            params, stream);
    } else {
        sm75_paged_fa2_detail::dispatch_causal<sm75_paged_fa2_detail::Hd128Traits>(
            params, stream);
    }

    if (swap_groups) {
        auto out = out_kernel.transpose(1, 2).reshape(
            {batch, original_q_len, original_heads, head_dim});
        auto lse = lse_kernel.reshape({batch, original_heads, original_q_len});
        return {out, lse};
    }
    return {out_kernel, lse_kernel};
}
