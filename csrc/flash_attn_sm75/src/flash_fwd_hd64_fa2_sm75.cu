#define FLASHATTN_ENABLE_SM75 1
#define FLASHATTN_SM75_VECTOR_GMEM 1
#define FLASHATTN_SM75_VECTOR_GMEM_O 1

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "../../flash_attn/src/flash_fwd_launch_template.h"

namespace sm75_hd64_fa2_detail {

using Hd64Traits = ::Flash_fwd_kernel_traits<
    64,
    128,
    128,
    8,
    false,
    false,
    cutlass::half_t>;

static_assert(Hd64Traits::kSmemSize == 48 * 1024);

template <typename KernelTraits, bool IsCausal, bool IsEvenMN>
__global__ void flash_fwd_head_major(const flash::Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    flash::compute_attn_1rowblock<
        KernelTraits,
        false,
        IsCausal,
        false,
        false,
        IsEvenMN,
        true,
        false,
        false>(params, blockIdx.z, blockIdx.x, blockIdx.y);
#endif
}

template <bool IsCausal, bool IsEvenMN>
void launch_hd64_fa2(
    flash::Flash_fwd_params &params,
    cudaStream_t stream,
    const bool head_major) {
    constexpr int kThreads = Hd64Traits::kNThreads;
    constexpr size_t kSmemSize = Hd64Traits::kSmemSize;
    const int m_blocks = cute::ceil_div(params.seqlen_q, Hd64Traits::kBlockM);
    if (head_major) {
        dim3 grid(params.h, m_blocks, params.b);
        auto kernel = &flash_fwd_head_major<Hd64Traits, IsCausal, IsEvenMN>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        kernel<<<grid, kThreads, kSmemSize, stream>>>(params);
    } else {
        dim3 grid(m_blocks, params.b, params.h);
        auto kernel = &flash::flash_fwd_kernel<
            Hd64Traits,
            false,
            IsCausal,
            false,
            false,
            IsEvenMN,
            true,
            false,
            false>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        kernel<<<grid, kThreads, kSmemSize, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sm75_hd64_fa2_detail

std::vector<at::Tensor> mha_fwd_hd64_fa2_sm75(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    const float softmax_scale,
    const bool causal,
    const bool head_major) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
                "SM75 hd64 FA2 requires FP16 inputs");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, and v must have shape [batch, seqlen, heads, head_dim]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch sizes must match");
    TORCH_CHECK(k.size(1) == v.size(1) && k.size(2) == v.size(2), "K and V shapes must match");
    TORCH_CHECK(q.size(3) == 64 && k.size(3) == 64 && v.size(3) == 64,
                "SM75 hd64 FA2 requires matching head_dim 64");
    TORCH_CHECK(q.size(2) == k.size(2), "SM75 hd64 FA2 MHA route requires matching Q and KV heads");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "the head-dimension stride must be 1");

    auto out = torch::empty(q.sizes(), q.options());
    auto lse = torch::empty(
        {q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));

    flash::Flash_fwd_params params{};
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_row_stride = q.stride(1);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(1);
    params.q_head_stride = q.stride(2);
    params.k_head_stride = k.stride(2);
    params.v_head_stride = v.stride(2);
    params.o_head_stride = out.stride(2);

    params.b = q.size(0);
    params.h = q.size(2);
    params.h_k = k.size(2);
    params.h_h_k_ratio = 1;
    params.seqlen_q = q.size(1);
    params.seqlen_k = k.size(1);
    params.seqlen_q_rounded = cute::ceil_div(params.seqlen_q, 128) * 128;
    params.seqlen_k_rounded = cute::ceil_div(params.seqlen_k, 128) * 128;
    params.d = 64;
    params.d_rounded = 64;
    params.total_q = params.b * params.seqlen_q;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    params.window_size_left = -1;
    params.window_size_right = causal ? 0 : -1;
    params.is_causal = causal;
    params.is_seqlens_k_cumulative = true;
    params.philox_args = at::PhiloxCudaState(0, 0);

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool even_mn = params.seqlen_q % sm75_hd64_fa2_detail::Hd64Traits::kBlockM == 0
        && params.seqlen_k % sm75_hd64_fa2_detail::Hd64Traits::kBlockN == 0;
    if (causal && even_mn) {
        sm75_hd64_fa2_detail::launch_hd64_fa2<true, true>(params, stream, head_major);
    } else if (causal) {
        sm75_hd64_fa2_detail::launch_hd64_fa2<true, false>(params, stream, head_major);
    } else if (even_mn) {
        sm75_hd64_fa2_detail::launch_hd64_fa2<false, true>(params, stream, head_major);
    } else {
        sm75_hd64_fa2_detail::launch_hd64_fa2<false, false>(params, stream, head_major);
    }
    return {out, lse};
}

std::vector<at::Tensor> mha_fwd_qkvpacked_hd64_fa2_sm75(
    at::Tensor qkv,
    const float softmax_scale,
    const bool causal) {
    TORCH_CHECK(qkv.is_cuda(), "qkv must be a CUDA tensor");
    TORCH_CHECK(qkv.scalar_type() == at::kHalf, "SM75 packed hd64 FA2 requires FP16 input");
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(2) == 3 && qkv.size(4) == 64,
                "qkv must have shape [batch, seqlen, 3, heads, 64]");
    TORCH_CHECK(qkv.stride(4) == 1, "the packed head-dimension stride must be 1");

    // Selects are metadata-only views. Keeping them inside this pybind call avoids
    // Python launch starvation on small packed-attention workloads.
    return mha_fwd_hd64_fa2_sm75(
        qkv.select(2, 0),
        qkv.select(2, 1),
        qkv.select(2, 2),
        softmax_scale,
        causal,
        true);
}
