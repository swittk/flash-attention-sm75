#define FLASHATTN_ENABLE_SM75 1
#define FLASHATTN_SM75_VECTOR_GMEM 1
#define FLASHATTN_SM75_VECTOR_GMEM_O 1

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "../../flash_attn/src/flash_fwd_launch_template.h"

namespace sm75_hd128_fa2_detail {

using Hd128Traits = ::Flash_fwd_kernel_traits<
    128,
    64,
    64,
    4,
    false,
    false,
    cutlass::half_t>;

template <bool IsCausal, bool IsEvenMN>
void launch(flash::Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t kSmemSize = Hd128Traits::kSmemSize;
    const int m_blocks = cute::ceil_div(params.seqlen_q, Hd128Traits::kBlockM);
    dim3 grid(m_blocks, params.b, params.h);
    auto kernel = &flash::flash_fwd_kernel<
        Hd128Traits,
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
    kernel<<<grid, Hd128Traits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sm75_hd128_fa2_detail

std::vector<at::Tensor> mha_fwd_hd128_fa2_sm75(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    const float softmax_scale,
    const bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must be on the same CUDA device");
    at::cuda::CUDAGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
                "SM75 hd128 FA2 requires FP16 inputs");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, and v must have shape [batch, seqlen, heads, head_dim]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch sizes must match");
    TORCH_CHECK(k.size(1) == v.size(1) && k.size(2) == v.size(2), "K and V shapes must match");
    TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128 && v.size(3) == 128,
                "SM75 hd128 FA2 requires matching head_dim 128");
    TORCH_CHECK(q.size(2) % k.size(2) == 0, "query heads must be divisible by KV heads");
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
    params.h_h_k_ratio = params.h / params.h_k;
    params.seqlen_q = q.size(1);
    params.seqlen_k = k.size(1);
    params.seqlen_q_rounded = cute::ceil_div(params.seqlen_q, 128) * 128;
    params.seqlen_k_rounded = cute::ceil_div(params.seqlen_k, 128) * 128;
    params.d = 128;
    params.d_rounded = 128;
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
    const bool even_mn = params.seqlen_q % sm75_hd128_fa2_detail::Hd128Traits::kBlockM == 0
        && params.seqlen_k % sm75_hd128_fa2_detail::Hd128Traits::kBlockN == 0;
    if (causal && even_mn) {
        sm75_hd128_fa2_detail::launch<true, true>(params, stream);
    } else if (causal) {
        sm75_hd128_fa2_detail::launch<true, false>(params, stream);
    } else if (even_mn) {
        sm75_hd128_fa2_detail::launch<false, true>(params, stream);
    } else {
        sm75_hd128_fa2_detail::launch<false, false>(params, stream);
    }
    return {out, lse};
}
