
#pragma once
#include "flash.h"
#include "flash_fwd_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>
#include "static_switch.h"
using half_t = cutlass::half_t;

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ __launch_bounds__(256)
// for some reason changing this into params struc is 10% slower for hdim = 128
void flash_fwd_kernel(half_t* __restrict__ q,
                          half_t* __restrict__ k,
                          half_t* __restrict__ v,
                          half_t* __restrict__ o,
                          float* __restrict__ l,
                          int* __restrict__ cu_seqlens_q,
                          int* __restrict__ cu_seqlens_k,
                          int batch_size,
                          int seqlen_q,
                          int seqlen_k,
                          int num_heads,
                          int num_heads_k,
                          int h_h_k_ratio,
                          int head_dim,
                          int q_batch_stride,
                          int k_batch_stride,
                          int v_batch_stride,
                          int o_batch_stride,
                          int q_row_stride,
                          int k_row_stride,
                          int v_row_stride,
                          int o_row_stride,
                          int q_head_stride,
                          int k_head_stride,
                          int v_head_stride,
                          int o_head_stride,
                          float softmax_scale,
                          int is_casual)
{
    compute_attn<Kernel_traits, Is_causal, Is_even_MN>(q,
                                           k,
                                           v,
                                           o,
                                           l,
                                           cu_seqlens_q,
                                           cu_seqlens_k,
                                           batch_size,
                                           seqlen_q,
                                           seqlen_k,
                                           num_heads,
                                           num_heads_k,
                                           h_h_k_ratio,
                                           head_dim,
                                           q_batch_stride,
                                           k_batch_stride,
                                           v_batch_stride,
                                           o_batch_stride,
                                           q_row_stride,
                                           k_row_stride,
                                           v_row_stride,
                                           o_row_stride,
                                           q_head_stride,
                                           k_head_stride,
                                           v_head_stride,
                                           o_head_stride,
                                           softmax_scale,
                                           is_casual);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params) {


    //auto kernel = flash_fwd_kernel<Kernel_traits, Is_causal>;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;

    const bool is_even_MN  = params.cu_seqlens_q == nullptr &&
                             params.cu_seqlens_k == nullptr &&
                             params.seqlen_q % kBlockM == 0 &&
                             params.seqlen_k % kBlockN == 0;

    dim3 dimGrid(num_m_block, params.b, params.h);
    dim3 dimBlock(kNWarps * 32);
    int maxbytes = 65536;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();


    BOOL_SWITCH(is_even_MN, Is_even_MN, [&] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            maxbytes));
        flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN><<<dimGrid, dimBlock, maxbytes, stream>>>(params.q_ptr,
                                                                                    params.k_ptr,
                                                                                    params.v_ptr,
                                                                                    params.o_ptr,
                                                                                    params.l_ptr,
                                                                                    params.cu_seqlens_q,
                                                                                    params.cu_seqlens_k,
                                                                                    params.b,
                                                                                    params.seqlen_q,
                                                                                    params.seqlen_k,
                                                                                    params.h,
                                                                                    params.h_k,
                                                                                    params.h_h_k_ratio,
                                                                                    params.d,
                                                                                    params.q_batch_stride,
                                                                                    params.k_batch_stride,
                                                                                    params.v_batch_stride,
                                                                                    params.o_batch_stride,
                                                                                    params.q_row_stride,
                                                                                    params.k_row_stride,
                                                                                    params.v_row_stride,
                                                                                    params.o_row_stride,
                                                                                    params.q_head_stride,
                                                                                    params.k_head_stride,
                                                                                    params.v_head_stride,
                                                                                    params.o_head_stride,
                                                                                    params.softmax_scale,
                                                                                    params.is_causal);

    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

}



template<bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}

template<bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params) {
    constexpr static int Headdim = 192;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 32;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);
}

template<bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}
