#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <vector>

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ float hload(const half *ptr) {
    return __half2float(*ptr);
}

__device__ __forceinline__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

__device__ __forceinline__ bool keep_dropout(int b, int h, int i, int j, float keep_prob) {
    if (keep_prob >= 1.0f) {
        return true;
    }
    unsigned int seed = static_cast<unsigned int>(b * 1315423911u)
                      ^ static_cast<unsigned int>(h * 2654435761u)
                      ^ static_cast<unsigned int>(i * 374761393u)
                      ^ static_cast<unsigned int>(j * 668265263u);
    const float u = static_cast<float>(hash_u32(seed) & 0x00ffffffU) * (1.0f / 16777216.0f);
    return u < keep_prob;
}

__device__ __forceinline__ bool key_is_visible(
    int row,
    int col,
    int seqlen_q,
    int seqlen_k,
    bool causal,
    int window_left,
    int window_right) {
    const int aligned_row = row + seqlen_k - seqlen_q;
    if (window_left >= 0 || window_right >= 0) {
        const int left = window_left < 0 ? seqlen_k : window_left;
        const int right = window_right < 0 ? seqlen_k : window_right;
        return col >= aligned_row - left && col <= aligned_row + right;
    }
    if (causal) {
        return col <= aligned_row;
    }
    return true;
}

__device__ float score_qk(
    const half *q,
    const half *k,
    int b,
    int row,
    int col,
    int head_q,
    int head_k,
    int seqlen_q,
    int seqlen_k,
    int nheads_q,
    int nheads_k,
    int headdim) {
    const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
    const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
    float score = 0.0f;
    for (int d = 0; d < headdim; ++d) {
        score += hload(q_ptr + d) * hload(k_ptr + d);
    }
    return score;
}

__device__ __forceinline__ float alibi_bias(
    const float *alibi_slopes,
    int alibi_batch_stride,
    int b,
    int head_q,
    int row,
    int col,
    int seqlen_q,
    int seqlen_k) {
    if (alibi_slopes == nullptr) {
        return 0.0f;
    }
    const float slope = alibi_slopes[(alibi_batch_stride == 0 ? 0 : b * alibi_batch_stride) + head_q];
    const int aligned_row = row + seqlen_k - seqlen_q;
    return -slope * static_cast<float>(abs(aligned_row - col));
}

__device__ __forceinline__ float transform_score(
    float dot,
    float softmax_scale,
    float softcap,
    const float *alibi_slopes,
    int alibi_batch_stride,
    int b,
    int head_q,
    int row,
    int col,
    int seqlen_q,
    int seqlen_k) {
    float score = dot * softmax_scale;
    if (softcap > 0.0f) {
        score = tanhf(score / softcap) * softcap;
    }
    return score + alibi_bias(alibi_slopes, alibi_batch_stride, b, head_q, row, col, seqlen_q, seqlen_k);
}

__device__ __forceinline__ float qk_grad_scale(float dot, float softmax_scale, float softcap) {
    if (softcap <= 0.0f) {
        return softmax_scale;
    }
    const float t = tanhf((dot * softmax_scale) / softcap);
    return softmax_scale * (1.0f - t * t);
}

__device__ float dot_do_v(
    const half *dout,
    const half *v,
    int b,
    int row,
    int col,
    int head_q,
    int head_k,
    int seqlen_q,
    int seqlen_k,
    int nheads_q,
    int nheads_k,
    int headdim) {
    const half *do_ptr = dout + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
    const half *v_ptr = v + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
    float acc = 0.0f;
    for (int d = 0; d < headdim; ++d) {
        acc += hload(do_ptr + d) * hload(v_ptr + d);
    }
    return acc;
}

__global__ void sm75_generic_bwd_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    const half *__restrict__ dout,
    half *__restrict__ dq,
    half *__restrict__ dk,
    half *__restrict__ dv,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int nheads_q,
    int nheads_k,
    int headdim,
    float softmax_scale,
    bool causal,
    int window_left,
    int window_right,
    float dropout_p,
    const float *__restrict__ alibi_slopes,
    int alibi_batch_stride,
    float softcap) {
    const int block = blockIdx.x;
    const int row = block % seqlen_q;
    const int head_q = (block / seqlen_q) % nheads_q;
    const int b = block / (seqlen_q * nheads_q);
    const int head_k = head_q / (nheads_q / nheads_k);
    const float keep_prob = 1.0f - dropout_p;

    extern __shared__ float smem[];
    float *reduce = smem;

    float local_max = -FLT_MAX;
    for (int col = threadIdx.x; col < seqlen_k; col += blockDim.x) {
        if (key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
            const float dot = score_qk(q, k, b, row, col, head_q, head_k,
                                       seqlen_q, seqlen_k, nheads_q, nheads_k, headdim);
            local_max = fmaxf(local_max, transform_score(dot, softmax_scale, softcap, alibi_slopes,
                                                         alibi_batch_stride, b, head_q, row, col,
                                                         seqlen_q, seqlen_k));
        }
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduce[0];

    float local_denom = 0.0f;
    if (row_max != -FLT_MAX) {
        for (int col = threadIdx.x; col < seqlen_k; col += blockDim.x) {
            if (key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
                const float dot = score_qk(q, k, b, row, col, head_q, head_k,
                                           seqlen_q, seqlen_k, nheads_q, nheads_k, headdim);
                const float s = transform_score(dot, softmax_scale, softcap, alibi_slopes,
                                                alibi_batch_stride, b, head_q, row, col,
                                                seqlen_q, seqlen_k);
                local_denom += expf(s - row_max);
            }
        }
    }
    reduce[threadIdx.x] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = reduce[0];

    float local_rowsum = 0.0f;
    if (denom > 0.0f) {
        for (int col = threadIdx.x; col < seqlen_k; col += blockDim.x) {
            if (!key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
                continue;
            }
            const float dot = score_qk(q, k, b, row, col, head_q, head_k,
                                       seqlen_q, seqlen_k, nheads_q, nheads_k, headdim);
            const float s = transform_score(dot, softmax_scale, softcap, alibi_slopes,
                                            alibi_batch_stride, b, head_q, row, col,
                                            seqlen_q, seqlen_k);
            const float p = expf(s - row_max) / denom;
            const bool keep = keep_dropout(b, head_q, row, col, keep_prob);
            const float dprob = keep ? dot_do_v(dout, v, b, row, col, head_q, head_k,
                                                seqlen_q, seqlen_k, nheads_q, nheads_k, headdim)
                                           / keep_prob
                                     : 0.0f;
            local_rowsum += p * dprob;
        }
    }
    reduce[threadIdx.x] = local_rowsum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float rowsum = reduce[0];

    for (int d = threadIdx.x; d < headdim; d += blockDim.x) {
        float dq_acc = 0.0f;
        if (denom > 0.0f) {
            for (int col = 0; col < seqlen_k; ++col) {
                if (!key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
                    continue;
                }
                const float dot = score_qk(q, k, b, row, col, head_q, head_k,
                                           seqlen_q, seqlen_k, nheads_q, nheads_k, headdim);
                const float s = transform_score(dot, softmax_scale, softcap, alibi_slopes,
                                                alibi_batch_stride, b, head_q, row, col,
                                                seqlen_q, seqlen_k);
                const float p = expf(s - row_max) / denom;
                const bool keep = keep_dropout(b, head_q, row, col, keep_prob);
                const float dprob = keep ? dot_do_v(dout, v, b, row, col, head_q, head_k,
                                                    seqlen_q, seqlen_k, nheads_q, nheads_k, headdim)
                                               / keep_prob
                                         : 0.0f;
                const float ds = p * (dprob - rowsum) * qk_grad_scale(dot, softmax_scale, softcap);
                const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
                dq_acc += ds * hload(k_ptr + d);
            }
        }
        dq[(((b * seqlen_q + row) * nheads_q + head_q) * headdim) + d] = __float2half_rn(dq_acc);
    }

    for (int idx = threadIdx.x; idx < seqlen_k * headdim; idx += blockDim.x) {
        const int col = idx / headdim;
        const int d = idx % headdim;
        if (!key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right) || denom <= 0.0f) {
            continue;
        }
        const float dot = score_qk(q, k, b, row, col, head_q, head_k,
                                   seqlen_q, seqlen_k, nheads_q, nheads_k, headdim);
        const float s = transform_score(dot, softmax_scale, softcap, alibi_slopes,
                                        alibi_batch_stride, b, head_q, row, col,
                                        seqlen_q, seqlen_k);
        const float p = expf(s - row_max) / denom;
        const bool keep = keep_dropout(b, head_q, row, col, keep_prob);
        const float p_drop = keep ? p / keep_prob : 0.0f;
        const float dprob = keep ? dot_do_v(dout, v, b, row, col, head_q, head_k,
                                            seqlen_q, seqlen_k, nheads_q, nheads_k, headdim)
                                       / keep_prob
                                 : 0.0f;
        const float ds = p * (dprob - rowsum) * qk_grad_scale(dot, softmax_scale, softcap);
        const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
        const half *do_ptr = dout + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
        half *dk_ptr = dk + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        half *dv_ptr = dv + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        atomicAdd(dk_ptr + d, __float2half_rn(ds * hload(q_ptr + d)));
        atomicAdd(dv_ptr + d, __float2half_rn(p_drop * hload(do_ptr + d)));
    }
}

}  // namespace

std::vector<at::Tensor> mha_bwd_generic_sm75(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor dout,
    const float softmax_scale,
    bool causal,
    int window_left,
    int window_right,
    float dropout_p,
    at::Tensor alibi_slopes,
    float softcap) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && dout.is_cuda(), "q, k, v, dout must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16
                && v.scalar_type() == torch::kFloat16 && dout.scalar_type() == torch::kFloat16,
                "generic SM75 backward expects FP16 tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && dout.dim() == 4,
                "q, k, v, dout must be rank-4 tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && dout.is_contiguous(),
                "q, k, v, dout must be contiguous");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must match q shape");
    TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0), "batch size mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v sequence length mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "k/v head count mismatch");
    TORCH_CHECK(q.size(3) == k.size(3) && k.size(3) == v.size(3), "head_dim mismatch");
    TORCH_CHECK(q.size(2) % k.size(2) == 0, "query heads must be divisible by key heads");
    TORCH_CHECK(dropout_p >= 0.0f && dropout_p < 1.0f, "dropout_p must be in [0, 1)");
    TORCH_CHECK(!alibi_slopes.defined() || alibi_slopes.numel() == 0
                || (alibi_slopes.is_cuda() && alibi_slopes.scalar_type() == torch::kFloat32 && alibi_slopes.is_contiguous()),
                "ALiBi slopes must be contiguous CUDA fp32 or empty");
    TORCH_CHECK(softcap >= 0.0f, "softcap must be non-negative");

    auto dq = torch::empty_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    const int batch = q.size(0);
    const int seqlen_q = q.size(1);
    const int seqlen_k = k.size(1);
    const int nheads_q = q.size(2);
    const int nheads_k = k.size(2);
    const int headdim = q.size(3);
    const int blocks = batch * nheads_q * seqlen_q;
    const size_t smem = kThreads * sizeof(float);
    const bool has_alibi = alibi_slopes.defined() && alibi_slopes.numel() > 0;
    const int alibi_batch_stride = has_alibi && alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    sm75_generic_bwd_kernel<<<blocks, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half *>(q.data_ptr()),
        reinterpret_cast<const half *>(k.data_ptr()),
        reinterpret_cast<const half *>(v.data_ptr()),
        reinterpret_cast<const half *>(dout.data_ptr()),
        reinterpret_cast<half *>(dq.data_ptr()),
        reinterpret_cast<half *>(dk.data_ptr()),
        reinterpret_cast<half *>(dv.data_ptr()),
        batch,
        seqlen_q,
        seqlen_k,
        nheads_q,
        nheads_k,
        headdim,
        softmax_scale,
        causal,
        window_left,
        window_right,
        dropout_p,
        has_alibi ? reinterpret_cast<const float *>(alibi_slopes.data_ptr()) : nullptr,
        alibi_batch_stride,
        softcap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv};
}
