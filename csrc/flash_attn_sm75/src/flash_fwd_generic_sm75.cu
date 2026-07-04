#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <vector>

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ float load_half(const half *ptr) {
    return __half2float(*ptr);
}

__device__ __forceinline__ half store_half(float value) {
    return __float2half_rn(value);
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

__global__ void sm75_generic_fwd_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    half *__restrict__ out,
    float *__restrict__ lse,
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
    float softcap,
    half *__restrict__ p_out) {
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
        if (!key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
            continue;
        }
        float score = 0.0f;
        const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
        const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        for (int d = 0; d < headdim; ++d) {
            score += load_half(q_ptr + d) * load_half(k_ptr + d);
        }
        local_max = fmaxf(local_max, transform_score(score, softmax_scale, softcap, alibi_slopes,
                                                     alibi_batch_stride, b, head_q, row, col,
                                                     seqlen_q, seqlen_k));
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

    float local_sum = 0.0f;
    if (row_max != -FLT_MAX) {
        for (int col = threadIdx.x; col < seqlen_k; col += blockDim.x) {
            if (!key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
                continue;
            }
            float score = 0.0f;
            const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
            const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
            for (int d = 0; d < headdim; ++d) {
                score += load_half(q_ptr + d) * load_half(k_ptr + d);
            }
            float p = expf(transform_score(score, softmax_scale, softcap, alibi_slopes,
                                           alibi_batch_stride, b, head_q, row, col,
                                           seqlen_q, seqlen_k) - row_max);
            local_sum += p;
        }
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = reduce[0];
    if (threadIdx.x == 0) {
        lse[(b * nheads_q + head_q) * seqlen_q + row] =
            denom > 0.0f ? row_max + logf(denom) : INFINITY;
    }

    for (int d = threadIdx.x; d < headdim; d += blockDim.x) {
        float acc = 0.0f;
        if (denom > 0.0f && row_max != -FLT_MAX) {
            for (int col = 0; col < seqlen_k; ++col) {
                if (!key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
                    continue;
                }
                float score = 0.0f;
                const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
                const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
                for (int dot_d = 0; dot_d < headdim; ++dot_d) {
                    score += load_half(q_ptr + dot_d) * load_half(k_ptr + dot_d);
                }
                float p = expf(transform_score(score, softmax_scale, softcap, alibi_slopes,
                                               alibi_batch_stride, b, head_q, row, col,
                                               seqlen_q, seqlen_k) - row_max);
                if (!keep_dropout(b, head_q, row, col, keep_prob)) {
                    p = 0.0f;
                }
                p *= (keep_prob < 1.0f ? (1.0f / keep_prob) : 1.0f) / denom;
                if (p_out != nullptr && d == 0) {
                    p_out[((b * nheads_q + head_q) * seqlen_q + row) * seqlen_k + col] = store_half(p);
                }
                const half *v_ptr = v + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
                acc += p * load_half(v_ptr + d);
            }
        }
        out[(((b * seqlen_q + row) * nheads_q + head_q) * headdim) + d] = store_half(acc);
    }
}

}  // namespace

std::vector<at::Tensor> mha_fwd_generic_sm75(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    const float softmax_scale,
    bool causal,
    int window_left,
    int window_right,
    float dropout_p,
    at::Tensor alibi_slopes,
    float softcap,
    bool return_softmax) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16,
                "generic SM75 forward expects FP16 tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, v must be contiguous");
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

    auto out = torch::empty_like(q);
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(torch::kFloat32));
    auto p = return_softmax
        ? torch::zeros({q.size(0), q.size(2), q.size(1), k.size(1)}, q.options().dtype(torch::kFloat16))
        : torch::empty({0}, q.options().dtype(torch::kFloat16));

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
    sm75_generic_fwd_kernel<<<blocks, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half *>(q.data_ptr()),
        reinterpret_cast<const half *>(k.data_ptr()),
        reinterpret_cast<const half *>(v.data_ptr()),
        reinterpret_cast<half *>(out.data_ptr()),
        reinterpret_cast<float *>(lse.data_ptr()),
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
        softcap,
        return_softmax ? reinterpret_cast<half *>(p.data_ptr()) : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse, p};
}
