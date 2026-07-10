#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpRows = 8;

__device__ __forceinline__ float load_half(const half *ptr) {
    return __half2float(*ptr);
}

__device__ __forceinline__ float2 load_half2_as_float2(const half2 *ptr) {
    return __half22float2(*ptr);
}

__device__ __forceinline__ half store_half(float value) {
    return __float2half_rn(value);
}

__device__ __forceinline__ float block_sum_256(float value, float *smem) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane_id == 0) {
        smem[warp_id] = value;
    }
    __syncthreads();

    float sum = 0.0f;
    if (warp_id == 0) {
        sum = lane_id < (blockDim.x >> 5) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            smem[0] = sum;
        }
    }
    __syncthreads();
    sum = smem[0];
    __syncthreads();
    return sum;
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

__device__ __forceinline__ half rotary_value(
    const half *__restrict__ x,
    const half *__restrict__ rotary_cos,
    const half *__restrict__ rotary_sin,
    int position,
    int dim,
    int rotary_half_dim,
    bool rotary_interleaved) {
    const int rotary_dim = rotary_half_dim * 2;
    const float cur = load_half(x + dim);
    if (dim >= rotary_dim) {
        return store_half(cur);
    }
    if (rotary_interleaved) {
        const int pair_dim = dim >> 1;
        const int mate_dim = (dim & 1) == 0 ? dim + 1 : dim - 1;
        const float mate = load_half(x + mate_dim);
        const float c = load_half(rotary_cos + position * rotary_half_dim + pair_dim);
        const float s = load_half(rotary_sin + position * rotary_half_dim + pair_dim);
        return (dim & 1) == 0
            ? store_half(cur * c - mate * s)
            : store_half(cur * c + mate * s);
    }
    const int pair_dim = dim < rotary_half_dim ? dim : dim - rotary_half_dim;
    const int mate_dim = dim < rotary_half_dim ? dim + rotary_half_dim : dim - rotary_half_dim;
    const float mate = load_half(x + mate_dim);
    const float c = load_half(rotary_cos + position * rotary_half_dim + pair_dim);
    const float s = load_half(rotary_sin + position * rotary_half_dim + pair_dim);
    return dim < rotary_half_dim
        ? store_half(cur * c - mate * s)
        : store_half(cur * c + mate * s);
}

__global__ void sm75_paged_kvcache_update_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    half *__restrict__ q_out,
    half *__restrict__ k_cache,
    half *__restrict__ v_cache,
    const int *__restrict__ block_table,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ cache_batch_idx,
    const half *__restrict__ rotary_cos,
    const half *__restrict__ rotary_sin,
    int *__restrict__ cache_seqlens_after,
    int batch,
    int seqlen_q,
    int seqlen_new,
    int nheads_q,
    int nheads_k,
    int headdim,
    int page_size,
    int max_blocks_per_seq,
    int rotary_half_dim,
    bool rotary_interleaved,
    bool sequential_q_positions) {
    const bool has_rotary = rotary_half_dim > 0;
    const int64_t q_elems = has_rotary
        ? static_cast<int64_t>(batch) * seqlen_q * nheads_q * headdim
        : 0;
    const int64_t kv_elems = static_cast<int64_t>(batch) * seqlen_new * nheads_k * headdim;
    const int64_t total = static_cast<int64_t>(batch) + q_elems + kv_elems;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride) {
        if (linear < batch) {
            cache_seqlens_after[linear] = cache_seqlens[linear] + seqlen_new;
            continue;
        }
        int64_t idx = linear - batch;
        if (idx < q_elems) {
            const int dim = idx % headdim;
            idx /= headdim;
            const int head = idx % nheads_q;
            idx /= nheads_q;
            const int token = idx % seqlen_q;
            const int b = idx / seqlen_q;
            const int base_pos = cache_seqlens[b];
            const int rotary_pos = base_pos + (sequential_q_positions ? token : 0);
            const half *q_ptr = q + (((b * seqlen_q + token) * nheads_q + head) * headdim);
            q_out[(((b * seqlen_q + token) * nheads_q + head) * headdim) + dim] =
                rotary_value(q_ptr, rotary_cos, rotary_sin, rotary_pos, dim, rotary_half_dim, rotary_interleaved);
            continue;
        }

        idx -= q_elems;
        const int dim = idx % headdim;
        idx /= headdim;
        const int head = idx % nheads_k;
        idx /= nheads_k;
        const int token = idx % seqlen_new;
        const int b = idx / seqlen_new;
        const int cache_b = cache_batch_idx == nullptr ? b : cache_batch_idx[b];
        const int pos = cache_seqlens[b] + token;
        const int block_slot = pos / page_size;
        const int block_offset = pos - block_slot * page_size;
        const int block_id = block_table[cache_b * max_blocks_per_seq + block_slot];

        const half *k_ptr = k + (((b * seqlen_new + token) * nheads_k + head) * headdim);
        const half *v_ptr = v + (((b * seqlen_new + token) * nheads_k + head) * headdim);
        half *k_dst = k_cache + (((block_id * page_size + block_offset) * nheads_k + head) * headdim);
        half *v_dst = v_cache + (((block_id * page_size + block_offset) * nheads_k + head) * headdim);
        k_dst[dim] = has_rotary
            ? rotary_value(k_ptr, rotary_cos, rotary_sin, pos, dim, rotary_half_dim, rotary_interleaved)
            : k_ptr[dim];
        v_dst[dim] = v_ptr[dim];
    }
}

__device__ __forceinline__ int64_t div_floor_i64(int64_t &value, int64_t divisor) {
    const int64_t remainder = value % divisor;
    value /= divisor;
    return remainder;
}

__device__ __forceinline__ half load_strided_or_zero(
    const half *__restrict__ src,
    int64_t b,
    int64_t s,
    int64_t h,
    int64_t d,
    int64_t src_dim,
    int64_t stride_b,
    int64_t stride_s,
    int64_t stride_h,
    int64_t stride_d) {
    return d < src_dim
        ? src[b * stride_b + s * stride_s + h * stride_h + d * stride_d]
        : __float2half_rn(0.0f);
}

__global__ void sm75_pad_qkv_head_dim_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    half *__restrict__ q_out,
    half *__restrict__ k_out,
    half *__restrict__ v_out,
    int64_t q_total,
    int64_t k_total,
    int64_t v_total,
    int64_t q_b,
    int64_t q_s,
    int64_t q_h,
    int64_t k_b,
    int64_t k_s,
    int64_t k_h,
    int64_t src_dim,
    int64_t dst_dim,
    int64_t q_stride_b,
    int64_t q_stride_s,
    int64_t q_stride_h,
    int64_t q_stride_d,
    int64_t k_stride_b,
    int64_t k_stride_s,
    int64_t k_stride_h,
    int64_t k_stride_d,
    int64_t v_stride_b,
    int64_t v_stride_s,
    int64_t v_stride_h,
    int64_t v_stride_d) {
    const int64_t total = q_total + k_total + v_total;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride) {
        if (linear < q_total) {
            int64_t idx = linear;
            const int64_t d = div_floor_i64(idx, dst_dim);
            const int64_t h = div_floor_i64(idx, q_h);
            const int64_t s = div_floor_i64(idx, q_s);
            const int64_t b = idx;
            q_out[linear] = load_strided_or_zero(
                q, b, s, h, d, src_dim,
                q_stride_b, q_stride_s, q_stride_h, q_stride_d);
        } else if (linear < q_total + k_total) {
            int64_t idx = linear - q_total;
            const int64_t d = div_floor_i64(idx, dst_dim);
            const int64_t h = div_floor_i64(idx, k_h);
            const int64_t s = div_floor_i64(idx, k_s);
            const int64_t b = idx;
            k_out[linear - q_total] = load_strided_or_zero(
                k, b, s, h, d, src_dim,
                k_stride_b, k_stride_s, k_stride_h, k_stride_d);
        } else {
            int64_t idx = linear - q_total - k_total;
            const int64_t d = div_floor_i64(idx, dst_dim);
            const int64_t h = div_floor_i64(idx, k_h);
            const int64_t s = div_floor_i64(idx, k_s);
            const int64_t b = idx;
            v_out[linear - q_total - k_total] = load_strided_or_zero(
                v, b, s, h, d, src_dim,
                v_stride_b, v_stride_s, v_stride_h, v_stride_d);
        }
    }
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
    const int d = threadIdx.x;
    const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float acc = 0.0f;
    for (int col = 0; col < seqlen_k; ++col) {
        const bool visible = key_is_visible(row, col, seqlen_q, seqlen_k, causal,
                                            window_left, window_right);
        const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        float local_dot = 0.0f;
        if (visible && d < headdim) {
            local_dot = load_half(q_ptr + d) * load_half(k_ptr + d);
        }
        const float dot = block_sum_256(local_dot, reduce);

        if (!visible) {
            continue;
        }
        const float score = transform_score(dot, softmax_scale, softcap, alibi_slopes,
                                            alibi_batch_stride, b, head_q, row, col,
                                            seqlen_q, seqlen_k);
        const float next_max = fmaxf(row_max, score);
        const float old_scale = row_max == -FLT_MAX ? 0.0f : expf(row_max - next_max);
        const float p = expf(score - next_max);
        const bool keep = keep_dropout(b, head_q, row, col, keep_prob);
        const float p_out_scale = keep ? (keep_prob < 1.0f ? (1.0f / keep_prob) : 1.0f) : 0.0f;
        const half *v_ptr = v + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        if (d < headdim) {
            acc = acc * old_scale + p * p_out_scale * load_half(v_ptr + d);
        }
        denom = denom * old_scale + p;
        row_max = next_max;
    }

    if (p_out != nullptr) {
        for (int col = 0; col < seqlen_k; ++col) {
            const bool visible = key_is_visible(row, col, seqlen_q, seqlen_k, causal,
                                                window_left, window_right);
            const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
            float local_dot = 0.0f;
            if (visible && d < headdim) {
                local_dot = load_half(q_ptr + d) * load_half(k_ptr + d);
            }
            const float dot = block_sum_256(local_dot, reduce);
            if (threadIdx.x == 0) {
                float p_store = 0.0f;
                if (visible && denom > 0.0f && row_max != -FLT_MAX) {
                    p_store = expf(transform_score(dot, softmax_scale, softcap, alibi_slopes,
                                                   alibi_batch_stride, b, head_q, row, col,
                                                   seqlen_q, seqlen_k) - row_max);
                    if (!keep_dropout(b, head_q, row, col, keep_prob)) {
                        p_store = 0.0f;
                    }
                    p_store *= (keep_prob < 1.0f ? (1.0f / keep_prob) : 1.0f) / denom;
                }
                p_out[((b * nheads_q + head_q) * seqlen_q + row) * seqlen_k + col] = store_half(p_store);
            }
        }
    }

    if (threadIdx.x == 0) {
        lse[(b * nheads_q + head_q) * seqlen_q + row] =
            denom > 0.0f ? row_max + logf(denom) : INFINITY;
    }

    if (d < headdim) {
        out[(((b * seqlen_q + row) * nheads_q + head_q) * headdim) + d] =
            denom > 0.0f ? store_half(acc / denom) : store_half(0.0f);
    }
}

__global__ void sm75_splitkv_partial_fwd_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    float *__restrict__ partial_o,
    float *__restrict__ partial_m,
    float *__restrict__ partial_l,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int nheads_q,
    int nheads_k,
    int headdim,
    int split_size,
    int num_splits,
    float softmax_scale,
    bool causal) {
    const int row_block = blockIdx.x;
    const int split = blockIdx.y;
    const int row = row_block % seqlen_q;
    const int head_q = (row_block / seqlen_q) % nheads_q;
    const int b = row_block / (seqlen_q * nheads_q);
    const int head_k = head_q / (nheads_q / nheads_k);
    const int d = threadIdx.x;
    const int col_begin = split * split_size;
    const int col_end = min(seqlen_k, col_begin + split_size);

    extern __shared__ float smem[];
    float *reduce = smem;
    const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float acc = 0.0f;
    for (int col = col_begin; col < col_end; ++col) {
        const bool visible = key_is_visible(row, col, seqlen_q, seqlen_k, causal, -1, -1);
        const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        float local_dot = 0.0f;
        if (visible && d < headdim) {
            local_dot = load_half(q_ptr + d) * load_half(k_ptr + d);
        }
        const float dot = block_sum_256(local_dot, reduce);
        if (!visible) {
            continue;
        }
        const float score = dot * softmax_scale;
        const float next_max = fmaxf(row_max, score);
        const float old_scale = row_max == -FLT_MAX ? 0.0f : expf(row_max - next_max);
        const float p = expf(score - next_max);
        const half *v_ptr = v + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        if (d < headdim) {
            acc = acc * old_scale + p * load_half(v_ptr + d);
        }
        denom = denom * old_scale + p;
        row_max = next_max;
    }

    const int partial_base = (row_block * num_splits + split) * headdim;
    if (d < headdim) {
        partial_o[partial_base + d] = acc;
    }
    if (threadIdx.x == 0) {
        partial_m[row_block * num_splits + split] = row_max;
        partial_l[row_block * num_splits + split] = denom;
    }
}

__global__ void sm75_splitkv_combine_fwd_kernel(
    const float *__restrict__ partial_o,
    const float *__restrict__ partial_m,
    const float *__restrict__ partial_l,
    half *__restrict__ out,
    float *__restrict__ lse,
    int seqlen_q,
    int nheads_q,
    int headdim,
    int num_splits) {
    const int row_block = blockIdx.x;
    const int row = row_block % seqlen_q;
    const int head_q = (row_block / seqlen_q) % nheads_q;
    const int b = row_block / (seqlen_q * nheads_q);
    const int d = threadIdx.x;

    extern __shared__ float smem[];
    if (threadIdx.x == 0) {
        float row_max = -FLT_MAX;
        for (int split = 0; split < num_splits; ++split) {
            row_max = fmaxf(row_max, partial_m[row_block * num_splits + split]);
        }
        float denom = 0.0f;
        if (row_max != -FLT_MAX) {
            for (int split = 0; split < num_splits; ++split) {
                const float m = partial_m[row_block * num_splits + split];
                const float l = partial_l[row_block * num_splits + split];
                if (l > 0.0f && m != -FLT_MAX) {
                    denom += l * expf(m - row_max);
                }
            }
        }
        smem[0] = row_max;
        smem[1] = denom;
        lse[row_block] = denom > 0.0f ? row_max + logf(denom) : INFINITY;
    }
    __syncthreads();

    const float row_max = smem[0];
    const float denom = smem[1];
    if (d < headdim) {
        float acc = 0.0f;
        if (denom > 0.0f && row_max != -FLT_MAX) {
            for (int split = 0; split < num_splits; ++split) {
                const float m = partial_m[row_block * num_splits + split];
                const float l = partial_l[row_block * num_splits + split];
                if (l > 0.0f && m != -FLT_MAX) {
                    acc += partial_o[(row_block * num_splits + split) * headdim + d] * expf(m - row_max);
                }
            }
            acc /= denom;
        }
        out[(((b * seqlen_q + row) * nheads_q + head_q) * headdim) + d] = store_half(acc);
    }
}

__global__ __launch_bounds__(kThreads) void sm75_warp_wide_fwd_kernel(
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
    bool causal) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpRows + warp_id;
    const int b = blockIdx.y;
    const int head_q = blockIdx.z;
    if (row >= seqlen_q) {
        return;
    }
    const int head_k = head_q / (nheads_q / nheads_k);

    float q_frag[8];
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int dim = lane_id + i * 32;
        q_frag[i] = dim < headdim
            ? load_half(q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim + dim))
            : 0.0f;
        acc[i] = 0.0f;
    }

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    int col_end = seqlen_k;
    if (causal) {
        col_end = row + seqlen_k - seqlen_q + 1;
        col_end = col_end < 0 ? 0 : (col_end > seqlen_k ? seqlen_k : col_end);
    }
    for (int col = 0; col < col_end; ++col) {
        float dot = 0.0f;
        const half *k_ptr = k + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int dim = lane_id + i * 32;
            if (dim < headdim) {
                dot += q_frag[i] * load_half(k_ptr + dim);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        dot = __shfl_sync(0xffffffff, dot, 0);

        const float score = dot * softmax_scale;
        const float next_max = fmaxf(row_max, score);
        const float old_scale = row_max == -FLT_MAX ? 0.0f : expf(row_max - next_max);
        const float p = expf(score - next_max);
        const half *v_ptr = v + (((b * seqlen_k + col) * nheads_k + head_k) * headdim);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int dim = lane_id + i * 32;
            if (dim < headdim) {
                acc[i] = acc[i] * old_scale + p * load_half(v_ptr + dim);
            }
        }
        denom = denom * old_scale + p;
        row_max = next_max;
    }

    if (lane_id == 0) {
        lse[(b * nheads_q + head_q) * seqlen_q + row] =
            denom > 0.0f ? row_max + logf(denom) : INFINITY;
    }
    half *out_ptr = out + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int dim = lane_id + i * 32;
        if (dim < headdim) {
            out_ptr[dim] = denom > 0.0f ? store_half(acc[i] / denom) : store_half(0.0f);
        }
    }
}

__global__ __launch_bounds__(kThreads) void sm75_warp256_half2_fwd_kernel(
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
    float softmax_scale,
    bool causal) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpRows + warp_id;
    const int b = blockIdx.y;
    const int head_q = blockIdx.z;
    if (row >= seqlen_q) {
        return;
    }
    const int head_k = head_q / (nheads_q / nheads_k);

    const half2 *q_ptr = reinterpret_cast<const half2 *>(
        q + (((b * seqlen_q + row) * nheads_q + head_q) * 256));
    float2 q_frag[4];
    float2 acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim2 = lane_id + i * 32;
        q_frag[i] = load_half2_as_float2(q_ptr + dim2);
        acc[i] = make_float2(0.0f, 0.0f);
    }

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    int col_end = seqlen_k;
    if (causal) {
        col_end = row + seqlen_k - seqlen_q + 1;
        col_end = col_end < 0 ? 0 : (col_end > seqlen_k ? seqlen_k : col_end);
    }
    for (int col = 0; col < col_end; ++col) {
        float dot = 0.0f;
        const half2 *k_ptr = reinterpret_cast<const half2 *>(
            k + (((b * seqlen_k + col) * nheads_k + head_k) * 256));
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int dim2 = lane_id + i * 32;
            const float2 kval = load_half2_as_float2(k_ptr + dim2);
            dot += q_frag[i].x * kval.x + q_frag[i].y * kval.y;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        dot = __shfl_sync(0xffffffff, dot, 0);

        const float score = dot * softmax_scale;
        const float next_max = fmaxf(row_max, score);
        const float old_scale = row_max == -FLT_MAX ? 0.0f : expf(row_max - next_max);
        const float p = expf(score - next_max);
        const half2 *v_ptr = reinterpret_cast<const half2 *>(
            v + (((b * seqlen_k + col) * nheads_k + head_k) * 256));
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int dim2 = lane_id + i * 32;
            const float2 vval = load_half2_as_float2(v_ptr + dim2);
            acc[i].x = acc[i].x * old_scale + p * vval.x;
            acc[i].y = acc[i].y * old_scale + p * vval.y;
        }
        denom = denom * old_scale + p;
        row_max = next_max;
    }

    if (lane_id == 0) {
        lse[(b * nheads_q + head_q) * seqlen_q + row] =
            denom > 0.0f ? row_max + logf(denom) : INFINITY;
    }
    half2 *out_ptr = reinterpret_cast<half2 *>(
        out + (((b * seqlen_q + row) * nheads_q + head_q) * 256));
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim2 = lane_id + i * 32;
        out_ptr[dim2] = denom > 0.0f
            ? __floats2half2_rn(acc[i].x / denom, acc[i].y / denom)
            : __float2half2_rn(0.0f);
    }
}

__global__ __launch_bounds__(kThreads) void sm75_paged_kvcache_warp_fwd_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k_cache,
    const half *__restrict__ v_cache,
    const int *__restrict__ block_table,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ cache_batch_idx,
    const int *__restrict__ cache_leftpad,
    half *__restrict__ out,
    float *__restrict__ lse,
    int batch,
    int seqlen_q,
    int nheads_q,
    int nheads_k,
    int headdim,
    int page_size,
    int max_blocks_per_seq,
    float softmax_scale,
    bool causal,
    int window_left,
    int window_right) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpRows + warp_id;
    const int b = blockIdx.y;
    const int head_q = blockIdx.z;
    if (row >= seqlen_q) {
        return;
    }
    const int cache_b = cache_batch_idx == nullptr ? b : cache_batch_idx[b];
    const int leftpad = cache_leftpad == nullptr ? 0 : cache_leftpad[b];
    const int seqlen_total = cache_seqlens[b];
    const int seqlen_k = seqlen_total - leftpad;
    const int head_k = head_q / (nheads_q / nheads_k);

    float q_frag[4];
    float acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim = lane_id + i * 32;
        q_frag[i] = dim < headdim
            ? load_half(q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim + dim))
            : 0.0f;
        acc[i] = 0.0f;
    }

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    for (int col = 0; col < seqlen_k; ++col) {
        if (!key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
            continue;
        }
        const int pos = leftpad + col;
        const int block_slot = pos / page_size;
        const int block_offset = pos - block_slot * page_size;
        const int block_id = block_table[cache_b * max_blocks_per_seq + block_slot];
        const half *k_ptr = k_cache + (((block_id * page_size + block_offset) * nheads_k + head_k) * headdim);
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int dim = lane_id + i * 32;
            if (dim < headdim) {
                dot += q_frag[i] * load_half(k_ptr + dim);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        dot = __shfl_sync(0xffffffff, dot, 0);

        const float score = dot * softmax_scale;
        const float next_max = fmaxf(row_max, score);
        const float old_scale = row_max == -FLT_MAX ? 0.0f : expf(row_max - next_max);
        const float p = expf(score - next_max);
        const half *v_ptr = v_cache + (((block_id * page_size + block_offset) * nheads_k + head_k) * headdim);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int dim = lane_id + i * 32;
            if (dim < headdim) {
                acc[i] = acc[i] * old_scale + p * load_half(v_ptr + dim);
            }
        }
        denom = denom * old_scale + p;
        row_max = next_max;
    }

    if (lane_id == 0) {
        lse[(b * nheads_q + head_q) * seqlen_q + row] =
            denom > 0.0f ? row_max + logf(denom) : INFINITY;
    }
    half *out_ptr = out + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int dim = lane_id + i * 32;
        if (dim < headdim) {
            out_ptr[dim] = denom > 0.0f ? store_half(acc[i] / denom) : store_half(0.0f);
        }
    }
}

template <int kHeadDim>
__global__ __launch_bounds__(64) void sm75_paged_kvcache_wmma_split_fwd_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k_cache,
    const half *__restrict__ v_cache,
    const int *__restrict__ block_table,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ cache_batch_idx,
    const int *__restrict__ cache_leftpad,
    float *__restrict__ partial_o,
    float *__restrict__ partial_m,
    float *__restrict__ partial_l,
    int batch,
    int seqlen_q,
    int nheads_q,
    int nheads_k,
    int page_size,
    int max_blocks_per_seq,
    int split_size,
    int num_splits,
    float softmax_scale,
    bool causal,
    int window_left,
    int window_right) {
    constexpr int kRows = 16;
    constexpr int kCols = 16;
    constexpr int kWarps = 2;
    constexpr int kDimTiles = kHeadDim / 16;
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int group_size = nheads_q / nheads_k;
    const int rows_per_kv_head = group_size * seqlen_q;
    const int chunks_per_kv_head = (rows_per_kv_head + kRows - 1) / kRows;
    const int chunk_groups_per_kv_head = (chunks_per_kv_head + kWarps - 1) / kWarps;
    int block = blockIdx.x;
    const int chunk_group = block % chunk_groups_per_kv_head;
    block /= chunk_groups_per_kv_head;
    const int head_k = block % nheads_k;
    const int b = block / nheads_k;
    if (b >= batch) {
        return;
    }

    const int split = blockIdx.y;
    const int cache_b = cache_batch_idx == nullptr ? b : cache_batch_idx[b];
    const int leftpad = cache_leftpad == nullptr ? 0 : cache_leftpad[b];
    const int seqlen_k = cache_seqlens[b] - leftpad;
    const int col_begin = split * split_size;
    const int col_end = min(seqlen_k, col_begin + split_size);
    const int chunk = chunk_group * kWarps + warp_id;
    const bool warp_active = chunk < chunks_per_kv_head;
    const int row_base = chunk * kRows;

    __shared__ __align__(16) half q_smem[kWarps * kRows * kHeadDim];
    __shared__ __align__(16) half k_smem[kCols * kHeadDim];
    __shared__ __align__(16) half v_smem[kCols * kHeadDim];
    __shared__ __align__(16) float tile_smem[kWarps * kRows * kCols];
    __shared__ __align__(16) half p_smem[kWarps * kRows * kCols];
    __shared__ __align__(16) float out_smem[kWarps * kRows * kHeadDim];
    __shared__ float row_scale[kWarps * kRows];
    __shared__ int block_ids[kCols];
    extern __shared__ float pv_smem[];

    half *q_warp = q_smem + warp_id * kRows * kHeadDim;
    float *tile_warp = tile_smem + warp_id * kRows * kCols;
    half *p_warp = p_smem + warp_id * kRows * kCols;
    float *out_warp = out_smem + warp_id * kRows * kHeadDim;
    float *row_scale_warp = row_scale + warp_id * kRows;
    float *pv_warp = pv_smem + warp_id * kRows * kHeadDim;

    for (int linear = lane_id; linear < kRows * kHeadDim; linear += 32) {
        const int row_in_tile = linear / kHeadDim;
        const int dim = linear - row_in_tile * kHeadDim;
        const int query_linear = row_base + row_in_tile;
        if (warp_active && query_linear < rows_per_kv_head) {
            const int head_in_group = query_linear / seqlen_q;
            const int row = query_linear - head_in_group * seqlen_q;
            const int head_q = head_k * group_size + head_in_group;
            q_warp[linear] = q[
                ((static_cast<int64_t>(b) * seqlen_q + row) * nheads_q + head_q) * kHeadDim + dim];
        } else {
            q_warp[linear] = __float2half_rn(0.0f);
        }
    }
    __syncthreads();

    const int lane_query_linear = row_base + lane_id;
    const bool lane_active =
        warp_active && lane_id < kRows && lane_query_linear < rows_per_kv_head;
    const int lane_head_in_group = lane_active ? lane_query_linear / seqlen_q : 0;
    const int lane_row = lane_active ? lane_query_linear - lane_head_in_group * seqlen_q : 0;
    const int lane_head_q = head_k * group_size + lane_head_in_group;
    float row_max = -FLT_MAX;
    float denom = 0.0f;

    for (int linear = lane_id; linear < kRows * kHeadDim; linear += 32) {
        out_warp[linear] = 0.0f;
    }
    __syncthreads();

    for (int tile_begin = col_begin; tile_begin < col_end; tile_begin += kCols) {
        if (threadIdx.x < kCols) {
            const int col = tile_begin + threadIdx.x;
            if (col < col_end) {
                const int pos = leftpad + col;
                block_ids[threadIdx.x] = block_table[
                    cache_b * max_blocks_per_seq + pos / page_size];
            } else {
                block_ids[threadIdx.x] = -1;
            }
        }
        __syncthreads();
        for (int linear = threadIdx.x; linear < kCols * kHeadDim; linear += blockDim.x) {
            const int key = linear / kHeadDim;
            const int dim = linear - key * kHeadDim;
            const int block_id = block_ids[key];
            if (block_id >= 0) {
                const int pos = leftpad + tile_begin + key;
                const int block_offset = pos % page_size;
                const int64_t cache_offset =
                    ((static_cast<int64_t>(block_id) * page_size + block_offset) * nheads_k + head_k)
                    * kHeadDim + dim;
                k_smem[linear] = k_cache[cache_offset];
                v_smem[linear] = v_cache[cache_offset];
            } else {
                k_smem[linear] = __float2half_rn(0.0f);
                v_smem[linear] = __float2half_rn(0.0f);
            }
        }
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> score_frag;
        nvcuda::wmma::fill_fragment(score_frag, 0.0f);
        #pragma unroll
        for (int dim = 0; dim < kHeadDim; dim += 16) {
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> q_frag;
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> k_frag;
            nvcuda::wmma::load_matrix_sync(q_frag, q_warp + dim, kHeadDim);
            nvcuda::wmma::load_matrix_sync(k_frag, k_smem + dim, kHeadDim);
            nvcuda::wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
        }
        nvcuda::wmma::store_matrix_sync(
            tile_warp, score_frag, kCols, nvcuda::wmma::mem_row_major);
        __syncthreads();

        if (lane_id < kRows) {
            float tile_max = -FLT_MAX;
            if (lane_active) {
                #pragma unroll
                for (int key = 0; key < kCols; ++key) {
                    const int col = tile_begin + key;
                    if (col < col_end && key_is_visible(
                            lane_row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
                        tile_max = fmaxf(
                            tile_max,
                            tile_warp[lane_id * kCols + key] * softmax_scale);
                    }
                }
            }
            const float next_max = fmaxf(row_max, tile_max);
            const float old_scale = tile_max == -FLT_MAX
                ? 1.0f
                : (row_max == -FLT_MAX ? 0.0f : expf(row_max - next_max));
            float tile_sum = 0.0f;
            #pragma unroll
            for (int key = 0; key < kCols; ++key) {
                const int col = tile_begin + key;
                float p = 0.0f;
                if (lane_active && col < col_end && key_is_visible(
                        lane_row, col, seqlen_q, seqlen_k, causal, window_left, window_right)) {
                    p = expf(tile_warp[lane_id * kCols + key] * softmax_scale - next_max);
                    tile_sum += p;
                }
                p_warp[lane_id * kCols + key] = __float2half_rn(p);
            }
            denom = denom * old_scale + tile_sum;
            row_max = next_max;
            row_scale_warp[lane_id] = old_scale;
        }
        __syncthreads();

        nvcuda::wmma::fragment<
            nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> p_frag;
        nvcuda::wmma::load_matrix_sync(p_frag, p_warp, kCols);
        #pragma unroll
        for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
            nvcuda::wmma::fragment<
                nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> v_frag;
            nvcuda::wmma::fragment<
                nvcuda::wmma::accumulator, 16, 16, 16, float> pv_frag;
            nvcuda::wmma::load_matrix_sync(
                v_frag, v_smem + dim_tile * 16, kHeadDim);
            nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
            nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            nvcuda::wmma::store_matrix_sync(
                pv_warp + dim_tile * kCols,
                pv_frag,
                kHeadDim,
                nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        for (int linear = lane_id; linear < kRows * kHeadDim; linear += 32) {
            const int row_in_tile = linear / kHeadDim;
            out_warp[linear] =
                out_warp[linear] * row_scale_warp[row_in_tile] + pv_warp[linear];
        }
        __syncthreads();
    }

    if (lane_active) {
        const int row_block =
            (b * nheads_q + lane_head_q) * seqlen_q + lane_row;
        partial_m[row_block * num_splits + split] = row_max;
        partial_l[row_block * num_splits + split] = denom;
    }
    __syncthreads();

    for (int linear = lane_id; linear < kRows * kHeadDim; linear += 32) {
        const int row_in_tile = linear / kHeadDim;
        const int dim = linear - row_in_tile * kHeadDim;
        const int query_linear = row_base + row_in_tile;
        if (query_linear < rows_per_kv_head) {
            const int head_in_group = query_linear / seqlen_q;
            const int row = query_linear - head_in_group * seqlen_q;
            const int head_q = head_k * group_size + head_in_group;
            const int row_block = (b * nheads_q + head_q) * seqlen_q + row;
            partial_o[
                (static_cast<int64_t>(row_block) * num_splits + split) * kHeadDim + dim] =
                out_warp[linear];
        }
    }
}

__global__ void sm75_paged_splitkv_partial_fwd_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k_cache,
    const half *__restrict__ v_cache,
    const int *__restrict__ block_table,
    const int *__restrict__ cache_seqlens,
    const int *__restrict__ cache_batch_idx,
    const int *__restrict__ cache_leftpad,
    float *__restrict__ partial_o,
    float *__restrict__ partial_m,
    float *__restrict__ partial_l,
    int batch,
    int seqlen_q,
    int nheads_q,
    int nheads_k,
    int headdim,
    int page_size,
    int max_blocks_per_seq,
    int split_size,
    int num_splits,
    float softmax_scale,
    bool causal,
    int window_left,
    int window_right) {
    const int row_block = blockIdx.x;
    const int split = blockIdx.y;
    const int row = row_block % seqlen_q;
    const int head_q = (row_block / seqlen_q) % nheads_q;
    const int b = row_block / (seqlen_q * nheads_q);
    const int cache_b = cache_batch_idx == nullptr ? b : cache_batch_idx[b];
    const int leftpad = cache_leftpad == nullptr ? 0 : cache_leftpad[b];
    const int seqlen_total = cache_seqlens[b];
    const int seqlen_k = seqlen_total - leftpad;
    const int head_k = head_q / (nheads_q / nheads_k);
    const int d = threadIdx.x;
    const int col_begin = split * split_size;
    const int col_end = min(seqlen_k, col_begin + split_size);

    extern __shared__ float smem[];
    float *reduce = smem;
    const half *q_ptr = q + (((b * seqlen_q + row) * nheads_q + head_q) * headdim);

    float row_max = -FLT_MAX;
    float denom = 0.0f;
    float acc = 0.0f;
    for (int col = col_begin; col < col_end; ++col) {
        const bool visible = key_is_visible(row, col, seqlen_q, seqlen_k, causal, window_left, window_right);
        const int pos = leftpad + col;
        const int block_slot = pos / page_size;
        const int block_offset = pos - block_slot * page_size;
        const int block_id = block_table[cache_b * max_blocks_per_seq + block_slot];
        const half *k_ptr = k_cache + (((block_id * page_size + block_offset) * nheads_k + head_k) * headdim);
        float local_dot = 0.0f;
        if (visible && d < headdim) {
            local_dot = load_half(q_ptr + d) * load_half(k_ptr + d);
        }
        const float dot = block_sum_256(local_dot, reduce);
        if (!visible) {
            continue;
        }
        const float score = dot * softmax_scale;
        const float next_max = fmaxf(row_max, score);
        const float old_scale = row_max == -FLT_MAX ? 0.0f : expf(row_max - next_max);
        const float p = expf(score - next_max);
        const half *v_ptr = v_cache + (((block_id * page_size + block_offset) * nheads_k + head_k) * headdim);
        if (d < headdim) {
            acc = acc * old_scale + p * load_half(v_ptr + d);
        }
        denom = denom * old_scale + p;
        row_max = next_max;
    }

    const int partial_base = (row_block * num_splits + split) * headdim;
    if (d < headdim) {
        partial_o[partial_base + d] = acc;
    }
    if (threadIdx.x == 0) {
        partial_m[row_block * num_splits + split] = row_max;
        partial_l[row_block * num_splits + split] = denom;
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
    const bool has_alibi = alibi_slopes.defined() && alibi_slopes.numel() > 0;
    const bool can_warp_wide = !return_softmax
        && dropout_p == 0.0f
        && window_left == -1
        && window_right == -1
        && !has_alibi
        && softcap == 0.0f
        && headdim >= 224
        && headdim <= 256;
    if (can_warp_wide) {
        if (headdim == 256) {
            dim3 grid((seqlen_q + kWarpRows - 1) / kWarpRows, batch, nheads_q);
            sm75_warp256_half2_fwd_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
                softmax_scale,
                causal);
        } else {
            dim3 grid((seqlen_q + kWarpRows - 1) / kWarpRows, batch, nheads_q);
            sm75_warp_wide_fwd_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
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
                causal);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {out, lse, p};
    }
    const bool can_splitkv = !return_softmax
        && dropout_p == 0.0f
        && window_left == -1
        && window_right == -1
        && !has_alibi
        && softcap == 0.0f
        && headdim >= 224
        && headdim <= 256
        && seqlen_k >= 128;
    if (can_splitkv) {
        constexpr int kSplitSize = 256;
        const int num_splits = (seqlen_k + kSplitSize - 1) / kSplitSize;
        const int64_t partial_elements = static_cast<int64_t>(blocks) * num_splits * headdim;
        // Keep this bridge path bounded; larger rows remain on the one-pass generic kernel
        // until a true wide-head tiled kernel exists.
        if (partial_elements <= 80000000LL) {
            auto partial_o = torch::empty({blocks, num_splits, headdim}, q.options().dtype(torch::kFloat32));
            auto partial_m = torch::empty({blocks, num_splits}, q.options().dtype(torch::kFloat32));
            auto partial_l = torch::empty({blocks, num_splits}, q.options().dtype(torch::kFloat32));
            const size_t partial_smem = kThreads * sizeof(float);
            dim3 partial_grid(blocks, num_splits);
            sm75_splitkv_partial_fwd_kernel<<<partial_grid, kThreads, partial_smem, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k.data_ptr()),
                reinterpret_cast<const half *>(v.data_ptr()),
                reinterpret_cast<float *>(partial_o.data_ptr()),
                reinterpret_cast<float *>(partial_m.data_ptr()),
                reinterpret_cast<float *>(partial_l.data_ptr()),
                batch,
                seqlen_q,
                seqlen_k,
                nheads_q,
                nheads_k,
                headdim,
                kSplitSize,
                num_splits,
                softmax_scale,
                causal);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            sm75_splitkv_combine_fwd_kernel<<<blocks, kThreads, 2 * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const float *>(partial_o.data_ptr()),
                reinterpret_cast<const float *>(partial_m.data_ptr()),
                reinterpret_cast<const float *>(partial_l.data_ptr()),
                reinterpret_cast<half *>(out.data_ptr()),
                reinterpret_cast<float *>(lse.data_ptr()),
                seqlen_q,
                nheads_q,
                headdim,
                num_splits);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return {out, lse, p};
        }
    }
    const size_t smem = kThreads * sizeof(float);
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

std::vector<at::Tensor> paged_kvcache_update_sm75(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor block_table,
    at::Tensor cache_seqlens,
    at::Tensor cache_batch_idx,
    at::Tensor rotary_cos,
    at::Tensor rotary_sin,
    bool rotary_interleaved,
    bool sequential_q_positions) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
    TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "paged k/v cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda() && cache_seqlens.is_cuda(), "block_table/cache_seqlens must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16,
                "paged SM75 update expects FP16 q/k/v");
    TORCH_CHECK(k_cache.scalar_type() == torch::kFloat16 && v_cache.scalar_type() == torch::kFloat16,
                "paged SM75 update expects FP16 cache tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,S,H,D]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "paged cache must be [blocks,page,H,D]");
    TORCH_CHECK(block_table.dim() == 2 && block_table.scalar_type() == torch::kInt32, "block_table must be int32 [Bcache,max_blocks]");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be int32 [B]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous for paged update");
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(), "paged cache tensors must be contiguous");
    TORCH_CHECK(block_table.is_contiguous() && cache_seqlens.is_contiguous(), "block_table/cache_seqlens must be contiguous");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache/v_cache shape mismatch");
    TORCH_CHECK(k.sizes() == v.sizes(), "new k/v shape mismatch");
    TORCH_CHECK(q.size(0) == k.size(0), "q and new k/v batch mismatch");
    TORCH_CHECK(k.size(2) == k_cache.size(2), "new k head count must match cache");
    TORCH_CHECK(k.size(3) == k_cache.size(3) && q.size(3) == k_cache.size(3), "head_dim mismatch");
    TORCH_CHECK(q.size(2) % k_cache.size(2) == 0, "query heads must be divisible by KV heads");
    TORCH_CHECK(cache_seqlens.numel() == q.size(0), "cache_seqlens must have one entry per query batch");
    TORCH_CHECK(!cache_batch_idx.defined() || cache_batch_idx.numel() == 0
                || (cache_batch_idx.is_cuda() && cache_batch_idx.scalar_type() == torch::kInt32 && cache_batch_idx.is_contiguous()),
                "cache_batch_idx must be empty or contiguous CUDA int32");

    const bool has_rotary = rotary_cos.defined() && rotary_cos.numel() > 0;
    int rotary_half_dim = 0;
    if (has_rotary) {
        TORCH_CHECK(rotary_sin.defined() && rotary_sin.numel() > 0, "rotary_sin is required with rotary_cos");
        TORCH_CHECK(rotary_cos.is_cuda() && rotary_sin.is_cuda(), "rotary tensors must be CUDA tensors");
        TORCH_CHECK(rotary_cos.scalar_type() == torch::kFloat16 && rotary_sin.scalar_type() == torch::kFloat16,
                    "SM75 rotary update expects FP16 rotary tensors");
        TORCH_CHECK(rotary_cos.dim() == 2 && rotary_sin.sizes() == rotary_cos.sizes(),
                    "rotary tensors must be [seqlen, rotary_dim / 2]");
        TORCH_CHECK(rotary_cos.is_contiguous() && rotary_sin.is_contiguous(), "rotary tensors must be contiguous");
        TORCH_CHECK(rotary_cos.size(1) * 2 <= q.size(3), "rotary_dim must be <= head_dim");
        rotary_half_dim = rotary_cos.size(1);
    }

    auto q_out = has_rotary ? torch::empty_like(q) : q;
    auto cache_seqlens_after = torch::empty_like(cache_seqlens);
    const int64_t q_elems = has_rotary
        ? q.numel()
        : 0;
    const int64_t kv_elems = k.numel();
    const int64_t total = cache_seqlens.numel() + q_elems + kv_elems;
    int64_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks < 1) {
        blocks = 1;
    } else if (blocks > 4096) {
        blocks = 4096;
    }

    sm75_paged_kvcache_update_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half *>(q.data_ptr()),
        reinterpret_cast<const half *>(k.data_ptr()),
        reinterpret_cast<const half *>(v.data_ptr()),
        reinterpret_cast<half *>(q_out.data_ptr()),
        reinterpret_cast<half *>(k_cache.data_ptr()),
        reinterpret_cast<half *>(v_cache.data_ptr()),
        reinterpret_cast<const int *>(block_table.data_ptr()),
        reinterpret_cast<const int *>(cache_seqlens.data_ptr()),
        (cache_batch_idx.defined() && cache_batch_idx.numel() > 0) ? reinterpret_cast<const int *>(cache_batch_idx.data_ptr()) : nullptr,
        has_rotary ? reinterpret_cast<const half *>(rotary_cos.data_ptr()) : nullptr,
        has_rotary ? reinterpret_cast<const half *>(rotary_sin.data_ptr()) : nullptr,
        reinterpret_cast<int *>(cache_seqlens_after.data_ptr()),
        q.size(0),
        q.size(1),
        k.size(1),
        q.size(2),
        k_cache.size(2),
        q.size(3),
        k_cache.size(1),
        block_table.size(1),
        rotary_half_dim,
        rotary_interleaved,
        sequential_q_positions);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_out, cache_seqlens_after};
}

std::vector<at::Tensor> pad_qkv_head_dim_sm75(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    int64_t padded_head_dim) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 && v.scalar_type() == torch::kFloat16,
                "SM75 qkv padding expects FP16 tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be rank-4 tensors");
    TORCH_CHECK(k.sizes() == v.sizes(), "k/v shapes must match");
    TORCH_CHECK(q.size(0) == k.size(0), "q/k/v batch sizes must match");
    TORCH_CHECK(q.size(3) == k.size(3), "q/k/v source head_dim must match");
    TORCH_CHECK(padded_head_dim > q.size(3), "padded_head_dim must be greater than source head_dim");
    TORCH_CHECK(padded_head_dim <= 256, "padded_head_dim must be <= 256");

    auto q_out = torch::empty({q.size(0), q.size(1), q.size(2), padded_head_dim}, q.options());
    auto k_out = torch::empty({k.size(0), k.size(1), k.size(2), padded_head_dim}, k.options());
    auto v_out = torch::empty({v.size(0), v.size(1), v.size(2), padded_head_dim}, v.options());

    const int64_t q_total = q_out.numel();
    const int64_t k_total = k_out.numel();
    const int64_t v_total = v_out.numel();
    const int64_t total = q_total + k_total + v_total;
    int64_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks < 1) {
        blocks = 1;
    } else if (blocks > 4096) {
        blocks = 4096;
    }

    sm75_pad_qkv_head_dim_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half *>(q.data_ptr()),
        reinterpret_cast<const half *>(k.data_ptr()),
        reinterpret_cast<const half *>(v.data_ptr()),
        reinterpret_cast<half *>(q_out.data_ptr()),
        reinterpret_cast<half *>(k_out.data_ptr()),
        reinterpret_cast<half *>(v_out.data_ptr()),
        q_total,
        k_total,
        v_total,
        q.size(0),
        q.size(1),
        q.size(2),
        k.size(0),
        k.size(1),
        k.size(2),
        q.size(3),
        padded_head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_out, k_out, v_out};
}

std::vector<at::Tensor> mha_paged_kvcache_fwd_sm75(
    at::Tensor q,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor block_table,
    at::Tensor cache_seqlens,
    at::Tensor cache_batch_idx,
    at::Tensor cache_leftpad,
    const float softmax_scale,
    bool causal,
    int window_left,
    int window_right,
    int max_seqlen_k) {
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "q and paged k/v cache must be CUDA tensors");
    TORCH_CHECK(block_table.is_cuda() && cache_seqlens.is_cuda(), "block_table and cache_seqlens must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k_cache.scalar_type() == torch::kFloat16 && v_cache.scalar_type() == torch::kFloat16,
                "direct paged SM75 KV-cache forward expects FP16 tensors");
    TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4 && v_cache.dim() == 4, "q must be [B,S,H,D], paged cache must be [blocks,page,H,D]");
    TORCH_CHECK(block_table.dim() == 2 && block_table.scalar_type() == torch::kInt32, "block_table must be int32 [Bcache,max_blocks]");
    TORCH_CHECK(cache_seqlens.dim() == 1 && cache_seqlens.scalar_type() == torch::kInt32, "cache_seqlens must be int32 [B]");
    TORCH_CHECK(!cache_batch_idx.defined() || cache_batch_idx.numel() == 0 || (cache_batch_idx.is_cuda() && cache_batch_idx.scalar_type() == torch::kInt32),
                "cache_batch_idx must be empty or CUDA int32");
    TORCH_CHECK(!cache_leftpad.defined() || cache_leftpad.numel() == 0 || (cache_leftpad.is_cuda() && cache_leftpad.scalar_type() == torch::kInt32),
                "cache_leftpad must be empty or CUDA int32");
    TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous(), "direct paged KV-cache tensors must be contiguous");
    TORCH_CHECK(block_table.is_contiguous() && cache_seqlens.is_contiguous(), "block_table/cache_seqlens must be contiguous");
    TORCH_CHECK(k_cache.size(0) == v_cache.size(0) && k_cache.size(1) == v_cache.size(1)
                && k_cache.size(2) == v_cache.size(2) && k_cache.size(3) == v_cache.size(3),
                "k_cache and v_cache shapes must match");
    TORCH_CHECK(q.size(0) == cache_seqlens.numel(), "cache_seqlens must have one entry per query batch");
    TORCH_CHECK(q.size(2) % k_cache.size(2) == 0, "query heads must be divisible by key/value heads");
    TORCH_CHECK(q.size(3) == k_cache.size(3), "q/cache head_dim must match");
    TORCH_CHECK(q.size(1) <= 8, "direct paged SM75 KV-cache path supports q_len <= 8");
    TORCH_CHECK(q.size(3) <= 128, "direct paged SM75 KV-cache path supports head_dim <= 128");
    TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k must be positive");

    auto out = torch::empty_like(q);
    auto lse = torch::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(torch::kFloat32));
    const int row_blocks = q.size(0) * q.size(1) * q.size(2);
    int split_size = 256;
    if (const char *split_size_env = std::getenv("FLASH_ATTN_SM75_PAGED_SPLIT_SIZE")) {
        const int requested = std::atoi(split_size_env);
        if (requested >= 16 && requested % 16 == 0) {
            split_size = requested;
        }
    }
    const int num_splits = (max_seqlen_k + split_size - 1) / split_size;
    const int64_t partial_elements = static_cast<int64_t>(row_blocks) * num_splits * q.size(3);
    const bool can_wmma_split =
        (q.size(3) == 64 || q.size(3) == 128) && partial_elements <= 80000000LL;
    if (can_wmma_split) {
        auto partial_o = torch::empty(
            {row_blocks, num_splits, q.size(3)}, q.options().dtype(torch::kFloat32));
        auto partial_m = torch::empty(
            {row_blocks, num_splits}, q.options().dtype(torch::kFloat32));
        auto partial_l = torch::empty(
            {row_blocks, num_splits}, q.options().dtype(torch::kFloat32));
        const int group_size = q.size(2) / k_cache.size(2);
        const int chunks_per_kv_head = (group_size * q.size(1) + 15) / 16;
        const int chunk_groups_per_kv_head = (chunks_per_kv_head + 1) / 2;
        dim3 partial_grid(
            q.size(0) * k_cache.size(2) * chunk_groups_per_kv_head,
            num_splits);
        if (q.size(3) == 64) {
            constexpr int kDynamicSmem = 2 * 16 * 64 * sizeof(float);
            auto kernel = sm75_paged_kvcache_wmma_split_fwd_kernel<64>;
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmem));
            kernel<<<partial_grid, 64, kDynamicSmem, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                reinterpret_cast<const int *>(block_table.data_ptr()),
                reinterpret_cast<const int *>(cache_seqlens.data_ptr()),
                (cache_batch_idx.defined() && cache_batch_idx.numel() > 0)
                    ? reinterpret_cast<const int *>(cache_batch_idx.data_ptr()) : nullptr,
                (cache_leftpad.defined() && cache_leftpad.numel() > 0)
                    ? reinterpret_cast<const int *>(cache_leftpad.data_ptr()) : nullptr,
                reinterpret_cast<float *>(partial_o.data_ptr()),
                reinterpret_cast<float *>(partial_m.data_ptr()),
                reinterpret_cast<float *>(partial_l.data_ptr()),
                q.size(0),
                q.size(1),
                q.size(2),
                k_cache.size(2),
                k_cache.size(1),
                block_table.size(1),
                split_size,
                num_splits,
                softmax_scale,
                causal,
                window_left,
                window_right);
        } else {
            constexpr int kDynamicSmem = 2 * 16 * 128 * sizeof(float);
            auto kernel = sm75_paged_kvcache_wmma_split_fwd_kernel<128>;
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmem));
            kernel<<<partial_grid, 64, kDynamicSmem, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const half *>(q.data_ptr()),
                reinterpret_cast<const half *>(k_cache.data_ptr()),
                reinterpret_cast<const half *>(v_cache.data_ptr()),
                reinterpret_cast<const int *>(block_table.data_ptr()),
                reinterpret_cast<const int *>(cache_seqlens.data_ptr()),
                (cache_batch_idx.defined() && cache_batch_idx.numel() > 0)
                    ? reinterpret_cast<const int *>(cache_batch_idx.data_ptr()) : nullptr,
                (cache_leftpad.defined() && cache_leftpad.numel() > 0)
                    ? reinterpret_cast<const int *>(cache_leftpad.data_ptr()) : nullptr,
                reinterpret_cast<float *>(partial_o.data_ptr()),
                reinterpret_cast<float *>(partial_m.data_ptr()),
                reinterpret_cast<float *>(partial_l.data_ptr()),
                q.size(0),
                q.size(1),
                q.size(2),
                k_cache.size(2),
                k_cache.size(1),
                block_table.size(1),
                split_size,
                num_splits,
                softmax_scale,
                causal,
                window_left,
                window_right);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        sm75_splitkv_combine_fwd_kernel<<<
            row_blocks, kThreads, 2 * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float *>(partial_o.data_ptr()),
            reinterpret_cast<const float *>(partial_m.data_ptr()),
            reinterpret_cast<const float *>(partial_l.data_ptr()),
            reinterpret_cast<half *>(out.data_ptr()),
            reinterpret_cast<float *>(lse.data_ptr()),
            q.size(1),
            q.size(2),
            q.size(3),
            num_splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {out, lse};
    }
    if (max_seqlen_k >= 16384 && row_blocks <= 128 && partial_elements <= 80000000LL) {
        auto partial_o = torch::empty({row_blocks, num_splits, q.size(3)}, q.options().dtype(torch::kFloat32));
        auto partial_m = torch::empty({row_blocks, num_splits}, q.options().dtype(torch::kFloat32));
        auto partial_l = torch::empty({row_blocks, num_splits}, q.options().dtype(torch::kFloat32));
        const size_t partial_smem = kThreads * sizeof(float);
        dim3 partial_grid(row_blocks, num_splits);
        sm75_paged_splitkv_partial_fwd_kernel<<<partial_grid, kThreads, partial_smem, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half *>(q.data_ptr()),
            reinterpret_cast<const half *>(k_cache.data_ptr()),
            reinterpret_cast<const half *>(v_cache.data_ptr()),
            reinterpret_cast<const int *>(block_table.data_ptr()),
            reinterpret_cast<const int *>(cache_seqlens.data_ptr()),
            (cache_batch_idx.defined() && cache_batch_idx.numel() > 0) ? reinterpret_cast<const int *>(cache_batch_idx.data_ptr()) : nullptr,
            (cache_leftpad.defined() && cache_leftpad.numel() > 0) ? reinterpret_cast<const int *>(cache_leftpad.data_ptr()) : nullptr,
            reinterpret_cast<float *>(partial_o.data_ptr()),
            reinterpret_cast<float *>(partial_m.data_ptr()),
            reinterpret_cast<float *>(partial_l.data_ptr()),
            q.size(0),
            q.size(1),
            q.size(2),
            k_cache.size(2),
            q.size(3),
            k_cache.size(1),
            block_table.size(1),
            split_size,
            num_splits,
            softmax_scale,
            causal,
            window_left,
            window_right);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        sm75_splitkv_combine_fwd_kernel<<<row_blocks, kThreads, 2 * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float *>(partial_o.data_ptr()),
            reinterpret_cast<const float *>(partial_m.data_ptr()),
            reinterpret_cast<const float *>(partial_l.data_ptr()),
            reinterpret_cast<half *>(out.data_ptr()),
            reinterpret_cast<float *>(lse.data_ptr()),
            q.size(1),
            q.size(2),
            q.size(3),
            num_splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {out, lse};
    }

    dim3 grid((q.size(1) + kWarpRows - 1) / kWarpRows, q.size(0), q.size(2));
    sm75_paged_kvcache_warp_fwd_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half *>(q.data_ptr()),
        reinterpret_cast<const half *>(k_cache.data_ptr()),
        reinterpret_cast<const half *>(v_cache.data_ptr()),
        reinterpret_cast<const int *>(block_table.data_ptr()),
        reinterpret_cast<const int *>(cache_seqlens.data_ptr()),
        (cache_batch_idx.defined() && cache_batch_idx.numel() > 0) ? reinterpret_cast<const int *>(cache_batch_idx.data_ptr()) : nullptr,
        (cache_leftpad.defined() && cache_leftpad.numel() > 0) ? reinterpret_cast<const int *>(cache_leftpad.data_ptr()) : nullptr,
        reinterpret_cast<half *>(out.data_ptr()),
        reinterpret_cast<float *>(lse.data_ptr()),
        q.size(0),
        q.size(1),
        q.size(2),
        k_cache.size(2),
        q.size(3),
        k_cache.size(1),
        block_table.size(1),
        softmax_scale,
        causal,
        window_left,
        window_right);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}
