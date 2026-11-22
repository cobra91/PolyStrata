// src/poly/kernels_inkling.cu - Inkling 2 Mini's kernels; see include/strata/poly/inkling_kernels.hpp for the math
// and where it comes from.  Plain float arithmetic throughout, the caches' halves (or Q8_0 codes) the only rounding,
// but for the prompt batches' attention on the tensor cores (attn_mma_kernel), whose products take halves.
#include "strata/poly/inkling_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace strata::poly {
namespace {

constexpr int WARP = 32;

inline cudaStream_t cs(void* stream) { return (cudaStream_t) stream; }
inline int grid(size_t n, int block) { return int((n + block - 1) / block); }

void check_launch(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o, WARP);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, WARP));
    return x;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- native_mmvq.cu's arithmetic -------------------------------------------------------------------------------
//
// native_mmvq.cu is built with --use_fast_math: its float operations flush denormals to zero and its divisions are
// approximate.  The launches here that do its Q8_1 quantization or its RMS norm write those operations out as the
// instructions it compiles them to, so that their bits are its bits (this file is built without fast math).  A HIP
// build has no PTX and compiles native_mmvq.cu without fast math, so there they are the plain IEEE operations.
namespace fm {
#if defined(__HIPCC__)
__device__ __forceinline__ float add(float a, float b) { return a + b; }
__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float div_rn(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float div(float a, float b) { return a / b; }
__device__ __forceinline__ float max(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ float fma_rn(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float sqrt_rn(float a) { return __fsqrt_rn(a); }
__device__ __forceinline__ float abs(float a) { return fabsf(a); }
#else
#define FM_BINARY(name, op)                                                  \
    __device__ __forceinline__ float name(float a, float b) {                \
        float r;                                                             \
        asm(op " %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));                   \
        return r;                                                            \
    }
FM_BINARY(add, "add.ftz.f32")         // a + b
FM_BINARY(add_rn, "add.rn.ftz.f32")   // __fadd_rn
FM_BINARY(mul_rn, "mul.rn.ftz.f32")   // __fmul_rn
FM_BINARY(div_rn, "div.rn.ftz.f32")   // __fdiv_rn
FM_BINARY(div, "div.approx.ftz.f32")  // a / b
FM_BINARY(max, "max.ftz.f32")         // fmaxf
#undef FM_BINARY
__device__ __forceinline__ float fma_rn(float a, float b, float c) {   // __fmaf_rn
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
__device__ __forceinline__ float sqrt_rn(float a) {   // __fsqrt_rn
    float r;
    asm("sqrt.rn.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float abs(float a) {   // fabsf
    float r;
    asm("abs.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
#endif
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = add(x, __shfl_xor_sync(0xffffffffu, x, o, WARP));
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = max(x, __shfl_xor_sync(0xffffffffu, x, o, WARP));
    return x;
}
constexpr float kHalfMax = 65504.0f;
__device__ __forceinline__ float finite(float v) { return abs(v) > kHalfMax ? copysignf(kHalfMax, v) : v; }

// A Q8_1 block: native_mmvq.cu's Q81Block.
struct Q81 {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "a Q8_1 block");

// A warp's 32 values, lane i value i of the block, as native_quantize_q8_1_kernel quantizes them (q8_1_finite.hpp's
// scale, value and sums).
__device__ __forceinline__ void q8_1(float xi, Q81* blk) {
    const int lane = threadIdx.x % WARP;
    const float amax = warp_max(abs(xi));
    const float sum = warp_sum(xi);
    const float d = finite(div(amax, 127.0f));
    int8_t q = 0;
    if (amax != 0.0f) {
        const float r = roundf(div(xi, d));
        q = (int8_t) (r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r);
    }
    blk->qs[lane] = q;
    if (lane == 0) blk->ds = make_half2(__float2half(finite(d)), __float2half(finite(sum)));
}

// native_add_rms_norm_q8_1_kernel<SUM>'s row without add: y = x / rms(x) * w and its Q8_1 blocks.  The block's
// threads (a multiple of 32, fewer than SUM or not) take the kernel's SUM threads' sums in turn, a warp one of its
// warps; part: SUM / WARP floats of shared memory.
template <int SUM>
__device__ __forceinline__ void rms_norm_q8_1(const float* x, const float* w, float* y, Q81* yq, int n, float eps,
                                              float* part) {
    for (int t0 = 0; t0 < SUM; t0 += blockDim.x) {
        const int t = t0 + threadIdx.x;
        if (t >= SUM) break;
        float ss = 0.0f;
        for (int i = t; i < n; i += SUM) {
            const float v = x[i];
            ss = fma_rn(v, v, ss);
        }
        ss = warp_sum(ss);
        if ((t & 31) == 0) part[t >> 5] = ss;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < SUM / WARP; ++k) sum = add_rn(sum, part[k]);
    const float scale = div_rn(1.0f, sqrt_rn(add_rn(div_rn(sum, float(n)), eps)));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = mul_rn(mul_rn(x[i], scale), w[i]);
        y[i] = xi;
        q8_1(xi, yq + i / 32);
    }
    __syncthreads();   // part, for a row after this one
}
}  // namespace fm

// upstream: cast(pos) -> scale_bias(1, 1) -> scale(1 / floor) -> clamp(1, inf) -> log -> scale_bias(alpha, 1)
__device__ __forceinline__ float tau_of(float alpha, int floor_, long long p) {
    float v = (float) p * 1.0f + 1.0f;
    v = v * (1.0f / (float) floor_);
    v = v < 1.0f ? 1.0f : v;
    return logf(v) * alpha + 1.0f;
}

// ---- clip ------------------------------------------------------------------------------------------------------

// A warp a head of a token.
__device__ __forceinline__ void clip_group(const float* __restrict__ x, int x_stride, const float* __restrict__ scale,
                                           float* y, int dim, int heads, int gi, float threshold, float alpha,
                                           int floor_, long long pos) {
    const int lane = threadIdx.x % WARP;
    const int t = gi / heads, h = gi % heads;
    const float* xr = x + size_t(t) * x_stride + size_t(h) * dim;
    float* yr = y + (size_t(t) * heads + h) * dim;
    const float* sr = scale ? scale + size_t(h) * dim : nullptr;
    constexpr int MAXV = 1024 / WARP;
    float v[MAXV];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int i = lane + k * WARP;
        v[k] = 0.0f;
        if (i < dim) {
            v[k] = sr ? xr[i] * sr[i] : xr[i];
            ss += v[k] * v[k];
        }
    }
    ss = warp_sum(ss);
    const float norm = sqrtf(ss) + 1e-6f;
    const float factor = fminf(fmaxf(threshold / norm, 0.0f), 1.0f);
    const float tau = floor_ > 0 ? tau_of(alpha, floor_, pos + t) : 1.0f;
#pragma unroll
    for (int k = 0; k < MAXV; ++k) {
        const int i = lane + k * WARP;
        if (i < dim) yr[i] = floor_ > 0 ? v[k] * factor * tau : v[k] * factor;
    }
}

__global__ void clip_kernel(const float* __restrict__ x, int x_stride, const float* __restrict__ scale,
                            float* y, int dim, int heads, int groups, float threshold, float alpha, int floor_,
                            long long pos) {
    const int gi = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (gi >= groups) return;
    clip_group(x, x_stride, scale, y, dim, heads, gi, threshold, alpha, floor_, pos);
}

struct ClipJobs {
    InklingClipJob j[4];
    int first[5];   // the jobs' first groups, and the end
};

// Where a call's tokens are: one sequence's, from a position on (inkling_clip_many()) given by value or read on the
// card when the launch runs, or each row a sequence's of its own at a position of its own (inkling_clip_many_seqs()).
// pos(t) - t is handed to clip_group(), which adds t.
struct ClipFrom {
    long long pos;
    __device__ long long first(int) const { return pos; }
};
struct ClipAt {
    const long long* at;
    __device__ long long first(int) const { return *at; }
};
struct ClipSeqs {
    long long pos[kInklingSeqRows];
    __device__ long long first(int t) const { return pos[t] - t; }
};

template <class From>
__device__ __forceinline__ void clip_many_group(const ClipJobs& c, int count, const From& from, int gi) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (k >= count) return;
        if (gi < c.first[k + 1]) {
            const InklingClipJob& j = c.j[k];
            const int g = gi - c.first[k];
            clip_group(j.x, j.x_stride ? j.x_stride : j.heads * j.dim, j.scale, j.y, j.dim, j.heads, g, j.threshold,
                       j.tau.alpha, j.tau.floor, from.first(g / j.heads));
            return;
        }
    }
}

template <class From>
__global__ void clip_many_kernel(ClipJobs c, int count, From from) {
    clip_many_group(c, count, from, blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP);
}

// ---- attention -------------------------------------------------------------------------------------------------

constexpr int AT_DIM = 128, AT_PER = 3, AT_KEYS = 32, AT_THREADS = 128, AT_MAX_REL = 32;
constexpr int AT_KPAD = AT_DIM + 1;   // a key row in shared memory, padded so that 32 keys are 32 banks
constexpr int AT_MIN_SPLIT = 32;      // the split path's smallest split: short contexts still fill the card

struct AttnArgs {
    const float* q;
    const half* kc;
    const half* vc;
    Kvq8 kq;
    Kvq8 vq;
    InklingRel rel;
    const float* sinks;
    long long pos;
    int rows, n_head, kv_heads, window, ring;
    float scale;
    int split;   // the split path's keys a split (counted from position 0)
    int nslot;   // the split path's splits a token's partials have room for
    int i8_from;  // the split path's rows of splits of this many keys or more take attn_dec_i8_kernel (0: none)
    float* out;
    const float* gate;   // the split path's output gate: out = o * sigmoid(gate) (null: none)
    void* out_q8_1;      // the split path's output's Q8_1 blocks too (null: none)
    float* work;
};

template <bool Q8>
__device__ __forceinline__ float cache_value(const half* c, const Kvq8& q8, size_t i) {
    if constexpr (Q8) return (float) q8.codes[i] * __half2float(__ushort_as_half(q8.scales[i / 32]));
    else return __half2float(c[i]);
}

constexpr size_t attn_smem(int tq) {
    const int r = tq * AT_PER;
    return sizeof(float) * (size_t(r) * AT_DIM + AT_KEYS * AT_KPAD + AT_KEYS * AT_DIM + size_t(r) * AT_KEYS +
                            size_t(r) * AT_MAX_REL + 3 * r);
}

// The unsplit path: a block takes TQ tokens from blockIdx.z * TQ, the PER heads of group blockIdx.y, and all of
// the tokens' keys.  The keys go in tiles of AT_KEYS counted from position 0; a row's online softmax only takes
// the keys the row attends, so a key the row does not see changes nothing (its weight is exactly 0 and the row's
// max stays).
template <int TQ, bool KQ8, bool VQ8>
__global__ void __launch_bounds__(AT_THREADS) attn_kernel(AttnArgs a) {
    constexpr int R = TQ * AT_PER;
    constexpr int RPT = (R * AT_KEYS + AT_THREADS - 1) / AT_THREADS;   // scores a thread per tile
    extern __shared__ float smem[];
    float* Qs = smem;                        // [R][DIM]
    float* Ks = Qs + R * AT_DIM;             // [KEYS][KPAD]
    float* Vs = Ks + AT_KEYS * AT_KPAD;      // [KEYS][DIM]
    float* Ps = Vs + AT_KEYS * AT_DIM;       // [R][KEYS]
    float* Rs = Ps + R * AT_KEYS;            // [R][MAX_REL]
    float* Ms = Rs + R * AT_MAX_REL;         // [R]
    float* Ls = Ms + R;                      // [R]
    float* As = Ls + R;                      // [R]

    const int tid = threadIdx.x, lane = tid % WARP, warp = tid / WARP;
    const int g = blockIdx.y, t0 = blockIdx.z * TQ;
    const int nt = min(TQ, a.rows - t0);
    const int n_rel = a.rel.n_rel;
    const long long p0 = a.pos + t0, plast = p0 + nt - 1;
    const long long kb = a.window > 0 && p0 >= a.window ? p0 - a.window + 1 : 0, ke = plast + 1;

    // the queries, their relative-position projections; the rows' running max, sum and output
    for (int i = tid; i < R * AT_DIM; i += AT_THREADS) {
        const int r = i / AT_DIM, d = i % AT_DIM, t = r / AT_PER, h = g * AT_PER + r % AT_PER;
        Qs[i] = t < nt ? a.q[((size_t(t0) + t) * a.n_head + h) * AT_DIM + d] : 0.0f;
    }
    for (int i = tid; i < R * AT_MAX_REL; i += AT_THREADS) {
        const int r = i / AT_MAX_REL, j = i % AT_MAX_REL, t = r / AT_PER, h = g * AT_PER + r % AT_PER;
        Rs[i] = t < nt && j < n_rel ? a.rel.r[((size_t(t0) + t) * a.n_head + h) * n_rel + j] : 0.0f;
    }
    if (tid < R) {
        Ms[tid] = -INFINITY;
        Ls[tid] = 0.0f;
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    __syncthreads();

    for (long long kt = kb < ke ? kb / AT_KEYS * AT_KEYS : ke; kt < ke; kt += AT_KEYS) {
        // the tile's keys and values; a key outside [kb, ke) is zeros (its cache row may be anything)
        for (int j = 0; j < AT_KEYS; ++j) {
            const long long kp = kt + j;
            const bool live = kp >= kb && kp < ke;
            const long long row = a.ring > 0 ? kp % a.ring : kp;
            const size_t base = (size_t(row) * a.kv_heads + g) * AT_DIM + tid;
            Ks[j * AT_KPAD + tid] = live ? cache_value<KQ8>(a.kc, a.kq, base) : 0.0f;
            Vs[j * AT_DIM + tid] = live ? cache_value<VQ8>(a.vc, a.vq, base) : 0.0f;
        }
        __syncthreads();

        // scores: a thread has key `lane` of the tile and the rows warp, warp + 4, ...
        {
            float s[RPT];
#pragma unroll
            for (int i = 0; i < RPT; ++i) s[i] = 0.0f;
            const float* kr = Ks + lane * AT_KPAD;
#pragma unroll 4
            for (int d = 0; d < AT_DIM; ++d) {
                const float kv = kr[d];
#pragma unroll
                for (int i = 0; i < RPT; ++i) {
                    const int r = warp + i * (AT_THREADS / WARP);
                    if (r < R) s[i] += Qs[r * AT_DIM + d] * kv;
                }
            }
            const long long kp = kt + lane;
#pragma unroll
            for (int i = 0; i < RPT; ++i) {
                const int r = warp + i * (AT_THREADS / WARP);
                if (r >= R) continue;
                const int t = r / AT_PER;
                const long long p = p0 + t;
                const long long dist = p - kp;
                const bool seen = t < nt && kp >= kb && kp < ke && dist >= 0 && (a.window <= 0 || dist < a.window);
                float v = -INFINITY;
                if (seen) {
                    float bias = 0.0f;
                    if (dist < a.rel.extent) {
                        const float* tb = a.rel.table + size_t(dist) * n_rel;
                        const float* rr = Rs + r * AT_MAX_REL;
                        for (int j = 0; j < n_rel; ++j) bias += tb[j] * rr[j];
                    }
                    v = s[i] * a.scale + bias;
                }
                Ps[r * AT_KEYS + lane] = v;
            }
        }
        __syncthreads();

        // the online softmax: a warp a row
        for (int r = warp; r < R; r += AT_THREADS / WARP) {
            const float s = Ps[r * AT_KEYS + lane];
            const float m_old = Ms[r];
            const float m_new = fmaxf(m_old, warp_max(s));
            const float p = s == -INFINITY ? 0.0f : expf(s - m_new);
            const float sum = warp_sum(p);
            Ps[r * AT_KEYS + lane] = p;
            if (lane == 0) {
                const float alpha = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                As[r] = alpha;
                Ms[r] = m_new;
                Ls[r] = Ls[r] * alpha + sum;
            }
        }
        __syncthreads();

        // the values: a thread a dimension, every row
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] *= As[r];
        for (int j = 0; j < AT_KEYS; ++j) {
            const float v = Vs[j * AT_DIM + tid];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] += Ps[r * AT_KEYS + j] * v;
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int t = r / AT_PER, h = g * AT_PER + r % AT_PER;
        if (t >= nt) continue;
        const size_t tok = size_t(t0) + t;
        const float m = Ms[r], l = Ls[r];
        const float sink = a.sinks ? a.sinks[tok * a.n_head + h] : -INFINITY;
        const float mx = fmaxf(m, sink);
        const float f = m == -INFINITY ? 0.0f : expf(m - mx);
        const float denom = l * f + (sink == -INFINITY ? 0.0f : expf(sink - mx));
        a.out[(tok * a.n_head + h) * AT_DIM + tid] = acc[r] * f / denom;
    }
}

// The split path's rows: a call's tokens one after the other in one sequence (inkling_attend()), or each row a
// sequence's of its own with its own caches, position and splits (inkling_attend_seqs()).  AttnArgs::nslot is then
// the most splits of a row, and what a token's partials are strided by.
struct AttnRow {
    const half* kc;
    const half* vc;
    Kvq8 kq, vq;
    long long pos;
    int split, nslot;
};
struct AttnOne {
    __device__ AttnRow row(const AttnArgs& a, int t) const {
        return {a.kc, a.vc, a.kq, a.vq, a.pos + t, a.split, a.nslot};
    }
};
struct AttnAt {   // AttnOne's from the position on the card
    const long long* at;
    __device__ AttnRow row(const AttnArgs& a, int t) const {
        return {a.kc, a.vc, a.kq, a.vq, *at + t, a.split, a.nslot};
    }
};
struct AttnSeqs {
    AttnRow r[kInklingSeqRows];
    __device__ AttnRow row(const AttnArgs&, int t) const { return r[t]; }
};
// the rows of a call whose splits are of `from` keys or more
inline int rows_from(const AttnOne&, const AttnArgs& a, int ntok, int from) { return a.split >= from ? ntok : 0; }
inline int rows_from(const AttnAt&, const AttnArgs& a, int ntok, int from) { return a.split >= from ? ntok : 0; }
inline int rows_from(const AttnSeqs& s, const AttnArgs&, int ntok, int from) {
    int n = 0;
    for (int t = 0; t < ntok; ++t) n += s.r[t].split >= from;
    return n;
}

// The splits of a token's head merged, in the splits' order, with its sink.  A block a head of a token, a thread a
// value dimension.  A row of one split has its output from attn_dec_kernel.
template <class Rows>
__global__ void attn_join_kernel(AttnArgs a, Rows rows) {
    const int th = blockIdx.x, d = threadIdx.x;
    const int nslot = rows.row(a, th / a.n_head).nslot;
    if (nslot <= 1) return;
    const float* sinks = a.sinks;
    const float* w = a.work + size_t(th) * a.nslot * (AT_DIM + 2);
    float mx = sinks ? sinks[th] : -INFINITY;
    for (int s = 0; s < nslot; ++s) mx = fmaxf(mx, w[size_t(s) * (AT_DIM + 2)]);
    float denom = sinks ? expf(sinks[th] - mx) : 0.0f, sum = 0.0f;
    for (int s = 0; s < nslot; ++s) {
        const float* ws = w + size_t(s) * (AT_DIM + 2);
        if (ws[0] == -INFINITY) continue;
        const float f = expf(ws[0] - mx);
        denom += ws[1] * f;
        sum += ws[2 + d] * f;
    }
    const size_t i = size_t(th) * AT_DIM + d;
    const float o = sum / denom, y = a.gate ? o * sigmoidf(a.gate[i]) : o;
    a.out[i] = y;
    if (a.out_q8_1) fm::q8_1(y, static_cast<fm::Q81*>(a.out_q8_1) + i / 32);
}

// The split path's kernel: one token of one group, the keys of one split.  No tiles in shared memory and no
// barriers while it walks: each warp takes DEC_KEYS keys at a time (the block's keys go to the warps in turn,
// counted from the split's first key).  A key's score is the sum of 32 shares, a share four of its values' dot with
// a head's query (values 4j .. 4j + 3) and bias weight j, added up pairwise as a butterfly over 32 lanes would (j
// with j + 16, then 8 apart, 4, 2, 1).  A lane takes the four shares of one key that sum first (j, j + 8, j + 16,
// j + 24), so the four keys' scores are made at once by 8 lanes each with three shuffles a head.  Its values: a
// lane four of each key's, as the output is kept.  Each warp keeps its own running max, sum and output for the
// group's heads; the warps are merged in their order at the end, and the block's result is the token's output when
// its call has one split, else a partial for attn_join_kernel.  Nothing depends on the other tokens of the call.
constexpr int DEC_WARPS = 4, DEC_KEYS = 4, DEC_LANES = WARP / DEC_KEYS;   // DEC_LANES: a key's lanes for its score
constexpr int DEC_SHARES = WARP / DEC_LANES;                              // a lane's shares of a score
static_assert(DEC_WARPS * WARP == AT_DIM, "attn_dec_kernel's merge takes a thread a value dimension");
static_assert(AT_MAX_REL <= WARP, "a share has one bias weight at most");

// Four values of a cache from value i: Q8_0 codes times their block's scale, or halves.
template <bool Q8>
__device__ __forceinline__ void load4(const half* c, const Kvq8& q8, size_t i, float* x) {
    if constexpr (Q8) {
        const int codes = *reinterpret_cast<const int*>(q8.codes + i);
        const float d = __half2float(__ushort_as_half(q8.scales[i / 32]));
#pragma unroll
        for (int j = 0; j < 4; ++j) x[j] = (float) (signed char) (codes >> (8 * j)) * d;
    } else {
        const uint2 bits = *reinterpret_cast<const uint2*>(c + i);
        const __half2 a = *reinterpret_cast<const __half2*>(&bits.x), b = *reinterpret_cast<const __half2*>(&bits.y);
        x[0] = __low2float(a), x[1] = __high2float(a), x[2] = __low2float(b), x[3] = __high2float(b);
    }
}

template <bool KQ8, bool VQ8, class Rows>
__global__ void __launch_bounds__(DEC_WARPS * WARP, 5) attn_dec_kernel(AttnArgs a, Rows rows) {
    __shared__ float sm_m[DEC_WARPS][AT_PER], sm_l[DEC_WARPS][AT_PER];
    __shared__ float sm_o[DEC_WARPS][AT_PER][AT_DIM];
    __shared__ __align__(16) float sm_q[AT_PER][AT_DIM];
    const int lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
    const int sub = lane % DEC_LANES, mine = lane / DEC_LANES;   // the score's lanes: key `mine` of a step
    const int g = blockIdx.y, t = blockIdx.z, n_rel = a.rel.n_rel;
    const AttnRow r = rows.row(a, t);
    if ((int) blockIdx.x >= r.nslot) return;   // a row with fewer splits than the call's most
    if (a.i8_from > 0 && r.split >= a.i8_from) return;   // attn_dec_i8_kernel's row
    const int p = int(r.pos);
    const int lo = a.window > 0 && p >= a.window ? p - a.window + 1 : 0;
    const int s = lo / r.split + blockIdx.x;
    const int kb = max(lo, s * r.split), ke = min(p + 1, (s + 1) * r.split);
    // the queries in shared memory; share j = sub + DEC_LANES * k of each head's bias weight
    float rr[AT_PER][DEC_SHARES], m[AT_PER], l[AT_PER], acc[AT_PER][4];
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        sm_q[h][threadIdx.x] = a.q[th * AT_DIM + threadIdx.x];
#pragma unroll
        for (int k = 0; k < DEC_SHARES; ++k) {
            const int j = sub + DEC_LANES * k;
            rr[h][k] = j < n_rel ? a.rel.r[th * n_rel + j] : 0.0f;
        }
        m[h] = -INFINITY, l[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.0f;
    }
    // A step of the warp's DEC_KEYS keys from k0, at cache rows from row0.  A full step (its keys all live and not
    // across a ring's end: all but a split's last) goes without those checks, and a far one (its keys all past the
    // bias's extent: most of a long context) loads no bias weights (theirs are 0, as the general step finds).
    const size_t key_stride = size_t(a.kv_heads) * AT_DIM;
    auto step = [&](int k0, int row0, auto full, auto far) {
        constexpr bool FULL = decltype(full)::value, FAR = decltype(far)::value;
        size_t at[DEC_KEYS];   // each key's row
        bool live[DEC_KEYS];
#pragma unroll
        for (int c = 0; c < DEC_KEYS; ++c) {
            if constexpr (FULL) {
                at[c] = (size_t(row0) * a.kv_heads + g) * AT_DIM + c * key_stride;
                live[c] = true;
            } else {
                int row = row0 + c;
                if (a.ring > 0 && row >= a.ring) row -= a.ring;
                at[c] = (size_t(row) * a.kv_heads + g) * AT_DIM;
                live[c] = k0 + c < ke;
            }
        }
        // the values: a lane four of each key
        float vx[DEC_KEYS][4];
#pragma unroll
        for (int c = 0; c < DEC_KEYS; ++c) {
            if (live[c]) {
                load4<VQ8>(r.vc, r.vq, at[c] + 4 * lane, vx[c]);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) vx[c][i] = 0.0f;
            }
        }
        // the score of key `mine`: this lane's four shares of it, as the butterfly's first two rounds add them
        float sc[AT_PER];
        {
            const int kp = k0 + mine, dist = p - kp;
            const bool here = FULL || kp < ke;
            const bool near = !FAR && here && dist < a.rel.extent;
            size_t base = at[0];
#pragma unroll
            for (int c = 1; c < DEC_KEYS; ++c)
                if (mine == c) base = at[c];
            float sh[AT_PER][DEC_SHARES];
#pragma unroll
            for (int k = 0; k < DEC_SHARES; ++k) {
                const int j = sub + DEC_LANES * k;
                float kx[4];
                if (here) {
                    load4<KQ8>(r.kc, r.kq, base + 4 * j, kx);
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) kx[i] = 0.0f;
                }
                const float tb = near && j < n_rel ? a.rel.table[size_t(dist) * n_rel + j] : 0.0f;
#pragma unroll
                for (int h = 0; h < AT_PER; ++h) {
                    const float4 qv = *reinterpret_cast<const float4*>(&sm_q[h][4 * j]);
                    float d = __fmul_rn(qv.y, kx[1]);   // (the first two products: the old kernel's contraction)
                    d = __fmaf_rn(qv.x, kx[0], d);
                    d = __fmaf_rn(qv.z, kx[2], d);
                    d = __fmaf_rn(qv.w, kx[3], d);
                    sh[h][k] = __fmaf_rn(d, a.scale, __fmul_rn(tb, rr[h][k]));
                }
            }
#pragma unroll
            for (int h = 0; h < AT_PER; ++h) {
                // shares j and j + 16, j + 8 and j + 24, then those two
                float v = __fadd_rn(__fadd_rn(sh[h][0], sh[h][2]), __fadd_rn(sh[h][1], sh[h][3]));
#pragma unroll
                for (int o = DEC_LANES / 2; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o, WARP));
                sc[h] = v;
            }
        }
        float s[AT_PER][DEC_KEYS];
#pragma unroll
        for (int h = 0; h < AT_PER; ++h)
#pragma unroll
            for (int c = 0; c < DEC_KEYS; ++c) s[h][c] = __shfl_sync(0xffffffffu, sc[h], c * DEC_LANES, WARP);
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < DEC_KEYS; ++c)
                if (live[c]) mx = fmaxf(mx, s[h][c]);
            const float m_new = fmaxf(m[h], mx);
            if (m_new == -INFINITY) continue;   // no live key yet
            const float alpha = m[h] == -INFINITY ? 0.0f : expf(m[h] - m_new);
            float pc[DEC_KEYS], sum = 0.0f;
#pragma unroll
            for (int c = 0; c < DEC_KEYS; ++c) {
                pc[c] = live[c] ? expf(s[h][c] - m_new) : 0.0f;
                sum = __fadd_rn(sum, pc[c]);
            }
            m[h] = m_new;
            l[h] = __fmaf_rn(l[h], alpha, sum);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = __fmaf_rn(acc[h][i], alpha, __fmul_rn(pc[0], vx[0][i]));
#pragma unroll
                for (int c = 1; c < DEC_KEYS; ++c) y = __fmaf_rn(pc[c], vx[c][i], y);
                acc[h][i] = y;
            }
        }
    };
    __syncthreads();   // the queries
    // a ring's wraps are found by comparing, not dividing, as the keys go on
    int row0 = kb + warp * DEC_KEYS;
    if (a.ring > 0) row0 %= a.ring;
    for (int k0 = kb + warp * DEC_KEYS; k0 < ke; k0 += DEC_WARPS * DEC_KEYS) {
        const bool full = k0 + DEC_KEYS <= ke && (a.ring <= 0 || row0 + DEC_KEYS <= a.ring);
        if (full && p - (k0 + DEC_KEYS - 1) >= a.rel.extent) step(k0, row0, std::true_type{}, std::true_type{});
        else if (full) step(k0, row0, std::true_type{}, std::false_type{});
        else step(k0, row0, std::false_type{}, std::false_type{});
        row0 += DEC_WARPS * DEC_KEYS;
        if (a.ring > 0 && row0 >= a.ring) row0 %= a.ring;
    }
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        if (lane == 0) sm_m[warp][h] = m[h], sm_l[warp][h] = l[h];
#pragma unroll
        for (int i = 0; i < 4; ++i) sm_o[warp][h][4 * lane + i] = acc[h][i];
    }
    __syncthreads();
    const int d = threadIdx.x;   // a thread a value dimension
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < DEC_WARPS; ++w) M = fmaxf(M, sm_m[w][h]);
        float L = 0.0f, O = 0.0f;
#pragma unroll
        for (int w = 0; w < DEC_WARPS; ++w) {
            const float f = sm_m[w][h] == -INFINITY ? 0.0f : expf(sm_m[w][h] - M);
            L += sm_l[w][h] * f;
            O += sm_o[w][h][d] * f;
        }
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        if (r.nslot > 1) {
            float* wp = a.work + (th * a.nslot + blockIdx.x) * (AT_DIM + 2);
            if (d == 0) wp[0] = M, wp[1] = L;
            wp[2 + d] = O;
        } else {
            const float sink = a.sinks ? a.sinks[th] : -INFINITY;
            const float mx = fmaxf(M, sink);
            const float f = M == -INFINITY ? 0.0f : expf(M - mx);
            const float denom = L * f + (sink == -INFINITY ? 0.0f : expf(sink - mx));
            const size_t i = th * AT_DIM + d;
            const float o = O * f / denom, y = a.gate ? o * sigmoidf(a.gate[i]) : o;
            a.out[i] = y;
            if (a.out_q8_1) fm::q8_1(y, static_cast<fm::Q81*>(a.out_q8_1) + i / 32);
        }
    }
}

// The split path over Q8_0 keys and values with the queries quantized too (inkling_attend()'s int8_dot) for a
// full layer's rows of splits of I8_MIN_SPLIT keys or more: the same splits, partials and join as attn_dec_kernel,
// other arithmetic.  A window layer's splits are 32 keys (its window's), so it never comes here: no window, no
// ring.  A key's 128 values go to 8 lanes, 16 each (one 16-byte load of its codes), so a warp reads 4 keys at once;
// each 8-lane group takes I8_KEYS keys a step (the block's keys go to the warps in turn, counted from the split's
// first key, and a warp's to its groups in turn) and keeps its own running max, sum and output for the group's
// heads.  A score is the sum over the key's 32-value blocks of (its query's codes . the key's codes) * the query
// block's delta * the key block's, the dot of 16 values four __dp4a; the query is quantized here about as
// llama.cpp's flash-attention kernels quantize theirs for a quantized K (delta = the block's largest |value| / 127,
// codes the values times 1 / delta rounded to the nearest, where llama.cpp divides).  The groups of a warp are
// merged in their order at the end (lanes 8 and 16 apart), then the warps in theirs through shared memory.
constexpr int I8_WARPS = 4, I8_LANES = 8, I8_KEYS = 2;
// the shortest split it takes (a full layer's room of 32K): attn_dec_kernel was the faster on an RTX 2070 Super for
// window layers (a token at 32767: 19.3 us against 26.3 here) and short rooms (at 4096: 52.3 against 55.0), this
// one from 32K (at 32767: 254 us against 286)
constexpr int I8_MIN_SPLIT = 512;
constexpr int I8_DIMS = AT_DIM / I8_LANES;                        // a lane's values of a key: half a block
constexpr int I8_WARP_KEYS = WARP / I8_LANES * I8_KEYS;           // a warp's keys a step
constexpr int I8_STEP = I8_WARPS * I8_WARP_KEYS;                  // the block's
constexpr int I8_RELS = (AT_MAX_REL + I8_LANES - 1) / I8_LANES;   // a lane's relative-position weights a head
static_assert(I8_WARPS * WARP == AT_DIM, "attn_dec_i8_kernel's merge takes a thread a value dimension");
static_assert(I8_DIMS * 2 == 32, "a lane's values are half of a Q8_0 block");

// Four codes, unscaled, as floats: a code's byte offset by 128 under the bits of 2^23, less 2^23 + 128 (exact).
__device__ __forceinline__ void codes4(int w, float* x) {
    const unsigned u = unsigned(w) ^ 0x80808080u;
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __uint_as_float(__byte_perm(u, 0x4b000000u, 0x7540 + j)) - 8388736.0f;
}

template <class Rows>
__global__ void __launch_bounds__(I8_WARPS * WARP) attn_dec_i8_kernel(AttnArgs a, Rows rows) {
    __shared__ float sm_m[I8_WARPS][AT_PER], sm_l[I8_WARPS][AT_PER];
    __shared__ float sm_o[I8_WARPS][AT_PER][AT_DIM];
    const int lane = threadIdx.x % WARP, warp = threadIdx.x / WARP, sub = lane % I8_LANES;
    const int g = blockIdx.y, t = blockIdx.z, n_rel = a.rel.n_rel;
    const AttnRow r = rows.row(a, t);
    if ((int) blockIdx.x >= r.nslot || r.split < a.i8_from) return;   // fewer splits, or attn_dec_kernel's row
    const int p = int(r.pos);
    const int kb = blockIdx.x * r.split, ke = min(p + 1, kb + r.split);
    // this lane's 16 query values of each head as codes, their block's delta times the score's scale
    int qc[AT_PER][I8_DIMS / 4];
    float qd[AT_PER], rr[AT_PER][I8_RELS], m[AT_PER], l[AT_PER], acc[AT_PER][I8_DIMS];
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        float x[I8_DIMS], amax = 0.0f;
#pragma unroll
        for (int i = 0; i < I8_DIMS / 4; ++i) {
            const float4 v = *reinterpret_cast<const float4*>(a.q + th * AT_DIM + I8_DIMS * sub + 4 * i);
            x[4 * i] = v.x, x[4 * i + 1] = v.y, x[4 * i + 2] = v.z, x[4 * i + 3] = v.w;
        }
#pragma unroll
        for (int i = 0; i < I8_DIMS; ++i) amax = fmaxf(amax, fabsf(x[i]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, WARP));   // the block's other half
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
#pragma unroll
        for (int i = 0; i < I8_DIMS / 4; ++i) {
            int w = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) w |= (int(roundf(x[4 * i + j] * id)) & 0xff) << (8 * j);
            qc[h][i] = w;
        }
        qd[h] = d * a.scale;
#pragma unroll
        for (int i = 0; i < I8_RELS; ++i) {
            const int j = sub + I8_LANES * i;
            rr[h][i] = j < n_rel ? a.rel.r[th * n_rel + j] : 0.0f;
        }
        m[h] = -INFINITY, l[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < I8_DIMS; ++i) acc[h][i] = 0.0f;
    }
    int k0 = kb + warp * I8_WARP_KEYS + lane / I8_LANES * I8_KEYS;   // this lane's first key
    for (int w0 = kb + warp * I8_WARP_KEYS; w0 < ke; w0 += I8_STEP) {
        int4 kc[I8_KEYS], vc[I8_KEYS];
        float kd[I8_KEYS], vd[I8_KEYS];
        bool live[I8_KEYS];
#pragma unroll
        for (int c = 0; c < I8_KEYS; ++c) {
            live[c] = k0 + c < ke;
            const size_t base = (size_t(k0 + c) * a.kv_heads + g) * AT_DIM + I8_DIMS * sub;
            if (live[c]) {
                kc[c] = *reinterpret_cast<const int4*>(r.kq.codes + base);
                vc[c] = *reinterpret_cast<const int4*>(r.vq.codes + base);
                kd[c] = __half2float(__ushort_as_half(r.kq.scales[base / 32]));
                vd[c] = __half2float(__ushort_as_half(r.vq.scales[base / 32]));
            } else {
                kc[c] = vc[c] = make_int4(0, 0, 0, 0);
                kd[c] = vd[c] = 0.0f;
            }
        }
        // a lane's share of each score: its 16 values' dot, scaled, and its share of the bias for keys near enough
        float sc[AT_PER][I8_KEYS];
#pragma unroll
        for (int c = 0; c < I8_KEYS; ++c)
#pragma unroll
            for (int h = 0; h < AT_PER; ++h) {
                int dot = STRATA_DP4A(qc[h][0], kc[c].x, 0);
                dot = STRATA_DP4A(qc[h][1], kc[c].y, dot);
                dot = STRATA_DP4A(qc[h][2], kc[c].z, dot);
                dot = STRATA_DP4A(qc[h][3], kc[c].w, dot);
                sc[h][c] = (float) dot * (qd[h] * kd[c]);
            }
        if (p - (k0 + I8_KEYS - 1) < a.rel.extent) {   // no shuffles in here: the groups may differ
#pragma unroll
            for (int c = 0; c < I8_KEYS; ++c) {
                const int dist = p - (k0 + c);
                if (!live[c] || dist >= a.rel.extent) continue;
                const float* tb = a.rel.table + size_t(dist) * n_rel;
#pragma unroll
                for (int i = 0; i < I8_RELS; ++i) {
                    const int j = sub + I8_LANES * i;
                    if (j < n_rel) {
                        const float w = tb[j];
#pragma unroll
                        for (int h = 0; h < AT_PER; ++h) sc[h][c] += w * rr[h][i];
                    }
                }
            }
        }
#pragma unroll
        for (int o = I8_LANES / 2; o > 0; o >>= 1)
#pragma unroll
            for (int h = 0; h < AT_PER; ++h)
#pragma unroll
                for (int c = 0; c < I8_KEYS; ++c) sc[h][c] += __shfl_xor_sync(0xffffffffu, sc[h][c], o, WARP);
        // the online softmax, then the values: each key's 16 values converted once for the three heads
        float alpha[AT_PER], pv[AT_PER][I8_KEYS];
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            float mx = -INFINITY;
#pragma unroll
            for (int c = 0; c < I8_KEYS; ++c)
                if (live[c]) mx = fmaxf(mx, sc[h][c]);
            const float m_new = fmaxf(m[h], mx);
            alpha[h] = m[h] == -INFINITY ? 0.0f : expf(m[h] - m_new);
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < I8_KEYS; ++c) {
                const float pc = live[c] ? expf(sc[h][c] - m_new) : 0.0f;   // m_new is finite with a live key
                sum += pc;
                pv[h][c] = pc * vd[c];
            }
            m[h] = m_new;
            l[h] = l[h] * alpha[h] + sum;
        }
#pragma unroll
        for (int h = 0; h < AT_PER; ++h)
#pragma unroll
            for (int i = 0; i < I8_DIMS; ++i) acc[h][i] *= alpha[h];
#pragma unroll
        for (int c = 0; c < I8_KEYS; ++c) {
            float x[I8_DIMS];
            codes4(vc[c].x, x), codes4(vc[c].y, x + 4), codes4(vc[c].z, x + 8), codes4(vc[c].w, x + 12);
#pragma unroll
            for (int h = 0; h < AT_PER; ++h)
#pragma unroll
                for (int i = 0; i < I8_DIMS; ++i) acc[h][i] += pv[h][c] * x[i];
        }
        k0 += I8_STEP;
    }
    // the warp's groups merged: group 0 with 1 and 2 with 3, then the pairs (the earlier group's terms first)
#pragma unroll
    for (int o = I8_LANES; o < WARP; o <<= 1)
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            const float mo = __shfl_xor_sync(0xffffffffu, m[h], o, WARP);
            const float lo_ = __shfl_xor_sync(0xffffffffu, l[h], o, WARP);
            const bool first = (lane & o) == 0;
            const float M = fmaxf(m[h], mo);
            const float fa = m[h] == -INFINITY ? 0.0f : expf(m[h] - M);
            const float fb = mo == -INFINITY ? 0.0f : expf(mo - M);
            const float f0 = first ? fa : fb, f1 = first ? fb : fa;
            const float l0 = first ? l[h] : lo_, l1 = first ? lo_ : l[h];
            l[h] = l0 * f0 + l1 * f1;
#pragma unroll
            for (int i = 0; i < I8_DIMS; ++i) {
                const float ao = __shfl_xor_sync(0xffffffffu, acc[h][i], o, WARP);
                const float a0 = first ? acc[h][i] : ao, a1 = first ? ao : acc[h][i];
                acc[h][i] = a0 * f0 + a1 * f1;
            }
            m[h] = M;
        }
    if (lane < I8_LANES) {
#pragma unroll
        for (int h = 0; h < AT_PER; ++h) {
            if (lane == 0) sm_m[warp][h] = m[h], sm_l[warp][h] = l[h];
#pragma unroll
            for (int i = 0; i < I8_DIMS; ++i) sm_o[warp][h][I8_DIMS * sub + i] = acc[h][i];
        }
    }
    __syncthreads();
    const int d = threadIdx.x;   // a thread a value dimension
#pragma unroll
    for (int h = 0; h < AT_PER; ++h) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < I8_WARPS; ++w) M = fmaxf(M, sm_m[w][h]);
        float L = 0.0f, O = 0.0f;
#pragma unroll
        for (int w = 0; w < I8_WARPS; ++w) {
            const float f = sm_m[w][h] == -INFINITY ? 0.0f : expf(sm_m[w][h] - M);
            L += sm_l[w][h] * f;
            O += sm_o[w][h][d] * f;
        }
        const size_t th = size_t(t) * a.n_head + g * AT_PER + h;
        if (r.nslot > 1) {
            float* wp = a.work + (th * a.nslot + blockIdx.x) * (AT_DIM + 2);
            if (d == 0) wp[0] = M, wp[1] = L;
            wp[2 + d] = O;
        } else {
            const float sink = a.sinks ? a.sinks[th] : -INFINITY;
            const float mx = fmaxf(M, sink);
            const float f = M == -INFINITY ? 0.0f : expf(M - mx);
            const float denom = L * f + (sink == -INFINITY ? 0.0f : expf(sink - mx));
            const size_t i = th * AT_DIM + d;
            const float o = O * f / denom, y = a.gate ? o * sigmoidf(a.gate[i]) : o;
            a.out[i] = y;
            if (a.out_q8_1) fm::q8_1(y, static_cast<fm::Q81*>(a.out_q8_1) + i / 32);
        }
    }
}

// ---- attention on the tensor cores (a prompt batch) -------------------------------------------------------------

// The tensor cores' m16n8k16 step and ldmatrix (kernels.cu's): Turing runs the step as two of k = 8 on the same
// registers.  AMD builds and cards before Turing have neither; there the host never launches attn_mma_kernel
// (mma_attention()).
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750)
#define INKLING_MMA 0
#else
#define INKLING_MMA 1
#endif

#if INKLING_MMA
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t half2_bits(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
#endif

// attn_mma_kernel's shape: MMA_WARPS warps of 16 query rows, row r of the block token r / 3 and head r % 3 of the
// group (MMA_TOKENS tokens; the last row of the last warp is unused), keys MMA_KEYS at a time.
constexpr int MMA_WARPS = 4, MMA_KEYS = 32, MMA_TOKENS = 16 * MMA_WARPS / AT_PER;
constexpr int MMA_KROW = AT_DIM + 8;                                // a key or value row in shared memory, halves
constexpr int MMA_BIAS = MMA_KEYS + 8;                              // a warp's distances a tile (5 tokens + keys)
constexpr int MMA_TROW = 16 + 8;                                    // a row of the bias table, halves
constexpr int MMA_SLICE = 16 * (MMA_WARPS - 1) / AT_PER + MMA_BIAS;  // the block's distances a tile
static_assert(16 * MMA_WARPS / AT_PER * AT_PER >= 16 * MMA_WARPS - 2, "attn_mma_kernel: at most 2 unused rows");

// The unsplit path on the tensor cores, the way kernels.cu's attn_flash128_kernel goes: the queries (scaled to log2
// units and rounded to halves) stay in registers, a tile's keys and values go to shared memory as halves (a Q8_0
// cache's codes times their scales, rounded), S = Q K^T and O += P V are m16n8k16 steps that sum floats, and each
// row keeps its online softmax in floats.  The relative-position bias is a product too: the warp's rows' R (16
// values, rounded to halves) times the table's rows of the distances the tile spans (halves), a warp's 16 rows by
// MMA_BIAS distances, put in shared memory as floats and read back at each score's distance.  Only the tiles
// that hold a distance below the extent compute it.  The tiles are counted from position 0, so a row's sums do
// not depend on the other rows of the block.  The sink is added at the end, as one more weight of no value.
template <bool KQ8, bool VQ8>
__global__ void __launch_bounds__(MMA_WARPS * WARP) attn_mma_kernel(AttnArgs a) {
#if INKLING_MMA
    __shared__ __align__(16) half ks[MMA_KEYS][MMA_KROW], vs[MMA_KEYS][MMA_KROW];
    __shared__ __align__(16) half slice[MMA_SLICE][MMA_TROW];
    __shared__ __align__(16) float bias[MMA_WARPS][16][MMA_BIAS];
    constexpr float LOG2E = 1.44269504f;
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, gid = lane / 4, tig = lane % 4;
    const int g = blockIdx.y, t0 = (gridDim.x - 1 - blockIdx.x) * MMA_TOKENS;   // the longest blocks first
    const int nt = min(MMA_TOKENS, a.rows - t0), wt0 = 16 * w / AT_PER;
    const long long p0 = a.pos + t0;
    int tok[2], head[2];   // the thread's rows gid and gid + 8 of the warp's, as tokens of the block
    bool row_ok[2];
    for (int i = 0; i < 2; ++i) {
        const int r = 16 * w + gid + 8 * i;
        tok[i] = r / AT_PER;
        head[i] = g * AT_PER + r % AT_PER;
        row_ok[i] = tok[i] < nt;
    }
    uint32_t qa[8][4], ra[4];   // a block of 16 values a step, in log2 units; R the same
    for (int i = 0; i < 2; ++i) {
        const size_t th = (size_t(t0) + tok[i]) * a.n_head + head[i];
        for (int u = 0; u < 2; ++u) {   // values 2 tig.. and 8 + 2 tig.. of each block of 16
            for (int k = 0; k < 8; ++k) {
                float2 x = make_float2(0.0f, 0.0f);
                if (row_ok[i]) x = *reinterpret_cast<const float2*>(a.q + th * AT_DIM + 16 * k + 8 * u + 2 * tig);
                qa[k][i + 2 * u] = half2_bits(x.x * a.scale * LOG2E, x.y * a.scale * LOG2E);
            }
            const float* rr = a.rel.r + th * 16 + 8 * u + 2 * tig;
            ra[i + 2 * u] = row_ok[i] ? half2_bits(rr[0] * LOG2E, rr[1] * LOG2E) : 0u;
        }
    }
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = p0 + nt, lo = a.window > 0 ? max(0LL, p0 - a.window + 1) : 0;
    for (long long k0 = lo / MMA_KEYS * MMA_KEYS; k0 < hi; k0 += MMA_KEYS) {
        __syncthreads();   // the last tile's keys, values and bias are read
        // the tile's keys and values, 8 values a thread at a time; positions outside [lo, hi) are zeros (their
        // cache rows may be anything), which no weight reads
        for (int i = threadIdx.x; i < MMA_KEYS * AT_DIM / 8; i += MMA_WARPS * WARP) {
            const int r = i / (AT_DIM / 8), c = 8 * (i % (AT_DIM / 8));
            const long long key = k0 + r;
            uint4 kx = make_uint4(0, 0, 0, 0), vx = kx;
            if (key >= lo && key < hi) {
                const size_t at = (size_t(a.ring > 0 ? key % a.ring : key) * a.kv_heads + g) * AT_DIM + c;
                const auto load = [&](const half* f16, const Kvq8& q8, bool quant) {
                    if (!quant) return *reinterpret_cast<const uint4*>(f16 + at);
                    const uint2 codes = *reinterpret_cast<const uint2*>(q8.codes + at);
                    const float d = __half2float(__ushort_as_half(q8.scales[at / 32]));
                    uint32_t h[4];
                    for (int j = 0; j < 4; ++j) {
                        const uint32_t word = j < 2 ? codes.x : codes.y, sh = 16 * (j % 2);
                        h[j] = half2_bits((float) (signed char) (word >> sh) * d,
                                          (float) (signed char) (word >> (sh + 8)) * d);
                    }
                    return make_uint4(h[0], h[1], h[2], h[3]);
                };
                kx = load(a.kc, a.kq, KQ8);
                vx = load(a.vc, a.vq, VQ8);
            }
            *reinterpret_cast<uint4*>(&ks[r][c]) = kx;
            *reinterpret_cast<uint4*>(&vs[r][c]) = vx;
        }
        // the table's rows of the block's distances d = p0 - (k0 + MMA_KEYS - 1) + e, zeros past the extent and
        // before 0 (the padded row, and keys after the token, which the mask removes)
        const long long d0 = p0 - (k0 + MMA_KEYS - 1);
        const bool near = d0 < a.rel.extent;
        if (near)
            for (int i = threadIdx.x; i < MMA_SLICE * 8; i += MMA_WARPS * WARP) {
                const int e = i / 8, j = 2 * (i % 8);
                const long long d = d0 + e;
                float2 x = make_float2(0.0f, 0.0f);
                if (d >= 0 && d < a.rel.extent) x = *reinterpret_cast<const float2*>(a.rel.table + d * 16 + j);
                *reinterpret_cast<uint32_t*>(&slice[e][j]) = half2_bits(x.x, x.y);
            }
        __syncthreads();
        float s[MMA_KEYS / 8][4];
        for (int n = 0; n < MMA_KEYS / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ldmatrix_x4(b, &ks[8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                mma16816(s[n], qa[k], b);
                mma16816(s[n], qa[k + 1], b + 2);
            }
        }
        if (near) {
            // the warp's rows times its distances: column e is distance p0 + wt0 - (k0 + MMA_KEYS - 1) + e
            for (int n = 0; n < MMA_BIAS / 8; ++n) {
                const half* tr = &slice[wt0 + 8 * n + gid][2 * tig];
                const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(tr),
                                       *reinterpret_cast<const uint32_t*>(tr + 8)};
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma16816(c, ra, b);
                *reinterpret_cast<float2*>(&bias[w][gid][8 * n + 2 * tig]) = make_float2(c[0], c[1]);
                *reinterpret_cast<float2*>(&bias[w][gid + 8][8 * n + 2 * tig]) = make_float2(c[2], c[3]);
            }
            __syncwarp();
        }
        for (int i = 0; i < 2; ++i) {
            const long long own = p0 + tok[i];   // the row's token attends positions (own - window, own]
            const int e0 = tok[i] - wt0 + MMA_KEYS - 1;
            float mx = -INFINITY;
            for (int n = 0; n < MMA_KEYS / 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const int c = 8 * n + 2 * tig + j % 2;
                    const long long key = k0 + c;
                    if (!row_ok[i] || key > own || (a.window > 0 && key <= own - a.window))
                        s[n][j] = -INFINITY;
                    else if (near)
                        s[n][j] += bias[w][gid + 8 * i][e0 - c];
                    mx = fmaxf(mx, s[n][j]);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = exp2f(m[i] - base);
            m[i] = now;
            l[i] *= f;
            for (int n = 0; n < 16; ++n) {
                o[n][2 * i] *= f;
                o[n][2 * i + 1] *= f;
            }
            for (int n = 0; n < MMA_KEYS / 8; ++n)
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = exp2f(s[n][j] - base);
                    l[i] += s[n][j];
                }
        }
        for (int k = 0; k < MMA_KEYS / 16; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {half2_bits(s[2 * k][0], s[2 * k][1]), half2_bits(s[2 * k][2], s[2 * k][3]),
                                    half2_bits(s[2 * k + 1][0], s[2 * k + 1][1]),
                                    half2_bits(s[2 * k + 1][2], s[2 * k + 1][3])};
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ldmatrix_x4(b, &vs[16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                mma16816(o[n], pa, b);
                mma16816(o[n + 1], pa, b + 2);
            }
        }
    }
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (!row_ok[i]) continue;
        const size_t th = (size_t(t0) + tok[i]) * a.n_head + head[i];
        float f = 1.0f;
        if (a.sinks) {   // the sink, in log2 units like the scores
            const float sink = a.sinks[th] * LOG2E, now = fmaxf(m[i], sink);
            f = exp2f(m[i] - now);
            l[i] = l[i] * f + exp2f(sink - now);
        }
        float* y = a.out + th * AT_DIM + 2 * tig;
        for (int n = 0; n < 16; ++n)
            *reinterpret_cast<float2*>(y + 8 * n) = make_float2(o[n][2 * i] * f / l[i], o[n][2 * i + 1] * f / l[i]);
    }
#else
    (void) a;   // never launched without the tensor cores (mma_attention())
#endif
}

// Whether attn_mma_kernel runs as such here: built for Turing or newer, and not from an older card's PTX compiled
// for this one (whose instructions are that card's).
bool mma_attention() {
#if defined(__HIPCC__)
    return false;
#else
    cudaFuncAttributes fa{};
    return cudaFuncGetAttributes(&fa, attn_mma_kernel<false, false>) == cudaSuccess && fa.ptxVersion >= 75;
#endif
}

template <int TQ, bool KQ8, bool VQ8>
void launch_attn(const AttnArgs& a, dim3 g, cudaStream_t st) {
    constexpr size_t smem = attn_smem(TQ);
    if constexpr (smem > 48 * 1024) {
        // the device's own setting: asked on whichever card the call is for
        const cudaError_t e =
            cudaFuncSetAttribute(attn_kernel<TQ, KQ8, VQ8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("inkling_attend: shared memory: ") + cudaGetErrorString(e));
    }
    attn_kernel<TQ, KQ8, VQ8><<<g, AT_THREADS, smem, st>>>(a);
}

template <int TQ>
void launch_attn_q(const AttnArgs& a, dim3 g, cudaStream_t st, bool kq8, bool vq8) {
    if (kq8 && vq8) launch_attn<TQ, true, true>(a, g, st);
    else if (kq8) launch_attn<TQ, true, false>(a, g, st);
    else if (vq8) launch_attn<TQ, false, true>(a, g, st);
    else launch_attn<TQ, false, false>(a, g, st);
}

int split_keys(long long room) {
    const long long s = (room + kInklingMaxSplits - 1) / kInklingMaxSplits;
    return (int) std::max<long long>(AT_MIN_SPLIT, (s + AT_KEYS - 1) / AT_KEYS * AT_KEYS);
}

// The split path's keys a split and splits of a token whose positions + 1 are at most span_room.
void splits_of(long long span_room, int window, int& split, int& nslot) {
    // a window layer's tokens see `window` keys whatever the room: its splits are sized for those
    split = split_keys(window > 0 ? std::min<long long>(span_room, window) : span_room);
    long long n = (span_room + split - 1) / split;
    if (window > 0) n = std::min<long long>(n, (window + split - 1) / split + 1);
    nslot = (int) std::max<long long>(n, 1);
}

template <class Rows>
void launch_dec(const AttnArgs& a, const Rows& rows, int ntok, bool kq, bool vq, bool int8_dot, cudaStream_t st) {
    const dim3 gd((unsigned) a.nslot, a.kv_heads, ntok);
    if (int8_dot && kq && vq && a.window <= 0) {   // a full layer's rows of long splits by __dp4a, the others not
        AttnArgs b = a;
        b.i8_from = I8_MIN_SPLIT;
        const int big = rows_from(rows, a, ntok, I8_MIN_SPLIT);
        if (big > 0) attn_dec_i8_kernel<Rows><<<gd, I8_WARPS * WARP, 0, st>>>(b, rows);
        if (big < ntok) attn_dec_kernel<true, true, Rows><<<gd, DEC_WARPS * WARP, 0, st>>>(b, rows);
    } else if (kq && vq) attn_dec_kernel<true, true, Rows><<<gd, DEC_WARPS * WARP, 0, st>>>(a, rows);
    else if (kq) attn_dec_kernel<true, false, Rows><<<gd, DEC_WARPS * WARP, 0, st>>>(a, rows);
    else if (vq) attn_dec_kernel<false, true, Rows><<<gd, DEC_WARPS * WARP, 0, st>>>(a, rows);
    else attn_dec_kernel<false, false, Rows><<<gd, DEC_WARPS * WARP, 0, st>>>(a, rows);
    check_launch("inkling_attend");
    if (a.nslot > 1) {   // a row of one split: the block wrote the token's output itself
        attn_join_kernel<Rows><<<ntok * a.n_head, AT_DIM, 0, st>>>(a, rows);
        check_launch("inkling_attend (join)");
    }
}

// ---- convolution -----------------------------------------------------------------------------------------------

// The convolution of one token of one channel from its inputs s0 (oldest) .. s2 and v (the token's), in
// upstream's order: ssm_conv's sum, then the four input-dependent taps lag 0..3, then the residual and the scale.
__device__ __forceinline__ float conv_one(float s0, float s1, float s2, float v, const float* wc,
                                          const float* tp, float scale) {
    float y = 0.0f;
    y += s0 * wc[0];
    y += s1 * wc[1];
    y += s2 * wc[2];
    y += v * wc[3];
    if (tp) {
        y += v * tp[0];
        y += s2 * tp[1];
        y += s1 * tp[2];
        y += s0 * tp[3];
    }
    return (v + y) * scale;
}

// The convolution's scale: 1/3 with taps, 1/sqrt(5) without.
inline float conv_scale(const float* taps) { return 1.0f / sqrtf(float((taps ? 2 : 1) * 4 + 1)); }

// Where conv_walk_kernel's tokens are: the call's tokens one after the other in one sequence, with one state (a
// thread walks them all), or each row a sequence's of its own with its own state (row blockIdx.y).
struct ConvOne {
    float* state;
    int rows;
    __device__ int first() const { return 0; }
    __device__ int end() const { return rows; }
    __device__ float* at(int) const { return state; }
};
struct ConvSeqs {
    float* state[kInklingSeqRows];
    __device__ int first() const { return blockIdx.y; }
    __device__ int end() const { return blockIdx.y + 1; }
    __device__ float* at(int t) const { return state[t]; }
};

template <bool ACC, class Rows>
__global__ void conv_walk_kernel(const float* __restrict__ x, Rows rows, const float* __restrict__ w,
                                 const float* __restrict__ taps, int heads, float* __restrict__ out, int channels,
                                 float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    const int h = taps ? c / (channels / heads) : 0;
    const int r0 = rows.first(), r1 = rows.end();
    float* st = rows.at(r0) + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int row = r0; row < r1; ++row) {
        const size_t i = size_t(row) * channels + c;
        const float v = x[i];
        const float y = conv_one(s0, s1, s2, v, wc, taps ? taps + (size_t(row) * heads + h) * 4 : nullptr, scale);
        out[i] = ACC ? __fadd_rn(out[i], y) : y;   // no fma with the scale: axpy's bits
        s0 = s1;
        s1 = s2;
        s2 = v;
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

template <bool ACC>
__global__ void conv_rows_kernel(const float* __restrict__ x, const float* __restrict__ state,
                                 const float* __restrict__ w, const float* __restrict__ taps, int heads,
                                 float* __restrict__ out, int channels, int row0, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int row = row0 + blockIdx.y;
    const float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    const int h = taps ? c / (channels / heads) : 0;
    const float y = conv_one(in(row - 3), in(row - 2), in(row - 1), in(row), w + 4 * size_t(c),
                             taps ? taps + (size_t(row) * heads + h) * 4 : nullptr, scale);
    float* o = out + size_t(row) * channels + c;
    *o = ACC ? __fadd_rn(*o, y) : y;
}

__global__ void conv_state_kernel(const float* __restrict__ x, float* __restrict__ state, int channels, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    const float a = in(rows - 3), b = in(rows - 2), d = in(rows - 1);
    st[0] = a;
    st[1] = b;
    st[2] = d;
}

// The state as it is after token t of x (t < rows) for each of the call's snapshot tokens (Snaps), row blockIdx.y
// of them into snap + blockIdx.y * stride: what conv_state_kernel leaves after t + 1 tokens.  Before the call
// advances the state.
constexpr int SNAPS_MAX = 32;
struct Snaps {
    int t[SNAPS_MAX];
};
__global__ void conv_snaps_kernel(const float* __restrict__ x, const float* __restrict__ state, int channels,
                                  Snaps snaps, float* __restrict__ snap, size_t stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int t = snaps.t[blockIdx.y];
    const float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    float* to = snap + blockIdx.y * stride + size_t(c) * 3;
    to[0] = in(t - 2);
    to[1] = in(t - 1);
    to[2] = in(t);
}

constexpr int CONV_ROWS = 16;

// ---- mix -------------------------------------------------------------------------------------------------------

constexpr int MIX_THREADS = 512;
constexpr int MIX_SUMS = kInklingMaxStates * 5;   // a state's sum of squares and its dot with each of 4 queries

// A block a token, in one pass over the states: every state's sum of squares and its dots with the raw queries
// together (s_hat . q = inv * (s . q)), the block's sums of all of them at once, the softmaxes, then the outputs.
__global__ void __launch_bounds__(MIX_THREADS) mix_kernel(InklingMix m, int n, float eps) {
    __shared__ float part[MIX_THREADS / WARP][MIX_SUMS];
    __shared__ float tot[MIX_SUMS];
    __shared__ float wts[4][kInklingMaxStates];
    const size_t t = blockIdx.x;
    const int ns = m.n_state, nq = m.n_query, lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
    float acc[MIX_SUMS];
#pragma unroll
    for (int j = 0; j < MIX_SUMS; ++j) acc[j] = 0.0f;
    for (int i = threadIdx.x; i < n; i += MIX_THREADS) {
        float qv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) qv[q] = q < nq ? m.query[size_t(q) * n + i] : 0.0f;
#pragma unroll
        for (int k = 0; k < kInklingMaxStates; ++k) {
            if (k >= ns) break;
            const float x = m.state[k][t * n + i];
            acc[k * 5] += x * x;
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[k * 5 + 1 + q] += x * qv[q];
        }
    }
#pragma unroll
    for (int j = 0; j < MIX_SUMS; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) part[warp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x < MIX_SUMS) {
        float v = 0.0f;
        for (int w = 0; w < MIX_THREADS / WARP; ++w) v += part[w][threadIdx.x];
        tot[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int q = threadIdx.x;
        const float scale = 1.0f / sqrtf((float) n);
        float sc[kInklingMaxStates], mx = -INFINITY;
        for (int k = 0; k < ns; ++k) {
            const float inv = 1.0f / sqrtf(tot[k * 5] / float(n) + eps);
            sc[k] = tot[k * 5 + 1 + q] * inv * scale;
            mx = fmaxf(mx, sc[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < ns; ++k) sum += (sc[k] = expf(sc[k] - mx));
        // the weights times the states' norms, so that the sum below reads the raw states
        for (int k = 0; k < ns; ++k) wts[q][k] = sc[k] / sum / sqrtf(tot[k * 5] / float(n) + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += MIX_THREADS) {
        float xs[kInklingMaxStates];
#pragma unroll
        for (int k = 0; k < kInklingMaxStates; ++k) xs[k] = k < ns ? m.state[k][t * n + i] : 0.0f;
        for (int q = 0; q < nq; ++q) {
            float y = 0.0f;
#pragma unroll
            for (int k = 0; k < kInklingMaxStates; ++k)
                if (k < ns) y += wts[q][k] * xs[k];
            m.out[q][t * n + i] = y;
        }
    }
}

constexpr int MIX4_MAX = 768;   // threads: a float4 each, so n <= 3072
// mix_kernel for n a multiple of 4 up to 4 * MIX4_MAX, NS states: a thread a float4 of every state, read once into
// registers and kept for the output, so that a token's whole read is on its way at once.  The same sums otherwise.
// States a thread of mix4_body() has in registers already (a convolution's output before it): their values at
// the thread's float4.
struct Given {
    const float* p[2] = {};
    float4 v[2];
    __device__ float4 at(const float* s, int i) const {
        return s == p[0] ? v[0] : s == p[1] ? v[1] : reinterpret_cast<const float4*>(s)[i];
    }
};

constexpr int MIX4_SUMS = kInklingMaxStates * 5;
struct Mix4Smem {
    float part[MIX4_MAX / WARP][MIX4_SUMS];
    float tot[MIX4_SUMS];
    float wts[4][kInklingMaxStates];
};

// mix4_body()'s sums of a thread's float4 of each state: its square, then its products with the queries'.
template <int NS>
__device__ __forceinline__ void mix4_sums(const float4* xs, const float4* qv, float* acc) {
#pragma unroll
    for (int k = 0; k < NS; ++k) {
        const float4 x = xs[k];
        acc[k * 5] = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
#pragma unroll
        for (int q = 0; q < 4; ++q)
            acc[k * 5 + 1 + q] = x.x * qv[q].x + x.y * qv[q].y + x.z * qv[q].z + x.w * qv[q].w;
    }
}

// Query q's weights w[NS] from the block's sums tot: the softmax of the normed states' scores, times their norms, so
// that the output's sum reads the raw states.
template <int NS>
__device__ __forceinline__ void mix4_weights(const float* tot, int q, int n, float eps, float* w) {
    const float scale = 1.0f / sqrtf((float) n);
    float sc[NS], inv[NS], mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < NS; ++k) {
        inv[k] = 1.0f / sqrtf(tot[k * 5] / float(n) + eps);
        sc[k] = tot[k * 5 + 1 + q] * inv[k] * scale;
        mx = fmaxf(mx, sc[k]);
    }
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < NS; ++k) sum += (sc[k] = expf(sc[k] - mx));
#pragma unroll
    for (int k = 0; k < NS; ++k) w[k] = sc[k] / sum * inv[k];
}

// An output value from the states' values x there.
template <int NS>
__device__ __forceinline__ float mix4_one(const float* w, const float* x) {
    float y = 0.0f;
#pragma unroll
    for (int k = 0; k < NS; ++k) y += w[k] * x[k];
    return y;
}

template <int NS>
__device__ __forceinline__ float4 mix4_out(const float* w, const float4* xs) {
    float x[NS], y[NS], z[NS], v[NS];
#pragma unroll
    for (int k = 0; k < NS; ++k) x[k] = xs[k].x, y[k] = xs[k].y, z[k] = xs[k].z, v[k] = xs[k].w;
    return make_float4(mix4_one<NS>(w, x), mix4_one<NS>(w, y), mix4_one<NS>(w, z), mix4_one<NS>(w, v));
}

// mix4_kernel's token t; `given` only for one token (t 0).
template <int NS>
__device__ __forceinline__ void mix4_body(const InklingMix& m, int n, float eps, size_t t, const Given& given,
                                          Mix4Smem& sm) {
    constexpr int SUMS = NS * 5;
    auto& part = sm.part;
    auto& tot = sm.tot;
    auto& wts = sm.wts;
    const int nq = m.n_query, lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
    const int i = threadIdx.x, n4 = n / 4;
    const bool own = i < n4;
    float4 xs[NS];
    float acc[SUMS];
#pragma unroll
    for (int j = 0; j < SUMS; ++j) acc[j] = 0.0f;
    if (own) {
        float4 qv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
            qv[q] = q < nq ? reinterpret_cast<const float4*>(m.query + size_t(q) * n)[i] : make_float4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < NS; ++k) xs[k] = given.at(m.state[k] + t * n, i);
        mix4_sums<NS>(xs, qv, acc);
    }
#pragma unroll
    for (int j = 0; j < SUMS; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) part[warp][j] = v;
    }
    __syncthreads();
    const int warps = blockDim.x / WARP;
    for (int j = threadIdx.x; j < SUMS; j += blockDim.x) {   // a block of n <= 128 has fewer threads than sums
        float v = 0.0f;
        for (int w = 0; w < warps; ++w) v += part[w][j];
        tot[j] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) mix4_weights<NS>(tot, threadIdx.x, n, eps, wts[threadIdx.x]);
    __syncthreads();
    if (!own) return;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (q >= nq) break;
        reinterpret_cast<float4*>(m.out[q] + t * n)[i] = mix4_out<NS>(wts[q], xs);
    }
}

template <int NS>
__global__ void __launch_bounds__(MIX4_MAX) mix4_kernel(InklingMix m, int n, float eps) {
    __shared__ Mix4Smem sm;
    mix4_body<NS>(m, n, eps, blockIdx.x, Given{}, sm);
}

void launch_mix4(const InklingMix& m, int n, int rows, float eps, cudaStream_t st) {
    const int threads = (n / 4 + WARP - 1) / WARP * WARP;
    switch (m.n_state) {
        case 1: mix4_kernel<1><<<rows, threads, 0, st>>>(m, n, eps); break;
        case 2: mix4_kernel<2><<<rows, threads, 0, st>>>(m, n, eps); break;
        case 3: mix4_kernel<3><<<rows, threads, 0, st>>>(m, n, eps); break;
        case 4: mix4_kernel<4><<<rows, threads, 0, st>>>(m, n, eps); break;
        case 5: mix4_kernel<5><<<rows, threads, 0, st>>>(m, n, eps); break;
        case 6: mix4_kernel<6><<<rows, threads, 0, st>>>(m, n, eps); break;
        case 7: mix4_kernel<7><<<rows, threads, 0, st>>>(m, n, eps); break;
        default: mix4_kernel<8><<<rows, threads, 0, st>>>(m, n, eps); break;
    }
}

// ---- a token's post-convolution in the launch that reads it -----------------------------------------------------

struct ConvIn {
    const float* x;
    const float* parts;
    const float* wts;
    int n_parts;
    float* state;
    const float* w;
    float* acc;
    float* save;
    bool fresh;
    float scale;
};

// Channel ch of InklingConvIn's convolution for one token, given acc's value there: the input (with parts, the sum
// moe_sum() leaves in a zeroed buffer), conv_walk_kernel's convolution and state, and acc's new value.
__device__ __forceinline__ float conv_in_one(const ConvIn& c, int n, int ch, float old) {
    float v;
    if (c.parts) {
        float s = 0.0f;
        for (int k = 0; k < c.n_parts; ++k) s += c.wts[k] * c.parts[size_t(k) * n + ch];
        v = __fadd_rn(0.0f, s);
    } else {
        v = c.x[ch];
    }
    float* st = c.state + size_t(ch) * 3;
    const float s0 = st[0], s1 = st[1], s2 = st[2];
    const float y = conv_one(s0, s1, s2, v, c.w + 4 * size_t(ch), nullptr, c.scale);
    st[0] = s1, st[1] = s2, st[2] = v;
    return c.fresh ? y : __fadd_rn(old, y);
}

__global__ void conv_in_kernel(ConvIn c, int n) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n) return;
    const float old = c.acc[ch];
    if (c.save) c.save[ch] = old;
    c.acc[ch] = conv_in_one(c, n, ch, old);
}

constexpr int MIX_CONV_STATES = 5;   // inkling_mix_conv()'s most states a mix
struct MixPair {
    InklingMix m[2];
};

struct MixNorms {
    const float* w[4];
    float* y[4];
    fm::Q81* yq[4];
    float eps;
};

// One token's mixes, a block each, mix4_kernel's work; the first block's threads first do the convolution of
// their float4 of c.acc, which its mix then reads from their registers.  With SUM, the first mix's outputs are then
// normed as native_add_rms_norm_q8_1_kernel<SUM> norms them, each with its own weight.
template <int SUM>
__global__ void __launch_bounds__(MIX4_MAX) mix_conv_kernel(MixPair mp, ConvIn c, MixNorms nr, int n, float eps) {
    __shared__ Mix4Smem sm;
    __shared__ float part[SUM > 0 ? SUM / WARP : 1];
    const InklingMix& m = mp.m[blockIdx.x];
    const int i = threadIdx.x;
    Given g;
    if (blockIdx.x == 0 && c.acc && i < n / 4) {
        float4* a4 = reinterpret_cast<float4*>(c.acc) + i;
        const float4 old = *a4;
        if (c.save) reinterpret_cast<float4*>(c.save)[i] = old;
        float4 y;
        y.x = conv_in_one(c, n, 4 * i, old.x);
        y.y = conv_in_one(c, n, 4 * i + 1, old.y);
        y.z = conv_in_one(c, n, 4 * i + 2, old.z);
        y.w = conv_in_one(c, n, 4 * i + 3, old.w);
        *a4 = y;
        g.p[0] = c.acc, g.v[0] = y;
        g.p[1] = c.save, g.v[1] = old;
    }
    switch (m.n_state) {
        case 1: mix4_body<1>(m, n, eps, 0, g, sm); break;
        case 2: mix4_body<2>(m, n, eps, 0, g, sm); break;
        case 3: mix4_body<3>(m, n, eps, 0, g, sm); break;
        case 4: mix4_body<4>(m, n, eps, 0, g, sm); break;
        default: mix4_body<5>(m, n, eps, 0, g, sm); break;
    }
    if constexpr (SUM > 0) {
        if (blockIdx.x != 0) return;
        __syncthreads();   // the mix's outputs
        for (int q = 0; q < m.n_query; ++q) fm::rms_norm_q8_1<SUM>(m.out[q], nr.w[q], nr.y[q], nr.yq[q], n, nr.eps, part);
    }
}
static_assert(MIX_CONV_STATES == 5, "mix_conv_kernel's cases");

// ---- one token's mixes over many blocks ------------------------------------------------------------------------
//
// mix_conv_kernel's work in blocks of MIX_SPREAD threads, thread g of a mix's blocks doing what thread g of its block
// does there (the same float4 of the same states and the same warp's sums), so that every value is the same
// operations in the same order.  A mix's blocks meet in device memory (grid_meet()) after their warps' sums, which
// each block then adds up as mix4_body() does for the weights; with norms, the first mix's blocks meet again after
// the norms' warps' sums, a norm's warp recomputing the outputs it reads from the states.

constexpr int MIX_SPREAD = 32;

struct MixSpreadWork {
    unsigned meet[8];   // a counter a mix (grid_meet())
    float part[2][MIX4_MAX / WARP][MIX_CONV_STATES * 5];
    float norm[4][1024 / WARP];
};

struct MixSpreadSmem {
    float tot[MIX_CONV_STATES * 5];
    float wts[4][MIX_CONV_STATES];
    __align__(16) float outs[4][4 * MIX_SPREAD];   // the norms' inputs, the block's part of them
};

// A load past the SM's cache, of what other blocks of the launch wrote before a grid_meet().
__device__ __forceinline__ float load_peer(const float* p) { return *(volatile const float*) p; }

// The `blocks` blocks of a group past this point together, their stores before it seen by their load_peer()s after
// it.  The arrivals add 2^31 to the counter in all, which flips its top bit with the last one: from 0, it is 0 or
// 2^31 between meetings.
__device__ __forceinline__ void grid_meet(unsigned* count, int blocks, bool first) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned old = atomicAdd(count, first ? 0x80000000u - unsigned(blocks - 1) : 1u);
        while (((old ^ *(volatile unsigned*) count) & 0x80000000u) == 0) {
        }
        __threadfence();
    }
    __syncthreads();
}

// conv_in_one() of channels 4i .. 4i + 3, all their loads before their stores; g gets acc's and save's float4.
__device__ __forceinline__ void conv_in4(const ConvIn& c, int n, int i, Given& g) {
    float4* a4 = reinterpret_cast<float4*>(c.acc) + i;
    const float4 old = *a4;
    const float o[4] = {old.x, old.y, old.z, old.w};
    float v[4];
    if (c.parts) {
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
        for (int k = 0; k < c.n_parts; ++k) {
            const float wk = c.wts[k];
            const float* p = c.parts + size_t(k) * n + 4 * size_t(i);
#pragma unroll
            for (int j = 0; j < 4; ++j) s[j] += wk * p[j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __fadd_rn(0.0f, s[j]);
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = c.x[4 * i + j];
    }
    float* st = c.state + size_t(i) * 12;
    float s0[4], s1[4], s2[4], y[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s0[j] = st[3 * j], s1[j] = st[3 * j + 1], s2[j] = st[3 * j + 2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j] = conv_one(s0[j], s1[j], s2[j], v[j], c.w + 16 * size_t(i) + 4 * j, nullptr, c.scale);
        y[j] = c.fresh ? y[j] : __fadd_rn(o[j], y[j]);
    }
    if (c.save) reinterpret_cast<float4*>(c.save)[i] = old;
#pragma unroll
    for (int j = 0; j < 4; ++j) st[3 * j] = s1[j], st[3 * j + 1] = s2[j], st[3 * j + 2] = v[j];
    const float4 y4 = make_float4(y[0], y[1], y[2], y[3]);
    *a4 = y4;
    g.p[0] = c.acc, g.v[0] = y4;
    g.p[1] = c.save, g.v[1] = old;
}

// Block b of a mix's nb; with SUM, the first mix's norms after it (fm::rms_norm_q8_1<SUM>()'s sums).
template <int NS, int SUM>
__device__ __forceinline__ void mix_spread_body(const InklingMix& m, const MixNorms& nr, int n, float eps,
                                                const Given& given, int mi, int b, int nb, MixSpreadWork* wk,
                                                MixSpreadSmem& sm) {
    constexpr int SUMS = NS * 5;
    auto& tot = sm.tot;
    auto& wts = sm.wts;
    auto& outs = sm.outs;
    const int nq = m.n_query, lane = threadIdx.x % WARP, n4 = n / 4;
    const int i = b * MIX_SPREAD + threadIdx.x, warp = i / WARP;   // mix4_body()'s thread and warp
    const bool own = i < n4;
    float4 xs[NS];
    float acc[SUMS];
#pragma unroll
    for (int j = 0; j < SUMS; ++j) acc[j] = 0.0f;
    if (own) {
        float4 qv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
            qv[q] = q < nq ? reinterpret_cast<const float4*>(m.query + size_t(q) * n)[i] : make_float4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < NS; ++k) xs[k] = given.at(m.state[k], i);
        mix4_sums<NS>(xs, qv, acc);
    }
#pragma unroll
    for (int j = 0; j < SUMS; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) wk->part[mi][warp][j] = v;
    }
    grid_meet(&wk->meet[mi], nb, b == 0);
    const int warps = (n4 + WARP - 1) / WARP;
    for (int j = threadIdx.x; j < SUMS; j += blockDim.x) {
        float v = 0.0f;
        for (int w = 0; w < warps; ++w) v += load_peer(&wk->part[mi][w][j]);
        tot[j] = v;
    }
    __syncthreads();
    if (threadIdx.x < nq) mix4_weights<NS>(tot, threadIdx.x, n, eps, wts[threadIdx.x]);
    __syncthreads();
    if (own) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q >= nq) break;
            const float4 y = mix4_out<NS>(wts[q], xs);
            reinterpret_cast<float4*>(m.out[q])[i] = y;
            if constexpr (SUM > 0) reinterpret_cast<float4*>(outs[q])[threadIdx.x] = y;
        }
    }
    if constexpr (SUM > 0) {
        if (mi != 0) return;
        // the norms' sums: their thread t (warp t / 32) of each output's values t, t + SUM, ...
        const int nw = nb * MIX_SPREAD / WARP;
        for (int k = warp; k < SUM / WARP; k += nw) {
            const int t = k * WARP + lane;
            float ss[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int e = t; e < n; e += SUM) {
                float x[NS];
#pragma unroll
                for (int s = 0; s < NS; ++s) x[s] = load_peer(m.state[s] + e);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (q >= nq) break;
                    const float v = mix4_one<NS>(wts[q], x);
                    ss[q] = fm::fma_rn(v, v, ss[q]);
                }
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (q >= nq) break;
                const float v = fm::warp_sum(ss[q]);
                if (lane == 0) wk->norm[q][k] = v;
            }
        }
        grid_meet(&wk->meet[0], nb, b == 0);
        const int base = 4 * MIX_SPREAD * b, count = min(4 * MIX_SPREAD, n - base);   // a multiple of 32
        for (int q = 0; q < nq; ++q) {
            float sum = 0.0f;
            for (int k = 0; k < SUM / WARP; ++k) sum = fm::add_rn(sum, load_peer(&wk->norm[q][k]));
            const float scale = fm::div_rn(1.0f, fm::sqrt_rn(fm::add_rn(fm::div_rn(sum, float(n)), nr.eps)));
            for (int e = threadIdx.x; e < count; e += blockDim.x) {
                const int j = base + e;
                const float xi = fm::mul_rn(fm::mul_rn(outs[q][e], scale), nr.w[q][j]);
                nr.y[q][j] = xi;
                fm::q8_1(xi, nr.yq[q] + j / 32);
            }
        }
    }
}

// grid: the first mix's blocks (blocks0), then the second's.
template <int SUM>
__global__ void __launch_bounds__(MIX_SPREAD) mix_spread_kernel(MixPair mp, ConvIn c, MixNorms nr, int n, float eps,
                                                                int blocks0, MixSpreadWork* wk) {
    const int mi = blockIdx.x < blocks0 ? 0 : 1;
    const int b = blockIdx.x - mi * blocks0, nb = mi ? gridDim.x - blocks0 : blocks0;
    const InklingMix& m = mp.m[mi];
    const int i = b * MIX_SPREAD + threadIdx.x;
    __shared__ MixSpreadSmem sm;
    Given g;
    if (mi == 0 && c.acc && i < n / 4) conv_in4(c, n, i, g);
    switch (m.n_state) {
        case 1: mix_spread_body<1, SUM>(m, nr, n, eps, g, mi, b, nb, wk, sm); break;
        case 2: mix_spread_body<2, SUM>(m, nr, n, eps, g, mi, b, nb, wk, sm); break;
        case 3: mix_spread_body<3, SUM>(m, nr, n, eps, g, mi, b, nb, wk, sm); break;
        case 4: mix_spread_body<4, SUM>(m, nr, n, eps, g, mi, b, nb, wk, sm); break;
        default: mix_spread_body<5, SUM>(m, nr, n, eps, g, mi, b, nb, wk, sm); break;
    }
}

// ---- controller, SiTU ------------------------------------------------------------------------------------------

__global__ void ctrl_mix_kernel(const float* cur, const float* __restrict__ msg, const float* __restrict__ gain,
                                const float* __restrict__ forget, float* out, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = cur[i];
    out[i] = (c - tanhf(forget[i]) * c) + tanhf(gain[i]) * msg[i];
}

__device__ __forceinline__ float situ_one(float g, float u, float beta, float lb) {
    const float act = tanhf(g * (1.0f / beta)) * beta * sigmoidf(g);
    const float up = lb > 0.0f ? tanhf(u * (1.0f / lb)) * lb : u;
    return act * up;
}

__device__ __forceinline__ float2 row_scales(const ExpertScales& s, int row) {
    if (!s.gate && !s.up) return make_float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return make_float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

__global__ void situ_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n, float beta,
                                 float lb, ExpertScales s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float2 sc = row_scales(s, blockIdx.y);
    out[size_t(blockIdx.y) * n + i] = situ_one(row[i] * sc.x, row[n + i] * sc.y, beta, lb);
}

// situ_rows_kernel's values, not stored, into their Q8_1 blocks (n a multiple of 32: a warp is a block).
__global__ void situ_q8_1_kernel(const float* __restrict__ gate_up, fm::Q81* __restrict__ q, int n, float beta,
                                 float lb, ExpertScales s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float2 sc = row_scales(s, blockIdx.y);
    const size_t at = size_t(blockIdx.y) * n + i;
    fm::q8_1(situ_one(row[i] * sc.x, row[n + i] * sc.y, beta, lb), q + at / 32);
}

// ---- fused decode steps ------------------------------------------------------------------------------------------

__device__ __forceinline__ float4 ld4(const float* p, size_t i) { return *reinterpret_cast<const float4*>(p + i); }
__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

constexpr int CTRL_ROWS = 4, CTRL_THREADS = 256, CTRL_OUT = 32;   // a block's outputs in the second launch
static_assert(kInklingMaxRank == 2 * WARP, "ctrl_up_kernel: a lane two of the rank's values");

struct CtrlArgs {
    const float* cur[4];
    const float* msg[4];
    const float* down[4];
    const float* gain[4];
    const float* forget[4];
    float* out[4];
    int n, rank, rows;
    float eps;
    float* work;
};

// The norms of cur and msg and output o of the down projection, for every row: down[o] . [cur / rms, msg / rms] is
// (down_left[o] . cur) / rms(cur) + (down_right[o] . msg) / rms(msg).  A block an output of a controller.
__global__ void __launch_bounds__(CTRL_THREADS) ctrl_down_kernel(CtrlArgs a) {
    __shared__ float part[CTRL_THREADS / WARP][4 * CTRL_ROWS];
    const int o = blockIdx.x, k = blockIdx.y, n = a.n, rows = a.rows;
    const float* cur = a.cur[k];
    const float* msg = a.msg[k];
    const float* wr = a.down[k] + size_t(o) * 2 * n;
    float acc[4 * CTRL_ROWS];   // per row: down_left . cur, down_right . msg, cur . cur, msg . msg
#pragma unroll
    for (int j = 0; j < 4 * CTRL_ROWS; ++j) acc[j] = 0.0f;
    for (int i = 4 * threadIdx.x; i < n; i += 4 * CTRL_THREADS) {
        const float4 wl = ld4(wr, i), wm = ld4(wr, size_t(n) + i);
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t) {
            if (t >= rows) break;
            const float4 c = ld4(cur, size_t(t) * n + i), m = ld4(msg, size_t(t) * n + i);
            acc[4 * t] += dot4(wl, c);
            acc[4 * t + 1] += dot4(wm, m);
            acc[4 * t + 2] += dot4(c, c);
            acc[4 * t + 3] += dot4(m, m);
        }
    }
    const int lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
#pragma unroll
    for (int j = 0; j < 4 * CTRL_ROWS; ++j) {
        const float v = warp_sum(acc[j]);
        if (lane == 0) part[warp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x < rows) {
        const int t = threadIdx.x;
        float v[4] = {0, 0, 0, 0};
        for (int w = 0; w < CTRL_THREADS / WARP; ++w)
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += part[w][4 * t + j];
        const float ic = 1.0f / sqrtf(v[2] / float(n) + a.eps), im = 1.0f / sqrtf(v[3] / float(n) + a.eps);
        a.work[(size_t(k) * rows + t) * a.rank + o] = v[0] * ic + v[1] * im;
    }
}

// gain and forget of CTRL_OUT outputs of a controller (a warp four of them, a lane two of the rank's values), and
// the combine: out = (cur - tanh(forget) * cur) + tanh(gain) * msg.
__global__ void __launch_bounds__(CTRL_THREADS) ctrl_up_kernel(CtrlArgs a) {
    __shared__ float r[CTRL_ROWS][kInklingMaxRank];
    const int k = blockIdx.y, n = a.n, rows = a.rows, rank = a.rank;
    for (int i = threadIdx.x; i < rows * rank; i += CTRL_THREADS)
        r[i / rank][i % rank] = a.work[size_t(k) * rows * rank + i];
    __syncthreads();
    const float* gw = a.gain[k];
    const float* fw = a.forget[k];
    const int lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
    constexpr int PER = CTRL_OUT / (CTRL_THREADS / WARP);
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int o = blockIdx.x * CTRL_OUT + warp * PER + j;
        if (o >= n) break;
        const bool l0 = lane < rank, l1 = lane + WARP < rank;
        const float g0 = l0 ? gw[size_t(o) * rank + lane] : 0.0f;
        const float g1 = l1 ? gw[size_t(o) * rank + lane + WARP] : 0.0f;
        const float f0 = l0 ? fw[size_t(o) * rank + lane] : 0.0f;
        const float f1 = l1 ? fw[size_t(o) * rank + lane + WARP] : 0.0f;
        float gs[CTRL_ROWS], fs[CTRL_ROWS];
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t) {
            const float r0 = t < rows && l0 ? r[t][lane] : 0.0f, r1 = t < rows && l1 ? r[t][lane + WARP] : 0.0f;
            gs[t] = warp_sum(g0 * r0 + g1 * r1);
            fs[t] = warp_sum(f0 * r0 + f1 * r1);
        }
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t)
            if (lane == t && t < rows) {
                const size_t i = size_t(t) * n + o;
                const float c = a.cur[k][i];
                a.out[k][i] = (c - tanhf(fs[t]) * c) + tanhf(gs[t]) * a.msg[k][i];
            }
    }
}

// y[t][o] = w[o] . x[t], a warp an output, for up to CTRL_ROWS tokens.
constexpr int GEMV_WARPS = 4;
__global__ void __launch_bounds__(GEMV_WARPS * WARP) gemv_rows_kernel(const float* __restrict__ w,
                                                                      const float* __restrict__ x,
                                                                      float* __restrict__ y, int n_in, int n_out,
                                                                      int rows) {
    const int lane = threadIdx.x % WARP, o = blockIdx.x * GEMV_WARPS + threadIdx.x / WARP;
    if (o >= n_out) return;
    const float* wr = w + size_t(o) * n_in;
    float acc[CTRL_ROWS] = {0, 0, 0, 0};
#pragma unroll 4
    for (int i = 4 * lane; i < n_in; i += 4 * WARP) {
        const float4 wv = ld4(wr, i);
#pragma unroll
        for (int t = 0; t < CTRL_ROWS; ++t)
            if (t < rows) acc[t] += dot4(wv, ld4(x, size_t(t) * n_in + i));
    }
#pragma unroll
    for (int t = 0; t < CTRL_ROWS; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0 && t < rows) y[size_t(t) * n_out + o] = v;
    }
}

// inkling_kv_store(): a block a head of K (blockIdx.y 0) or of V (1), a thread a value of it; the tokens in turn.
constexpr int KV_ROWS = 16;
struct KvArgs {
    const float* raw[2];
    float* state[2];
    const float* w[2];
    const float* taps[2];
    float* out[2];
    InklingKvDst dst;
    int kv_heads, rows;
    float k_clip, eps;
    long long pos;
};

// Where kv_store_kernel's tokens go: the call's tokens one after the other in one sequence (a block takes them all),
// from a position given by value or read on the card when the launch runs, or each row a sequence's of its own (a
// block a row, blockIdx.z).
struct KvOne {
    __device__ int first(const KvArgs&) const { return 0; }
    __device__ int end(const KvArgs& a) const { return a.rows; }
    __device__ float* state(const KvArgs& a, int which, int) const { return a.state[which]; }
    __device__ const InklingKvDst& dst(const KvArgs& a, int) const { return a.dst; }
    __device__ long long pos(const KvArgs& a, int t) const { return a.pos + t; }
};
struct KvAt : KvOne {
    const long long* at;
    __device__ long long pos(const KvArgs&, int t) const { return *at + t; }
};
struct KvSeqs {
    InklingKvSeq row[kInklingSeqRows];
    __device__ int first(const KvArgs&) const { return blockIdx.z; }
    __device__ int end(const KvArgs&) const { return blockIdx.z + 1; }
    __device__ float* state(const KvArgs&, int which, int t) const { return which ? row[t].v_state : row[t].k_state; }
    __device__ const InklingKvDst& dst(const KvArgs&, int t) const { return row[t].dst; }
    __device__ long long pos(const KvArgs&, int t) const { return (long long) row[t].pos; }
};

// A block's head g of K (which 0) or of V (1); part: AT_DIM / WARP floats of shared memory.
template <class Rows>
__device__ __forceinline__ void kv_store_head(const KvArgs& a, const Rows& rows, int g, int which, float* part) {
    const int d = threadIdx.x, lane = d % WARP;
    const int channels = a.kv_heads * AT_DIM, c = g * AT_DIM + d;
    const float* x = a.raw[which];
    const float* taps = a.taps[which];
    const float* wc = a.w[which] + 4 * size_t(c);
    const int t0 = rows.first(a), t1 = rows.end(a);
    float* st = rows.state(a, which, t0) + size_t(c) * 3;
    const float scale = 1.0f / sqrtf(float((taps ? 2 : 1) * 4 + 1));
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int t = t0; t < t1; ++t) {
        const InklingKvDst& dst = rows.dst(a, t);
        const bool q8 = which ? dst.vc != nullptr : dst.kc != nullptr;
        const float v = x[size_t(t) * channels + c];
        const float y = conv_one(s0, s1, s2, v, wc, taps ? taps + (size_t(t) * a.kv_heads + g) * 4 : nullptr, scale);
        s0 = s1, s1 = s2, s2 = v;
        // the head's sum of squares: the warps' sums, then the four in order
        float ss = warp_sum(y * y);
        if (lane == 0) part[d / WARP] = ss;
        __syncthreads();
        ss = part[0] + part[1] + part[2] + part[3];
        __syncthreads();
        float val;
        if (which == 0) {
            const float factor = fminf(fmaxf(a.k_clip / (sqrtf(ss) + 1e-6f), 0.0f), 1.0f);
            val = y * factor;
        } else {
            val = y * (1.0f / sqrtf(ss / float(AT_DIM) + a.eps));
        }
        if (a.out[which]) a.out[which][size_t(t) * channels + c] = val;
        const long long p = rows.pos(a, t);
        const long long row = dst.ring > 0 ? p % dst.ring : p;
        if (!q8) {
            half* cache = reinterpret_cast<half*>(which ? dst.v : dst.k);
            cache[size_t(row) * channels + c] = __float2half(val);
        } else {
            // a warp is a Q8_0 block of 32 values: store_q8's rule
            float amax = fabsf(val);
#pragma unroll
            for (int o = WARP / 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, WARP));
            const float dl = amax / 127.0f;
            const float id = dl > 0.f ? 1.0f / dl : 0.0f;
            int8_t* codes = which ? dst.vc : dst.kc;
            half* scales = reinterpret_cast<half*>(which ? dst.vs : dst.ks);
            const size_t at = size_t(row) * channels + c;
            if (lane == 0) scales[at / 32] = __float2half(dl);
            const float q = fminf(fmaxf(rintf(val * id), -127.0f), 127.0f);
            codes[at] = (int8_t) q;
        }
    }
    st[0] = s0, st[1] = s1, st[2] = s2;
}

template <class Rows>
__global__ void __launch_bounds__(AT_DIM) kv_store_kernel(KvArgs a, Rows rows) {
    __shared__ float part[AT_DIM / WARP];
    kv_store_head(a, rows, blockIdx.x, blockIdx.y, part);
}

// kv_store_kernel's blocks (a head of K, then of V, by blockIdx.x) and then clip_many_kernel's warps, AT_DIM / WARP
// a block, in one launch: the same work, a block or a warp as each of the two would give it.
template <class Rows, class From>
__global__ void __launch_bounds__(AT_DIM) kv_clip_kernel(KvArgs a, Rows rows, ClipJobs c, int count, From from) {
    __shared__ float part[AT_DIM / WARP];
    const int b = blockIdx.x, kv = 2 * a.kv_heads;
    if (b < kv) {
        kv_store_head(a, rows, b % a.kv_heads, b / a.kv_heads, part);
        return;
    }
    clip_many_group(c, count, from, (b - kv) * (AT_DIM / WARP) + int(threadIdx.x) / WARP);
}

}  // namespace

// ================================================================================================================

void inkling_clip(const float* x, int x_stride, const float* scale, float* y, int dim, int heads, int rows,
                  float threshold, InklingTau tau, int64_t pos, void* stream) {
    if (rows <= 0) return;
    if (dim <= 0 || dim > 1024) throw std::invalid_argument("inkling_clip: dim must be 1 to 1024");
    const int groups = rows * heads, per_block = 256 / WARP;
    clip_kernel<<<grid(groups, per_block), 256, 0, cs(stream)>>>(x, x_stride ? x_stride : heads * dim, scale, y, dim,
                                                                 heads, groups, threshold, tau.alpha, tau.floor,
                                                                 (long long) pos);
    check_launch("inkling_clip");
}

size_t inkling_attend_work_floats(int n_head, int dim) {
    return size_t(kInklingSplitRows) * n_head * (kInklingMaxSplits + 2) * (dim + 2);
}

namespace {
// inkling_attend()'s and inkling_attend_seqs()'s checks, and the arguments every row shares
AttnArgs attn_args(const char* who, const float* q, const InklingRel& rel, const float* sinks, int rows, float* out,
                   int dim, int n_head, int kv_heads, float scale, int window, int ring, float* work) {
    if (dim != AT_DIM || kv_heads <= 0 || n_head != AT_PER * kv_heads)
        throw std::invalid_argument(std::string(who) + ": heads of 128 values in groups of 3 only");
    if (rel.n_rel < 0 || rel.n_rel > AT_MAX_REL || (rel.n_rel > 0 && !rel.r) || !rel.table)
        throw std::invalid_argument(std::string(who) + ": bad relative-position bias");
    AttnArgs a{};
    a.q = q;
    a.rel = rel;
    a.sinks = sinks;
    a.rows = rows;
    a.n_head = n_head;
    a.kv_heads = kv_heads;
    a.window = window;
    a.ring = ring;
    a.scale = scale;
    a.out = out;
    a.work = work;
    return a;
}
}  // namespace

void inkling_attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, Kvq8 kq8, Kvq8 vq8,
                    InklingRel rel, const float* sinks, int64_t pos, int rows, float* out, int dim, int n_head,
                    int kv_heads, float scale, int window, int ring, float* work, size_t work_floats, void* stream,
                    int64_t room, bool int8_dot, const long long* at, const float* gate, void* out_q8_1) {
    if (rows <= 0) return;
    if (at && room <= 0) throw std::invalid_argument("inkling_attend: the position on the card needs a room");
    AttnArgs a =
        attn_args("inkling_attend", q, rel, sinks, rows, out, dim, n_head, kv_heads, scale, window, ring, work);
    a.kc = (const half*) kcache;
    a.vc = (const half*) vcache;
    a.kq = kq8;
    a.vq = vq8;
    a.pos = (long long) pos;
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    const cudaStream_t st = cs(stream);
    // positions + 1, at most: the caller's room when it gives one, else pos + rows
    const long long span_room = room > 0 ? (long long) room : (long long) pos + rows;

    if (work && rows <= kInklingSplitRows && span_room > 0) {
        splits_of(span_room, window, a.split, a.nslot);
        const size_t need = size_t(rows) * n_head * a.nslot * (AT_DIM + 2);
        if (need <= work_floats) {
            a.gate = gate;
            a.out_q8_1 = out_q8_1;
            if (at) launch_dec(a, AttnAt{at}, rows, kq, vq, int8_dot, st);
            else launch_dec(a, AttnOne{}, rows, kq, vq, int8_dot, st);
            return;
        }
    }
    if (at) throw std::invalid_argument("inkling_attend: the position on the card takes the split path and a room");
    if (rows >= kInklingMmaRows && rel.n_rel == 16 && mma_attention()) {
        const dim3 gm(grid(rows, MMA_TOKENS), kv_heads);
        if (kq && vq) attn_mma_kernel<true, true><<<gm, MMA_WARPS * WARP, 0, st>>>(a);
        else if (kq) attn_mma_kernel<true, false><<<gm, MMA_WARPS * WARP, 0, st>>>(a);
        else if (vq) attn_mma_kernel<false, true><<<gm, MMA_WARPS * WARP, 0, st>>>(a);
        else attn_mma_kernel<false, false><<<gm, MMA_WARPS * WARP, 0, st>>>(a);
        check_launch("inkling_attend");
        if (gate) mul_sigmoid(out, gate, n_head * dim, rows, n_head * dim, false, stream);
        if (out_q8_1) kernels::native_quantize_q8_1(out, out_q8_1, n_head * dim * rows, 1, stream);
        return;
    }
    const int tq = rows >= 8 ? 8 : rows >= 4 ? 4 : rows >= 2 ? 2 : 1;
    const dim3 g(1, kv_heads, (rows + tq - 1) / tq);
    switch (tq) {
        case 8: launch_attn_q<8>(a, g, st, kq, vq); break;
        case 4: launch_attn_q<4>(a, g, st, kq, vq); break;
        case 2: launch_attn_q<2>(a, g, st, kq, vq); break;
        default: launch_attn_q<1>(a, g, st, kq, vq); break;
    }
    check_launch("inkling_attend");
    if (gate) mul_sigmoid(out, gate, n_head * dim, rows, n_head * dim, false, stream);
    if (out_q8_1) kernels::native_quantize_q8_1(out, out_q8_1, n_head * dim * rows, 1, stream);
}

void inkling_attend_seqs(const float* q, const InklingAttnSeq* seqs, int rows, InklingRel rel, const float* sinks,
                         float* out, int dim, int n_head, int kv_heads, float scale, int window, int ring,
                         float* work, size_t work_floats, void* stream, bool int8_dot) {
    if (rows <= 0) return;
    if (rows > kInklingSeqRows || !work)
        throw std::invalid_argument("inkling_attend_seqs: 1 to " + std::to_string(kInklingSeqRows) +
                                    " rows, and work");
    AttnArgs a = attn_args("inkling_attend_seqs", q, rel, sinks, rows, out, dim, n_head, kv_heads, scale, window, ring,
                           work);
    a.nslot = 1;
    const bool kq = seqs[0].kq8.codes != nullptr, vq = seqs[0].vq8.codes != nullptr;
    AttnSeqs r{};
    for (int t = 0; t < rows; ++t) {
        const InklingAttnSeq& c = seqs[t];
        if ((c.kq8.codes != nullptr) != kq || (c.vq8.codes != nullptr) != vq)
            throw std::invalid_argument("inkling_attend_seqs: every row's caches of one kind");
        AttnRow& w = r.r[t];
        w.kc = (const half*) c.kcache, w.vc = (const half*) c.vcache, w.kq = c.kq8, w.vq = c.vq8;
        w.pos = (long long) c.pos;
        splits_of(c.room > 0 ? (long long) c.room : w.pos + 1, window, w.split, w.nslot);
        a.nslot = std::max(a.nslot, w.nslot);
    }
    if (size_t(rows) * n_head * a.nslot * (AT_DIM + 2) > work_floats)
        throw std::invalid_argument("inkling_attend_seqs: work too small");
    launch_dec(a, r, rows, kq, vq, int8_dot, cs(stream));
}

namespace {
template <bool ACC>
void conv_any(const float* x, float* state, const float* w, const float* taps, int heads, float* out, int channels,
              int rows, void* stream) {
    if (rows <= 0) return;
    if (taps && (heads <= 0 || channels % heads))
        throw std::invalid_argument("inkling_conv: channels must split evenly into the taps' heads");
    const float scale = conv_scale(taps);
    if (rows >= CONV_ROWS) {
        for (int r = 0; r < rows; r += 65535)   // a grid has at most 65535 rows
            conv_rows_kernel<ACC><<<dim3(grid(channels, 256), std::min(65535, rows - r)), 256, 0, cs(stream)>>>(
                x, state, w, taps, heads, out, channels, r, scale);
        conv_state_kernel<<<grid(channels, 256), 256, 0, cs(stream)>>>(x, state, channels, rows);
    } else {
        conv_walk_kernel<ACC><<<grid(channels, 256), 256, 0, cs(stream)>>>(x, ConvOne{state, rows}, w, taps, heads,
                                                                           out, channels, scale);
    }
    check_launch("inkling_conv");
}
}  // namespace

void inkling_conv(const float* x, float* state, const float* w, const float* taps, int heads, float* out,
                  int channels, int rows, void* stream) {
    conv_any<false>(x, state, w, taps, heads, out, channels, rows, stream);
}

void inkling_conv_acc(const float* x, float* state, const float* w, const float* taps, int heads, float* acc,
                      int channels, int rows, void* stream) {
    conv_any<true>(x, state, w, taps, heads, acc, channels, rows, stream);
}

void inkling_conv_snaps(const float* x, const float* state, int channels, int rows, unsigned bits, float* snap,
                        size_t stride, void* stream) {
    Snaps snaps{};
    int n = 0;
    for (int t = 0; t < rows && t < SNAPS_MAX; ++t)
        if (bits >> t & 1u) snaps.t[n++] = t;
    if (n == 0) return;
    conv_snaps_kernel<<<dim3(grid(channels, 256), n), 256, 0, cs(stream)>>>(x, state, channels, snaps, snap, stride);
    check_launch("inkling_conv_snaps");
}

namespace {
template <bool ACC>
void conv_seqs(const float* x, float* const* states, const float* w, float* out, int channels, int rows,
               void* stream) {
    if (rows <= 0) return;
    if (rows > kInklingSeqRows)
        throw std::invalid_argument("inkling_conv_seqs: at most " + std::to_string(kInklingSeqRows) + " rows");
    ConvSeqs s{};
    for (int t = 0; t < rows; ++t) s.state[t] = states[t];
    conv_walk_kernel<ACC><<<dim3(grid(channels, 256), rows), 256, 0, cs(stream)>>>(x, s, w, nullptr, 0, out, channels,
                                                                                   conv_scale(nullptr));
    check_launch("inkling_conv_seqs");
}
}  // namespace

void inkling_conv_seqs(const float* x, float* const* states, const float* w, float* out, int channels, int rows,
                       void* stream) {
    conv_seqs<false>(x, states, w, out, channels, rows, stream);
}

void inkling_conv_acc_seqs(const float* x, float* const* states, const float* w, float* acc, int channels, int rows,
                           void* stream) {
    conv_seqs<true>(x, states, w, acc, channels, rows, stream);
}

void inkling_mix(const InklingMix& m, int n, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    if (m.n_state < 1 || m.n_state > kInklingMaxStates || m.n_query < 1 || m.n_query > 4)
        throw std::invalid_argument("inkling_mix: 1 to 8 states and 1 to 4 queries");
    bool aligned = n % 4 == 0 && n / 4 <= MIX4_MAX && (uintptr_t(m.query) & 15) == 0;
    for (int k = 0; k < m.n_state; ++k) aligned = aligned && (uintptr_t(m.state[k]) & 15) == 0;
    for (int q = 0; q < m.n_query; ++q) aligned = aligned && (uintptr_t(m.out[q]) & 15) == 0;
    if (aligned)
        launch_mix4(m, n, rows, eps, cs(stream));
    else
        mix_kernel<<<rows, MIX_THREADS, 0, cs(stream)>>>(m, n, eps);
    check_launch("inkling_mix");
}

namespace {
ConvIn conv_in(const InklingConvIn& c) {
    if (!c.acc || !c.state || !c.w || (!c.x && !(c.parts && c.weights && c.n_parts > 0)))
        throw std::invalid_argument("inkling_conv_in: acc, state, w and an input");
    return ConvIn{c.x, c.parts, c.weights, c.n_parts, c.state, c.w, c.acc, c.save, c.fresh, conv_scale(nullptr)};
}
bool aligned16(const void* p) { return (uintptr_t(p) & 15) == 0; }
}  // namespace

void inkling_conv_in(const InklingConvIn& c, int n, void* stream) {
    conv_in_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(conv_in(c), n);
    check_launch("inkling_conv_in");
}

size_t inkling_mix_conv_work_bytes() { return sizeof(MixSpreadWork); }

namespace {
// The blocks of mix_spread_kernel the current card holds at once (the least over its SUMs): a launch's blocks wait
// for each other in grid_meet(), so a launch takes no more than these.  Other streams' kernels on the card can only
// delay its blocks, as they run to their end without them.
int spread_room() {
    static std::mutex mu;
    static std::vector<int> rooms;   // a device's, 0: not yet asked
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    const std::lock_guard<std::mutex> lock(mu);
    if (rooms.size() <= size_t(dev)) rooms.resize(size_t(dev) + 1, 0);
    if (!rooms[dev]) {
        int sms = 0, least = INT_MAX;
        for (const void* k : {(const void*) mix_spread_kernel<0>, (const void*) mix_spread_kernel<128>,
                              (const void*) mix_spread_kernel<256>, (const void*) mix_spread_kernel<512>,
                              (const void*) mix_spread_kernel<1024>}) {
            int per_sm = 0;
            if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, MIX_SPREAD, 0) != cudaSuccess) per_sm = 0;
            least = std::min(least, per_sm);
        }
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) sms = 0;
        cudaGetLastError();
        rooms[dev] = least * sms > 0 ? least * sms : -1;   // -1: none
    }
    return rooms[dev];
}
}  // namespace

void inkling_mix_conv(const InklingMix* mixes, int count, const InklingConvIn* conv, int n, float eps, void* stream,
                      const InklingMixNorm* norm, void* work) {
    if (count < 1 || count > 2) throw std::invalid_argument("inkling_mix_conv: 1 or 2 mixes");
    bool fits = n % 4 == 0 && n / 4 <= MIX4_MAX;
    MixPair mp{};
    for (int k = 0; k < count; ++k) {
        const InklingMix& m = mixes[k];
        if (m.n_state < 1 || m.n_state > kInklingMaxStates || m.n_query < 1 || m.n_query > 4)
            throw std::invalid_argument("inkling_mix_conv: 1 to 8 states and 1 to 4 queries");
        fits = fits && m.n_state <= MIX_CONV_STATES && aligned16(m.query);
        for (int j = 0; j < m.n_state; ++j) fits = fits && aligned16(m.state[j]);
        for (int q = 0; q < m.n_query; ++q) fits = fits && aligned16(m.out[q]);
        mp.m[k] = m;
    }
    if (conv) fits = fits && aligned16(conv->acc) && aligned16(conv->save);
    if (norm && n % WARP) throw std::invalid_argument("inkling_mix_conv: the norms take n a multiple of 32");
    if (!fits) {   // the calls one after the other
        if (conv) inkling_conv_in(*conv, n, stream);
        for (int k = 0; k < count; ++k) inkling_mix(mixes[k], n, 1, eps, stream);
        for (int q = 0; norm && q < mixes[0].n_query; ++q)
            kernels::native_add_rms_norm_q8_1(mixes[0].out[q], nullptr, norm->w[q], norm->y[q], norm->y_q8_1[q], n, 1,
                                              norm->eps, norm->threads, stream);
        return;
    }
    const int threads = (n / 4 + WARP - 1) / WARP * WARP;   // launch_mix4()'s
    const ConvIn c = conv ? conv_in(*conv) : ConvIn{};
    MixNorms nr{};
    if (norm) {
        for (int q = 0; q < mixes[0].n_query; ++q)
            nr.w[q] = norm->w[q], nr.y[q] = norm->y[q], nr.yq[q] = static_cast<fm::Q81*>(norm->y_q8_1[q]);
        nr.eps = norm->eps;
    }
    const cudaStream_t st = cs(stream);
    const int blocks = (threads + MIX_SPREAD - 1) / MIX_SPREAD;   // a mix's, spread
    if (work && count * blocks <= spread_room()) {
        auto* wk = static_cast<MixSpreadWork*>(work);
        const auto spread = [&](auto kernel) {
            kernel<<<count * blocks, MIX_SPREAD, 0, st>>>(mp, c, nr, n, eps, blocks, wk);
        };
        switch (norm ? norm->threads : 0) {
            case 0: spread(mix_spread_kernel<0>); break;
            case 128: spread(mix_spread_kernel<128>); break;
            case 256: spread(mix_spread_kernel<256>); break;
            case 512: spread(mix_spread_kernel<512>); break;
            case 1024: spread(mix_spread_kernel<1024>); break;
            default: throw std::invalid_argument("inkling_mix_conv: norms of 128, 256, 512 or 1024 threads");
        }
        check_launch("inkling_mix_conv");
        return;
    }
    switch (norm ? norm->threads : 0) {
        case 0: mix_conv_kernel<0><<<count, threads, 0, st>>>(mp, c, nr, n, eps); break;
        case 128: mix_conv_kernel<128><<<count, threads, 0, st>>>(mp, c, nr, n, eps); break;
        case 256: mix_conv_kernel<256><<<count, threads, 0, st>>>(mp, c, nr, n, eps); break;
        case 512: mix_conv_kernel<512><<<count, threads, 0, st>>>(mp, c, nr, n, eps); break;
        case 1024: mix_conv_kernel<1024><<<count, threads, 0, st>>>(mp, c, nr, n, eps); break;
        default: throw std::invalid_argument("inkling_mix_conv: norms of 128, 256, 512 or 1024 threads");
    }
    check_launch("inkling_mix_conv");
}

void inkling_control(void* blas, const float* cur, const float* msg, const float* down, const float* gain,
                     const float* forget, float* work, float* out, int n, int rank, int rows, float eps,
                     void* stream) {
    if (rows <= 0) return;
    float* z = work;
    float* r = z + size_t(rows) * 2 * n;
    float* g = r + size_t(rows) * rank;
    float* f = g + size_t(rows) * n;
    rms_norm(cur, nullptr, z, n, rows, eps, stream, 2 * n);
    rms_norm(msg, nullptr, z + n, n, rows, eps, stream, 2 * n);
    if (rows < 8 || !blas) {
        gemv_f32(down, z, r, 2 * n, rank, rank, rows, stream);
        gemv_f32(gain, r, g, rank, n, n, rows, stream);
        gemv_f32(forget, r, f, rank, n, n, rows, stream);
    } else {
        gemm_f32(blas, down, z, r, 2 * n, rank, rank, rows);
        gemm_f32(blas, gain, r, g, rank, n, n, rows);
        gemm_f32(blas, forget, r, f, rank, n, n, rows);
    }
    const size_t count = size_t(rows) * n;
    ctrl_mix_kernel<<<grid(count, 256), 256, 0, cs(stream)>>>(cur, msg, g, f, out, count);
    check_launch("inkling_control");
}

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float lb, void* stream,
               const ExpertScales* scales) {
    if (n <= 0 || rows <= 0) return;
    for (int r = 0; r < rows; r += 65535) {   // a grid has at most 65535 rows
        ExpertScales s = scales ? *scales : ExpertScales{};
        if (s.ids) s.ids += r;
        situ_rows_kernel<<<dim3(grid(n, 256), std::min(65535, rows - r)), 256, 0, cs(stream)>>>(
            gate_up + size_t(r) * 2 * n, out + size_t(r) * n, n, beta, lb, s);
    }
    check_launch("situ_rows");
}

void situ_rows_q8_1(const float* gate_up, void* q8_1, int n, int rows, float beta, float lb, void* stream,
                    const ExpertScales* scales) {
    if (n <= 0 || rows <= 0) return;
    if (n % WARP) throw std::invalid_argument("situ_rows_q8_1: n must be a multiple of 32");
    for (int r = 0; r < rows; r += 65535) {
        ExpertScales s = scales ? *scales : ExpertScales{};
        if (s.ids) s.ids += r;
        situ_q8_1_kernel<<<dim3(grid(n, 256), std::min(65535, rows - r)), 256, 0, cs(stream)>>>(
            gate_up + size_t(r) * 2 * n, static_cast<fm::Q81*>(q8_1) + size_t(r) * n / 32, n, beta, lb, s);
    }
    check_launch("situ_rows_q8_1");
}

void inkling_control_fused(const InklingCtrl* ctrl, int count, int n, int rank, int rows, float eps, float* work,
                           void* stream) {
    if (rows <= 0 || count <= 0) return;
    if (count > 4 || rank <= 0 || rank > kInklingMaxRank || n % 4)
        throw std::invalid_argument("inkling_control_fused: 1 to 4 controllers, rank 1 to " +
                                    std::to_string(kInklingMaxRank) + ", n a multiple of 4");
    const cudaStream_t st = cs(stream);
    for (int r0 = 0; r0 < rows; r0 += CTRL_ROWS) {
        CtrlArgs a{};
        a.n = n, a.rank = rank, a.rows = std::min(CTRL_ROWS, rows - r0), a.eps = eps, a.work = work;
        for (int k = 0; k < count; ++k) {
            const size_t off = size_t(r0) * n;
            a.cur[k] = ctrl[k].cur + off;
            a.msg[k] = ctrl[k].msg + off;
            a.out[k] = ctrl[k].out + off;
            a.down[k] = ctrl[k].down, a.gain[k] = ctrl[k].gain, a.forget[k] = ctrl[k].forget;
        }
        const dim3 g1(rank, count), g2((n + CTRL_OUT - 1) / CTRL_OUT, count);
        ctrl_down_kernel<<<g1, CTRL_THREADS, 0, st>>>(a);
        ctrl_up_kernel<<<g2, CTRL_THREADS, 0, st>>>(a);
        check_launch("inkling_control_fused");
    }
}

void inkling_gemv(const float* w, const float* x, float* y, int n_in, int n_out, int rows, void* stream) {
    if (rows <= 0 || n_out <= 0) return;
    if (n_in % 4) throw std::invalid_argument("inkling_gemv: n_in must be a multiple of 4");
    for (int r0 = 0; r0 < rows; r0 += CTRL_ROWS) {
        const int m = std::min(CTRL_ROWS, rows - r0);
        const float* xr = x + size_t(r0) * n_in;
        float* yr = y + size_t(r0) * n_out;
        gemv_rows_kernel<<<grid(n_out, GEMV_WARPS), GEMV_WARPS * WARP, 0, cs(stream)>>>(w, xr, yr, n_in, n_out, m);
    }
    check_launch("inkling_gemv");
}

namespace {
ClipJobs clip_jobs(const InklingClipJob* jobs, int count, int rows) {
    if (count > 4) throw std::invalid_argument("inkling_clip_many: at most 4 jobs");
    ClipJobs c{};
    c.first[0] = 0;
    for (int k = 0; k < count; ++k) {
        if (jobs[k].dim <= 0 || jobs[k].dim > 1024) throw std::invalid_argument("inkling_clip_many: dim 1 to 1024");
        c.j[k] = jobs[k];
        c.first[k + 1] = c.first[k] + rows * jobs[k].heads;
    }
    return c;
}

template <class From>
void clip_many(const InklingClipJob* jobs, int count, int rows, const From& from, void* stream) {
    if (rows <= 0 || count <= 0) return;
    const ClipJobs c = clip_jobs(jobs, count, rows);
    const int per_block = 256 / WARP;
    clip_many_kernel<From><<<grid(c.first[count], per_block), 256, 0, cs(stream)>>>(c, count, from);
    check_launch("inkling_clip_many");
}
}  // namespace

void inkling_clip_many(const InklingClipJob* jobs, int count, int rows, int64_t pos, void* stream,
                       const long long* at) {
    if (at) clip_many(jobs, count, rows, ClipAt{at}, stream);
    else clip_many(jobs, count, rows, ClipFrom{(long long) pos}, stream);
}

void inkling_clip_many_seqs(const InklingClipJob* jobs, int count, int rows, const int64_t* pos, void* stream) {
    if (rows > kInklingSeqRows)
        throw std::invalid_argument("inkling_clip_many_seqs: at most " + std::to_string(kInklingSeqRows) + " rows");
    ClipSeqs from{};
    for (int t = 0; t < rows; ++t) from.pos[t] = (long long) pos[t];
    clip_many(jobs, count, rows, from, stream);
}

namespace {
KvArgs kv_args(const float* kraw, const float* vraw, float* k_state, float* v_state, const float* k_w, const float* v_w,
               const float* k_taps, const float* v_taps, int kv_heads, int rows, float k_clip, float eps,
               const InklingKvDst& dst, int64_t pos, float* k_out, float* v_out) {
    KvArgs a{};
    a.raw[0] = kraw, a.raw[1] = vraw;
    a.state[0] = k_state, a.state[1] = v_state;
    a.w[0] = k_w, a.w[1] = v_w;
    a.taps[0] = k_taps, a.taps[1] = v_taps;
    a.out[0] = k_out, a.out[1] = v_out;
    a.dst = dst;
    a.kv_heads = kv_heads, a.rows = rows, a.k_clip = k_clip, a.eps = eps;
    a.pos = (long long) pos;
    return a;
}
}  // namespace

void inkling_kv_store(const float* kraw, const float* vraw, float* k_state, float* v_state, const float* k_w,
                      const float* v_w, const float* k_taps, const float* v_taps, int kv_heads, int dim, int rows,
                      float k_clip, float eps, const InklingKvDst& dst, int64_t pos, float* k_out, float* v_out,
                      void* stream, const long long* at) {
    if (rows <= 0) return;
    if (dim != AT_DIM) throw std::invalid_argument("inkling_kv_store: heads of 128 values only");
    const int width = kv_heads * dim;
    if (rows > KV_ROWS) {   // a prompt batch: the separate calls
        if (at) throw std::invalid_argument("inkling_kv_store: the position on the card takes at most 16 tokens");
        if (!k_out || !v_out) throw std::invalid_argument("inkling_kv_store: more than 16 tokens need k_out and v_out");
        inkling_conv(kraw, k_state, k_w, k_taps, kv_heads, k_out, width, rows, stream);
        inkling_conv(vraw, v_state, v_w, v_taps, kv_heads, v_out, width, rows, stream);
        inkling_clip(k_out, 0, nullptr, k_out, dim, kv_heads, rows, k_clip, InklingTau{}, pos, stream);
        rms_norm(v_out, nullptr, v_out, dim, rows * kv_heads, eps, stream);
        for (int side = 0; side < 2; ++side) {
            const float* x = side ? v_out : k_out;
            int8_t* codes = side ? dst.vc : dst.kc;
            uint16_t* scales = side ? dst.vs : dst.ks;
            uint16_t* cache = side ? dst.v : dst.k;
            for (int done = 0; done < rows;) {
                const int64_t p = pos + done;
                const int64_t r = dst.ring > 0 ? p % dst.ring : p;
                const int m = dst.ring > 0 ? (int) std::min<int64_t>(rows - done, dst.ring - r) : rows - done;
                if (codes) store_q8(x + size_t(done) * width, codes, scales, width, r, m, stream);
                else store_f16(x + size_t(done) * width, cache, width, r, m, stream);
                done += m;
            }
        }
        check_launch("inkling_kv_store");
        return;
    }
    const KvArgs a = kv_args(kraw, vraw, k_state, v_state, k_w, v_w, k_taps, v_taps, kv_heads, rows, k_clip, eps, dst,
                             pos, k_out, v_out);
    if (at) kv_store_kernel<<<dim3(kv_heads, 2), AT_DIM, 0, cs(stream)>>>(a, KvAt{{}, at});
    else kv_store_kernel<<<dim3(kv_heads, 2), AT_DIM, 0, cs(stream)>>>(a, KvOne{});
    check_launch("inkling_kv_store");
}

void inkling_kv_store_clip(const float* kraw, const float* vraw, float* k_state, float* v_state, const float* k_w,
                           const float* v_w, const float* k_taps, const float* v_taps, int kv_heads, int dim, int rows,
                           float k_clip, float eps, const InklingKvDst& dst, int64_t pos, float* k_out, float* v_out,
                           const InklingClipJob* jobs, int count, void* stream, const long long* at) {
    if (rows <= 0) return;
    if (rows > KV_ROWS || count <= 0) {
        inkling_kv_store(kraw, vraw, k_state, v_state, k_w, v_w, k_taps, v_taps, kv_heads, dim, rows, k_clip, eps, dst,
                         pos, k_out, v_out, stream, at);
        inkling_clip_many(jobs, count, rows, pos, stream, at);
        return;
    }
    if (dim != AT_DIM) throw std::invalid_argument("inkling_kv_store_clip: heads of 128 values only");
    const KvArgs a = kv_args(kraw, vraw, k_state, v_state, k_w, v_w, k_taps, v_taps, kv_heads, rows, k_clip, eps, dst,
                             pos, k_out, v_out);
    const ClipJobs c = clip_jobs(jobs, count, rows);
    const int blocks = 2 * kv_heads + grid(c.first[count], AT_DIM / WARP);
    if (at) kv_clip_kernel<<<blocks, AT_DIM, 0, cs(stream)>>>(a, KvAt{{}, at}, c, count, ClipAt{at});
    else kv_clip_kernel<<<blocks, AT_DIM, 0, cs(stream)>>>(a, KvOne{}, c, count, ClipFrom{(long long) pos});
    check_launch("inkling_kv_store_clip");
}

void inkling_kv_store_seqs(const float* kraw, const float* vraw, const InklingKvSeq* seqs, const float* k_w,
                           const float* v_w, const float* k_taps, const float* v_taps, int kv_heads, int dim,
                           int rows, float k_clip, float eps, float* k_out, float* v_out, void* stream) {
    if (rows <= 0) return;
    if (dim != AT_DIM) throw std::invalid_argument("inkling_kv_store_seqs: heads of 128 values only");
    if (rows > kInklingSeqRows)
        throw std::invalid_argument("inkling_kv_store_seqs: at most " + std::to_string(kInklingSeqRows) + " rows");
    KvArgs a{};
    a.raw[0] = kraw, a.raw[1] = vraw;
    a.w[0] = k_w, a.w[1] = v_w;
    a.taps[0] = k_taps, a.taps[1] = v_taps;
    a.out[0] = k_out, a.out[1] = v_out;
    a.kv_heads = kv_heads, a.rows = rows, a.k_clip = k_clip, a.eps = eps;
    KvSeqs r{};
    for (int t = 0; t < rows; ++t) r.row[t] = seqs[t];
    kv_store_kernel<<<dim3(kv_heads, 2, rows), AT_DIM, 0, cs(stream)>>>(a, r);
    check_launch("inkling_kv_store_seqs");
}

}  // namespace strata::poly
