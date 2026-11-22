// src/poly/mmq_tiles.cuh - Mmq::mul_tiles(): llama.cpp's MMQ tiles (ggml-cuda mmq.cuh, MIT) launched over a table of
// tiles instead of a grid of every matrix's column tiles.  Included only by the per-type instance TUs (CMake:
// mmq_tiles_instance.cu.in); mmq.cu sees the declaration below.
//
// The kernel is the branch of llama.cpp's mul_mat_q that gives each block one whole tile (the one it takes where a
// launch does not use stream-k), with three changes: a block's tile - its matrix, first row, rows and width - comes
// from the table; a matrix is found by its byte offset from one base pointer; and the tile's width is the table's,
// one of kWidths, so one launch has tiles of several widths.  Each width runs llama.cpp's own
// mul_mat_q_process_tile, loaders, vec_dot and write_back for that width.  A block sums its tile over the whole of K
// in K's order, as llama.cpp does every tile it does not split, and an output's sum does not depend on the width of
// its tile: the results are the same to the bit however the rows are cut into tiles and the tiles into launches
// (poly_mmq_tiles_test checks them against llama.cpp's MMQ run so that it splits nothing).
#pragma once

#include "common.cuh"
#include "mmq.cuh"
#include "mmq.hpp"

#include <algorithm>

namespace strata::poly {

struct MmqTilesArgs {
    const char* x;           // the base the matrices' offsets count from
    const int64_t* at;       // per matrix: bytes from x
    const int* y;            // the rows, as Mmq::quantize() wrote them
    const int32_t* ids;      // row i - row0 of y goes to row ids[i - row0] of dst
    const Mmq::Tile* tiles;  // on the device
    const Mmq::Tile* host_tiles;  // the same, here: where a launch's anchor changes
    float* dst;
    int n_tiles, nrows_x, ncols_x, ncols_y, nrows_dst, row0;
};

template <ggml_type type> void mul_mat_q_tiles_case(const MmqTilesArgs& a, int cc, size_t smpbo, cudaStream_t s);

namespace mmq_tiles {

constexpr int kWidths[] = {16, 32, 48, 64, 96, 128};
constexpr int kNWidths = (int) (sizeof(kWidths) / sizeof(kWidths[0]));

// A launch has one block shape - threads and rows of the weights a block - and takes the tiles of every width that
// llama.cpp configures with the shape of its anchor, the widest such width in kWidths.  One anchor (128) on NVIDIA,
// where every width has 256 threads and 128 rows; two on RDNA4 (up to 64 columns: 128 threads, 64 rows).
constexpr __device__ bool same_shape(ggml_type type, int J, int A) {
    const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, J, false);
    const ggml_cuda_mmq_config w = ggml_cuda_mmq_get_config(type, A, false);
    return c.type != GGML_TYPE_COUNT && w.type != GGML_TYPE_COUNT && c.nthreads == w.nthreads && c.I == w.I;
}

constexpr __device__ bool anchor(ggml_type type, int A) {
    if (!same_shape(type, A, A)) return false;
    for (const int W : kWidths)
        if (W > A && same_shape(type, W, A)) return false;
    return true;
}

template <ggml_type type, int J, int A>
static __device__ __forceinline__ void tile(const char* __restrict__ x, const int* __restrict__ y,
                                            const int32_t* __restrict__ ids, float* __restrict__ dst,
                                            const Mmq::Tile t, const int it, const int blocks_per_ne00,
                                            const int nrows_x, const int stride_row_x, const int ncols_y,
                                            const int stride_col_dst, const int row0) {
    if constexpr (!same_shape(type, J, A)) {
        NO_DEVICE_CODE;
    } else {
        constexpr int warp_size = ggml_cuda_get_physical_warp_size();
        constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, false) / warp_size;
        constexpr int I = ggml_cuda_mmq_get_I(type, J, false);
        extern __shared__ int ids_dst_shared[];   // the first J ints of mul_mat_q_process_tile's shared memory
        const int r = t.row - row0;
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
            const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
            if (j0 + nwarps * warp_size > J && j >= J) break;
            ids_dst_shared[j] = j < t.rows ? ids[r + j] : 0;
        }
        __syncthreads();
        const int offset_y = r * (int) (sizeof(block_q8_1_mmq) / sizeof(int));
        mul_mat_q_process_tile<type, J, false, false>(x, it * I * stride_row_x, y + offset_y, ids_dst_shared,
                                                      dst + it * I, nullptr, nullptr, stride_row_x, ncols_y,
                                                      stride_col_dst, nrows_x - it * I - 1, t.rows - 1, 0,
                                                      blocks_per_ne00);
    }
}

// grid: (row tiles of the matrices, tiles of the table); a block reads its tile from the table.  Compiled only for an
// anchor of the architecture it is built for (a stub elsewhere).
template <ggml_type type, int A>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, A, false), ggml_cuda_mmq_get_occupancy(type, A, false))
static __global__ void mul_mat_q_tiles(const char* __restrict__ x, const int64_t* __restrict__ at,
                                       const int* __restrict__ y, const int32_t* __restrict__ ids,
                                       const Mmq::Tile* __restrict__ tiles, float* __restrict__ dst,
                                       const int blocks_per_ne00, const int nrows_x, const int stride_row_x,
                                       const int ncols_y, const int stride_col_dst, const int row0) {
    if constexpr (!anchor(type, A)) {
        NO_DEVICE_CODE;
    } else {
        const Mmq::Tile t = tiles[blockIdx.y];
        const char* xm = x + at[t.mat];
        const int it = blockIdx.x;
#define STRATA_TILE(J)                                                                                             \
    case J:                                                                                                        \
        tile<type, J, A>(xm, y, ids, dst, t, it, blocks_per_ne00, nrows_x, stride_row_x, ncols_y, stride_col_dst, \
                         row0);                                                                                    \
        break;
        switch (t.width) {
            STRATA_TILE(16)
            STRATA_TILE(32)
            STRATA_TILE(48)
            STRATA_TILE(64)
            STRATA_TILE(96)
            STRATA_TILE(128)
            default: break;
        }
#undef STRATA_TILE
    }
}

// The widths of the type on a card (host side): for each of kWidths, out[k] = its anchor, or 0 where llama.cpp has no
// configuration for it or the card has not its shared memory.  The shared memory of an anchor's launch: the most of
// its widths'.  Returns how many widths there are.
inline int widths(ggml_type type, int cc, size_t smpbo, int* anchor_of, size_t* shared_of) {
    int n = 0;
    for (int k = 0; k < kNWidths; ++k) {
        anchor_of[k] = 0;
        shared_of[k] = 0;
    }
    for (int k = 0; k < kNWidths; ++k) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, kWidths[k], false, cc);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        int a = k;   // the widest width of the same shape
        for (int m = k + 1; m < kNWidths; ++m) {
            const ggml_cuda_mmq_config d = ggml_cuda_mmq_get_config(type, kWidths[m], false, cc);
            if (d.type != GGML_TYPE_COUNT && d.nthreads == c.nthreads && d.I == c.I) a = m;
        }
        anchor_of[k] = kWidths[a];
        shared_of[a] = std::max(shared_of[a], mmq_get_nbytes_shared(c, cc));
        ++n;
    }
    return n;
}

template <ggml_type type, int A>
void launch(const MmqTilesArgs& a, int first, int count, size_t shared, int cc, cudaStream_t s) {
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_tiles<type, A>), shared);
    const ggml_cuda_mmq_config w = ggml_cuda_mmq_get_config(type, A, false, cc);
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 blocks((unsigned) (a.nrows_x / w.I), (unsigned) count, 1);
    const dim3 threads((unsigned) warp_size, (unsigned) (w.nthreads / warp_size), 1);
    const int bpr = a.ncols_x / ggml_cuda_type_traits<type>::qk;
    mul_mat_q_tiles<type, A><<<blocks, threads, shared, s>>>(a.x, a.at, a.y, a.ids, a.tiles + first, a.dst, bpr,
                                                             a.nrows_x, bpr, a.ncols_y, a.nrows_dst, a.row0);
}

}  // namespace mmq_tiles

// A launch for each run of tiles of one anchor in the table (plan_tiles() puts them one anchor after the other).
template <ggml_type type> void mul_mat_q_tiles_case(const MmqTilesArgs& a, int cc, size_t smpbo, cudaStream_t s) {
    int anchor_of[mmq_tiles::kNWidths];
    size_t shared_of[mmq_tiles::kNWidths];
    GGML_ASSERT(mmq_tiles::widths(type, cc, smpbo, anchor_of, shared_of) > 0);
    for (int i = 0; i < a.n_tiles;) {
        const int A = anchor_of[std::find(mmq_tiles::kWidths, mmq_tiles::kWidths + mmq_tiles::kNWidths,
                                          a.host_tiles[i].width) - mmq_tiles::kWidths];
        int j = i + 1;
        while (j < a.n_tiles &&
               anchor_of[std::find(mmq_tiles::kWidths, mmq_tiles::kWidths + mmq_tiles::kNWidths,
                                   a.host_tiles[j].width) - mmq_tiles::kWidths] == A)
            ++j;
        const size_t shared =
            shared_of[std::find(mmq_tiles::kWidths, mmq_tiles::kWidths + mmq_tiles::kNWidths, A) - mmq_tiles::kWidths];
        switch (A) {
#define STRATA_ANCHOR(W) case W: mmq_tiles::launch<type, W>(a, i, j - i, shared, cc, s); break;
            STRATA_ANCHOR(16)
            STRATA_ANCHOR(32)
            STRATA_ANCHOR(48)
            STRATA_ANCHOR(64)
            STRATA_ANCHOR(96)
            STRATA_ANCHOR(128)
#undef STRATA_ANCHOR
            default: GGML_ABORT("mmq tiles: width %d", a.host_tiles[i].width);
        }
        i = j;
    }
}

}  // namespace strata::poly
