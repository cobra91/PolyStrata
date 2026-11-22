// src/poly/mmq.hpp - llama.cpp's quantized matrix products (ggml-cuda mmq.cuh, MIT) for the prompt path: the rows
// of many tokens against matrices that stay in their GGUF type.  Built for the types the model's GGUFs keep
// their matrices in: the K-quants, the I-quants and the older Q4, Q5 and Q8.
//
// Every call enqueues on the given CUDA stream.  Pointers are device pointers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::poly {

class Mmq {
public:
    Mmq();
    ~Mmq();
    Mmq(const Mmq&) = delete;
    Mmq& operator=(const Mmq&) = delete;

    /// The products a caller lets a matrix take besides the default ones.
    ///   q2_0: Q2_0's tiles (Engine::lowbit_experts_).
    ///   nvfp4_int8: on HIP, NVFP4's int8 tiles (mmq_nvfp4.cu) unless STRATA_PREFILL_NVFP4 keeps them out
    ///     (Engine::nvfp4_int8_); CUDA takes them for NVFP4 always.
    enum Allow : unsigned { none = 0, q2_0 = 1u << 0, nvfp4_int8 = 1u << 2 };

    /// This build multiplies ggml_type, and the card has a tile for a matrix of [w_rows, w_cols] of it.
    static bool fits(int ggml_type, int64_t w_rows, int64_t w_cols, Allow allow = none);
    /// Bytes of `rows` rows of `cols` values as quantize() writes them.
    static size_t q8_bytes(int64_t rows, int64_t cols);
    /// quantize() keeps the rows' block sums in halves for a matrix of ggml_type (guard_half_sums()).
    static bool half_sums(int ggml_type);
    /// A row's product with a matrix of ggml_type is the same to the bit however the rows are split between
    /// launches.  Not on a card where llama.cpp splits a tile's sum between blocks by the launch's tile count
    /// (stream-k: NVIDIA from Volta, AMD CDNA): there another split adds the partial sums in another order.
    static bool split_exact(int ggml_type, int64_t w_rows);

    /// The rows of a product with a matrix of ggml_type: row i of xq is row ids[i] of x, or row i when ids is null.
    /// x has ld floats a row, 16-byte aligned.  The rows are interleaved in xq, so a product takes the xq of one
    /// call whole.
    static void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld,
                         int64_t rows, void* stream);
    /// The scratch the products take at their first launch with a split of tiles (llama.cpp's stream-k fixup, at
    /// most a tile of floats a multiprocessor), taken now: it is VRAM a card that later has none to spare would
    /// fail to get (Inkling 2 at a long prompt after short ones).
    void reserve_fixup();

    /// One launch over n matrices of [w_rows, w_cols], `stride` bytes apart from w: matrix e times the rows
    /// [bounds[e], bounds[e + 1]) of xq, which quantize() wrote with total_rows rows.  Row i's product goes to row
    /// ids[i] of dst, ld_dst floats a row.  bounds (n + 1 values) and ids are on the device.  max_rows is the most
    /// rows one of the matrices has.  With one matrix, bounds and ids may both be null: it takes every row, and
    /// row i goes to row i.  Throws std::runtime_error.
    void mul(const void* w, int ggml_type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
             const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
             int64_t ld_dst, void* stream);

    /// A tile of mul_tiles(): rows [row, row + rows) of the call's rows times matrix `mat`, in a tile `width`
    /// columns wide (a width plan_tiles() chose, rows <= width).
    struct alignas(16) Tile {
        int32_t row, rows, mat, width;
    };
    /// mul_tiles() has kernels for ggml_type on this card, and a matrix of [w_rows, w_cols] of it takes them.  Q2_0
    /// only when `q2_0`, as fits().  CUDA builds only.
    static bool tiles_fit(int ggml_type, int64_t w_rows, int64_t w_cols, bool q2_0 = false);
    /// Appends to `out` the tiles of matrix mat[i] times the rows [first[i], first[i] + rows[i]), for each i < n with
    /// rows: as many 128 rows wide as it fills, then one of the narrowest width that holds the rest; the matrices
    /// with the most rows first, a matrix's tiles one after the other (they read the same weights).  On a card whose
    /// narrow tiles take another block shape (RDNA4), those come after the others, a launch of their own.
    static void plan_tiles(int ggml_type, const int32_t* first, const int32_t* rows, const int32_t* mat, int n,
                           std::vector<Tile>& out);
    /// The product over a table of n_tiles tiles, on the device and here (one launch, or one for each block shape
    /// the table has: see plan_tiles()): matrix m is [w_rows, w_cols] at w + at[m] bytes (at
    /// on the device); a tile's rows are rows of xq counted from row0, which quantize() wrote with xq_rows rows, and
    /// row i's product goes to row ids[i - row0] of dst, ld_dst floats a row.  Each output is one block's sum over
    /// the whole of K, in the order llama.cpp's MMQ adds a tile it does not split (stream-k): the same to the bit
    /// whatever the tiles' widths and however the tiles are shared out between launches.  Throws std::runtime_error.
    void mul_tiles(const void* w, const int64_t* at, int ggml_type, int64_t w_rows, int64_t w_cols, const void* xq,
                   int64_t row0, int64_t xq_rows, const Tile* tiles, const Tile* host_tiles, int n_tiles,
                   const int32_t* ids, float* dst, int64_t ld_dst, void* stream);

private:
    void* ctx_ = nullptr;   ///< llama.cpp's launch context: a small scratch pool
};

}  // namespace strata::poly
