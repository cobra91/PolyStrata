// src/poly/mmq_tiles_test.cu - two ways of cutting a prompt batch's expert products into launches, against one launch:
//
// 1. The size-class launches of STRATA_POLY_STAGED_TILES (engine.cpp, moe_rows) give the same results as one launch:
//    to the bit, except on a card whose MMQ uses stream-k (Mmq::split_exact()), where they are the same products added
//    in another order and must agree within rounding.  llama.cpp's MMQ takes a launch's tile width from the largest
//    expert in it, so the engine launches the experts of up to N rows apart from the larger ones, each class with
//    bounds in which the other classes' experts have no rows, over rows laid out class after class.  That layout is
//    built here the way the engine builds it, for row counts with many small experts, a few large, empty ones and
//    equal ones.
// 2. Mmq::mul_tiles() (STRATA_POLY_EXPERT_TILES), for each type it has kernels for, against llama.cpp's MMQ launched
//    so that it splits no tile (a grid of exactly the card's multiprocessors times the row tiles, every tile its own
//    block): to the bit on every card, with the tiles in one launch and in several, from row 0 and from a later row.
//    How many rows differ from llama.cpp's own launch (stream-k on most cards) is printed, not checked.
//
// usage: poly_mmq_tiles_test            (exit 0: every case agrees; 77: no card)
//        poly_mmq_tiles_test bench FILE  Inkling 2 Mini's expert shapes, rows per expert from FILE (a line of counts
//                                        a layer call), llama.cpp's launches against mul_tiles(): microseconds, TOPS
#include "mmq.hpp"

#include <cuda_runtime.h>

#include "ggml.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <random>
#include <vector>

using strata::poly::Mmq;

namespace {

struct Quant { const char* name; int type; size_t block; };   // ggml_type, bytes of a block of 256 values
constexpr Quant kQuants[] = {{"Q4_K", 12, 144}, {"Q5_K", 13, 176}, {"Q6_K", 14, 210}};
constexpr int kRows = 256, kCols = 512;   // a matrix of the shapes the experts have, 256 rows a multiple of 128

bool ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("%s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

int tile_of(const std::vector<int>& t, int rows) {
    int k = 0;
    while (k < (int) t.size() && rows > t[(size_t) k]) ++k;
    return k;
}

// one case: the rows of each expert, the classes' limits; returns the number of rows that differ from one launch,
// to the bit where `exact`, else by more than 8 float epsilons of the case's largest value
int run_case(const Quant& q, const std::vector<int>& rows, const std::vector<int>& limits, bool exact,
             std::mt19937& rng) {
    const int n = (int) rows.size();
    const size_t per = (size_t) kRows * (kCols / 256) * q.block;
    std::vector<uint8_t> w(per * n + 4096, 0);   // zero bytes behind the matrices (kTail)
    for (size_t i = 0; i < per * n; ++i) w[i] = (uint8_t) rng();
    for (size_t b = 0; b < per * n / q.block; ++b) {   // finite block scales: a half of 2^-3 where each type keeps its d
        uint8_t* p = &w[b * q.block];
        if (q.type == 14) { p[q.block - 2] = 0; p[q.block - 1] = 0x30; }
        else { p[0] = 0; p[1] = 0x30; p[2] = 0; p[3] = 0x30; }
    }
    const int total = std::accumulate(rows.begin(), rows.end(), 0);
    std::vector<float> x((size_t) total * kCols);
    for (auto& v : x) v = (float) ((rng() & 0xffff) / 32768.0 - 1.0);
    std::vector<int32_t> start(n + 1, 0);
    for (int e = 0; e < n; ++e) start[(size_t) e + 1] = start[(size_t) e] + rows[(size_t) e];

    // the reference: rows by expert, one launch
    std::vector<int32_t> iota(total), gather_ref(total);
    std::iota(iota.begin(), iota.end(), 0);
    std::iota(gather_ref.begin(), gather_ref.end(), 0);
    // the classes: rows class after class (an expert's rows in order), a bounds array for each class
    const int nclass = (int) limits.size() + 1;
    std::vector<int32_t> gather_cls, bounds((size_t) nclass * (n + 1)), most(nclass, 0);
    for (int k = 0, at = 0; k < nclass; ++k) {
        for (int e = 0; e < n; ++e) {
            bounds[(size_t) k * (n + 1) + e] = at;
            if (rows[(size_t) e] > 0 && tile_of(limits, rows[(size_t) e]) == k) {
                for (int r = 0; r < rows[(size_t) e]; ++r) gather_cls.push_back(start[(size_t) e] + r);
                at += rows[(size_t) e];
                most[(size_t) k] = std::max(most[(size_t) k], rows[(size_t) e]);
            }
        }
        bounds[(size_t) k * (n + 1) + n] = at;
    }
    std::vector<int32_t> ref_bounds(start.begin(), start.end());

    Mmq mmq;
    uint8_t* dw = (uint8_t*) upload(w);
    float* dx = upload(x);
    int32_t *d_iota = upload(iota), *d_gref = upload(gather_ref), *d_gcls = upload(gather_cls), *d_rb = upload(ref_bounds),
            *d_cb = upload(bounds);
    void* xq = nullptr;
    float *dref = nullptr, *dcls = nullptr;
    bool good = dw && dx && d_iota && d_gref && d_gcls && d_rb && d_cb;
    good = good && ok(cudaMalloc(&xq, Mmq::q8_bytes(total, kCols)), "xq") &&
           ok(cudaMalloc((void**) &dref, (size_t) total * kRows * 4), "ref") &&
           ok(cudaMalloc((void**) &dcls, (size_t) total * kRows * 4), "cls");
    int bad = -1;
    if (good) {
        cudaMemset(dref, 0, (size_t) total * kRows * 4);
        cudaMemset(dcls, 0, (size_t) total * kRows * 4);
        const int maxrows = *std::max_element(rows.begin(), rows.end());
        mmq.quantize(dx, d_gref, xq, q.type, kCols, kCols, total, nullptr);
        mmq.mul(dw, q.type, kRows, kCols, per, n, xq, d_rb, d_iota, total, maxrows, dref, kRows, nullptr);
        mmq.quantize(dx, d_gcls, xq, q.type, kCols, kCols, total, nullptr);
        for (int k = 0; k < nclass; ++k)
            if (most[(size_t) k] > 0)
                mmq.mul(dw, q.type, kRows, kCols, per, n, xq, d_cb + (size_t) k * (n + 1), d_iota, total, most[(size_t) k],
                        dcls, kRows, nullptr);
        if (ok(cudaDeviceSynchronize(), "run")) {
            std::vector<float> a((size_t) total * kRows), b((size_t) total * kRows);
            cudaMemcpy(a.data(), dref, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), dcls, b.size() * 4, cudaMemcpyDeviceToHost);
            float big = 0;
            for (float v : a) big = std::max(big, std::fabs(v));
            const float tol = 8 * FLT_EPSILON * big;
            const auto near = [tol](float u, float v) { return std::fabs(u - v) <= tol; };
            bad = 0;   // row i of the class layout is row gather_cls[i] of the reference
            for (int i = 0; i < total; ++i) {
                const float* c = &b[(size_t) i * kRows];
                const float* r = &a[(size_t) gather_cls[(size_t) i] * kRows];
                if (exact ? std::memcmp(c, r, kRows * 4) != 0 : !std::equal(c, c + kRows, r, near)) ++bad;
            }
        }
    }
    for (void* p : {(void*) dw, (void*) dx, (void*) d_iota, (void*) d_gref, (void*) d_gcls, (void*) d_rb, (void*) d_cb, xq,
                    (void*) dref, (void*) dcls})
        if (p) cudaFree(p);
    return bad;
}


// ---- 2. Mmq::mul_tiles() ----

struct TileType { const char* name; int type; };
constexpr TileType kTileTypes[] = {{"Q8_0", 8},   {"IQ1_S", 19}, {"IQ2_XXS", 16}, {"IQ2_XS", 17},
                                   {"IQ2_S", 22}, {"IQ4_NL", 20}, {"Q2_0", 42}};

// n matrices of [w_rows, w_cols] of ggml_type `type`, each `per` bytes, random blocks with a finite scale: every type
// here keeps its block's half d first, set to a half in [0.125, 0.25).  Zero bytes behind them (kTail).
std::vector<uint8_t> random_matrices(int type, int64_t w_rows, int64_t w_cols, int n, size_t* per, std::mt19937& rng) {
    const size_t bs = ggml_type_size((ggml_type) type);
    const int64_t bk = ggml_blck_size((ggml_type) type);
    *per = (size_t) w_rows * (size_t) (w_cols / bk) * bs;
    std::vector<uint8_t> w(*per * n + 4096, 0);
    for (size_t i = 0; i < *per * n; ++i) w[i] = (uint8_t) rng();
    for (size_t b = 0; b < *per * n / bs; ++b) w[b * bs + 1] = (uint8_t) (0x30 | (rng() & 3));
    return w;
}

struct Buffers {   // cudaFree on the way out
    std::vector<void*> p;
    template <typename T> T* add(T* q) { if (q) p.push_back((void*) q); return q; }
    void* alloc(size_t bytes) { void* q = nullptr; return cudaMalloc(&q, bytes) == cudaSuccess ? add(q) : nullptr; }
    ~Buffers() { for (void* q : p) cudaFree(q); }
};

// One case: matrix e times rows[e] rows.  Returns the rows that differ from the reference in any of the three ways of
// launching the tiles (-1: a CUDA error); *vs_launch gets the rows in which llama.cpp's own launch differs from it.
int run_tiles(const TileType& q, const std::vector<int>& rows, int nsm, std::mt19937& rng, int* vs_launch) {
    constexpr int64_t kR = 256, kC = 768;   // three of MMQ's K steps
    const int n = (int) rows.size();
    size_t per = 0;
    const std::vector<uint8_t> w = random_matrices(q.type, kR, kC, n, &per, rng);
    std::vector<int32_t> start(n + 1, 0), mat(n);
    for (int e = 0; e < n; ++e) start[(size_t) e + 1] = start[(size_t) e] + rows[(size_t) e];
    std::iota(mat.begin(), mat.end(), 0);
    const int total = start[(size_t) n];
    std::vector<float> x((size_t) total * kC);
    for (auto& v : x) v = (float) ((rng() & 0xffff) / 32768.0 - 1.0);
    std::vector<int32_t> iota(total);
    std::iota(iota.begin(), iota.end(), 0);
    std::vector<int64_t> at(n);
    for (int e = 0; e < n; ++e) at[(size_t) e] = (int64_t) e * (int64_t) per;
    // the tiles of every matrix, and of the second half's from a later row (as a staging round's down product)
    const int h = n / 2, row0 = start[(size_t) h];
    std::vector<Mmq::Tile> all, late;
    Mmq::plan_tiles(q.type, start.data(), rows.data(), mat.data(), n, all);
    Mmq::plan_tiles(q.type, start.data() + h, rows.data() + h, mat.data() + h, n - h, late);
    std::vector<Mmq::Tile> tiles(all);
    tiles.insert(tiles.end(), late.begin(), late.end());

    Buffers b;
    Mmq mmq;
    const size_t out = (size_t) total * kR * sizeof(float);
    uint8_t* dw = b.add(upload(w));
    float* dx = b.add(upload(x));
    int32_t *d_iota = b.add(upload(iota)), *d_start = b.add(upload(start));
    int64_t* d_at = b.add(upload(at));
    Mmq::Tile* d_tiles = b.add(upload(tiles));
    void *xq = b.alloc(Mmq::q8_bytes(total, kC)), *xq2 = b.alloc(Mmq::q8_bytes(total, kC));
    float *ref = (float*) b.alloc(out), *own = (float*) b.alloc(out), *one = (float*) b.alloc(out),
          *three = (float*) b.alloc(out), *from = (float*) b.alloc(out);
    if (!dw || !dx || !d_iota || !d_start || !d_at || !d_tiles || !xq || !xq2 || !ref || !own || !one || !three || !from)
        return -1;
    for (float* d : {ref, own, one, three, from}) cudaMemset(d, 0, out);
    const int most = *std::max_element(rows.begin(), rows.end());
    Mmq::quantize(dx, nullptr, xq, q.type, kC, kC, total, nullptr);
    Mmq::quantize(dx + (size_t) row0 * kC, nullptr, xq2, q.type, kC, kC, total - row0, nullptr);
    // the reference: as many column tiles as the card has multiprocessors, so llama.cpp's grid is every tile a block
    mmq.mul(dw, q.type, kR, kC, per, n, xq, d_start, d_iota, total, 128 * (int64_t) nsm, ref, kR, nullptr);
    mmq.mul(dw, q.type, kR, kC, per, n, xq, d_start, d_iota, total, most, own, kR, nullptr);
    const int nt = (int) all.size();
    mmq.mul_tiles(dw, d_at, q.type, kR, kC, xq, 0, total, d_tiles, tiles.data(), nt, d_iota, one, kR, nullptr);
    for (int k = 0; k < 3; ++k) {   // the same tiles in three launches
        const int a = nt * k / 3, z = nt * (k + 1) / 3;
        mmq.mul_tiles(dw, d_at, q.type, kR, kC, xq, 0, total, d_tiles + a, tiles.data() + a, z - a, d_iota, three, kR,
                      nullptr);
    }
    mmq.mul_tiles(dw, d_at, q.type, kR, kC, xq2, row0, total - row0, d_tiles + nt, tiles.data() + nt, (int) late.size(),
                  d_iota + row0, from, kR, nullptr);
    if (!ok(cudaDeviceSynchronize(), "run")) return -1;
    std::vector<float> r((size_t) total * kR), o(r.size()), u(r.size()), t(r.size()), f(r.size());
    cudaMemcpy(r.data(), ref, out, cudaMemcpyDeviceToHost);
    cudaMemcpy(o.data(), own, out, cudaMemcpyDeviceToHost);
    cudaMemcpy(u.data(), one, out, cudaMemcpyDeviceToHost);
    cudaMemcpy(t.data(), three, out, cudaMemcpyDeviceToHost);
    cudaMemcpy(f.data(), from, out, cudaMemcpyDeviceToHost);
    int bad = 0;
    *vs_launch = 0;
    for (int i = 0; i < total; ++i) {
        const size_t a = (size_t) i * kR, bytes = kR * sizeof(float);
        const bool same = std::memcmp(&u[a], &r[a], bytes) == 0 && std::memcmp(&t[a], &r[a], bytes) == 0 &&
                          (i < row0 || std::memcmp(&f[a], &r[a], bytes) == 0);
        bad += !same;
        *vs_launch += std::memcmp(&o[a], &r[a], bytes) != 0;
    }
    return bad;
}

int test_tiles(std::mt19937& rng) {
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    struct Dist { const char* name; std::vector<int> rows; };
    std::vector<Dist> dists;
    dists.push_back({"widths' edges", {1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 0,
                                       255, 256, 257, 383}});
    {   // a layer of a prompt batch: 1,024 tokens of 8 experts in 128, a few popular
        std::vector<int> r(128);
        for (int i = 0; i < 1024 * 8; ++i) {
            const double u = (double) (rng() % 1000000) / 1000000.0;
            ++r[(size_t) std::min(127, (int) (128 * u * u))];
        }
        dists.push_back({"skewed layer", r});
    }
    {
        std::vector<int> r;
        for (int e = 0; e < 40; ++e) r.push_back(rng() % 3 == 0 ? 0 : 1 + (int) (rng() % 200));
        dists.push_back({"random", r});
    }
    int failures = 0, cases = 0;
    for (const TileType& q : kTileTypes) {
        if (!Mmq::tiles_fit(q.type, 256, 768, true)) {
            std::printf("tiles %s: no kernel in this build or on this card: skipped\n", q.name);
            continue;
        }
        for (const Dist& d : dists) {
            int vs = 0;
            const int bad = run_tiles(q, d.rows, nsm, rng, &vs);
            ++cases;
            const int total = std::accumulate(d.rows.begin(), d.rows.end(), 0);
            if (bad != 0) ++failures;
            std::printf("%s tiles %s %s: %d of %d rows differ from llama.cpp's unsplit tiles; llama.cpp's own launch "
                        "differs from them in %d\n", bad == 0 ? "ok  " : "FAIL", q.name, d.name, bad, total, vs);
        }
    }
    if (failures) std::printf("FAIL: %d of %d tile cases\n", failures, cases);
    return failures;
}

// ---- bench ----

struct Product { const char* name; int type; int64_t w_rows, w_cols; };

double time_ms(const std::function<void()>& f) {
    cudaEvent_t a, z;
    cudaEventCreate(&a);
    cudaEventCreate(&z);
    f();   // warm
    std::vector<float> t;
    for (int k = 0; k < 5; ++k) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(z);
        cudaEventSynchronize(z);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, z);
        t.push_back(ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(z);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Inkling 2 Mini's expert products (n_embd 2304, n_ff_exp 1152, 128 experts) for each line of counts: the engine's
// default launches - a launch for each class of rows (up to 32, up to 128, more; the rows laid out class after class)
// over all the matrices, and over runs of 1 to 3 matrices as scattered cache slots make them - against one launch
// of mul_tiles().
int bench(const char* file) {
    std::ifstream in(file);
    std::vector<std::vector<int>> calls;
    for (std::string line; std::getline(in, line);) {
        std::istringstream ss(line);
        std::vector<int> r;
        for (int v; ss >> v;) r.push_back(v);
        if (r.size() == 128) calls.push_back(r);
    }
    if (calls.empty()) { std::printf("bench: no lines of 128 counts in %s\n", file); return 2; }
    const Product products[] = {{"IQ4_NL gate_up", 20, 2304, 2304}, {"IQ4_NL down", 20, 2304, 1152},
                                {"Q8_0 gate_up", 8, 2304, 2304},    {"Q8_0 down", 8, 2304, 1152},
                                {"IQ2_S gate_up", 22, 2304, 2304},  {"IQ2_XXS gate_up", 16, 2304, 2304},
                                {"IQ1_S gate_up", 19, 2304, 2304},  {"Q2_0 down", 42, 2304, 1152}};
    std::mt19937 rng(11);
    constexpr int n = 128, kLimits[] = {32, 128};
    for (const Product& p : products) {
        if (!Mmq::tiles_fit(p.type, p.w_rows, p.w_cols, true)) continue;
        size_t per = 0;
        Buffers b;
        Mmq mmq;
        const std::vector<uint8_t> w = random_matrices(p.type, p.w_rows, p.w_cols, n, &per, rng);
        uint8_t* dw = b.add(upload(w));
        const int most_total = 1024 * 8;
        std::vector<float> x((size_t) most_total * p.w_cols);
        for (auto& v : x) v = (float) ((rng() & 0xffff) / 32768.0 - 1.0);
        float* dx = b.add(upload(x));
        std::vector<int32_t> iota(most_total);
        std::iota(iota.begin(), iota.end(), 0);
        int32_t* d_iota = b.add(upload(iota));
        std::vector<int64_t> at(n);
        for (int e = 0; e < n; ++e) at[(size_t) e] = (int64_t) e * (int64_t) per;
        int64_t* d_at = b.add(upload(at));
        void* xq = b.alloc(Mmq::q8_bytes(most_total, p.w_cols));
        float* dst = (float*) b.alloc((size_t) most_total * p.w_rows * sizeof(float));
        Mmq::Tile* d_tiles = (Mmq::Tile*) b.alloc((size_t) (n + most_total / 128 + 1) * sizeof(Mmq::Tile));
        int32_t* d_cb = (int32_t*) b.alloc(3 * (n + 1) * sizeof(int32_t));
        if (!dw || !dx || !d_iota || !d_at || !xq || !dst || !d_tiles || !d_cb) { std::printf("bench: no VRAM\n"); return 1; }
        double sa = 0, sb = 0, sc = 0, ops = 0;
        int la = 0, lb = 0;
        for (const auto& rows : calls) {
            int total = 0;
            for (int r : rows) total += r;
            if (total > most_total) continue;
            const auto cls = [&](int r) { return r <= kLimits[0] ? 0 : r <= kLimits[1] ? 1 : 2; };
            // the engine's layout: class after class, an expert's rows together; per class its bounds
            std::vector<int32_t> cb(3 * (n + 1)), first(n), mat(n);
            int most[3] = {};
            for (int c = 0, a = 0; c < 3; ++c) {
                for (int e = 0; e < n; ++e) {
                    cb[(size_t) c * (n + 1) + e] = a;
                    if (rows[(size_t) e] > 0 && cls(rows[(size_t) e]) == c) {
                        first[(size_t) e] = a;
                        a += rows[(size_t) e];
                        most[c] = std::max(most[c], rows[(size_t) e]);
                    }
                }
                cb[(size_t) c * (n + 1) + n] = a;
            }
            std::iota(mat.begin(), mat.end(), 0);
            std::vector<Mmq::Tile> tiles;
            Mmq::plan_tiles(p.type, first.data(), rows.data(), mat.data(), n, tiles);
            cudaMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(Mmq::Tile), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cb, cb.data(), cb.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            Mmq::quantize(dx, nullptr, xq, p.type, p.w_cols, p.w_cols, total, nullptr);
            // runs of 1 to 3 matrices
            std::vector<std::pair<int, int>> runs;
            for (int j = 0; j < n;) {
                const int m = std::min(n - j, 1 + (int) (rng() % 3));
                runs.push_back({j, m});
                j += m;
            }
            int n_a = 0, n_b = 0;
            const double ta = time_ms([&] {
                n_a = 0;
                for (int c = 0; c < 3; ++c)
                    if (most[c] > 0) {
                        mmq.mul(dw, p.type, p.w_rows, p.w_cols, per, n, xq, d_cb + (size_t) c * (n + 1), d_iota, total,
                                most[c], dst, p.w_rows, nullptr);
                        ++n_a;
                    }
            });
            const double tb = time_ms([&] {
                n_b = 0;
                for (const auto& [j, m] : runs) {
                    int top[3] = {};
                    for (int e = j; e < j + m; ++e)
                        if (rows[(size_t) e] > 0) top[cls(rows[(size_t) e])] = std::max(top[cls(rows[(size_t) e])], rows[(size_t) e]);
                    for (int c = 0; c < 3; ++c)
                        if (top[c] > 0) {
                            mmq.mul(dw + (size_t) j * per, p.type, p.w_rows, p.w_cols, per, m, xq,
                                    d_cb + (size_t) c * (n + 1) + j, d_iota, total, top[c], dst, p.w_rows, nullptr);
                            ++n_b;
                        }
                }
            });
            const double tc = time_ms([&] {
                mmq.mul_tiles(dw, d_at, p.type, p.w_rows, p.w_cols, xq, 0, total, d_tiles, tiles.data(), (int) tiles.size(),
                              d_iota, dst, p.w_rows, nullptr);
            });
            sa += ta; sb += tb; sc += tc; la += n_a; lb += n_b;
            ops += 2.0 * total * (double) p.w_rows * (double) p.w_cols;
        }
        if (!ok(cudaDeviceSynchronize(), "bench")) return 1;
        const auto tops = [&](double ms) { return ops / (ms * 1e-3) / 1e12; };
        const double nc = (double) calls.size();
        std::printf("bench %-16s %zu calls: classes %.1f us (%.1f launches) %.1f TOPS | runs %.1f us (%.1f launches) "
                    "%.1f TOPS | tiles %.1f us (1 launch) %.1f TOPS\n", p.name, calls.size(), 1000 * sa / nc, la / nc,
                    tops(sa), 1000 * sb / nc, lb / nc, tops(sb), 1000 * sc / nc, tops(sc));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no card: skipped\n");
        return 77;
    }
    if (argc == 3 && std::strcmp(argv[1], "bench") == 0) return bench(argv[2]);
    std::mt19937 rng(7);
    struct Dist { const char* name; std::vector<int> rows; };
    std::vector<Dist> dists;
    {   // many small experts and a few large, as a prompt batch's staged experts are
        std::vector<int> r;
        for (int e = 0; e < 96; ++e) r.push_back(1 + (int) (rng() % 16) * (rng() % 4 == 0 ? 6 : 1));
        for (int e : {5, 40, 77}) r[(size_t) e] = 200 + (int) (rng() % 300);
        dists.push_back({"skewed", r});
    }
    {   // empty experts between the others, one with a single row
        std::vector<int> r(40, 0);
        for (int e = 0; e < 40; e += 3) r[(size_t) e] = 1 + (int) (rng() % 90);
        r[1] = 1;
        dists.push_back({"with empty", r});
    }
    dists.push_back({"all small", std::vector<int>(33, 7)});
    dists.push_back({"one class edge", std::vector<int>{8, 9, 16, 17, 32, 33, 64, 65, 8, 16, 32, 64}});
    const std::vector<std::vector<int>> limits = {{8, 16, 32}, {16}, {4, 8, 12, 16, 24, 32, 48}, {64}};
    int failures = 0, cases = 0;
    for (const Quant& q : kQuants) {
        const bool exact = Mmq::split_exact(q.type, kRows);
        if (!exact) std::printf("%s: this card's MMQ uses stream-k, compared within rounding\n", q.name);
        for (const Dist& d : dists)
            for (const auto& l : limits) {
                const int bad = run_case(q, d.rows, l, exact, rng);
                ++cases;
                if (bad != 0) {
                    ++failures;
                    std::printf("FAIL %s %s, %zu classes: %d rows differ%s\n", q.name, d.name, l.size() + 1, bad,
                                exact ? "" : " by more than rounding");
                }
            }
    }
    const int tile_failures = test_tiles(rng);
    if (failures || tile_failures) {
        std::printf("FAIL: %d of %d class cases, %d tile cases\n", failures, cases, tile_failures);
        return 1;
    }
    std::printf("PASS cases=%d\n", cases);
    return 0;
}
