// src/poly/inkling.cpp - the layers of Inkling 2 Mini.  The reference for the math is llama.cpp's
// src/models/inkling2.cpp and inkling2-loop.cpp (29b52d398b); the kernels' header has it written out
// (include/strata/poly/inkling_kernels.hpp).
#include "inkling.hpp"

#include "runtime.hpp"
#include "strata/poly/inkling_kernels.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

const char* const kCtrl[4] = {"q", "k", "v", "ffn"};

constexpr int kSparkRows = 256;   // the model's rows the DSpark drafter takes in a launch

// Unsloth's files (architecture "inkling2", the loop in its metadata) name the loop's parts otherwise; byte for byte
// they are the ggml-org file's: attn = q, mlp = ffn.
const Names kTensorNames[] = {
    {"loop_q_ctrl_down.weight", {"loop_attn_down.weight", nullptr}},
    {"loop_q_ctrl_gain.weight", {"loop_attn_gain.weight", nullptr}},
    {"loop_q_ctrl_forget.weight", {"loop_attn_forget.weight", nullptr}},
    {"loop_k_ctrl_down.weight", {"loop_k_down.weight", nullptr}},
    {"loop_k_ctrl_gain.weight", {"loop_k_gain.weight", nullptr}},
    {"loop_k_ctrl_forget.weight", {"loop_k_forget.weight", nullptr}},
    {"loop_v_ctrl_down.weight", {"loop_v_down.weight", nullptr}},
    {"loop_v_ctrl_gain.weight", {"loop_v_gain.weight", nullptr}},
    {"loop_v_ctrl_forget.weight", {"loop_v_forget.weight", nullptr}},
    {"loop_ffn_ctrl_down.weight", {"loop_mlp_down.weight", nullptr}},
    {"loop_ffn_ctrl_gain.weight", {"loop_mlp_gain.weight", nullptr}},
    {"loop_ffn_ctrl_forget.weight", {"loop_mlp_forget.weight", nullptr}},
};
const Names kMetaNames[] = {
    {"loop.start", {"loop.core_start", nullptr}},
    {"loop.end", {"loop.core_end", nullptr}},
    {"loop.count", {"num_loops", nullptr}},
};

// Where a visit's keys and values go, in its planes pl: fp16 keys and values, or q8_0 codes and scales.
InklingKvDst kv_dst(bool q8, uint8_t* const* pl, int ring) {
    InklingKvDst dst;
    if (q8) {
        dst.kc = (int8_t*) pl[0], dst.ks = (uint16_t*) pl[1];
        dst.vc = (int8_t*) pl[2], dst.vs = (uint16_t*) pl[3];
    } else {
        dst.k = (uint16_t*) pl[0], dst.v = (uint16_t*) pl[1];
    }
    dst.ring = ring;
    return dst;
}

// The caches the attention of a token at `pos` reads, in the planes pl.
InklingAttnSeq attn_seq(bool q8, uint8_t* const* pl, int64_t pos, int64_t room) {
    InklingAttnSeq c;
    if (q8) {
        c.kq8 = Kvq8{(const int8_t*) pl[0], (const uint16_t*) pl[1]};
        c.vq8 = Kvq8{(const int8_t*) pl[2], (const uint16_t*) pl[3]};
    } else {
        c.kcache = (const uint16_t*) pl[0];
        c.vcache = (const uint16_t*) pl[1];
    }
    c.pos = pos;
    c.room = room;
    return c;
}

}  // namespace

// ---- visits ----------------------------------------------------------------------------------------------------

int Inkling::phys_of(int v) const {
    const int core = core_end_ - core_start_, end = core_start_ + n_passes_ * core;
    if (v < core_start_) return v;
    if (v < end) return core_start_ + (v - core_start_) % core;
    return v - (n_passes_ - 1) * core;
}

int Inkling::pass_of(int v) const {
    const int core = core_end_ - core_start_, end = core_start_ + n_passes_ * core;
    return v < core_start_ || v >= end ? 0 : (v - core_start_) / core;
}

// llama.cpp's cache slot of a visit: the physical layer for the first pass, then n_phys + the later passes' runs.
int Inkling::slot_of(int v) const {
    const int p = phys_of(v), q = pass_of(v);
    return q == 0 ? p : n_phys_ + (q - 1) * (core_end_ - core_start_) + (p - core_start_);
}

void Inkling::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

const Names* Inkling::tensor_alias(const char* name) const {
    for (const Names& n : kTensorNames)
        if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

const Names* Inkling::meta_alias(const char* key) const {
    for (const Names& n : kMetaNames)
        if (std::strcmp(n.name, key) == 0) return &n;
    return nullptr;
}

void Inkling::split_layers(const Shards& file, int& n_layer, int& keep_lo, int& keep_hi) {
    const std::string pre = file.get("general.architecture")->s + ".";
    const auto num = [&](const char* key) {
        const MetaValue* v = file.get(pre + key);
        for (const Names& n : kMetaNames)
            if (!v && std::strcmp(n.name, key) == 0)
                for (const char* other : n.other)
                    if (other && !v) v = file.get(pre + other);
        if (!v) throw std::runtime_error(std::string("unsupported model: no ") + pre + key);
        return (int) v->u;
    };
    keep_lo = num("loop.start");
    keep_hi = num("loop.end");
    if (pre == "inkling2.") n_layer -= (num("loop.count") - 1) * (keep_hi - keep_lo);   // Unsloth's: every visit
}

void Inkling::load_geometry() {
    // --draft-model is the DSpark drafter here, a model of its own, not a file of this one with a block in it,
    // which is what load_shape() would look for
    std::unique_ptr<Shards> spark = std::move(dg_);
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };
    unsloth_ = gg_->get("general.architecture")->s == "inkling2";
    core_start_ = shape_i("loop.start");
    core_end_ = shape_i("loop.end");
    n_passes_ = shape_i("loop.count");
    // ggml-org's block_count is the physical layers, unsloth's every visit
    n_phys_ = unsloth_ ? n_layer_ - (n_passes_ - 1) * (core_end_ - core_start_) : n_layer_;
    rank_ = shape_i("loop.gate_rank");
    const MetaValue* varying = find("loop.varying");
    need(!varying || varying->u == 0, "a loop whose passes vary (inkling2_loop.loop.varying)");
    need(shape_i("attn_res.block_size") == kBlockSize, "attn_res.block_size is not 6");
    // the streams in r_ hold four blocks and one block of the pass before: Inkling 2 Mini's shape
    need(n_phys_ == 24 && core_start_ == 6 && core_end_ == 18 && n_passes_ >= 2 && n_passes_ <= kMaxPasses,
         "a loop other than layers 6 to 17 of 24, or a bad loop.count");
    // the controllers' decode kernel holds the rank's values in a warp's registers (inkling_control_fused())
    if (rank_ <= 0 || rank_ > kInklingMaxRank)
        throw std::runtime_error("unsupported model: loop.gate_rank is " + std::to_string(rank_) + ", 1 to " +
                                 std::to_string(kInklingMaxRank) + " is supported");
    n_layer_ = n_phys_ + (n_passes_ - 1) * (core_end_ - core_start_);

    const MetaValue& heads = must("attention.head_count");
    const MetaValue& kv = must("attention.head_count_kv");
    n_head_ = (int) of_layer(heads, 0);
    n_kv_ = (int) of_layer(kv, 0);
    for (int il = 0; il < n_phys_; ++il)
        need(of_layer(heads, il) == n_head_ && of_layer(kv, il) == n_kv_, "layers with different head counts");
    head_dim_ = shape_i("attention.key_length");
    need(shape_i("attention.value_length") == head_dim_, "the keys and values have different widths");
    need(head_dim_ == 128 && n_kv_ > 0 && n_head_ == 3 * n_kv_, "attention other than 3 query heads of 128 a key");
    n_rel_ = shape_i("attention.relative_length");
    extent_ = shape_i("attention.relative_extent");
    window_ = shape_i("attention.sliding_window");
    need(n_rel_ > 0 && n_rel_ <= 32 && extent_ > 0 && window_ > 0, "bad relative-position or window sizes");
    n_conv_ = shape_i("shortconv.l_cache");
    need(n_conv_ == 4, "shortconv.l_cache is not 4");
    const MetaValue& swa = must("attention.sliding_window_pattern");
    for (int il = 0; il < n_phys_; ++il) of_layer(swa, il);
    qk_clip_ = shape_f("attention.qk_clip");
    rel_clip_ = shape_f("attention.relative_clip");
    tau_floor_ = shape_i("attention.log_scale_floor");
    tau_alpha_ = shape_f("attention.log_scale_alpha");
    need(tau_floor_ > 0 && qk_clip_ > 0.f && rel_clip_ > 0.f, "bad clips or log scale");
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->f : 0.f;
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;
    reserve_mmq_ = true;   // MMQ's scratch before the expert cache takes the VRAM
    prestage_ = true;      // a read's next layer's experts come while a layer runs
    kernel_tables_ = true;    // and a call's tables do not wait behind the copies to the card
    lowbit_experts_ = true;   // Unsloth's files: IQ1_S gate_up and Q2_0 down experts
    // STRATA_POLY_EXPERT_TILES=1: a prompt batch's experts in tiles of their own widths (Engine::expert_tiles_).
    // Opt-in: no tile's sum is split over K where llama.cpp's stream-k splits some, so the sums differ in rounding.
    if (const char* e = std::getenv("STRATA_POLY_EXPERT_TILES"); e && std::atoi(e) != 0) expert_tiles_ = true;
#if !defined(STRATA_USE_HIP)
    lane_slots_ = true;       // the cached experts' launches through MmvqSlots::lanes' kernels (measured on CUDA)
#endif
    situ_ = Situ{shape_f("activation.situ_beta"), shape_f("activation.situ_linear_beta")};
    need(situ_->beta > 0.f, "bad activation.situ_beta");
    const MetaValue* real = find("vocab_size_unpadded");
    n_vocab_real_ = real ? (int) real->u : n_vocab_;
    need(n_vocab_real_ > 0 && n_vocab_real_ <= n_vocab_, "bad vocab_size_unpadded");
    conv_floats_ = (size_t) (n_conv_ - 1) * (2 * (size_t) n_kv_ * head_dim_ + 2 * (size_t) n_embd_);

    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;

    // step()'s rows are a sequence's each, and the _seqs kernels take kInklingSeqRows of them a launch
    if (opt_.batch + 1 > kInklingSeqRows)
        throw std::runtime_error("Inkling 2: --batch takes at most " + std::to_string(kInklingSeqRows - 1));
    if (opt_.kv_format != KvFmt::kFp16 && opt_.kv_format != KvFmt::kQ8_0)
        throw std::runtime_error("Inkling 2: the attention cache is fp16 or q8_0");
    // A card of a split has a run of physical layers (split_layers()), here its visits: the core's on one card.
    const auto visit = [&](int p) {
        if (p > core_start_ && p < core_end_) throw std::logic_error("Inkling 2: a card starts inside the loop");
        return p <= core_start_ ? p : p + (n_passes_ - 1) * (core_end_ - core_start_);
    };
    lo_ = visit(lo_);
    if (end_ >= 0) end_ = visit(end_);
    // The file has no block that guesses ahead.  With the DSpark drafter (--draft-model) a window is the token and three
    // of its guesses; without, the guesses come from the conversation's repeats (--lookup) only: opt-in with --window,
    // a token at a time by default.
    if (opt_.window == 0) opt_.window = spark ? 4 : 1;
    opt_.draft = false;
    if (spark) {
        if (opt_.batch > 0) throw std::runtime_error("Inkling 2: the DSpark drafter (--draft-model) is not for --batch");
        dg_ = std::move(spark);
        block_ = dg_.get();   // the file its tensors are read from (home())
        load_spark_shape();
        spark_ = draft_ = block_drafter_ = true;
    }
    std::fprintf(stderr, "strata-poly: %s: %d layers, %d to %d run %d times: %d visits\n",
                 unsloth_ ? "inkling2 (looped)" : "inkling2_loop", n_phys_, core_start_, core_end_ - 1, n_passes_,
                 n_layer_);
}

void Inkling::load_layer(int v) {
    Layer& L = layers_[(size_t) v];
    Block& B = blocks_[(size_t) v];
    const int p = phys_of(v);
    const int first = p < core_end_ ? p : p + (n_passes_ - 1) * (core_end_ - core_start_);
    if (v != first) {   // a later visit: the first one's weights, and caches of its own
        const Block& F = blocks_[(size_t) first];
        B = F;
        B.pass = pass_of(v);
        B.conv = nullptr;
        for (uint8_t*& q : B.plane) q = nullptr;
        L = layers_[(size_t) first];
        L.moe = false;   // the first visit's experts: one copy of each in the cache (Layer::alias)
        L.alias = first;
        L.slot.clear();
        return;
    }
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(p, n), blk(p, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(p, n), blk(p, n), a, b, kAnyMat); };
    const int64_t E = n_embd_, hq = (int64_t) n_head_ * head_dim_, hk = (int64_t) n_kv_ * head_dim_;

    if (v == lo_) {   // the model's own small tensors, beside the card's first layer: the input's and the output's
        if (lo_ == 0) tok_norm_ = upload(gg_->find("token_embd_norm.weight"), "token_embd_norm.weight", E, 1, kF32);
        if (hi_ == n_layer_)
            out_query_ = upload(gg_->find("output_res_query.weight"), "output_res_query.weight", E, 1, kF32);
    }
    B.phys = p;
    B.pass = 0;
    // a token alone and a window's tokens are recorded (Engine::run()), a recording for each number of tokens;
    // pictures and step()'s rows take the kernels' positions by value.  step() replays a fixed layer's head without
    // asking layer_graphs(), and its recordings are keyed by row, not position: fixed must stay off with --batch.
    L.fixed = opt_.batch == 0;
    B.swa = of_layer(*meta_in(*gg_, "attention.sliding_window_pattern"), p) != 0;
    B.wq = mat("attn_q.weight", E, hq);
    B.wk = mat("attn_k.weight", E, hk);
    B.wv = mat("attn_v.weight", E, hk);
    B.wo = mat("attn_output.weight", hq, E);
    B.wgate = mat("attn_gate.weight", E, hq);
    B.wrel = mat("attn_rel.weight", E, (int64_t) n_rel_ * n_head_);
    B.wsink = mat("attn_sink_proj.weight", E, n_head_);
    const TensorInfo* rb = tensor(p, "attn_rel_bias.weight");
    if (!rb || rb->shape.size() != 2 || (int) rb->shape[0] != n_rel_)
        throw std::runtime_error(blk(p, "attn_rel_bias.weight") + " is missing or not [" + std::to_string(n_rel_) +
                                 ", extent]");
    B.extent = (int) rb->shape[1];
    if (B.extent != (B.swa ? window_ : extent_))
        throw std::runtime_error(blk(p, "attn_rel_bias.weight") + " has " + std::to_string(B.extent) +
                                 " distances, expected " + std::to_string(B.swa ? window_ : extent_));
    B.rel_bias = f32("attn_rel_bias.weight", n_rel_, B.extent);
    B.q_clip = f32("attn_q_clip.weight", head_dim_, n_head_);
    B.rel_clip = f32("attn_rel_clip.weight", n_rel_, n_head_);
    B.k_conv = f32("attn_k_conv.weight", n_conv_, hk);
    B.v_conv = f32("attn_v_conv.weight", n_conv_, hk);
    B.post_conv = f32("attn_post_conv.weight", n_conv_, E);
    B.ffn_post_conv = f32("ffn_post_conv.weight", n_conv_, E);
    if (B.swa) {
        L.attn_norm = f32("attn_norm.weight", E);
    } else {
        B.q_norm = f32("attn_q_inp_norm.weight", E);
        B.k_norm = f32("attn_k_inp_norm.weight", E);
        B.v_norm = f32("attn_v_inp_norm.weight", E);
        B.k_conv_proj = mat("attn_k_conv_proj.weight", E, (int64_t) n_conv_ * n_kv_);
        B.v_conv_proj = mat("attn_v_conv_proj.weight", E, (int64_t) n_conv_ * n_kv_);
    }
    if (p > 0) B.attn_query = f32("attn_res_query.weight", E, B.swa ? 1 : 3);
    B.ffn_query = f32("ffn_res_query.weight", E);
    L.ffn_norm = f32("ffn_norm.weight", E);
    if (p >= core_start_ && p < core_end_) {
        if (unsloth_ && !tensor(p, "loop_msg_query.weight")) {
            // unsloth's queries are two tensors of the model, [2304, 24] each: row 2i and 2i + 1 of loop_res_query are
            // core layer start + i's q and ffn queries, of loop_kv_res_query its k and v ones
            const TensorInfo* rq = gg_->find("loop_res_query.weight");
            const TensorInfo* kq = gg_->find("loop_kv_res_query.weight");
            const int nc = core_end_ - core_start_;
            for (const TensorInfo* t : {rq, kq})
                if (!t || t->type != kTypeF32 || (int64_t) t->elements() != 2 * nc * E)
                    throw std::runtime_error("inkling2: loop_res_query / loop_kv_res_query missing or not F32 [2304, " +
                                             std::to_string(2 * nc) + "]");
            if (dense_) {   // the second pass of the load: VRAM of its own beside the arena
                if (!msg_mem_) msg_mem_ = dmalloc((size_t) nc * 4 * E * sizeof(float), "the loop's queries");
                const int i = p - core_start_;
                const float* r = (const float*) home(*rq).data(*rq);
                const float* k = (const float*) home(*kq).data(*kq);
                std::vector<float> q((size_t) 4 * E);
                std::memcpy(q.data(), r + (size_t) (2 * i) * E, E * sizeof(float));
                std::memcpy(q.data() + E, k + (size_t) (2 * i) * E, E * sizeof(float));
                std::memcpy(q.data() + 2 * E, k + (size_t) (2 * i + 1) * E, E * sizeof(float));
                std::memcpy(q.data() + 3 * E, r + (size_t) (2 * i + 1) * E, E * sizeof(float));
                float* d = (float*) msg_mem_ + (size_t) i * 4 * E;
                h2d(d, q.data(), q.size() * sizeof(float));
                B.msg_query = Dev{d, kTypeF32, (int) E, 4};
            }
        } else {
            B.msg_query = f32("loop_msg_query.weight", E, 4);
        }
        for (int c = 0; c < 4; ++c) {
            const std::string pre = std::string("loop_") + kCtrl[c] + "_ctrl_";
            B.ctrl_down[c] = f32((pre + "down.weight").c_str(), 2 * E, rank_);
            B.ctrl_gain[c] = f32((pre + "gain.weight").c_str(), rank_, E);
            B.ctrl_forget[c] = f32((pre + "forget.weight").c_str(), rank_, E);
        }
    }
    load_moe(v, false, p, true);   // Unsloth's files scale every expert's gate_up and down, whatever their type
    if (L.gate_inp.type != kTypeF32)   // router_scores() takes it as floats
        throw std::runtime_error(blk(p, "ffn_gate_inp.weight") + " is not F32");
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(p, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(p, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(p, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

size_t Inkling::recurrent_floats() const { return (size_t) n_layer_ * conv_floats_; }

// Each visit's convolution state in state_; a window visit's caches are a ring of the rows a call can read, a full
// one's a row a position.
void Inkling::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    size_t one = 0;   // a position's bytes in a visit's planes together
    for (int i = 0; i < planes(); ++i) one += plane_row(i);
    if (q8()) std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", one / 2);
    // STRATA_KV_INT8=1 reads the q8_0 cache of a decoded token with __dp4a, the queries quantized too (as Kolibri's
    // does).  Opt-in: other arithmetic, so other bits; fp16 caches and prompt batches are unchanged.
    if (const char* e = std::getenv("STRATA_KV_INT8"); e && std::atoi(e) != 0 && !int8_dot_) {
        int8_dot_ = q8();
        std::fprintf(stderr, "strata-poly: STRATA_KV_INT8: decode attention %s\n",
                     int8_dot_ ? "through __dp4a on the q8_0 cache" : "unchanged (the cache is not q8_0)");
    }
    size_t rings = 0;
    for (int v = 0; v < (int) blocks_.size(); ++v)
        if (runs(v) && blocks_[(size_t) v].swa) rings += (size_t) ring_ * one;
    uint8_t* kv = rings ? (uint8_t*) seq_alloc(rings, "the attention cache") : nullptr;
    float* st = state_;
    for (int v = 0; v < (int) blocks_.size(); ++v) {
        Block& B = blocks_[(size_t) v];
        if (!runs(v)) continue;
        B.conv = st;
        st += conv_floats_;
        for (int i = 0; i < planes(); ++i) {
            if (B.swa) {
                B.plane[i] = kv;
                kv += (size_t) ring_ * plane_row(i);
            } else {
                B.plane[i] = (uint8_t*) kv_alloc(plane_row(i), 1);
            }
        }
    }
    if (!mix_work_) {   // inkling_mix_conv()'s meeting place
        mix_work_ = dmalloc(inkling_mix_conv_work_bytes(), "scratch");
        ck(cudaMemset(mix_work_, 0, inkling_mix_conv_work_bytes()), "scratch");
    }
    // the drafter's: its window layers' rings, its other layer's rows, and the head's context of the last positions
    if (spark_) {
        const size_t row = (size_t) s_kv_ * s_dim_ * sizeof(uint16_t);
        for (SparkLayer& D : spark_layers_) {
            if (D.swa) {
                D.k = (uint16_t*) seq_alloc((size_t) s_ring_ * row, "the drafter's cache");
                D.v = (uint16_t*) seq_alloc((size_t) s_ring_ * row, "the drafter's cache");
            } else {
                D.k = (uint16_t*) kv_alloc(row, 1);
                D.v = (uint16_t*) kv_alloc(row, 1);
            }
        }
        s_rnn_ = (float*) seq_alloc((size_t) (s_block_ + 1) * n_embd_ * sizeof(float), "the drafter's cache");
    }
    // the padded vocabulary's scores, which finish_logits() puts out of reach
    if (!neg_inf_ && n_vocab_ > n_vocab_real_) {
        const size_t pad_n = (size_t) (n_vocab_ - n_vocab_real_);
        neg_inf_ = (float*) dmalloc(pad_n * sizeof(float), "scratch");
        std::vector<float> h(pad_n, -INFINITY);
        ck(cudaMemcpy(neg_inf_, h.data(), pad_n * sizeof(float), cudaMemcpyHostToDevice), "scratch");
    }
}

// Per-token buffers hold rows_ tokens, row after row.
void Inkling::assign_scratch() {
    const size_t n = (size_t) rows_, e = n * n_embd_, hq = n * n_head_ * head_dim_, hk = n * n_kv_ * head_dim_;
    res_ = fbuf(e);
    for (float*& b : blk_) b = fbuf(e);
    bank_ = fbuf(e);
    bank_out_ = fbuf(e);
    for (float*& m : msg_) m = fbuf(e);
    for (float*& m : mix_) m = fbuf(e);
    for (float*& m : in_) m = fbuf(e);
    for (float*& m : ctl_) m = fbuf(e);
    const size_t q8 = (n * kernels::native_q8_1_bytes(n_embd_) + sizeof(float) - 1) / sizeof(float);
    for (void*& m : inq_) m = fbuf(q8);
    attq_ = fbuf((n * kernels::native_q8_1_bytes(n_head_ * head_dim_) + sizeof(float) - 1) / sizeof(float));
    ctl_work_ = fbuf(n * (4 * (size_t) n_embd_ + rank_));
    q_ = fbuf(hq);
    gate_ = fbuf(hq);
    att_ = fbuf(hq);
    kraw_ = fbuf(hk);
    k_ = fbuf(hk);
    vraw_ = fbuf(hk);
    v_ = fbuf(hk);
    rel_ = fbuf(n * n_head_ * n_rel_);
    sink_ = fbuf(n * n_head_);
    ktap_ = fbuf(n * n_kv_ * n_conv_);
    vtap_ = fbuf(n * n_kv_ * n_conv_);
    o_ = fbuf(e);
    awork_n_ = inkling_attend_work_floats(n_head_, head_dim_);
    awork_ = awork_n_ ? fbuf(awork_n_) : nullptr;
    if (!spark_) return;
    if (rows_ < s_block_)
        throw std::runtime_error("Inkling 2: the DSpark drafter takes a block of " + std::to_string(s_block_) +
                                 " rows: a prompt batch of fewer cannot hold it");
    const size_t c = (size_t) std::min(rows_, kSparkRows), b = (size_t) s_block_, E = (size_t) n_embd_;
    const size_t sq = (size_t) s_head_ * s_dim_, sk = std::max(b, c) * s_kv_ * s_dim_, r = (size_t) s_rank_;
    for (float*& x : s_ctx_) x = fbuf(c * E);
    s_x_ = fbuf(b * E);
    s_xn_ = fbuf(b * E);
    s_q_ = fbuf(b * sq);
    s_qr_ = fbuf(b * sq);
    s_k_ = fbuf(sk);
    s_kr_ = fbuf(sk);
    s_v_ = fbuf(sk);
    s_att_ = fbuf(b * sq);
    s_g_ = fbuf(b * s_ff_);
    s_u_ = fbuf(b * s_ff_);
    s_cur_ = fbuf(b * E);
    s_base_ = fbuf((b - 1) * n_vocab_);
    s_bias_ = fbuf((size_t) n_vocab_);
    s_in_ = fbuf(2 * r + 2 * E);
    s_jo_ = fbuf(3 * r);
    s_state_ = fbuf(r);
    s_latent_ = fbuf(r);
    s_corr_ = fbuf(E);
    s_work_ = fbuf(inkling_block_attend_work_floats(s_block_, s_head_, s_dim_));
}

// ---- the streams in r_ -----------------------------------------------------------------------------------------

void Inkling::gather(int k, float* dst, int n, int first) {
    const size_t e = (size_t) n_embd_ * sizeof(float), rs = e * streams();
    ck(cudaMemcpy2DAsync(dst, e, r_ + (size_t) first * streams() * n_embd_ + (size_t) k * n_embd_, rs, e, (size_t) n,
                         cudaMemcpyDeviceToDevice, cs(st_)),
       "streams");
}

void Inkling::scatter(int k, const float* src, int n) {
    const size_t e = (size_t) n_embd_ * sizeof(float), rs = e * streams();
    ck(cudaMemcpy2DAsync(stream(k), rs, src, e, e, (size_t) n, cudaMemcpyDeviceToDevice, cs(st_)), "streams");
}

void Inkling::copy_stream(int from, int to, int n) {
    const size_t e = (size_t) n_embd_ * sizeof(float), rs = e * streams();
    ck(cudaMemcpy2DAsync(stream(to), rs, stream(from), rs, e, (size_t) n, cudaMemcpyDeviceToDevice, cs(st_)),
       "streams");
}

// --dump-layers (Engine::dbg()): a stage of n tokens, by llama.cpp's index of the visit (its cache slot)
void Inkling::dump(const char* name, int v, const float* dev, int width, int n) {
    dbg(name, v >= 0 ? slot_of(v) : -1, dev, width * n);
}

// ---- a layer ---------------------------------------------------------------------------------------------------

// The rows the attention of n tokens from pos_ is sized for: a full visit's in room()'s steps rather than pos + n,
// so that its splits, and so a token's bits, do not change with every position.  A window visit's serves every
// position: sized for the context, it gets the window's splits and the one more a window not starting on a split's
// edge spans (window + n - 1 rows would leave that one out).
int64_t Inkling::attn_room(const Block& B, int n) const {
    return B.swa ? (int64_t) opt_.max_context : Engine::room(n);
}

int Inkling::head_key(int il, int n) const { return blocks_[(size_t) il].swa ? 0 : room_key(n); }

// A post-convolution of n rows into acc (written when `fresh`, else added), its state at `state` in the bound
// sequence's state_, or with step()'s rows at the same place in each row's own sequence's.
void Inkling::post_conv(const float* x, float* state, const Dev& w, float* acc, int n, bool fresh, const Row* rows) {
    if (rows) {
        float* states[kInklingSeqRows];
        for (int i = 0; i < n; ++i) states[i] = seq_state(rows[i].seq, state);
        if (fresh) inkling_conv_seqs(x, states, fp(w), acc, n_embd_, n, st_);
        else inkling_conv_acc_seqs(x, states, fp(w), acc, n_embd_, n, st_);
    } else {
        conv_snaps(x, state, n_embd_, n);
        if (fresh) inkling_conv(x, state, fp(w), nullptr, 0, acc, n_embd_, n, st_);
        else inkling_conv_acc(x, state, fp(w), nullptr, 0, acc, n_embd_, n, st_);
    }
}

// A window's snapshots of a convolution's state (Engine::window()), before the call's n rows of x advance it.
void Inkling::conv_snaps(const float* x, const float* state, int channels, int n) {
    if (snap_bits_) inkling_conv_snaps(x, state, channels, n, snap_bits_, snap_at_ + (state - state_), rec_n_, st_);
}

// The attention from its three normed inputs (and their Q8_1 copies) to ao_, through the output's convolution.
void Inkling::attn(Block& B, int v, const float* xq, const float* xk, const float* xv, const void* const* xq8, int n,
                   bool fuse) {
    const int hk = n_kv_ * head_dim_, hq = n_head_ * head_dim_;
    // the projections of the three inputs, those of one format in one launch
    const LinJob proj[8] = {{&B.wq, xq, q_, xq8[0]},     {&B.wgate, xq, gate_, xq8[0]},
                            {&B.wrel, xq, rel_, xq8[0]}, {&B.wsink, xq, sink_, xq8[0]},
                            {&B.wk, xk, kraw_, xq8[1]},  {&B.wv, xv, vraw_, xq8[2]},
                            {&B.k_conv_proj, xk, ktap_, xq8[1]}, {&B.v_conv_proj, xv, vtap_, xq8[2]}};
    lin_jobs(proj, B.swa ? 6 : 8, n);
    const size_t v_at = (size_t) (n_conv_ - 1) * hk;   // the V convolution's state, after K's
    const InklingTau tau = B.swa ? InklingTau{} : InklingTau{tau_alpha_, tau_floor_};
    InklingClipJob jobs[2];
    jobs[0] = {q_, 0, fp(B.q_clip), q_, head_dim_, n_head_, qk_clip_, tau};
    jobs[1] = {rel_, 0, fp(B.rel_clip), rel_, n_rel_, n_head_, rel_clip_, tau};
    InklingRel rel;
    rel.r = rel_;
    rel.table = fp(B.rel_bias);
    rel.n_rel = n_rel_;
    rel.extent = B.extent;
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    // K and V from their projections into the cache in one launch, Q's and R's clips in another, then the attention
    if (rows_of_) {   // step(): each row a token of its own sequence, with its caches, convolution state and position
        InklingKvSeq kv[kInklingSeqRows];
        InklingAttnSeq at[kInklingSeqRows];
        int64_t pos[kInklingSeqRows];
        for (int i = 0; i < n; ++i) {
            const Row& r = rows_of_[i];
            uint8_t* pl[4] = {};
            for (int j = 0; j < planes(); ++j) pl[j] = (uint8_t*) seq_buf(r.seq, B.plane[j]);
            float* const conv = seq_state(r.seq, B.conv);
            kv[i].dst = kv_dst(q8(), pl, ring);
            kv[i].k_state = conv;
            kv[i].v_state = conv + v_at;
            kv[i].pos = pos[i] = r.pos;
            const int64_t keep = pos_;   // its room as alone (room() reads pos_)
            pos_ = r.pos;
            at[i] = attn_seq(q8(), pl, r.pos, attn_room(B, 1));
            pos_ = keep;
        }
        inkling_kv_store_seqs(kraw_, vraw_, kv, fp(B.k_conv), fp(B.v_conv), B.swa ? nullptr : ktap_,
                              B.swa ? nullptr : vtap_, n_kv_, head_dim_, n, qk_clip_, rms_eps_, k_, v_, st_);
        inkling_clip_many_seqs(jobs, 2, n, pos, st_);
        inkling_attend_seqs(q_, at, n, rel, sink_, att_, head_dim_, n_head_, n_kv_, scale, window, ring, awork_,
                            awork_n_, st_, int8_dot_);
    } else {
        const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the position from what at_ holds
        conv_snaps(kraw_, B.conv, hk, n);
        conv_snaps(vraw_, B.conv + v_at, hk, n);
        if (fuse) {
            inkling_kv_store_clip(kraw_, vraw_, B.conv, B.conv + v_at, fp(B.k_conv), fp(B.v_conv),
                                  B.swa ? nullptr : ktap_, B.swa ? nullptr : vtap_, n_kv_, head_dim_, n, qk_clip_,
                                  rms_eps_, kv_dst(q8(), B.plane, ring), pos_, k_, v_, jobs, 2, st_, at);
        } else {
            inkling_kv_store(kraw_, vraw_, B.conv, B.conv + v_at, fp(B.k_conv), fp(B.v_conv),
                             B.swa ? nullptr : ktap_, B.swa ? nullptr : vtap_, n_kv_, head_dim_, n, qk_clip_, rms_eps_,
                             kv_dst(q8(), B.plane, ring), pos_, k_, v_, st_, at);
            inkling_clip_many(jobs, 2, n, pos_, st_, at);
        }
        dump("Qcur", v, q_, hq, n);
        dump("Kcur", v, k_, hk, n);
        dump("Vcur", v, v_, hk, n);
        dump("Rcur", v, rel_, n_head_ * n_rel_, n);
        const InklingAttnSeq c = attn_seq(q8(), B.plane, pos_, attn_room(B, n));
        inkling_attend(q_, c.kcache, c.vcache, c.kq8, c.vq8, rel, sink_, pos_, n, att_, head_dim_, n_head_, n_kv_,
                       scale, window, ring, awork_, awork_n_, st_, c.room, int8_dot_, at, fuse ? gate_ : nullptr,
                       fuse ? attq_ : nullptr);
    }
    dump("kqv_out", v, att_, hq, n);
    if (!fuse) mul_sigmoid(att_, gate_, hq, n, hq, false, st_);
    lin(B.wo, att_, o_, n, fuse ? attq_ : nullptr);   // the post-convolution: layer_head(), with the residual
}

// A visit from its input to its router's copy.  The streams: see inkling.hpp.
void Inkling::layer_head(Layer& L, int v, int n) {
    Block& B = blocks_[(size_t) v];
    const int ip = B.phys;
    const size_t e = (size_t) n * n_embd_;
    // One token's streams lie one after the other in r_, as a buffer of one row each: the mixes read them there and
    // the stream is worked on in place.  More tokens' are gathered, a buffer a stream, and the stream goes back in the
    // tail.
    const bool direct = n == 1;
    float* const cres = work_stream(n);
    tail_rows_ = rows_of_;   // step()'s rows, for the tail's convolution
    // A token alone has its launches fused (fused()): the visit before's tail convolution goes in with the first
    // mix here, the attention's with the FFN's mix.
    const bool fuse = fused(n);
    // the visit before's, still to do (none before the card's first: a call cut short leaves its own behind)
    const InklingConvIn* last = tail_wait_ && v != lo_ ? &tail_conv_ : nullptr;
    tail_wait_ = false;
    const auto flush = [&] {
        if (last) inkling_conv_in(*last, n_embd_, st_);
        last = nullptr;
    };
    if (v == 0) {   // the embedding, normed, is the first block's start
        gather(kCur, o_, n);
        rms_norm(o_, fp(tok_norm_), res_, n_embd_, n, rms_eps_, st_);
        // a picture's rows go in as the encoder made them: llama.cpp norms token embeddings only (inkling2.cpp,
        // `if (ubatch.token)`).  A call can hold text and picture rows, so the raw rows are put back row by row.
        // step()'s rows have none (Engine::step()).
        if (!rows_of_)
            for (const Image& im : images_) {
                const int64_t a = std::max<int64_t>(im.at, pos_), b = std::min<int64_t>(im.at + im.n, pos_ + n);
                if (a < b)
                    ck(cudaMemcpyAsync(res_ + (size_t) (a - pos_) * n_embd_, o_ + (size_t) (a - pos_) * n_embd_,
                                       (size_t) (b - a) * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
                       "picture rows");
            }
        if (direct) ck(cudaMemcpyAsync(cres, res_, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "streams");
        else scatter(kCur, res_, n);
    } else {
        // a later pass of the core starts from layer 5's output, with what the pass before left for its messages:
        // the start of the block in the core's middle (its layer 11's output) and the stream (its layer 17's)
        if (B.pass > 0 && ip == core_start_) {
            flush();
            const int core_in = kBlock0 + core_start_ / kBlockSize;
            copy_stream(core_in + 1, kBank, n);
            copy_stream(kCur, kBankOut, n);
            copy_stream(core_in, kCur, n);
        }
        if (!direct) gather(kCur, res_, n);
    }
    dump("l_in", v, cres, n_embd_, n);
    if (spark_)   // the drafter reads the stream entering this visit (drafter_rows()), the visit before's tail in it
        for (int k = 0; k < kTaps; ++k) {
            if (taps_[k] != v) continue;
            flush();
            if (direct) ck(cudaMemcpyAsync(stream(kTap0 + k), cres, e * sizeof(float), cudaMemcpyDeviceToDevice,
                                           cs(st_)),
                           "streams");
            else scatter(kTap0 + k, res_, n);
        }
    const int nb = blocks_before(ip);
    const float* bp[4];   // the blocks' starts
    for (int k = 0; k < nb; ++k) {
        if (direct) {
            bp[k] = stream(kBlock0 + k);
        } else {
            gather(kBlock0 + k, blk_[k], n);
            bp[k] = blk_[k];
        }
    }

    // the attention's inputs: the stream, or its mix with the blocks (one for a window layer, Q, K and V apart
    // for a full one)
    const float* in[3] = {cres, cres, cres};
    // the mixes: the attention's inputs, and a later pass's messages (the pass before's mix, gated into each input)
    InklingMix mixes[2];
    int n_mix = 0;
    if (B.attn_query.p) {
        InklingMix& m = mixes[n_mix++];
        for (int k = 0; k < nb; ++k) m.state[k] = bp[k];
        m.state[nb] = cres;
        m.n_state = nb + 1;
        m.query = fp(B.attn_query);
        m.n_query = B.swa ? 1 : 3;
        for (int j = 0; j < m.n_query; ++j) m.out[j] = mix_[j];
        for (int j = 0; j < 3; ++j) in[j] = mix_[B.swa ? 0 : j];
    }
    if (B.pass > 0) {
        if (!direct) {
            gather(kBank, bank_, n);
            gather(kBankOut, bank_out_, n);
        }
        InklingMix& m = mixes[n_mix++];
        m.state[0] = direct ? stream(kBank) : bank_;
        m.state[1] = direct ? stream(kBankOut) : bank_out_;
        m.n_state = 2;
        m.query = fp(B.msg_query);
        m.n_query = 4;
        for (int j = 0; j < 4; ++j) m.out[j] = msg_[j];
    }
    const int threads = sum_threads(n_embd_);
    // the first pass's visits norm the mix's outputs in its launch (later passes' controllers come in between)
    const bool normed = fuse && n_mix > 0 && B.attn_query.p && B.pass == 0;
    if (fuse && n_mix > 0 && B.attn_query.p) {
        InklingMixNorm nm;
        for (int j = 0; j < mixes[0].n_query; ++j) {
            nm.w[j] = fp(B.swa ? L.attn_norm : j == 0 ? B.q_norm : j == 1 ? B.k_norm : B.v_norm);
            nm.y[j] = in_[j], nm.y_q8_1[j] = inq_[j];
        }
        nm.eps = rms_eps_, nm.threads = threads;
        inkling_mix_conv(mixes, n_mix, last, n_embd_, norm_eps_, st_, normed ? &nm : nullptr, mix_work_);
        last = nullptr;
    } else {
        flush();
        for (int k = 0; k < n_mix; ++k) inkling_mix(mixes[k], n_embd_, n, norm_eps_, st_);
    }
    if (B.pass > 0) {
        if (n <= kFusedRows) {   // the three controllers in two launches; each out its own buffer
            InklingCtrl c[3];
            for (int j = 0; j < 3; ++j)
                c[j] = {in[j], msg_[j], fp(B.ctrl_down[j]), fp(B.ctrl_gain[j]), fp(B.ctrl_forget[j]), ctl_[j]};
            inkling_control_fused(c, 3, n_embd_, rank_, n, norm_eps_, ctl_work_, st_);
            for (int j = 0; j < 3; ++j) in[j] = ctl_[j];
        } else {
            for (int j = 0; j < 3; ++j) {
                inkling_control(blas_, in[j], msg_[j], fp(B.ctrl_down[j]), fp(B.ctrl_gain[j]), fp(B.ctrl_forget[j]),
                                ctl_work_, ctl_[j], n_embd_, rank_, n, norm_eps_, st_);
                in[j] = ctl_[j];
            }
        }
    }
    const void* q8[3];
    const auto norm = [&](int j, const float* x, const Dev& w) {
        if (!normed)
            kernels::native_add_rms_norm_q8_1(const_cast<float*>(x), nullptr, fp(w), in_[j], inq_[j], n_embd_, n,
                                              rms_eps_, threads, st_);
        q8[j] = inq_[j];
    };
    const float* x[3];
    if (B.swa && B.pass == 0) {
        norm(0, in[0], L.attn_norm);
        x[0] = x[1] = x[2] = in_[0];
        q8[1] = q8[2] = q8[0];
    } else {
        const Dev* w[3] = {&L.attn_norm, &L.attn_norm, &L.attn_norm};
        if (!B.swa) w[0] = &B.q_norm, w[1] = &B.k_norm, w[2] = &B.v_norm;
        kernels::NormRows rows{};
        for (int j = 0; j < 3; ++j) {
            rows.x[j] = in[j], rows.w[j] = fp(*w[j]), rows.y[j] = in_[j], rows.y_q8_1[j] = inq_[j];
            x[j] = in_[j];
            q8[j] = inq_[j];
        }
        if (!normed && (!fuse || !kernels::native_rms_norm_q8_1_rows(rows, 3, n_embd_, rms_eps_, threads, st_)))
            for (int j = 0; j < 3; ++j) norm(j, in[j], *w[j]);
    }
    attn(B, v, x[0], x[1], x[2], q8, n, fuse);

    // the first layer of a block starts the stream afresh, and what it started from is the block's
    float* const post = B.conv + (size_t) 2 * (n_conv_ - 1) * n_kv_ * head_dim_;   // the attention-out's state
    InklingConvIn conv;   // a token alone's, done by the FFN's mix
    if (fuse) {
        conv_snaps(o_, post, n_embd_, n);
        conv.x = o_;
        conv.state = post;
        conv.w = fp(B.post_conv);
        conv.acc = cres;
        conv.fresh = ip % kBlockSize == 0;
        if (conv.fresh) {
            const int k = ip / kBlockSize;
            conv.save = stream(kBlock0 + k);
            bp[k] = conv.save;
        }
    } else if (ip % kBlockSize == 0) {
        const int k = ip / kBlockSize;
        if (direct) {
            ck(cudaMemcpyAsync(stream(kBlock0 + k), cres, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
               "streams");
            bp[k] = stream(kBlock0 + k);
        } else {
            scatter(kBlock0 + k, res_, n);
            ck(cudaMemcpyAsync(blk_[k], res_, e * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)), "streams");
            bp[k] = blk_[k];
        }
        post_conv(o_, post, B.post_conv, cres, n, true, rows_of_);
    } else {
        post_conv(o_, post, B.post_conv, cres, n, false, rows_of_);
    }
    dump("ffn_inp", v, cres, n_embd_, n);

    // the FFN's input: the stream mixed with the blocks, gated with its message in a later pass
    const int nf = blocks_at(ip);
    InklingMix m;
    for (int k = 0; k < nf; ++k) m.state[k] = bp[k];
    m.state[nf] = cres;
    m.n_state = nf + 1;
    m.query = fp(B.ffn_query);
    m.n_query = 1;
    m.out[0] = mix_[0];
    InklingMixNorm nm;   // the first pass's FFN input, normed in the mix's launch
    nm.w[0] = fp(L.ffn_norm), nm.y[0] = xn_, nm.y_q8_1[0] = xq_, nm.eps = rms_eps_, nm.threads = threads;
    if (fuse) inkling_mix_conv(&m, 1, &conv, n_embd_, norm_eps_, st_, B.pass == 0 ? &nm : nullptr, mix_work_);
    else inkling_mix(m, n_embd_, n, norm_eps_, st_);
    float* f = mix_[0];
    if (B.pass > 0) {
        if (n <= kFusedRows) {
            const InklingCtrl c{f, msg_[3], fp(B.ctrl_down[3]), fp(B.ctrl_gain[3]), fp(B.ctrl_forget[3]), ctl_[0]};
            inkling_control_fused(&c, 1, n_embd_, rank_, n, norm_eps_, ctl_work_, st_);
        } else {
            inkling_control(blas_, f, msg_[3], fp(B.ctrl_down[3]), fp(B.ctrl_gain[3]), fp(B.ctrl_forget[3]),
                            ctl_work_, ctl_[0], n_embd_, rank_, n, norm_eps_, st_);
        }
        f = ctl_[0];
    }
    if (!fuse || B.pass > 0)
        kernels::native_add_rms_norm_q8_1(f, nullptr, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dump("ffn_in", v, xn_, n_embd_, n);
    if (!batched(layers_[(size_t) home_of(v)], n)) moe_router(L, n);   // the experts' layer decides (Layer::alias)
}

// The routed experts' sum through its convolution, into the stream.
void Inkling::layer_tail(Layer&, int v, int n) {
    Block& B = blocks_[(size_t) v];
    const size_t hk = (size_t) n_kv_ * head_dim_;
    const bool parts = card_parts_;   // moe_card() left the experts' sum to this
    card_parts_ = false;
    dump("ffn_moe_out", v, fo_, n_embd_, n);
    float* const state = B.conv + (size_t) (n_conv_ - 1) * (2 * hk + n_embd_);
    // the stream at this half's rows: a window's halves (shift_rows()) run the other half's head in between
    float* const cres = work_stream(n);
    if (fused(n) && !tail_rows_) {   // a token alone: the next visit's first mix does the convolution
        InklingConvIn& c = tail_conv_;
        c = InklingConvIn{};
        if (parts) {
            c.parts = d_, c.weights = d_wts_, c.n_parts = n_used_;
        } else {
            conv_snaps(fo_, state, n_embd_, n);
            c.x = fo_;
        }
        c.state = state;
        c.w = fp(B.ffn_post_conv);
        c.acc = cres;
        if (layers_in_order_ && v + 1 < hi_) tail_wait_ = true;
        else inkling_conv_in(c, n_embd_, st_);
        return;
    }
    if (parts) moe_sum(d_, d_wts_, fo_, n_embd_, n_used_, n, st_);   // step()'s row of one sequence
    post_conv(fo_, state, B.ffn_post_conv, cres, n, false, tail_rows_);
    tail_rows_ = nullptr;
    if (cres == res_) scatter(kCur, res_, n);   // several tokens' stream back to r_ (one token's is worked on there)
    dump("l_out", v, cres, n_embd_, n);
}

// A half of a window at its rows: the engine's buffers and the stream a layer's head leaves for its tail.
void Inkling::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    res_ += by * n_embd_;
}

// The output's input: the stream mixed with the four blocks, normed.
void Inkling::output_rows(int first, int rows, float* h) {
    const int nb = blocks_at(n_phys_ - 1);
    InklingMix m;
    for (int k = 0; k < nb; ++k) {
        gather(kBlock0 + k, blk_[k], rows, first);
        m.state[k] = blk_[k];
    }
    gather(kCur, mix_[1], rows, first);
    m.state[nb] = mix_[1];
    m.n_state = nb + 1;
    m.query = fp(out_query_);
    m.n_query = 1;
    m.out[0] = mix_[0];
    inkling_mix(m, n_embd_, rows, norm_eps_, st_);
    rms_norm(mix_[0], fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
    dump("result_norm", -1, h, n_embd_, rows);
}

// The final softcap, and the padded vocabulary's rows out of reach.
void Inkling::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, softcap_, st_);
    if (!neg_inf_) return;
    const size_t pad_n = (size_t) (n_vocab_ - n_vocab_real_);
    for (int r = 0; r < rows; ++r)
        ck(cudaMemcpyAsync(logits + (size_t) r * n_vocab_ + n_vocab_real_, neg_inf_, pad_n * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "logits");
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, weights from the
// bare gates divided by their sum (+ 1e-20, expert_weights_norm_eps) and times expert_weights_scale.
void Inkling::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    sum += 1e-20f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= sum;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

// The router's F32 product: a decode step's few tokens with inkling_gemv(), a prompt batch as the engine does it.
void Inkling::router_scores(const Layer& L, int n) {
    if (n <= kFusedRows) inkling_gemv(fp(L.gate_inp), xn_, rl_, n_embd_, n_expert_, n, st_);
    else Engine::router_scores(L, n);
}

bool Inkling::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, B.probs_b_dev.p ? fp(B.probs_b_dev) : nullptr, nullptr, true,
               w_scale_, ids, w, st_);
    return true;
}

// SiTU straight into the down products' Q8_1 input (situ_rows_q8_1()): the same bits as situ_rows() then the
// quantization.
bool Inkling::act_rows_q8_1(const float* gate_up, void* q8, int rows, const ExpertScales* es) {
    situ_rows_q8_1(gate_up, q8, n_ff_exp_, rows, situ_->beta, situ_->lin_beta, st_, es);
    return true;
}

// No shared expert: the routed experts' sum starts from zero.
void Inkling::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// ---- a sequence's state and rows -------------------------------------------------------------------------------

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first
// ring row, its rows).
template <class F>
void Inkling::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

size_t Inkling::plane_row(int i) const {
    const size_t width = (size_t) n_kv_ * head_dim_;
    if (!q8()) return width * sizeof(uint16_t);
    return i % 2 == 0 ? width : width / 32 * sizeof(uint16_t);
}

// A sequence's rows are the full visits'; a window visit's are part of its state (save_layer()).  A visit's part of
// a row is its planes one after the other (fp16 keys, values; or q8_0 key codes, key scales, value codes, value
// scales, as Flint's), each plane's rows together.
size_t Inkling::card_row_bytes() const {
    size_t one = 0;
    for (int i = 0; i < planes(); ++i) one += plane_row(i);
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.plane[0] && !B.swa) ++layers;
    size_t spark = 0;   // the drafter's: its full layers' keys and values
    for (const SparkLayer& D : spark_layers_)
        if (!D.swa) spark += 2 * spark_row();
    return layers * one + spark;
}

void Inkling::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !B.plane[0]) continue;
        for (int i = 0; i < planes(); ++i) {
            const size_t row = plane_row(i);
            ck(cudaMemcpy(dst, B.plane[i] + (size_t) first * row, (size_t) n * row, cudaMemcpyDeviceToHost),
               "save rows");
            dst += (size_t) n * row;
        }
    }
    // on st_: the drafter writes its rows of a call behind the call's logits (Engine::eval())
    for (const SparkLayer& D : spark_layers_) {
        if (D.swa || n == 0) continue;
        for (const uint16_t* plane : {D.k, D.v}) {
            ck(cudaMemcpyAsync(dst, (const uint8_t*) plane + (size_t) first * spark_row(), (size_t) n * spark_row(),
                               cudaMemcpyDeviceToHost, cs(st_)),
               "save rows");
            dst += (size_t) n * spark_row();
        }
    }
    if (spark_) ck(cudaStreamSynchronize(cs(st_)), "save rows");
}

void Inkling::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) {
        if (B.swa || n == 0 || !B.plane[0]) continue;
        for (int i = 0; i < planes(); ++i) {
            const size_t row = plane_row(i);
            h2d(B.plane[i] + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row);
            src += (size_t) count * row;
        }
    }
    for (const SparkLayer& D : spark_layers_) {
        if (D.swa || n == 0) continue;
        for (uint16_t* plane : {D.k, D.v}) {
            h2d((uint8_t*) plane + (size_t) first * spark_row(), src + (size_t) (first - from) * spark_row(),
                (size_t) n * spark_row());
            src += (size_t) count * spark_row();
        }
    }
}

// What a window visit has of the sequence: the rows the next token attends, its planes one after the other.
size_t Inkling::extra_bytes() const {
    size_t one = 0;
    for (int i = 0; i < planes(); ++i) one += plane_row(i);
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.swa) ++layers;
    return layers * (size_t) keep_ * one + spark_extra();
}

// A visit's convolution state at its place in state_, and a window visit's rows behind the window visits before it.
void Inkling::save_layer(int v, int64_t pos, uint8_t* dst) {
    const Block& B = blocks_[(size_t) v];
    if (B.conv)
        ck(cudaMemcpy(dst + (size_t) (B.conv - state_) * sizeof(float), B.conv, conv_floats_ * sizeof(float),
                      cudaMemcpyDeviceToHost),
           "save state");
    if (!B.swa) return;
    size_t whole = 0;
    for (int i = 0; i < planes(); ++i) whole += (size_t) keep_ * plane_row(i);
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < v; ++j)
        if (blocks_[(size_t) j].swa) dst += whole;
    for (int i = 0; i < planes(); ++i) {
        const size_t one_row = plane_row(i);
        std::memset(dst, 0, (size_t) keep_ * one_row);   // the places of positions before the sequence's start
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(dst + slot * one_row, B.plane[i] + at * one_row, m * one_row, cudaMemcpyDeviceToHost),
               "save state");
        });
        dst += (size_t) keep_ * one_row;
    }
}

void Inkling::load_extra(const uint8_t* src, int64_t pos) {
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        for (int i = 0; i < planes(); ++i) {
            const size_t one_row = plane_row(i);
            window_runs(pos, [&](size_t slot, size_t at, size_t m) {
                h2d(B.plane[i] + at * one_row, src + slot * one_row, m * one_row);
            });
            src += (size_t) keep_ * one_row;
        }
    }
    // the drafter's, behind the model's (save_last())
    for (const SparkLayer& D : spark_layers_) {
        if (!D.swa) continue;
        for (uint16_t* plane : {D.k, D.v}) {
            spark_window(pos, [&](size_t slot, size_t at, size_t m) {
                h2d((uint8_t*) plane + at * spark_row(), src + slot * spark_row(), m * spark_row());
            });
            src += (size_t) s_window_ * spark_row();
        }
    }
    if (spark_) h2d(s_rnn_, src, (size_t) (s_block_ + 1) * n_embd_ * sizeof(float));
}

// The drafter's part of the state: its window layers' rows of the window before pos, and the head's last contexts.
size_t Inkling::spark_extra() const {
    if (!spark_) return 0;
    size_t rings = 0;
    for (const SparkLayer& D : spark_layers_)
        if (D.swa) rings += 2 * (size_t) s_window_ * spark_row();
    return rings + (size_t) (s_block_ + 1) * n_embd_ * sizeof(float);
}

// The window before pos in a drafter's ring: f(place in the saved rows, row of the ring, rows) for each run.
template <class F>
void Inkling::spark_window(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - s_window_); p < pos;) {
        const int64_t at = p % s_ring_, m = std::min<int64_t>(pos - p, s_ring_ - at);
        f((size_t) (p - (pos - s_window_)), (size_t) at, (size_t) m);
        p += m;
    }
}

void Inkling::save_last(uint8_t* dst, int64_t pos) {
    Engine::save_last(dst, pos);
    if (!spark_) return;
    dst += card_state_bytes() - spark_extra();
    for (const SparkLayer& D : spark_layers_) {
        if (!D.swa) continue;
        for (const uint16_t* plane : {D.k, D.v}) {
            std::memset(dst, 0, (size_t) s_window_ * spark_row());   // the places before the sequence's start
            spark_window(pos, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpyAsync(dst + slot * spark_row(), (const uint8_t*) plane + at * spark_row(),
                                   m * spark_row(), cudaMemcpyDeviceToHost, cs(st_)),
                   "save state");
            });
            dst += (size_t) s_window_ * spark_row();
        }
    }
    ck(cudaMemcpyAsync(dst, s_rnn_, (size_t) (s_block_ + 1) * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost,
                       cs(st_)),
       "save state");
    ck(cudaStreamSynchronize(cs(st_)), "save state");
}


// ---- the DSpark drafter ----------------------------------------------------------------------------------------
//
// Inkling 2 Mini's own block drafter (unsloth/Inkling-2-Mini-GGUF's dspark-Inkling-2-Mini-*.gguf, architecture
// "dflash"), as llama.cpp runs it in PR 32218 (88025eb): src/models/dspark.cpp graph_inkling2 and
// common/speculative.cpp's draft-dspark.
//
// Every row the model computes - a prompt's, a token's, a window's with its wrong guesses - goes in at its position:
// the streams entering visits 5, 23 and 41 and the normed output are mixed, a mix for each of the drafter's three
// layers and one for its head (inkling_dspark_fuse()), and each layer keeps its mix's keys and values in its cache
// at the row's position.  A wrong guess's row is written again when its position gets its real token, or by the
// block, before anything reads it: the block reads its own positions and the ones before them.
//
// To guess, a block of eight rows at pos() .. pos() + 7 goes through the three layers: the token at pos() (picked
// and not fed yet) as the model norms its embeddings, then seven copies of mask_embd.  Every row sees every row of
// the block and the cached rows before it, a window layer's from pos() - 1024 on (the first row's window, for all).
// The rows' outputs take the model's output matrix.  The guesses are rows 1, 2, ... in turn (row 0's is none): a
// small recurrent state takes the guess before (its markov_w1 row), the row's output and the head's mix of the row
// before pos(), and its correction goes through the output matrix again and onto the row's scores, which are capped
// as the model's.  Each guess costs a product with the output matrix, and the block one more.

void Inkling::load_spark_shape() {
    const Shards& g = *dg_;
    auto need = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(opt_.draft_model + " is not a DSpark drafter of this model: " + what);
    };
    const MetaValue* arch = g.get("general.architecture");
    need(arch && arch->s == "dflash" && g.find("dspark_hidden_out.weight"),
         "not a dflash file with dspark_hidden_out.weight");
    auto get = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = meta_in(g, key);
        need(v != nullptr, std::string("no dflash.") + key);
        return *v;
    };
    const int n = (int) get("block_count").u;
    need(n > 0 && n <= 8 && (int) get("embedding_length").u == n_embd_, "its layers or width");
    s_head_ = (int) get("attention.head_count").u;
    s_kv_ = (int) get("attention.head_count_kv").u;
    s_dim_ = (int) get("attention.key_length").u;
    s_ff_ = (int) get("feed_forward_length").u;
    need(s_dim_ == 128 && (int) get("attention.value_length").u == s_dim_ &&
             (int) get("rope.dimension_count").u == s_dim_ && s_kv_ > 0 && s_head_ % s_kv_ == 0 && s_ff_ > 0,
         "attention other than heads of 128 values, rotated whole");
    s_block_ = (int) get("block_size").u;
    // a block's queries of a key head go to inkling_block_attend() together
    need(s_block_ >= 2 && s_block_ * (s_head_ / s_kv_) <= 32,
         "a block of " + std::to_string(s_block_) + " rows");
    need(get("attention.causal").u == 0 && get("sample_from_anchor").u == 0,
         "a causal block, or one whose first row is a guess");
    need((int) get("vocab_size").u == n_vocab_ && (int) get("vocab_size_unpadded").u == n_vocab_real_,
         "another vocabulary");
    s_eps_ = (float) get("attention.layer_norm_rms_epsilon").f;
    s_cap_ = (float) get("final_logit_softcapping").f;
    s_window_ = (int) get("attention.sliding_window").u;
    s_theta_ = std::pow((float) get("rope.freq_base").f, -2.f / (float) s_dim_);
    need(s_eps_ > 0.f && s_cap_ > 0.f && s_window_ > 0, "its norms' epsilon, softcap or window");
    const MetaValue& targets = get("target_layers");
    need(targets.items.size() == kTaps + 1 && (int) targets.items[kTaps].u == n_layer_,
         "it reads other layers than three visits and the output");
    for (int k = 0; k < kTaps; ++k) {
        taps_[k] = (int) targets.items[(size_t) k].u;
        need(taps_[k] > 0 && taps_[k] < n_layer_ && (k == 0 || taps_[k] > taps_[k - 1]), "its target layers");
    }
    const MetaValue& pattern = get("attention.sliding_window_pattern");
    spark_layers_.assign((size_t) n, SparkLayer{});
    for (int il = 0; il < n; ++il) spark_layers_[(size_t) il].swa = of_layer(pattern, il) != 0;
    const TensorInfo* w1 = g.find("markov_w1.weight");
    need(w1 && w1->type == kTypeBF16 && w1->shape.size() == 2 && (int64_t) w1->shape[1] == n_vocab_,
         "markov_w1.weight is not BF16 [rank, vocabulary]");
    s_rank_ = (int) w1->shape[0];
    need(s_rank_ > 0 && s_rank_ <= n_embd_, "its recurrent rank");
    markov_ = (const uint16_t*) g.data(*w1);
    // softmax(dspark_fuse): a row of the four inputs' weights for each layer's mix and one for the head's
    const TensorInfo* fuse = g.find("dspark_fuse.weight");
    need(fuse && fuse->type == kTypeF32 && fuse->elements() == (uint64_t) (kTaps + 1) * (n + 1) && n + 1 == 4,
         "dspark_fuse.weight is not F32 [4, 4]");
    const float* f = (const float*) g.data(*fuse);
    for (int j = 0; j < 4; ++j) {
        const float* row = f + (size_t) j * (kTaps + 1);
        const float mx = *std::max_element(row, row + kTaps + 1);
        double sum = 0.0;
        for (int k = 0; k <= kTaps; ++k) sum += fuse_[j][k] = std::exp(row[k] - mx);
        for (int k = 0; k <= kTaps; ++k) fuse_[j][k] = (float) (fuse_[j][k] / sum);
    }
    s_ring_ = s_window_ + s_block_;
    std::fprintf(stderr, "strata-poly: DSpark drafter: %d layers, a block of %d, reads visits %d, %d, %d\n", n,
                 s_block_, taps_[0], taps_[1], taps_[2]);
}

// Twice, like the model's weights: the first time adds up the arena.
void Inkling::load_drafter() {
    const Shards& g = *dg_;
    auto up = [&](const std::string& name, int64_t a, int64_t b, Want w) { return upload(g.find(name), name, a, b, w); };
    const int64_t E = n_embd_, hq = (int64_t) s_head_ * s_dim_, hk = (int64_t) s_kv_ * s_dim_, r = s_rank_;
    s_out_norm_ = up("output_norm.weight", E, 1, kF32);
    s_mask_ = up("mask_embd.weight", E, 1, kF32);
    s_ctx_norm_ = up("dspark_norm.weight", E, 1, kF32);
    // the head's two products take one token: as floats, so that no BF16 matrix asks for lin()'s scratch
    s_joint_ = up("joint_proj.weight", 2 * r + 2 * E, 3 * r, kF32);
    s_joint_b_ = up("joint_proj.bias", 3 * r, 1, kF32);
    s_hout_ = up("dspark_hidden_out.weight", r, E, kF32);
    s_hout_b_ = up("dspark_hidden_out.bias", E, 1, kF32);
    for (int il = 0; il < (int) spark_layers_.size(); ++il) {
        SparkLayer& D = spark_layers_[(size_t) il];
        auto f32 = [&](const char* n, int64_t a) { return up(blk(il, n), a, 1, kF32); };
        auto mat = [&](const char* n, int64_t a, int64_t b) { return up(blk(il, n), a, b, kAnyMat); };
        D.attn_norm = f32("attn_norm.weight", E);
        D.ffn_norm = f32("ffn_norm.weight", E);
        D.ctx_norm = f32("dflare_norm.weight", E);
        D.q_norm = f32("attn_q_norm.weight", s_dim_);
        D.k_norm = f32("attn_k_norm.weight", s_dim_);
        D.k_ctx_norm = f32("attn_k_ctx_norm.weight", s_dim_);
        D.wq = mat("attn_q.weight", E, hq);
        D.wk = mat("attn_k.weight", E, hk);
        D.wv = mat("attn_v.weight", E, hk);
        D.wo = mat("attn_output.weight", hq, E);
        D.k_ctx = mat("attn_k_ctx.weight", E, hk);
        D.v_ctx = mat("attn_v_ctx.weight", E, hk);
        D.gate = mat("ffn_gate.weight", E, s_ff_);
        D.up = mat("ffn_up.weight", E, s_ff_);
        D.down = mat("ffn_down.weight", s_ff_, E);
    }
}

// n rows of keys and values from position pos into a layer's cache, around its ring in a window layer.
void Inkling::spark_store(const SparkLayer& D, const float* k, const float* v, int64_t pos, int n) {
    const int hk = s_kv_ * s_dim_;
    for (int at = 0; at < n;) {
        const int64_t row = D.swa ? (pos + at) % s_ring_ : pos + at;
        const int m = D.swa ? (int) std::min<int64_t>(n - at, s_ring_ - row) : n - at;
        store_f16(k + (size_t) at * hk, D.k, hk, row, m, st_);
        store_f16(v + (size_t) at * hk, D.v, hk, row, m, st_);
        at += m;
    }
}

// The model's n rows from position pos into the drafter: input k's row r at x[k] + r * stride[k].
void Inkling::spark_rows(int64_t pos, int n, const float* const* x, const int64_t* stride) {
    const int E = n_embd_;
    const int64_t last = pos + n - 1, ring = s_block_ + 1;
    for (int c0 = 0; c0 < n; c0 += kSparkRows) {
        const int m = std::min(kSparkRows, n - c0);
        InklingFuse f;
        for (int k = 0; k < 4; ++k) {
            f.x[k] = x[k] + (size_t) c0 * stride[k];
            f.stride[k] = stride[k];
            for (int j = 0; j < 4; ++j) f.w[j][k] = fuse_[j][k];
        }
        for (int j = 0; j < 4; ++j) {
            f.norm[j] = fp(j < 3 ? spark_layers_[(size_t) j].ctx_norm : s_ctx_norm_);
            f.out[j] = s_ctx_[j];
        }
        inkling_dspark_fuse(f, E, m, s_eps_, st_);
        for (int il = 0; il < (int) spark_layers_.size(); ++il) {
            const SparkLayer& D = spark_layers_[(size_t) il];
            lin(D.k_ctx, s_ctx_[il], s_k_, m);
            lin(D.v_ctx, s_ctx_[il], s_v_, m);
            head_norm_rope(s_k_, s_dim_, fp(D.k_ctx_norm), s_kr_, s_dim_, s_kv_, s_dim_, s_theta_, pos + c0, m, s_eps_,
                           st_);
            spark_store(D, s_kr_, s_v_, pos + c0, m);
        }
        // the head reads the mix of the row before a block's first: the last positions' are kept
        for (int t = std::max(0, (int) (last - s_block_ - pos) - c0); t < m; ++t)
            ck(cudaMemcpyAsync(s_rnn_ + (size_t) ((pos + c0 + t) % ring) * E, s_ctx_[3] + (size_t) t * E,
                               (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
               "drafter");
    }
}

void Inkling::drafter_rows(int64_t pos, int n, const float* h) {
    const int64_t rs = (int64_t) streams() * n_embd_;
    const float* x[4] = {r_ + (size_t) kTap0 * n_embd_, r_ + (size_t) (kTap0 + 1) * n_embd_,
                         r_ + (size_t) (kTap0 + 2) * n_embd_, h};
    const int64_t stride[4] = {rs, rs, rs, n_embd_};
    spark_rows(pos, n, x, stride);
}

int Inkling::draft_block(int32_t token, int most, Likeliest* out) { return spark_block(token, pos_, most, out); }

// Up to `most` guesses after `token` at pos0 (see above), into out.
int Inkling::spark_block(int32_t token, int64_t pos0, int most, Likeliest* out) {
    const int B = s_block_, E = n_embd_, r = s_rank_;
    most = std::min(most, B - 1);
    if (most <= 0 || pos0 < 1 || pos0 + B > opt_.max_context) return 0;
    cover(pos0 + B);
    if (!h_w1_) {
        h_w1_ = (float*) hmalloc((size_t) (r + E) * sizeof(float), "pinned memory");
        h_pick_ = (Likeliest*) hmalloc(sizeof(Likeliest), "pinned memory");
    }
    ck(cudaStreamSynchronize(cs(st_)), "drafter");   // h_w1_ is free
    // the block's rows: the token as the model norms its embeddings, then the masks.  The embedding goes behind the
    // markov_w1 row, which the head's first guess writes while the card may not have copied the embedding yet.
    float* const h_embd = h_w1_ + r;
    embed(token, h_embd);
    h2d(s_xn_, h_embd, (size_t) E * sizeof(float));
    rms_norm(s_xn_, fp(tok_norm_), s_x_, E, 1, rms_eps_, st_);
    for (int i = 1; i < B; ++i)
        ck(cudaMemcpyAsync(s_x_ + (size_t) i * E, fp(s_mask_), (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice,
                           cs(st_)),
           "drafter");
    const float scale = 1.f / std::sqrt((float) s_dim_);
    for (const SparkLayer& D : spark_layers_) {
        rms_norm(s_x_, fp(D.attn_norm), s_xn_, E, B, s_eps_, st_);
        lin(D.wq, s_xn_, s_q_, B);
        lin(D.wk, s_xn_, s_k_, B);
        lin(D.wv, s_xn_, s_v_, B);
        head_norm_rope(s_q_, s_dim_, fp(D.q_norm), s_qr_, s_dim_, s_head_, s_dim_, s_theta_, pos0, B, s_eps_, st_);
        head_norm_rope(s_k_, s_dim_, fp(D.k_norm), s_kr_, s_dim_, s_kv_, s_dim_, s_theta_, pos0, B, s_eps_, st_);
        spark_store(D, s_kr_, s_v_, pos0, B);
        const int64_t lo = D.swa ? std::max<int64_t>(0, pos0 - s_window_) : 0;
        inkling_block_attend(s_qr_, D.k, D.v, lo, pos0 + B - 1, D.swa ? s_ring_ : 0, B, s_att_, s_dim_, s_head_,
                             s_kv_, scale, s_work_, st_);
        lin(D.wo, s_att_, s_xn_, B);
        axpy(s_xn_, 1.f, s_x_, B * E, true, st_);
        rms_norm(s_x_, fp(D.ffn_norm), s_xn_, E, B, s_eps_, st_);
        lin(D.gate, s_xn_, s_g_, B);
        lin(D.up, s_xn_, s_u_, B);
        swiglu_clamp(s_g_, s_u_, s_g_, B * s_ff_, INFINITY, st_);
        lin(D.down, s_g_, s_xn_, B);
        axpy(s_xn_, 1.f, s_x_, B * E, true, st_);
    }
    rms_norm(s_x_, fp(s_out_norm_), s_cur_, E, B, s_eps_, st_);
    lin(output_, s_cur_ + E, s_base_, most);   // the guesses' rows' scores
    // the head, a guess at a time from the one before
    ck(cudaMemsetAsync(s_state_, 0, (size_t) r * sizeof(float), cs(st_)), "drafter");
    const float* context = s_rnn_ + (size_t) ((pos0 - 1) % (B + 1)) * E;
    int32_t prev = token;
    for (int i = 1; i <= most; ++i) {
        const uint16_t* w1 = markov_ + (size_t) prev * r;
        for (int j = 0; j < r; ++j) {
            const uint32_t bits = (uint32_t) w1[j] << 16;
            std::memcpy(h_w1_ + j, &bits, sizeof(float));
        }
        copy2(s_state_, s_in_, r, s_cur_ + (size_t) i * E, s_in_ + 2 * r, E, st_);
        h2d(s_in_ + r, h_w1_, (size_t) r * sizeof(float));
        ck(cudaMemcpyAsync(s_in_ + 2 * r + E, context, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
           "drafter");
        lin(s_joint_, s_in_, s_jo_, 1);
        add_bias(fp(s_joint_b_), s_jo_, 3 * r, 1, st_);
        inkling_dspark_gru(s_jo_, s_state_, s_latent_, r, st_);
        lin(s_hout_, s_latent_, s_corr_, 1);
        add_bias(fp(s_hout_b_), s_corr_, E, 1, st_);
        lin(output_, s_corr_, s_bias_, 1);
        float* logits = s_base_ + (size_t) (i - 1) * n_vocab_;
        axpy(s_bias_, 1.f, logits, n_vocab_, true, st_);
        softcap(logits, (size_t) n_vocab_, s_cap_, st_);
        likeliest(logits, n_vocab_real_, h_pick_, st_);
        ck(cudaStreamSynchronize(cs(st_)), "drafter");
        out[i - 1] = *h_pick_;
        prev = h_pick_->token;
    }
    return most;
}

void Inkling::spark_rows_for_test(int64_t pos, int n, const float* features) {
    if (!spark_) throw std::logic_error("spark_rows_for_test: no drafter");
    cover(pos + n);
    const size_t row = (size_t) 4 * n_embd_;
    float* d = nullptr;
    ck(cudaMalloc(&d, (size_t) n * row * sizeof(float)), "test");
    // on st_: a cudaMemcpy() from pageable memory can return before its copy lands, and st_ does not wait for it
    h2d(d, features, (size_t) n * row * sizeof(float));
    const float* x[4] = {d, d + n_embd_, d + 2 * (size_t) n_embd_, d + 3 * (size_t) n_embd_};
    const int64_t stride[4] = {(int64_t) row, (int64_t) row, (int64_t) row, (int64_t) row};
    spark_rows(pos, n, x, stride);
    ck(cudaStreamSynchronize(cs(st_)), "test");
    cudaFree(d);
}

void Inkling::spark_reload_for_test(int64_t pos) {
    if (!spark_) throw std::logic_error("spark_reload_for_test: no drafter");
    pos_ = pos;
    std::vector<uint8_t> state(state_bytes()), rows((size_t) pos * row_bytes());
    save_state(state.data());
    save_rows(0, pos, rows.data());
    for (const SparkLayer& D : spark_layers_)
        for (uint16_t* plane : {D.k, D.v}) zero_vram(plane, (size_t) (D.swa ? s_ring_ : pos) * spark_row());
    zero_vram(s_rnn_, (size_t) (s_block_ + 1) * n_embd_ * sizeof(float));
    load_rows(rows.data(), 0, pos, 0, pos);
    load_state(state.data(), pos);
    ck(cudaStreamSynchronize(cs(st_)), "test");
    pos_ = 0;
}

int Inkling::spark_block_for_test(int32_t token, int64_t pos0, int most, int32_t* tokens, float* logits) {
    if (!spark_) throw std::logic_error("spark_block_for_test: no drafter");
    std::vector<Likeliest> out((size_t) s_block_);
    const int n = spark_block(token, pos0, most, out.data());
    for (int i = 0; i < n; ++i) tokens[i] = out[(size_t) i].token;
    ck(cudaMemcpy2DAsync(logits, (size_t) n_vocab_real_ * sizeof(float), s_base_, (size_t) n_vocab_ * sizeof(float),
                         (size_t) n_vocab_real_ * sizeof(float), (size_t) n, cudaMemcpyDeviceToHost, cs(st_)),
       "test");
    ck(cudaStreamSynchronize(cs(st_)), "test");
    return n;
}

}  // namespace strata::poly
