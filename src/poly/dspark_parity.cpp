// src/poly/dspark_parity.cpp - Inkling 2 Mini's DSpark drafter against llama.cpp's on the same inputs.  Run by hand:
//     poly_dspark_parity MODEL.gguf DRAFTER.gguf RECORD [--max-context N] [--expert-profile FILE] [--reload]
// RECORD is what llama.cpp's drafter got and gave in a run of speculative decoding: after a header of four int32
// ('DSPK', n_embd, 4, 0), a 'B' for each of the model's calls (int32 n, n positions, n tokens, then n rows of the
// drafter's four inputs, 4 * n_embd floats a row) and a 'D' for each round of guesses (int32 pos0, the token at
// pos0, the guesses' count and the guesses, then seven rows of scores, one a slot of the block after the first).
// The calls' rows go into this engine's drafter as they are, and each round's block is guessed again: its slots'
// tokens (each the likeliest of its scores, from the guesses before) and scores are compared with llama.cpp's.
// --reload: before each round the conversation's state and rows are saved, the drafter's caches zeroed and the two
// loaded back (Inkling::spark_reload_for_test()); its lines are then the same as without it.
#include "engine.hpp"
#include "inkling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int32_t first_max(const float* l, int n) { return (int32_t) (std::max_element(l, l + n) - l); }

template <class T>
void get(std::FILE* f, T* p, size_t n) {
    if (std::fread(p, sizeof(T), n, f) != n) throw std::runtime_error("the record ends early");
}

}  // namespace

int main(int argc, char** argv) {
    using namespace strata::poly;
    if (argc < 4) {
        std::fprintf(stderr, "usage: poly_dspark_parity MODEL.gguf DRAFTER.gguf RECORD [--max-context N] "
                             "[--expert-profile FILE] [--reload]\n");
        return 2;
    }
    try {
        Options o;
        o.model = argv[1];
        o.draft_model = argv[2];
        o.max_context = 4096;
        bool reload = false;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--reload") reload = true;
            else if (a == "--max-context" && i + 1 < argc) o.max_context = std::atoi(argv[++i]);
            else if (a == "--expert-profile" && i + 1 < argc) o.profile = argv[++i];
            else throw std::runtime_error("unknown option " + a);
        }
        std::unique_ptr<Engine> e = Engine::open(o);
        Inkling* ink = dynamic_cast<Inkling*>(e.get());
        if (!ink) throw std::runtime_error("not an Inkling 2 model");
        const int nv = ink->n_vocab_real();
        std::FILE* f = std::fopen(argv[3], "rb");
        if (!f) throw std::runtime_error(std::string("cannot read ") + argv[3]);
        int32_t hdr[4];
        get(f, hdr, 4);
        const int E = hdr[1];
        if (hdr[0] != 0x4b505344 || E != e->n_embd()) throw std::runtime_error("not a record of this model");
        std::vector<float> feats, ref((size_t) 7 * nv), mine((size_t) 7 * nv);
        int rounds = 0, slots = 0, same = 0, chains = 0;
        double worst = 0.0;
        for (int c; (c = std::fgetc(f)) != EOF;) {
            if (c == 'B') {
                int32_t n;
                get(f, &n, 1);
                std::vector<int32_t> pos((size_t) n), tok((size_t) n);
                get(f, pos.data(), (size_t) n);
                get(f, tok.data(), (size_t) n);
                feats.resize((size_t) n * 4 * E);
                get(f, feats.data(), feats.size());
                for (int i = 1; i < n; ++i)
                    if (pos[(size_t) i] != pos[0] + i) throw std::runtime_error("a call's positions do not follow");
                ink->spark_rows_for_test(pos[0], n, feats.data());
                continue;
            }
            if (c != 'D') throw std::runtime_error("a bad record");
            int32_t head[3];
            get(f, head, 3);
            std::vector<int32_t> draft((size_t) head[2]);
            get(f, draft.data(), draft.size());
            get(f, ref.data(), ref.size());
            int32_t toks[8];
            if (reload) ink->spark_reload_for_test(head[0]);
            const int n = ink->spark_block_for_test(head[1], head[0], 7, toks, mine.data());
            // the slots compare while the guesses before them are the same: a slot's input is the guess before it
            int agree = 0;
            double d = 0.0;
            std::string line;
            for (int i = 0; i < n; ++i) {
                const float* r = ref.data() + (size_t) i * nv;
                const float* m = mine.data() + (size_t) i * nv;
                const int32_t rt = first_max(r, nv);
                for (int j = 0; j < nv; ++j) d = std::max(d, (double) std::fabs(r[j] - m[j]));
                ++slots;
                line += " " + std::to_string(rt) + (rt == toks[i] ? "=" : "/" + std::to_string(toks[i]));
                if (rt != toks[i]) break;
                ++agree;
            }
            same += agree;
            chains += agree == n;
            worst = std::max(worst, d);
            ++rounds;
            std::printf("pos %d anchor %d: %d of %d slots, max |d logit| %.4f:%s\n", head[0], head[1], agree, n, d,
                        line.c_str());
        }
        std::fclose(f);
        std::printf("%d rounds: %d of %d slots compared gave llama.cpp's token, %d rounds all 7; worst |d logit| "
                    "%.4f\n",
                    rounds, same, slots, chains, worst);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "poly_dspark_parity: %s\n", ex.what());
        return 1;
    }
}
