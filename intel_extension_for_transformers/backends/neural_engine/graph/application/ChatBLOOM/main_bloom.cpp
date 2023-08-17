#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "common.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

struct bloom_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t f16     = 1;
    float alibi_bias_max = 8;
};

struct bloom_layer {
    // normalization
    struct ne_tensor * attention_norm;
    struct ne_tensor * attention_norm_b;

    // attention
    struct ne_tensor * query_key_value;
    struct ne_tensor * query_key_value_b;
    struct ne_tensor * wo;
    struct ne_tensor * wo_b;

    // normalization
    struct ne_tensor * ffn_norm;
    struct ne_tensor * ffn_norm_b;

    // ff
    struct ne_tensor * w1;
    struct ne_tensor * w1_b;
    struct ne_tensor * w2;
    struct ne_tensor * w2_b;
};

struct bloom_model {
    bloom_hparams hparams;

    struct ne_tensor * tok_embeddings;
    struct ne_tensor * norm;
    struct ne_tensor * norm_b;

    struct ne_tensor * output_norm;
    struct ne_tensor * output_norm_b;
    struct ne_tensor * output;
    

    std::vector<bloom_layer> layers;

    // key + value memory
    struct ne_tensor * memory_k;
    struct ne_tensor * memory_v;

    //
    struct ne_context * ctx;
    std::map<std::string, struct ne_tensor *> tensors;
};

std::vector<gpt_vocab::id> bloom_tokenize(const gpt_vocab & vocab, const std::string & text, bool bos) {
    //auto res = gpt_tokenize(vocab, text);

    //if (bos) {
    //    res.insert(res.begin(), 1); // TODO: replace with vocab.bos
    //}

    std::vector<gpt_vocab::id> res;

    if (bos) {
        res.push_back(1); // TODO: replace with vocab.bos
    }

     //find the longest token that matches the text
    int pos = 0;
    while (true) {
        int l = 0;
        int t = 0;
        for (const auto & kv : vocab.id_to_token) {
            if (kv.second.size() < l) continue;
            if (kv.second.size() > text.size() - pos) continue;
            if (text.substr(pos, kv.second.size()) == kv.second) {
                l = kv.second.size();
                t = kv.first;
            }
        }

        if (l == 0) {
            break;
        }

        res.push_back(t);
        pos += l;
    }

    return res;
}

gpt_vocab::id bloom_sample_top_p(
        const gpt_vocab & vocab,
        const float * logits,
        std::vector<gpt_vocab::id> & last_n_tokens,
        double repeat_penalty,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    // return max token (greedy search)
    // {
    //     int max_idx = 0;
    //     float max_val = logits[0];
    //     // fprintf(stdout, "\nlogits[0]: %f", logits[0]);
    //     for (int i = 1; i < n_logits; ++i) {
    //         if (logits[i] > max_val) {
    //             max_val = logits[i];
    //             max_idx = i;
    //             // fprintf(stdout, "\nlogits[%d]: %f", i, logits[i]);
    //         }
    //     }
    //     return max_idx;
    // }

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/bloom/compare/main...shawwn:bloom:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (logits[i] < 0.0) {
                    logits_id.push_back(std::make_pair(logits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(logits[i]*scale/repeat_penalty, i));
                }                
            } else {
                logits_id.push_back(std::make_pair(logits[i]*scale, i));
            }
        }
    }

    std::sort(
            logits_id.begin(),
            logits_id.end(),
            [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
        return a.first > b.first;
    });

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}


// load the model's weights from a file
bool bloom_model_load(const std::string & fname, bloom_model & model, gpt_vocab & vocab, int n_ctx) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    int n_ff = 0;
    int n_parts = 0;

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;

        n_ff = ((4*hparams.n_embd + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
        // n_parts = BLOOM_N_PARTS.at(hparams.n_embd);
        n_parts = 1;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_mult  = %d\n", __func__, hparams.n_mult);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
        printf("%s: n_ff    = %d\n", __func__, n_ff);
        printf("%s: n_parts = %d\n", __func__, n_parts);
    }

    // load vocab
    {
        const int32_t n_vocab = model.hparams.n_vocab;

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;

            //if (i < 30000) {
            //    printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
            //}
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ne_type wtype = NE_TYPE_COUNT;
    switch (model.hparams.f16) {
        case 0: wtype = NE_TYPE_F32;  break;
        case 1: wtype = NE_TYPE_F16;  break;
        case 2: wtype = NE_TYPE_Q4_0; break;
        case 3: wtype = NE_TYPE_Q4_1; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            __func__, fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    const ne_type wtype2 = NE_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*n_vocab*ne_type_sizef(wtype); // tok_embeddings

        ctx_size += n_embd*ne_type_sizef(NE_TYPE_F32); // norm
        ctx_size += n_embd*ne_type_sizef(NE_TYPE_F32); // norm_b

        ctx_size += n_embd*ne_type_sizef(NE_TYPE_F32); // output_norm
        ctx_size += n_embd*ne_type_sizef(NE_TYPE_F32); // output_norm_b

        ctx_size += n_embd*n_vocab*ne_type_sizef(wtype); // output

        ctx_size += n_layer*(n_embd*ne_type_sizef(NE_TYPE_F32)); // attention_norm
        ctx_size += n_layer*(n_embd*ne_type_sizef(NE_TYPE_F32)); // attention_norm_b

        ctx_size += n_layer*(3*n_embd*n_embd*ne_type_sizef(wtype)); // query_key_value
        ctx_size += n_layer*(3*n_embd*ne_type_sizef(NE_TYPE_F32)); // query_key_value_b
        ctx_size += n_layer*(n_embd*n_embd*ne_type_sizef(wtype)); // wo
        ctx_size += n_layer*(n_embd*ne_type_sizef(NE_TYPE_F32)); // wo_b

        ctx_size += n_layer*(n_embd*ne_type_sizef(NE_TYPE_F32)); // ffn_norm
        ctx_size += n_layer*(n_embd*ne_type_sizef(NE_TYPE_F32)); // ffn_norm_b

        ctx_size += n_layer*(n_ff*n_embd*ne_type_sizef(wtype)); // w1
        ctx_size += n_layer*(n_ff*ne_type_sizef(NE_TYPE_F32)); // w1_b
        ctx_size += n_layer*(n_ff*n_embd*ne_type_sizef(wtype)); // w2
        ctx_size += n_layer*(n_ff*ne_type_sizef(NE_TYPE_F32)); // w2_b

        ctx_size += n_ctx*n_layer*n_embd*ne_type_sizef(NE_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ne_type_sizef(NE_TYPE_F32); // memory_v

        ctx_size += (5 + 10*n_layer)*256; // object overhead TODO:

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ne_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
        };

        model.ctx = ne_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ne_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.tok_embeddings = ne_new_tensor_2d(ctx, wtype, n_embd, n_vocab, NE_SIZE_CALC);
        model.norm   = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
        model.norm_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

        model.output_norm = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
        model.output_norm_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
        model.output = ne_new_tensor_2d(ctx, wtype,         n_embd, n_vocab, NE_SIZE_CALC);

        // map by name
        model.tensors["tok_embeddings.weight"] = model.tok_embeddings;
        model.tensors["norm.weight"]   = model.norm;
        model.tensors["norm.bias"]   = model.norm_b;
        
        model.tensors["output_norm.weight"] = model.output_norm;
        model.tensors["output_norm.bias"] = model.output_norm_b;
        model.tensors["output.weight"] = model.output;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.attention_norm = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
            layer.attention_norm_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

            layer.query_key_value = ne_new_tensor_2d(ctx, wtype, n_embd, 3*n_embd, NE_SIZE_CALC);
            layer.query_key_value_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, 3*n_embd, NE_SIZE_CALC);
            layer.wo = ne_new_tensor_2d(ctx, wtype, n_embd, n_embd, NE_SIZE_CALC);
            layer.wo_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

            layer.ffn_norm = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);
            layer.ffn_norm_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

            layer.w1 = ne_new_tensor_2d(ctx, wtype, n_embd,   n_ff, NE_SIZE_CALC);
            layer.w1_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_ff, NE_SIZE_CALC);
            layer.w2 = ne_new_tensor_2d(ctx, wtype,   n_ff, n_embd, NE_SIZE_CALC);
            layer.w2_b = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_embd, NE_SIZE_CALC);

            // map by name
            model.tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;
            model.tensors["layers." + std::to_string(i) + ".attention_norm.bias"] = layer.attention_norm_b;

            model.tensors["layers." + std::to_string(i) + ".attention.query_key_value.weight"] = layer.query_key_value;
            model.tensors["layers." + std::to_string(i) + ".attention.query_key_value.bias"] = layer.query_key_value_b;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;
            model.tensors["layers." + std::to_string(i) + ".attention.wo.bias"] = layer.wo_b;

            model.tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;
            model.tensors["layers." + std::to_string(i) + ".ffn_norm.bias"] = layer.ffn_norm_b;

            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w1.bias"] = layer.w1_b;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
            model.tensors["layers." + std::to_string(i) + ".feed_forward.w2.bias"] = layer.w2_b;
        }
    }

    // key + value memory
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_elements, NE_SIZE_CALC);
        model.memory_v = ne_new_tensor_1d(ctx, NE_TYPE_F32, n_elements, NE_SIZE_CALC);

        const size_t memory_size = ne_nbytes(model.memory_k) + ne_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    const size_t file_offset = fin.tellg();

    fin.close();

    std::vector<uint8_t> tmp;

    for (int i = 0; i < n_parts; ++i) {
        const int part_id = i;
        //const int part_id = n_parts - i - 1;

        std::string fname_part = fname;
        if (i > 0) {
            fname_part += "." + std::to_string(i);
        }

        printf("%s: loading model part %d/%d from '%s'\n", __func__, i+1, n_parts, fname_part.c_str());
        fin = std::ifstream(fname_part, std::ios::binary);
        fin.seekg(file_offset);

        // load weights
        {
            int n_tensors = 0;
            size_t total_size = 0;

            printf("%s: ", __func__);

            while (true) {
                int32_t n_dims;
                int32_t length;
                int32_t ftype;

                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                fin.read(reinterpret_cast<char *>(&length), sizeof(length));
                fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

                if (fin.eof()) {
                    break;
                }

                int32_t nelements = 1;
                int32_t ne[2] = { 1, 1 };
                for (int i = 0; i < n_dims; ++i) {
                    fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                    nelements *= ne[i];
                }

                std::string name(length, 0);
                fin.read(&name[0], length);

                if (model.tensors.find(name.data()) == model.tensors.end()) {
                    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                    return false;
                }

                // split_type = 0: split by columns
                // split_type = 1: split by rows
                int split_type = 0;

                // split_type = 0:
                // regex:
                //   - tok_embeddings.*
                //   - layers.*.attention.wo.weight
                //   - layers.*.feed_forward.w2.weight

                // split_type = 1:
                // regex:
                //   - output.*
                //   - layers.*.attention.wq.weight
                //   - layers.*.attention.wk.weight
                //   - layers.*.attention.wv.weight
                //   - layers.*.feed_forward.w1.weight
                //   - layers.*.feed_forward.w3.weight
                if (name.find("tok_embeddings") != std::string::npos) {
                    split_type = 0;
                } else if (name.find("layers") != std::string::npos) {
                    if (name.find("attention.wo.weight") != std::string::npos) {
                        split_type = 0;
                    } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                        split_type = 0;
                    } else {
                        split_type = 1;
                    }
                } else if (name.find("output") != std::string::npos) {
                    split_type = 1;
                }

                auto tensor = model.tensors[name.data()];

                if (n_dims == 1) {
                    if (ne_nelements(tensor) != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                } else {
                    if (ne_nelements(tensor)/n_parts != nelements) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                        return false;
                    }
                }

                if (n_dims == 1) {
                    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                        fprintf(stderr,
                                "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%d, %d]\n",
                                __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                        return false;
                    }
                } else {
                    if (split_type == 0) {
                        if (tensor->ne[0]/n_parts != ne[0] || tensor->ne[1] != ne[1]) {
                            fprintf(stderr,
                                    "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0] / n_parts, tensor->ne[1], ne[0], ne[1]);
                            return false;
                        }
                    } else {
                        if (tensor->ne[0] != ne[0] || tensor->ne[1]/n_parts != ne[1]) {
                            fprintf(stderr,
                                    "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%d, %d]\n",
                                    __func__, name.data(), tensor->ne[0], tensor->ne[1] / n_parts, ne[0], ne[1]);
                            return false;
                        }
                    }
                }

                if (0) {
                    static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                    printf("%24s - [%5d, %5d], type = %6s, split = %d\n", name.data(), ne[0], ne[1], ftype_str[ftype], split_type);
                }

                size_t bpe = 0;

                switch (ftype) {
                    case 0: bpe = ne_type_size(NE_TYPE_F32);  break;
                    case 1: bpe = ne_type_size(NE_TYPE_F16);  break;
                    case 2: bpe = ne_type_size(NE_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                    case 3: bpe = ne_type_size(NE_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                    default:
                            {
                                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                                return false;
                            }
                };

                if (n_dims == 1 || n_parts == 1) {
                    if ((nelements*bpe)/ne_blck_size(tensor->type) != ne_nbytes(tensor)) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ne_nbytes(tensor), nelements*bpe);
                        return false;
                    }

                    if (part_id == 0) {
                        fin.read(reinterpret_cast<char *>(tensor->data), ne_nbytes(tensor));
                    } else {
                        fin.seekg(ne_nbytes(tensor), std::ios::cur);
                    }

                    total_size += ne_nbytes(tensor);
                } else {
                    if ((nelements*bpe)/ne_blck_size(tensor->type) != ne_nbytes(tensor)/n_parts) {
                        fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                                __func__, name.data(), ne_nbytes(tensor)/n_parts, nelements*bpe);
                        return false;
                    }

                    if (split_type == 0) {
                        const int np0 = ne[0];

                        const size_t row_size = (tensor->ne[0]/ne_blck_size(tensor->type))*ne_type_size(tensor->type);
                        assert(row_size == tensor->nb[1]);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = i1*row_size;
                            const size_t offset = offset_row + ((part_id*np0)/ne_blck_size(tensor->type))*ne_type_size(tensor->type);
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
                        }
                    } else {
                        const int np1 = ne[1];

                        const size_t row_size = (tensor->ne[0]/ne_blck_size(tensor->type))*ne_type_size(tensor->type);

                        for (int i1 = 0; i1 < ne[1]; ++i1) {
                            const size_t offset_row = (i1 + part_id*np1)*row_size;
                            fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
                        }
                    }

                    total_size += ne_nbytes(tensor)/n_parts;
                }

                //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ne_nbytes(tensor)/1024.0/1024.0);
                if (++n_tensors % 8 == 0) {
                    printf(".");
                    fflush(stdout);
                }
            }

            printf(" done\n");

            printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
        }

        fin.close();
    }

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
bool bloom_eval(
        const bloom_model & model,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w,
              size_t                     & mem_per_token) {

    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;

    const int d_key = n_embd/n_head;

    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ne_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
    };

    struct ne_context * ctx0 = ne_init(params);
    ne_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ne_tensor * embd = ne_new_tensor_1d(ctx0, NE_TYPE_I32, N, NE_SIZE_CALC);
    memcpy(embd->data, embd_inp.data(), N*ne_element_size(embd));

    struct ne_tensor * inpL = ne_get_rows(ctx0, model.tok_embeddings, embd);

    // word embeddings norm
    {
        inpL = ne_norm(ctx0, inpL);
        inpL = ne_mul(ctx0, ne_repeat(ctx0, model.norm, inpL), inpL);
        inpL = ne_add(ctx0, ne_repeat(ctx0, model.norm_b, inpL), inpL);
    }

    for (int il = 0; il < n_layer; ++il) {
        struct ne_tensor * inpSA = inpL; //TODO: copy?

        struct ne_tensor * cur;

        // norm
        {
            cur = ne_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ne_mul(ctx0,
                        ne_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
            cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].attention_norm_b, cur), cur);
        }

        // attn
        {
            cur = ne_mul_mat(ctx0,model.layers[il].query_key_value, cur);

            cur = ne_add(ctx0,
                    ne_repeat(ctx0, model.layers[il].query_key_value_b, cur),
                    cur);
        }

        // cur = ggml_debug(ctx0, cur);

        // self-attention
        {
            struct ne_tensor * Qcur = ne_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ne_tensor * Kcur = ne_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd); //TODO: float or fp16?
            struct ne_tensor * Vcur = ne_view_2d(ctx0, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ne_tensor * k = ne_view_1d(ctx0, model.memory_k, N*n_embd, (ne_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ne_tensor * v = ne_view_1d(ctx0, model.memory_v, N*n_embd, (ne_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ne_build_forward_expand(&gf, ne_cpy(ctx0, Kcur, k));
                ne_build_forward_expand(&gf, ne_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ne_tensor * Q =
                ne_permute(ctx0,
                            ne_cpy(ctx0, Qcur,
                                ne_new_tensor_3d(ctx0, NE_TYPE_F32, n_embd/n_head, n_head, N, NE_SIZE_CALC)),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ne_tensor * K =
                ne_permute(ctx0, ne_reshape_3d(ctx0,
                                ne_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ne_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            struct ne_tensor * KQ = ne_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ne_tensor * KQ_scaled =
                ne_scale(ctx0,
                        KQ,
                        ne_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // Alibi
            // KQ_scaled_alibi = KQ_scaled + alibi_bias //TODO: optimize
            struct ne_tensor * KQ_scaled_alibi = ne_alibi(ctx0, KQ_scaled, n_past, n_head, model.hparams.alibi_bias_max);

            // KQ_masked = mask_past(KQ_scaled)
            struct ne_tensor * KQ_masked = ne_diag_mask_inf(ctx0, KQ_scaled_alibi, n_past);

            // KQ = soft_max(KQ_masked)
            struct ne_tensor * KQ_soft_max = ne_soft_max_inplace(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ne_tensor *V_trans =
                    ne_cpy(ctx0,
                             ne_permute(ctx0,
                                          ne_reshape_3d(ctx0,
                                                          ne_view_1d(ctx0, model.memory_v, (n_past + N) * n_embd,
                                                                       il * n_ctx * ne_element_size(model.memory_v) *
                                                                       n_embd),
                                                          n_embd / n_head, n_head, n_past + N),
                                          1, 2, 0, 3),
                             ne_new_tensor_3d(ctx0, model.memory_v->type, n_past + N, n_embd / n_head, n_head, NE_SIZE_CALC));
            // KQV = transpose(V) * KQ_soft_max
            struct ne_tensor * KQV = ne_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ne_tensor * KQV_merged = ne_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ne_cpy(ctx0,
                    KQV_merged,
                    ne_new_tensor_2d(ctx0, NE_TYPE_F32, n_embd, N, NE_SIZE_CALC));

            // projection
            cur = ne_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].wo_b, cur), cur);
        }

        struct ne_tensor * inpFF = ne_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ne_norm(ctx0, inpFF);

                // cur = ffn_norm*cur + ffn_norm_b
                cur = ne_mul(ctx0,
                        ne_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
                cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].ffn_norm_b, cur), cur);
            }

            cur = ne_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].w1_b, cur), cur);

            cur = ne_gelu(ctx0, cur);

            cur = ne_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            cur = ne_add(ctx0, ne_repeat(ctx0, model.layers[il].w2_b, cur), cur);
        }

        cur  = ne_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // norm
    {
        inpL = ne_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ne_mul(ctx0,
                    ne_repeat(ctx0, model.output_norm, inpL),
                    inpL);
        
        inpL = ne_add(ctx0, ne_repeat(ctx0, model.output_norm_b, inpL), inpL);
    }

    // lm_head
    {
        inpL = ne_mul_mat(ctx0, model.output, inpL);
    }

    // logits -> probs
    //inpL = ne_soft_max_inplace(ctx0, inpL);

    // run the computation
    ne_build_forward_expand(&gf, inpL);
    ne_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ne_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ne_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ne_used_mem(ctx0)/N;
    }
    //printf("used_mem = %zu\n", ne_used_mem(ctx0));

    ne_free(ctx0);

    return true;
}

int main(int argc, char ** argv) {
    ne_time_init();
    const int64_t t_main_start_us = ne_time_us();

    common_params params;
    params.model = "models/ggml-model-bloomz-7b1-f16-q4_0.bin";
    params.prompt = "Je vais";

    if (common_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

//    params.prompt = R"(// this function checks if the number n is prime
//bool is_prime(int n) {)";

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    bloom_model model;

    // load the model
    {
        const int64_t t_start_us = ne_time_us();
        const int n_ctx = 512;
        if (!bloom_model_load(params.model, model, vocab, n_ctx)) {  // TODO: set context from user input ??
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ne_time_us() - t_start_us;
    }
    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::bloom_tokenize(vocab, params.prompt, false); //TODO: set bos to true?
    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("\n");
    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < (int) embd_inp.size(); i++) {
        printf("%6d -> '%s'\n", embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    printf("\n");
    printf("sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    printf("\n\n");

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    bloom_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ne_time_us();

            if (!bloom_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) { // update logits
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ne_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ne_time_us();

                id = bloom_sample_top_p(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_p, temp, rng);

                // // print
                // printf("\ngenerated token: '%s' (%d)\n", vocab.id_to_token[id].c_str(), id);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ne_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 2) {
            printf(" [end of text]\n");
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ne_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ne_free(model.ctx);

    return 0;
}


