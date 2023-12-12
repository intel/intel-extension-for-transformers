//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <regex>
#include "models/model_utils/quant_utils.h"
#include "common.h"

// default hparams (Whisper tiny)
struct whisper_hparams {
    int32_t n_vocab       = 51864;
    int32_t n_audio_ctx   = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head  = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx    = 448;
    int32_t n_text_state  = 384;
    int32_t n_text_head   = 6;
    int32_t n_text_layer  = 4;
    int32_t n_mels        = 80;
    int32_t ftype         = 1;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// quantize a model
bool whisper_model_quantize(const std::string & fname_inp, const std::string & fname_out, ne_ftype ftype) {
    gpt_vocab vocab;

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic != NE_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        fout.write((char *) &magic, sizeof(magic));
    }

    whisper_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        finp.read((char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        finp.read((char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        finp.read((char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        finp.read((char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        finp.read((char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        finp.read((char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        finp.read((char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        finp.read((char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        finp.read((char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        finp.read((char *) &hparams.ftype,         sizeof(hparams.ftype));

        const int32_t qntvr_src =    hparams.ftype / 1000; //GGML_QNT_VERSION_FACTOR
        const int32_t ftype_dst = 2 * 1000 + ftype; //GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR

        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        fprintf(stderr, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        fprintf(stderr, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        fprintf(stderr, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        fprintf(stderr, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        fprintf(stderr, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        fprintf(stderr, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        fprintf(stderr, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        fprintf(stderr, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
        fprintf(stderr, "%s: ftype (src)   = %d\n", __func__, hparams.ftype);
        fprintf(stderr, "%s: qntvr (src)   = %d\n", __func__, qntvr_src);
        fprintf(stderr, "%s: ftype (dst)   = %d\n", __func__, ftype_dst);
        fprintf(stderr, "%s: qntvr (dst)   = %d\n", __func__, 2);

        fout.write((const char *) &hparams.n_vocab,       sizeof(hparams.n_vocab));
        fout.write((const char *) &hparams.n_audio_ctx,   sizeof(hparams.n_audio_ctx));
        fout.write((const char *) &hparams.n_audio_state, sizeof(hparams.n_audio_state));
        fout.write((const char *) &hparams.n_audio_head,  sizeof(hparams.n_audio_head));
        fout.write((const char *) &hparams.n_audio_layer, sizeof(hparams.n_audio_layer));
        fout.write((const char *) &hparams.n_text_ctx,    sizeof(hparams.n_text_ctx));
        fout.write((const char *) &hparams.n_text_state,  sizeof(hparams.n_text_state));
        fout.write((const char *) &hparams.n_text_head,   sizeof(hparams.n_text_head));
        fout.write((const char *) &hparams.n_text_layer,  sizeof(hparams.n_text_layer));
        fout.write((const char *) &hparams.n_mels,        sizeof(hparams.n_mels));
        fout.write((const char *) &ftype_dst,             sizeof(hparams.ftype));
    }

    // load mel filters
    {
        whisper_filters filters;

        finp.read ((char *) &filters.n_mel, sizeof(filters.n_mel));
        fout.write((char *) &filters.n_mel, sizeof(filters.n_mel));
        finp.read ((char *) &filters.n_fft, sizeof(filters.n_fft));
        fout.write((char *) &filters.n_fft, sizeof(filters.n_fft));

        filters.data.resize(filters.n_mel * filters.n_fft);
        finp.read ((char *) filters.data.data(), filters.data.size() * sizeof(float));
        fout.write((char *) filters.data.data(), filters.data.size() * sizeof(float));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        finp.read ((char *) &n_vocab, sizeof(n_vocab));
        fout.write((char *) &n_vocab, sizeof(n_vocab));

        //if (n_vocab != hparams.n_vocab) {
        //    fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
        //            __func__, fname_inp.c_str(), n_vocab, hparams.n_vocab);
        //    return false;
        //}

        char word[129];

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word[len] = '\0';

            finp.read ((char *) word, len);
            fout.write((char *) word, len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // regexes of tensor names to not be quantized
    const std::vector<std::string> to_skip = {
        //"encoder.*",
        "encoder.conv1.bias",
        "encoder.conv2.bias",
        "encoder.positional_embedding",
        "decoder.positional_embedding",
    };

    if (!model_quantize_special(finp, fout, ftype, { ".*" }, to_skip)) {
        fprintf(stderr, "%s: failed to quantize model '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    finp.close();
    fout.close();

    return true;
}

int main(int argc, char ** argv) {
    quant_params q_params;
    if (quant_params_parse(argc, argv, q_params) == false) {
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ne_init_params params = {0, NULL, false};
        struct ne_context* ctx = ne_init(params);
        ne_free(ctx);
    }
    const std::string fname_inp = q_params.model_file;
    const std::string fname_out = q_params.out_file;
    // printf("input_model_file:%s \n",fname_inp.c_str());

    const ne_ftype ftype = NE_FTYPE_MOSTLY_Q_JBLAS;
    // printf("*****model_ftype:%s \n*****",ftype); 

    const int64_t t_main_start_us = common_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = common_time_us();

        if (!whisper_model_quantize(fname_inp, fname_out, ne_ftype( ftype ))) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = common_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = common_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}

