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

#ifndef OPT_H
#define OPT_H

#include "models/model_utils/model_files.h"
#include "models/model_utils/model_types.h"

// n_ctx = 2048
// FFN activation = Relu
// word_embed_proj_dim != n_embd needss two proj gemm (in decoder block  and out decoder block)
enum opt_model {
  OPT_UNKNOWN,
  OPT_125M,    // layers = 12, n_embd = 768, n_head = 12, word_embed_proj_dim = 768
  OPT_350M,    // layers = 24, n_embd = 1024, n_head = 16, word_embed_proj_dim = 512
  OPT_1DOT3B,  // layers = 24, n_embd = 2048, n_head = 32, word_embed_proj_dim = 2048
  OPT_2DOT7B,  // layers = 32, n_embd = 2560, n_head = 32, word_embed_proj_dim = 2560
  OPT_6DOT7B,  // layers = 32, n_embd = 4096, n_head = 32, word_embed_proj_dim = 4096
  OPT_13B,     // layers = 40, n_embd = 5120, n_head = 40, word_embed_proj_dim = 5120
  OPT_30B,     // layers = 48, n_embd = 7168, n_head = 56, word_embed_proj_dim = 7168
  OPT_66B,     // layers = 64, n_embd = 9216, n_head = 72, word_embed_proj_dim = 9216
};

// TODO naive memory buffer size
static const model_scratch opt_mem_req(int n_layers) {
  switch (n_layers) {
    case 12:  // OPT_125M
      return {512ull * MB, 512ull * MB, 1024ull * MB};
    case 24:  // OPT_350M, OPT_1DOT3B
      return {1024ull * MB, 1024ull * MB, 2048ull * MB};
    case 32:  // OPT_2DOT7B OPT_6DOT7B
      return {2048ull * MB, 2048ull * MB, 4096ull * MB};
    case 40:
      return {2560ull * MB, 2560ull * MB, 5120ull * MB};
    case 48:
      return {3072ull * MB, 3072ull * MB, 6144ull * MB};
    case 64:
      return {4096ull * MB, 4096ull * MB, 8192ull * MB};
    default:
      MODEL_ASSERT(false);
  }
}

class OPT : public IModel {
 private:
  model_archs arch = MODEL_OPT;
  std::unique_ptr<model_model_loader> ml;
  uint32_t n_layer, n_embd, n_ff, n_vocab, word_embed_proj_dim, max_seq_len;
  bool do_layer_norm_before = true;
  int n_gpu_layer;
  ne_type memory_type;
  bool use_mmap, use_mlock, vocab_only;
  model_scratch scratch;

 public:
  void init(const char* path_model, model_context* ctx, int n_gpu_layers, bool use_mmap_, bool use_mlock_,
            bool vocab_only_) override;
  void load(model_context* ctx, model_progress_callback progress_callback, void* progress_callback_user_data) override;
};

#endif  // OPT_H
