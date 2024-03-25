//  Copyright (c) 2022 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_MATMUL_REDUCE_SCALE_QUANT_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_MATMUL_REDUCE_SCALE_QUANT_HPP_

#include "jit_generator.hpp"

namespace jd {
struct dynamic_quant_matmul_reduce_scale_quant_param_t {
  int n_block_num;
  int quant_m;
  int quant_n;
  int n;
};

struct dynamic_quant_matmul_reduce_scale_quant_data_t {
  void* mat_src;
  void* mat_dst;
  void* dst_scale;
  void* reduce_scale;
};

class jit_dynamic_quant_matmul_reduce_scale_quant_t : public jit_generator {
 public:
  explicit jit_dynamic_quant_matmul_reduce_scale_quant_t(const dynamic_quant_matmul_reduce_scale_quant_param_t& param)
      : jit_generator(), param_(param) {}
  virtual ~jit_dynamic_quant_matmul_reduce_scale_quant_t() {}

 private:
  dynamic_quant_matmul_reduce_scale_quant_param_t param_;

 private:
  void generate() override;
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_DYNAMIC_QUANT_MATMUL_REDUCE_SCALE_QUANT_HPP_
