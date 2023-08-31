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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GATHER_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GATHER_HPP_

#include <map>
#include <vector>

#include "jit_generator.hpp"
#include "src/cpu/jit_domain/jit_binary_injector.hpp"
#include "src/utils.hpp"

namespace jd {

class jit_gather_t : public jit_generator {
 public:
  struct param_t {
    bool use_avx512;
    data_type dt;
    int dt_size;
    int src_axis_size, dst_axis_size;
    int src_size, idx_size, outer_size, inner_size;
    const std::vector<binaryop_attr> binary_ops;
  };

  struct rt_data_t {
    const void* src;
    const void* idx;
    void* dst;
    const void* binaryop_addrs[16];

   public:
    rt_data_t(const void* a, const void* b, void* c) : src(a), idx(b), dst(c) {}
  };

  explicit jit_gather_t(const param_t& param) : jit_generator(), param_(param), binaryop_attrs_(param.binary_ops) {
    binary_injector.binary_injector_init(this);
  }
  virtual ~jit_gather_t() {}

 private:
  void generate() override;
  template <bool USE_AVX512>
  void generate_();

  param_t param_;
  jit_binary_injector binary_injector;
  const std::vector<jd::binaryop_attr> binaryop_attrs_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GATHER_HPP_
