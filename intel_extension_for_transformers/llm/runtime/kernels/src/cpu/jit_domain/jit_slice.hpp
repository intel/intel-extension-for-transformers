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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SLICE_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SLICE_HPP_

#include <vector>

#include "jit_binary_injector.hpp"
#include "jit_generator.hpp"
#include "regs_pool.hpp"

namespace jd {
class jit_slice_t : public jit_generator {
 public:
  struct param_t {
    bool use_avx512;
    int step;
    int src_axis_size;
    int inner_size;
    int copy_size;
    int dt_size;
  };
  struct rt_data_t {
    const void* src;
    void* dst;
  };

  explicit jit_slice_t(const param_t& param)
      : jit_generator(),
        use_avx512(param.use_avx512),
        step(param.step),
        src_axis_size(param.src_axis_size),
        inner_size(param.inner_size),
        copy_size(param.copy_size),
        dt_size(param.dt_size) {
    SPARSE_LOG_IF(FATAL, dt_size != 1 && dt_size != 2 && dt_size != 4) << "Unexpected dt_size: " << dt_size;
  }
  virtual ~jit_slice_t() {}

 private:
  void generate() override;

  template <bool USE_AVX512>
  void generate_();
  template <bool USE_AVX512>
  inline void copy_by_step(regs_pool* const rp, const Reg64 dst, const Reg64 src);
  template <bool USE_AVX512>
  inline void copy_continuously(regs_pool* const rp, const Reg64 dst, const Reg64 src);

  const bool use_avx512;
  const int step;
  const int src_axis_size;
  const int inner_size;
  const int copy_size;
  const int dt_size;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SLICE_HPP_
