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

#include <utility>
#include <vector>
#include <map>
#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "kernels/slice_types.hpp"
#include "jit_binary_injector.hpp"

#define SLICE_GET_OFF(field) offsetof(ssd::slice_data_t, field)

namespace jd {
class jit_slice_t : public jit_generator {
  using Zmm = Xbyak::Zmm;
  using Reg32 = Xbyak::Reg32;
  using Reg64 = Xbyak::Reg64;
  using Opmask = Xbyak::Opmask;

 public:
  explicit jit_slice_t(const ssd::slice_param_t& param) : jit_generator(), param_(param) { assign_regs(); }
  virtual ~jit_slice_t() {}

 private:
  void generate() override;
  void assign_regs();
  void copy_continuously();
  void copy_by_step();

  void load_params() {
    mov(src_addr, ptr[reg_param + SLICE_GET_OFF(src)]);
    mov(dst_addr, ptr[reg_param + SLICE_GET_OFF(dst)]);
  }

 private:
  ssd::slice_param_t param_;
  jit_binary_injector binary_injector;

  Reg64 reg_param;
  Reg64 src_addr;
  Reg64 dst_addr;
  Opmask extend_tail_mask;
  std::vector<Reg64> binaryop_addr;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_SLICE_HPP_
