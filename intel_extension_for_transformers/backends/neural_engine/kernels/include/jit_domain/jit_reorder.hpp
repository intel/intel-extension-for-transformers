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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_REORDER_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_REORDER_HPP_

#include <utility>
#include <vector>
#include <map>
#include "jit_generator.hpp"
#include "utils.hpp"
#include "kernels/reorder_types.hpp"
#include "jit_domain/jit_binary_injector.hpp"

#define REORDER_GET_OFF(field) offsetof(ssd::reorder_data_t, field)

namespace jd {
class jit_reorder_t : public jit_generator {
  using Zmm = Xbyak::Zmm;
  using Reg32 = Xbyak::Reg32;
  using Reg64 = Xbyak::Reg64;
  using Opmask = Xbyak::Opmask;

 public:
  explicit jit_reorder_t(const ssd::reorder_param_t& param) : jit_generator(), param_(param) { assign_regs(); }
  virtual ~jit_reorder_t() {}

 private:
  void generate() override;
  void assign_regs();
  void gen_load_offset();

  void load_params() {
    mov(src_addr, ptr[reg_param + REORDER_GET_OFF(src)]);
    mov(dst_addr, ptr[reg_param + REORDER_GET_OFF(dst)]);
  }

 private:
  ssd::reorder_param_t param_;

  Reg64 reg_param;
  Reg64 src_addr;
  Reg64 dst_addr;
  Reg64 reorder_idx;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_REORDER_HPP_
