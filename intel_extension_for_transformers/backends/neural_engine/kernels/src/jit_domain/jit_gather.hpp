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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_GATHER_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_GATHER_HPP_

#include <utility>
#include <vector>
#include <map>
#include "jit_generator.hpp"
#include "utils.hpp"
#include "kernels/gather_types.hpp"
#include "jit_domain/jit_binary_injector.hpp"

#define GATHER_GET_OFF(field) offsetof(ssd::gather_data_t, field)

namespace jd {
class jit_gather_t : public jit_generator {
  using Zmm = Xbyak::Zmm;
  using Reg32 = Xbyak::Reg32;
  using Reg64 = Xbyak::Reg64;
  using Opmask = Xbyak::Opmask;

 public:
  explicit jit_gather_t(const ssd::gather_param_t& param) : jit_generator(), param_(param) {
    assign_regs();
    binary_injector.binary_injector_init(this);
  }
  virtual ~jit_gather_t() {}

 private:
  void generate() override;
  void assign_regs();
  void gen_load_offset();

  void load_params() {
    mov(src_addr, ptr[reg_param + GATHER_GET_OFF(src)]);
    mov(idx_addr, ptr[reg_param + GATHER_GET_OFF(idx)]);
    mov(dst_addr, ptr[reg_param + GATHER_GET_OFF(dst)]);
    for (size_t i = 0; i < param_.binaryop_attrs.size(); i++) {
      mov(binaryop_addr[i], ptr[reg_param + GATHER_GET_OFF(binaryop_addrs)]);
    }
  }

 private:
  ssd::gather_param_t param_;
  jit_binary_injector binary_injector;

  Reg64 reg_param;
  Reg64 src_addr;
  Reg64 idx_addr;
  Reg64 dst_addr;
  Reg64 gather_idx;
  Reg64 next_gather_idx;
  Opmask tail_mask;
  Opmask extend_tail_mask;
  std::vector<Reg64> binaryop_addr;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_GATHER_HPP_
