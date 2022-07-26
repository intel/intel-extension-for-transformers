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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_LAYERNORM_BA_HPP_

#include "../jit_generator.hpp"
#include "utils.hpp"
#include "kernels/layernorm_ba_types.hpp"
#include "jit_eltwise_injector.hpp"

#define LNBA_GET_OFF(field) offsetof(ssd::layernorm_ba_data_t, field)

namespace jd {
class jit_layernorm_ba_t : public jit_generator {
  using Zmm = Xbyak::Zmm;
  using Reg64 = Xbyak::Reg64;
  using Opmask = Xbyak::Opmask;

 public:
  explicit jit_layernorm_ba_t(const ssd::layernorm_ba_param_t& param) : jit_generator(), param_(param) {
    // insert the affine op to the front of the postop-chain.
    // when we apply the true postops,the begin idx is row_num.
    if (param_.affine) {
      for (int i = param_.row_num - 1; i >= 0; i--) {
        postop_attr tmp = {param_.dt, postop_type::eltwise, postop_alg::linear, param_.alpha[i], param_.beta[i]};
        param_.postop_attrs.insert(param_.postop_attrs.begin(), tmp);
      }
    }

    eltwise_injector.init_tb_allocate_set(param_.postop_attrs);
    // 29=32(all zmm regs num)-3(zmm_one,mean_,var_)
    unroll_degree = 29 - eltwise_injector.max_zmm_allocate_num();
    assign_regs();
    eltwise_injector.eltwise_injector_init(this, param_.postop_attrs);
  }
  virtual ~jit_layernorm_ba_t() {}

 private:
  void generate() override;
  size_t get_offset(int col, int row);
  void reset_unroll_reg_idxs(int degree);
  std::pair<int, int> get_unroll_add_idx(int begin);
  bool check_unroll_add_done();
  void binary_add(int degree, Zmm dst);
  void prepare_mask();
  void assign_regs();
  void escape_regs(int degree);

  void load_params() {
    mov(src_addr, ptr[reg_param + LNBA_GET_OFF(martix)]);
    mov(one_div_n, ptr[reg_param + LNBA_GET_OFF(one_div_n)]);
    mov(one, ptr[reg_param + LNBA_GET_OFF(one)]);
    mov(eps, ptr[reg_param + LNBA_GET_OFF(eps)]);
  };

 private:
  ssd::layernorm_ba_param_t param_;
  jit_eltwise_injector eltwise_injector;
  int unroll_degree;
  std::vector<int> unroll_reg_idxs;
  std::map<reg_type, std::set<int>> reg_map;
  Zmm zmm_one;
  Zmm reg_mean;
  Zmm reg_var;
  Reg64 reg_param;
  Reg64 reg64_tmp;
  Reg64 src_addr;
  Reg64 one_div_n;
  Reg64 one;
  Reg64 eps;
  Opmask remain_task_mask;
};  // namespace jd
}  // namespace jd
#endif