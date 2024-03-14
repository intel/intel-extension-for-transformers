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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GROUPNORM_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GROUPNORM_HPP_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "jit_eltwise_injector.hpp"
#include "regs_pool.hpp"

namespace jd {

struct groupnorm_param_t {
  data_type dt;
  int64_t HW;
  int channels;
  int groups;
  float eps;
  std::vector<postop_attr> postop_attrs;
};

struct groupnorm_data_t {
  void* src;
  void* dst;
  float* sum_x_ptr;
  float* sum_powx_ptr;
  float* gamma;
  float* beta;
};

class jit_groupnorm_t : public jit_generator {
 public:
  explicit jit_groupnorm_t(const groupnorm_param_t& param) : jit_generator(), param_(param) {
    // all of numeral-calc-postop data type will be fp32.
    for (auto& i : param_.postop_attrs) {
      if (i.op_alg != postop_alg::quantize && i.op_alg != postop_alg::dequantize &&
          i.op_alg != postop_alg::eltop_int_lut) {
        i.dt = data_type::fp32;
      }
    }
    eltwise_injector_.eltwise_injector_init(this, param_.postop_attrs);
  }
  virtual ~jit_groupnorm_t() {}

  void prepare_mask(Reg64 reg_tmp, Opmask sum_write_mask);
  void sum_code_gen(regs_pool* rp, Reg64 reg_src, Reg64 reg_sum_x, Reg64 reg_sum_powx, Opmask sun_write_mask,
                    const Xbyak::Label& data_label, size_t sum_dim);
  void calc_scale_and_norm(regs_pool* rp, Reg64 reg_src, Reg64 reg_dst, Reg64 reg_sum_x, Reg64 reg_sum_powx,
                           Reg64 reg_gamma, Reg64 reg_beta, const Xbyak::Label& div_const_label,
                           const Xbyak::Label& eps_label, size_t channels_per_group = 1);

 protected:
  groupnorm_param_t param_;
  Opmask sum_write_mask;
  jit_eltwise_injector eltwise_injector_;
  int unroll;
  int reg_tmp_idx;

 private:
  void generate() override;
};

class jit_channelwise_sum_t : public jit_groupnorm_t {
 public:
  explicit jit_channelwise_sum_t(const groupnorm_param_t& param) : jit_groupnorm_t(param) {}
  virtual ~jit_channelwise_sum_t() {}

 private:
  void generate() override;
};

class jit_channelwise_norm_t : public jit_groupnorm_t {
 public:
  explicit jit_channelwise_norm_t(const groupnorm_param_t& param) : jit_groupnorm_t(param) {}
  virtual ~jit_channelwise_norm_t() {}

 private:
  void generate() override;
};

}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_DOMAIN_JIT_GROUPNORM_HPP_
