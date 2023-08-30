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

#include "jit_mean_var_reduce.hpp"

#define GET_OFF(field) offsetof(ssd::mean_var_reduce_data_t, field)

namespace jd {
void jit_mean_var_reduce_t::load_params() {
  mov(reg_mean_in, ptr[reg_param + GET_OFF(mean_in)]);
  mov(reg_var_in, ptr[reg_param + GET_OFF(var_in)]);
  mov(reg_mean_out, ptr[reg_param + GET_OFF(mean_out)]);
  mov(reg_var_out, ptr[reg_param + GET_OFF(var_out)]);
}

void jit_mean_var_reduce_t::calc_mean_var() {
  vmovups(avg_a, ptr[reg_mean_in]);  // zmm0
  vmovups(M_a, ptr[reg_var_in]);     // zmm2
  mov(num_tmp, 0);
  vpbroadcastd(n_a, num_tmp.cvt32());  // zmm5
  mov(num_tmp, param_.BM);
  vpbroadcastd(n_b, num_tmp.cvt32());  // zmm6
  mov(num_tmp.cvt32(), bit_cast<int, float>(reciprocal_M_));
  vpbroadcastd(reciprocal_M, num_tmp.cvt32());  // zmm11
  for (int i = 0; i < param_.element_num - 1; i++) {
    vmovups(avg_b, ptr[reg_mean_in + (i + 1) * param_.BN * sizeof(float)]);  // zmm1
    vmovups(M_b, ptr[reg_var_in + (i + 1) * param_.BN * sizeof(float)]);     // zmm3
    vaddps(M_a, M_a, M_b);                                                   // zmm2
    if (i == param_.element_num - 2 && tail_BM_ != 0) {
      mov(num_tmp, tail_BM_);
      vpbroadcastd(n_b, num_tmp.cvt32());  // zmm5
    }
    vpaddd(n_a, n_a, n_b);             // zmm4
    vsubps(delta, avg_b, avg_a);       // zmm7
    vmulps(pow2_delta, delta, delta);  // zmm8
    float scale_tmp;
    if (i != param_.element_num - 2 || tail_BM_ == 0) {
      scale_tmp = 1.f / (i + 2);
    } else {
      scale_tmp = tail_BM_ * 1.f / ((i + 1) * param_.BM + tail_BM_);
    }
    mov(num_tmp, bit_cast<int, float>(scale_tmp));
    vpbroadcastd(scale, num_tmp.cvt32());  // zmm9
    vcvtdq2ps(n_a_float, n_a);             // zmm5
    vmulps(scalee, scale, n_a_float);      // zmm10

    vfmadd231ps(M_a, pow2_delta, scalee);  // zmm3
    vfmadd231ps(avg_a, delta, scale);      // zmm0
  }
  vmovups(ptr[reg_mean_out], avg_a);  // zmm0
  vmulps(M_a, M_a, reciprocal_M);     // zmm2
  vmovups(ptr[reg_var_out], M_a);     // zmm2
}
void jit_mean_var_reduce_t::generate() {
  this->preamble();
  load_params();
  calc_mean_var();
  this->postamble();
}

}  // namespace jd
