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

#include "kernels/layernorm_ba.hpp"

namespace jd {

bool layernorm_ba_kd_t::init() {
  if (!isa_available(avx512_core)) return false;
  auto tensor_desc = op_desc_.tensor_descs();
  assert(tensor_desc.size() == 3);
  // TODO(zhe1wang): support more data_type.
  auto input_dt = tensor_desc[0].dtype();
  auto output_dt = tensor_desc[1].dtype();
  auto affine_dt = tensor_desc[2].dtype();
  assert(input_dt == data_type::fp32);
  assert(tensor_desc[0].ftype() == format_type::ba);
  auto tensor_shape = tensor_desc[0].shape();

  int row_num = 1;
  for (int i = 0; i < tensor_shape.size() - 1; i++) row_num *= tensor_shape[i];
  int col_num = tensor_shape.back();

  // init ptr
  one_div_n_ = reinterpret_cast<float*>(aligned_alloc(64, col_num * sizeof(float)));
  for (int i = 0; i < col_num; i++) one_div_n_[i] = 1.0 / row_num;

  // init params
  // TODO(zhe1wang): support col nums can't divded by 16.
  assert(col_num % 16 == 0);
  int max_eff_nthr = col_num / 16;
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize.
  params_.resize(max_eff_nthr);
  int col_per_thr = col_num / max_eff_nthr;
  for (int i = 0; i < max_eff_nthr; i++) {
    int thread_elt_offset = col_per_thr * i;
    ssd::layernorm_ba_param_t param;
    param.input_dt = input_dt;
    param.output_dt = output_dt;
    param.affine_dt = affine_dt;
    param.row_num = row_num;
    param.col_num = col_num;
    param.process_col = col_per_thr;
    param.thread_elt_offset = thread_elt_offset;
    param.postop_attrs = op_desc_.apply_postops_list();
    auto op_attr = op_desc_.attrs();
    params_[i] = param;
  }
  return true;
}

bool layernorm_ba_k_t::init() {
  auto op_desc = kd()->operator_desc();
  auto output_dt = op_desc.tensor_descs()[1].dtype();
  auto col_num = op_desc.tensor_descs()[0].shape().back();
  nthr_ = col_num / 16;
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize.
  jit_kers_.resize(nthr_);
  for (int i = 0; i < nthr_; i++) {
    td.push_back(new ssd::layernorm_ba_data_t());
    jit_layernorm_ba_t* ker = new jit_layernorm_ba_t(derived_kd()->params()[i]);
    if (ker == nullptr) return false;
    if (!(ker->create_kernel())) return false;
    jit_kers_[i] = ker;
  }

  return true;
}

bool layernorm_ba_k_t::execute(const std::vector<const void*>& rt_data) const {
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize and restore it at end of the function.
#pragma omp parallel for
  for (int i = 0; i < nthr_; i++) {
    const jit_layernorm_ba_t* jit_impl = jit_kers_[i];
    auto data_param = td[i];
    data_param->src = const_cast<void*>(rt_data[0]);
    data_param->dst = const_cast<void*>(rt_data[1]);
    data_param->alpha = reinterpret_cast<float*>(const_cast<void*>(rt_data[2]));
    data_param->beta = reinterpret_cast<float*>(const_cast<void*>(rt_data[3]));
    data_param->one_div_n = derived_kd()->one_div_n_ptr();
    (*jit_impl)(td[i]);
  }

  return true;
}

}  // namespace jd
