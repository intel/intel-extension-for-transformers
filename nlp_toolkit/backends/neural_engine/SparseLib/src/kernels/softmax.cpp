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

#include "kernels/softmax.hpp"

namespace jd {

bool softmax_kd_t::init() {
  if (!isa_available(avx512_core)) return false;
  auto op_attrs = op_desc_.attrs();
  if (op_attrs["spec_type"] == "lut") {
    // assert int8 dt as input.
    auto tensor_desc = op_desc_.tensor_descs();
    if (tensor_desc.size() != 2) LOG(ERROR) << "softmax lut kernel need 2 tensor descriptor:src & dst." << std::endl;
    auto input_dt = tensor_desc[0].dtype();
    auto output_dt = tensor_desc[1].dtype();
    if (output_dt == data_type::bf16 && !isa_available(avx512_core_bf16)) return false;
    assert(output_dt == data_type::bf16);  // TODO(zhe1wang): support more dt,current impl is for experiment only.
    if (get_data_size(input_dt) != 1)
      LOG(ERROR) << "softmax lut kernel only support int8 dtype as input currently." << std::endl;
    auto input_shape = tensor_desc[0].shape();

    // init param
    int vec_len = input_shape.back();
    int total_vec_num = 1;
    for (int i = 0; i < input_shape.size() - 1; i++) total_vec_num *= input_shape[i];
    int thr_num = omp_get_max_threads();
    int vec_num_per_thr = total_vec_num / thr_num;
    int vec_num_tail_thr = total_vec_num - (thr_num - 1) * vec_num_per_thr;
    param_.input_dt = input_dt;
    param_.output_dt = output_dt;
    param_.postop_attrs = op_desc_.apply_postops_list();
    postop_attr exp_attr{data_type::bf16, postop_type::eltwise, postop_alg::exp};
    postop_attr etlop_lut_attr{input_dt, postop_type::eltwise, postop_alg::eltop_int_lut, 16, 256};
    param_.postop_attrs.push_back(exp_attr);
    param_.postop_attrs.insert(param_.postop_attrs.begin(), etlop_lut_attr);
    param_.vec_align_len = vec_len / 32 * 32;
    param_.vec_num_per_thr = vec_num_per_thr;
    param_.vec_num_tail_thr = vec_num_tail_thr;
    param_.vec_tail_len = vec_len % 32;
    param_.sepc_type = ssd::spec_softmax_type::lut;
  } else {
    LOG(ERROR) << "do not supported specialization softmax type" << std::endl;
  }
  return true;
}

bool softmax_k_t::init() {
  int thr_num = omp_get_max_threads();
  nthr_ = thr_num;
  for (int i = 0; i < thr_num; i++) td.push_back(new ssd::softmax_data_t());
  jit_softmax_t* ker = new jit_softmax_t(derived_kd()->param());
  if (ker == nullptr) return false;
  if (!(ker->create_kernel())) return false;
  jit_ker_ = ker;
  return true;
}

bool softmax_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto param = derived_kd()->param();
#pragma omp parallel for
  for (int i = 0; i < nthr_; i++) {
    const jit_softmax_t* jit_impl = jit_ker_;
    auto data_param = td[i];
    data_param->src =
        const_cast<void*>(rt_data[0]) + i * param.vec_num_per_thr * (param.vec_align_len + param.vec_tail_len);
    data_param->dst =
        const_cast<void*>(rt_data[1]) + i * param.vec_num_per_thr * (param.vec_align_len + param.vec_tail_len);
    if (i != nthr_ - 1)
      data_param->process_vec_num = param.vec_num_per_thr;
    else
      data_param->process_vec_num = param.vec_num_tail_thr;
    (*jit_impl)(td[i]);
  }

  return true;
}

}  // namespace jd
