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

#include "softmax.hpp"

namespace jd {

bool softmax_kd_t::init() {
  auto op_attrs = op_desc_.attrs();
  if (op_attrs["spec_type"] == "lut") {
    if (!isa_available(avx512_core_vbmi)) {
      SPARSE_LOG(WARNING) << "vbmi ISA not available, dispatch to ref_impl.";
      return false;
    }
    prepare_lut_softmax_params();
  } else {
    SPARSE_LOG(ERROR) << "do not supported specialization softmax type";
  }
  return true;
}

bool softmax_k_t::init() {
  nthr_ = omp_get_max_threads();
  auto op_attrs = derived_kd()->get_operator_desc().attrs();
  auto param = derived_kd()->param();
  for (int i = 0; i < nthr_; i++) {
    td.push_back(new ssd::softmax_data_t());
    if (op_attrs["spec_type"] == "lut") {
      if (isa_available(avx512_core_bf16)) {
        td[i]->tmp = malloc(param.scalar_num * sizeof(bfloat16_t));
      } else {
        td[i]->tmp = malloc(param.scalar_num * sizeof(int32_t));
      }
      td[i]->one = bfloat16_t(1.0f);
    }
  }
  jit_softmax_t* ker = new jit_softmax_t(derived_kd()->param());
  if (ker == nullptr) return false;
  if (!(ker->create_kernel())) return false;
  jit_ker_ = ker;
  return true;
}

void softmax_kd_t::prepare_lut_softmax_params() {
  // assert int8 dt as input.
  auto tensor_desc = op_desc_.tensor_descs();
  if (tensor_desc.size() != 2) SPARSE_LOG(ERROR) << "softmax lut kernel need 2 tensor descriptor:src & dst.";
  auto input_dt = tensor_desc[0].dtype();
  auto output_dt = tensor_desc[1].dtype();
  if (get_data_size(input_dt) != 1) LOG(ERROR) << "softmax lut kernel only support int8 dtype as input." << std::endl;
  if (get_data_size(output_dt) == 1 && op_desc_.apply_postops_list().back().op_alg != postop_alg::quantize)
    LOG(WARNING) << "The result of softmax lut kernel need to be quantized when output_dt is int8." << std::endl;
  auto input_shape = tensor_desc[0].shape();

  // init param
  int vec_len = input_shape.back();
  int total_vec_num = 1;
  for (size_t i = 0; i < input_shape.size() - 1; i++) total_vec_num *= input_shape[i];
  param_.scalar_num = total_vec_num * vec_len;
  int thr_num = omp_get_max_threads();
  int vec_num_per_thr = total_vec_num / thr_num;
  int vec_num_tail_thr = total_vec_num - (thr_num - 1) * vec_num_per_thr;
  param_.input_dt = input_dt;
  param_.output_dt = output_dt;
  param_.postop_attrs = op_desc_.apply_postops_list();
  if (param_.postop_attrs.front().op_alg != postop_alg::dequantize)
    LOG(ERROR) << "lut softmax must append dequantize postop.";
  param_.postop_attrs.front().alpha = 0;  // (x-zp)*scale-(max-zp)*scale=(x-max)*scale,so we don't need zp
  param_.get_lut_exp_attrs.push_back(param_.postop_attrs.front());
  param_.postop_attrs.erase(param_.postop_attrs.begin());
  postop_attr exp_attr{data_type::bf16, postop_type::eltwise, postop_alg::exp};
  postop_attr etlop_lut_attr{input_dt, postop_type::eltwise, postop_alg::eltop_int_lut, 16, 256};
  param_.get_lut_exp_attrs.push_back(exp_attr);
  param_.get_lut_exp_attrs.insert(param_.get_lut_exp_attrs.begin(), etlop_lut_attr);
  param_.vec_align_len = vec_len / 32 * 32;
  param_.vec_num_per_thr = vec_num_per_thr;
  param_.vec_num_tail_thr = vec_num_tail_thr;
  param_.vec_tail_len = vec_len % 32;
  param_.sepc_type = ssd::spec_softmax_type::lut;
}

bool softmax_k_t::execute(const std::vector<const void*>& rt_data) const {
  auto param = derived_kd()->param();
  const jit_softmax_t* jit_impl = jit_ker_;

#pragma omp parallel for
  for (int i = 0; i < nthr_; i++) {
    auto data_param = td[i];
    data_param->src = const_cast<char*>(reinterpret_cast<const char*>(rt_data[0]) +
                                        i * param.vec_num_per_thr * (param.vec_align_len + param.vec_tail_len) *
                                            get_data_size(param.input_dt));
    data_param->dst = const_cast<char*>(reinterpret_cast<const char*>(rt_data[1]) +
                                        i * param.vec_num_per_thr * (param.vec_align_len + param.vec_tail_len) *
                                            get_data_size(param.output_dt));
    if (i != nthr_ - 1)
      data_param->process_vec_num = param.vec_num_per_thr;
    else
      data_param->process_vec_num = param.vec_num_tail_thr;
    (*jit_impl)(td[i]);
  }

  return true;
}

}  // namespace jd
