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
  SPARSE_LOG_IF(FATAL, tensor_desc.size() != 3) << "only support 3 rt_data";
  auto input_dt = tensor_desc[0].dtype();
  auto output_dt = tensor_desc[1].dtype();
  auto affine_dt = tensor_desc[2].dtype();
  SPARSE_LOG_IF(FATAL, input_dt != data_type::fp32) << "only support fp32";
  SPARSE_LOG_IF(FATAL, tensor_desc[0].ftype() != format_type::ba) << "only support transpose";
  auto tensor_shape = tensor_desc[0].shape();
  SPARSE_LOG_IF(FATAL, tensor_shape.size() < 2 || tensor_shape.size() > 3) << "layernorm support 2D/3D input";
  if (tensor_shape.size() == 2) tensor_shape.insert(tensor_shape.begin(), 1);

  int batch_num = tensor_shape[0];
  int row_num = tensor_shape[1];
  int col_num = tensor_shape[2];

  // init params
  // TODO(zhe1wang): support col nums can't divded by 16.
  SPARSE_LOG_IF(FATAL, col_num % 16 != 0) << "col nums should divded by 16 now";
  auto ker_num = col_num / 16;
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize.
  params_.resize(ker_num);
  int col_per_thr = col_num / ker_num;
  for (int i = 0; i < ker_num; i++) {
    int thread_elt_offset = col_per_thr * i;
    ssd::layernorm_ba_param_t param;
    param.input_dt = input_dt;
    param.output_dt = output_dt;
    param.affine_dt = affine_dt;
    param.row_num = row_num;
    param.col_num = col_num;
    param.process_col = col_per_thr;
    param.process_batch_per_ker = omp_get_max_threads() >= batch_num ? 1 : batch_num;
    param.ker_per_batch = ker_num;
    param.thread_elt_offset = thread_elt_offset;
    param.postop_attrs = op_desc_.apply_postops_list();
    param.binaryop_attrs = op_desc_.get_binaryop_list();
    params_[i] = param;
  }
  return true;
}

bool layernorm_ba_k_t::init() {
  auto op_desc = kd()->get_operator_desc();
  auto tensor_shape = op_desc.tensor_descs()[0].shape();
  src_dt = op_desc.tensor_descs()[0].dtype();
  dst_dt = op_desc.tensor_descs()[1].dtype();
  auto params = derived_kd()->params();
  per_ker_process_batch = params[0].process_batch_per_ker;
  ker_num = params[0].ker_per_batch;
  if (tensor_shape.size() == 2) tensor_shape.insert(tensor_shape.begin(), 1);
  batch_loop = tensor_shape[0] / per_ker_process_batch;
  row_num = tensor_shape[1];
  col_num = tensor_shape[2];
  one_div_n_ = reinterpret_cast<float*>(aligned_alloc(64, col_num * sizeof(float)));
  if (one_div_n_ == nullptr) {
    SPARSE_LOG(INFO) << "layernorm alloc one_div failed";
    return false;
  }
  for (int i = 0; i < col_num; i++) one_div_n_[i] = 1.0 / row_num;
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize.
  jit_kers_.resize(ker_num);
  for (int i = 0; i < batch_loop; i++) td.push_back(new ssd::layernorm_ba_data_t());
  for (int i = 0; i < ker_num; i++) {
    jit_layernorm_ba_t* ker = new jit_layernorm_ba_t(derived_kd()->params()[i]);
    if (ker == nullptr) return false;
    if (!(ker->create_kernel())) return false;
    jit_kers_[i] = ker;
  }

  return true;
}

bool layernorm_ba_k_t::execute(const std::vector<const void*>& rt_data) const {
  // TODO(zhe1wang): set most appreciate thread num when fuse with quantize and restore it at end of the function.
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch_loop; i++) {
    for (int j = 0; j < ker_num; j++) {
      const jit_layernorm_ba_t* jit_impl = jit_kers_[j];
      auto data_param = td[i];
      data_param->src =
          const_cast<char*>(reinterpret_cast<const char*>(rt_data[0]) + i * row_num * col_num * get_data_size(src_dt));
      data_param->dst =
          const_cast<char*>(reinterpret_cast<const char*>(rt_data[1]) + i * row_num * col_num * get_data_size(dst_dt));
      data_param->alpha = reinterpret_cast<float*>(const_cast<void*>(rt_data[2]));
      data_param->beta = reinterpret_cast<float*>(const_cast<void*>(rt_data[3]));
      data_param->one_div_n = one_div_n_ptr();
      (*jit_impl)(td[i]);
    }
  }
  return true;
}
}  // namespace jd
