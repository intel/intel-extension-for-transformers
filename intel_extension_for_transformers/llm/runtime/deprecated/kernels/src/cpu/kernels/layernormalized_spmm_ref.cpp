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

#include "layernormalized_spmm_ref.hpp"
#include "spmm_ref.hpp"
#include "layernorm_ba_ref.hpp"

namespace jd {

enum rf_data_idx {
  SPMM_WEI,
  SPMM_SRC,
  SPMM_BIAS,
  SPMM_DST,
  SPMM_SCALES,
  SPMM_MEAN,
  SPMM_VAR,
  SPMM_WORKSPACE,
  LNORM_DST1,
  LNORM_APLHA,
  LNORM_BETA,
  LNORM_DST2
};

bool layernormalized_spmm_ref_kd_t::init() {
  auto ts_desc = op_desc_.tensor_descs();
  SPARSE_LOG_IF(FATAL, ts_desc.size() != 5) << "layernormalized vnni spmm need 5 ts_desc";
  auto wei_desc = ts_desc[0];
  auto src_desc = ts_desc[1];
  auto bia_desc = ts_desc[2];
  //  when only need int8 output, make dt(dst_desc) = fp32.
  auto dst_desc = ts_desc[3];
  auto scales_desc = ts_desc[4];
  int M = wei_desc.shape()[0];
  int num_mbs = src_desc.shape()[0];
  int micro_bs = src_desc.shape().size() == 2 ? src_desc.shape()[1] : src_desc.shape()[2];
  // gen mean, var, workspace_desc
  tensor_desc mean_desc = {{num_mbs, micro_bs}, data_type::fp32, format_type::a};
  tensor_desc var_desc = {{num_mbs, micro_bs}, data_type::fp32, format_type::a};
  tensor_desc workspace_desc = {{M * 2, micro_bs * num_mbs}, data_type::fp32, format_type::a};
  std::vector<tensor_desc> spmm_ts_desc = {wei_desc,    src_desc,  bia_desc, dst_desc,
                                           scales_desc, mean_desc, var_desc, workspace_desc};
  std::vector<tensor_desc> lnorm_ts_desc = {dst_desc, dst_desc};
  auto op_attrs = op_desc_.attrs();
  std::unordered_map<std::string, std::string> spmm_op_attrs;
  spmm_op_attrs["append_sum"] = op_attrs["append_sum"];
  spmm_op_attrs["sparse_ptr"] = op_attrs["sparse_ptr"];
  spmm_op_attrs["sub_func"] = op_attrs["sub_func"];
  spmm_op_attrs["micro_oc"] = op_attrs["micro_oc"];
  spmm_op_attrs["welford"] = "true";
  std::unordered_map<std::string, std::string> lnorm_op_attrs;
  lnorm_op_attrs["spec_type"] = "direct";
  lnorm_op_attrs["split_output"] = op_attrs["split_output"];
  split_output = op_attrs["split_output"] == "true" ? true : false;
  spmm_desc = new operator_desc(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu,
                                spmm_ts_desc, spmm_op_attrs);
  lnorm_desc = new operator_desc(kernel_kind::layernorm_ba, kernel_prop::forward_inference, engine_kind::cpu,
                                 lnorm_ts_desc, lnorm_op_attrs, op_desc_.apply_postops_list());
  return true;
}

bool layernormalized_spmm_ref_k_t::init() {
  auto kd = derived_kd();
  split_output = kd->split_output;
  std::shared_ptr<const kernel_desc_t> spmm_ker_desc;
  std::shared_ptr<const kernel_desc_t> lnorm_ker_desc;
  if (!kernel_desc_t::create<spmm_ref_kd_t>(spmm_ker_desc, *(kd->spmm_desc))) return false;
  if (!create<spmm_ref_k_t, spmm_ref_kd_t>(kernels_[0], spmm_ker_desc)) return false;
  if (!kernel_desc_t::create<layernorm_ba_ref_kd_t>(lnorm_ker_desc, *(kd->lnorm_desc))) return false;
  if (!create<layernorm_ba_ref_k_t, layernorm_ba_ref_kd_t>(kernels_[1], lnorm_ker_desc)) return false;
  return true;
}

bool layernormalized_spmm_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  std::vector<const void*> spmm_rt_data = {rt_data[SPMM_WEI], rt_data[SPMM_SRC],      rt_data[SPMM_BIAS],
                                           rt_data[SPMM_DST], rt_data[SPMM_SCALES],   rt_data[SPMM_MEAN],
                                           rt_data[SPMM_VAR], rt_data[SPMM_WORKSPACE]};
  std::vector<const void*> lnorm_rt_data = {rt_data[SPMM_DST],   rt_data[LNORM_DST1], rt_data[LNORM_APLHA],
                                            rt_data[LNORM_BETA], rt_data[SPMM_MEAN],  rt_data[SPMM_VAR]};
  if (split_output) lnorm_rt_data.push_back(rt_data[LNORM_DST2]);
  kernels_[0]->execute(spmm_rt_data);
  kernels_[1]->execute(lnorm_rt_data);
  return true;
}

}  // namespace jd
