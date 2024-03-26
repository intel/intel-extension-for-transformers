//  Copyright (c) 2021 Intel Corporation
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

#include "group_norm.hpp"
namespace executor {
GroupNormOperator::GroupNormOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("epsilon");
  if (iter != attrs_map.end()) {
    epsilon_ = StringToNum<float>(attrs_map["epsilon"]);
  }
  iter = attrs_map.find("group");
  if (iter != attrs_map.end()) {
    group_ = StringToNum<int64_t>(attrs_map["group"]);
  }
  iter = attrs_map.find("channels");
  if (iter != attrs_map.end()) {
    channels_ = StringToNum<int64_t>(attrs_map["channels"]);
  }
  iter = attrs_map.find("append_op");
  swish_fusion_ = (iter != attrs_map.end() && iter->second == "swish") ? true : false;
  iter = attrs_map.find("swish_beta");
  if (iter != attrs_map.end()) swish_beta_ = StringToNum<float>(attrs_map["swish_beta"]);
}

void GroupNormOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  assert(src->dtype() == "fp32" || src->dtype() == "bf16");
  output[0]->set_dtype(src->dtype());
  Tensor* gamma = input[1];
  Tensor* beta = input[2];
  assert(gamma->shape()[0] == channels_);
  assert(beta->shape()[0] == channels_);
  const float* gamma_data = static_cast<const float*>(gamma->data());
  const float* beta_data = static_cast<const float*>(beta->data());
  for (int64_t i = 0; i < channels_; ++i) {
    if (gamma_data[i] != 1.f || beta_data[i] != 0.f) {
      affine_ = true;
      break;
    }
  }
}

void GroupNormOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t> src_shape = input[0]->shape();
  assert(src_shape.size() > 1);
  output[0]->set_shape(src_shape);
  src_desc_ = {src_shape, type2sparsemem_[input[0]->dtype()], jd::format_type::undef};
  dst_desc_ = {src_shape, type2sparsemem_[output[0]->dtype()], jd::format_type::undef};
  gamma_desc_ = {{}, jd::data_type::fp32, jd::format_type::undef};
  beta_desc_ = {{}, jd::data_type::fp32, jd::format_type::undef};
  vector<jd::tensor_desc> ts_descs = {src_desc_, dst_desc_, gamma_desc_, beta_desc_};
  std::unordered_map<std::string, std::string> op_attrs_;
  op_attrs_["eps"] = std::to_string(epsilon_);
  op_attrs_["groups"] = std::to_string(group_);
  op_attrs_["postop_list"] = "";
  vector<jd::postop_attr> postops;
  if (swish_fusion_) {
    jd::postop_attr swish_attr = {jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::swish,
                                  swish_beta_};  // swish calculation always will be fp32 in groupnorm postop.
    postops.push_back(swish_attr);
    op_attrs_["postop_list"] += "swish+" + std::to_string(swish_beta_);
  }
  jd::operator_desc op_desc(jd::kernel_kind::groupnorm, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs_, postops);
  jd::groupnorm_desc groupnorm_desc(op_desc);
  groupnorm_ker = jd::groupnorm(groupnorm_desc);
  if (work_space) {
    aligned_free(work_space);
  }
  work_space = reinterpret_cast<void*>(
      aligned_alloc(ALIGNMENT, (groupnorm_ker.get_workspace_size() / ALIGNMENT + 1) * ALIGNMENT));
}

void GroupNormOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  const vector<int64_t>& src_shape = src->shape();
  const float* src_data = static_cast<const float*>(src->data());
  Tensor* gamma = input[1];
  const float* gamma_data = static_cast<const float*>(gamma->data());
  Tensor* beta = input[2];
  const float* beta_data = static_cast<const float*>(beta->data());
  Tensor* dst = output[0];
  float* dst_data = static_cast<float*>(dst->mutable_data());
  std::vector<const void*> rt_data;
  rt_data = {src->data(), dst->data(), gamma->data(), beta->data(), work_space};
  groupnorm_ker.execute(rt_data);
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(GroupNorm);
}  // namespace executor
