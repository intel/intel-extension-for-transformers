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

#include "gather.hpp"
#include "common.hpp"

namespace executor {
#ifdef WITH_SPARSELIB
static unordered_map<string, jd::data_type> type2sparsemem{
    {"fp32", jd::data_type::fp32}, {"s32", jd::data_type::s32}, {"fp16", jd::data_type::fp16},
    {"u8", jd::data_type::u8},     {"s8", jd::data_type::s8},   {"bf16", jd::data_type::bf16}};

#endif

GatherOperator::GatherOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  idx_axis_ = (iter != attrs_map.end() && iter->second != "") ? iter->second : "0";
  iter = attrs_map.find("batch_dims");
  src_axis_ = (iter != attrs_map.end() && iter->second != "") ? iter->second : "0";
  iter = attrs_map.find("append_op");
  binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  iter = attrs_map.find("reshape");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&reshape_, attrs_map["reshape"], ",");
  }
  iter = attrs_map.find("reshape_dims");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&reshape_dims_, attrs_map["reshape_dims"], ",");
  }
  iter = attrs_map.find("mul");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&mul_, attrs_map["mul"], ",");
  }
}

void GatherOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  dst_ = output[0];
  idx_ = input[0];
  src_ = input[1];
  if (binary_add_) append_ = input[2];
}

void GatherOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  rt_data_.resize(input.size() + 1);
}

void GatherOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // todo: add Reshape
#ifdef WITH_SPARSELIB
  vector<int64_t> pre_dst_shape;
  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) {
      ref_shape = input.back()->shape();
    }
    pre_dst_shape = GetDstShape(reshape_, input[0]->size(), ref_shape, reshape_dims_);
    vector<int64_t> dst_shape(pre_dst_shape);
    if (!mul_.empty()) {
      dst_shape.clear();
      int j = 0;
      int64_t mul_size = 1;
      for (int i = 0; i < mul_.size(); ++i) mul_size *= pre_dst_shape[mul_[i]];
      for (int i = 0; i < pre_dst_shape.size(); ++i) {
        if (j < mul_.size() && i == mul_[j]) {
          if (j == 0) dst_shape.push_back(mul_size);
          j++;
        } else {
          dst_shape.push_back(pre_dst_shape[i]);
        }
      }
    }
    output[0]->set_shape(dst_shape);
  } else {
    auto dst_shape = src_->shape();
    dst_shape[stoi(src_axis_)] = idx_->shape()[stoi(idx_axis_)];
    dst_->set_shape(dst_shape);
  }
  dst_->set_dtype(src_->dtype());

  std::unordered_map<std::string, std::string> attr_map;
  attr_map["idx_axis"] = idx_axis_;
  attr_map["src_axis"] = src_axis_;
  std::vector<jd::tensor_desc> ts_descs;
  jd::data_type dt = type2sparsemem[src_->dtype()];
  std::vector<jd::binaryop_attr> binaryops;
  vector<int64_t> gather_dst_shape(dst_->shape());
  ts_descs.emplace_back(src_->shape(), dt, jd::format_type::undef);
  ts_descs.emplace_back(idx_->shape(), jd::data_type::s32, jd::format_type::undef);
  ts_descs.emplace_back(gather_dst_shape, dt, jd::format_type::undef);
  if (binary_add_) {
    attr_map["binaryop_list"] = "binary_add";
    binaryops.push_back({append_->mutable_data(), jd::binaryop_alg::add, dt});
    ts_descs.emplace_back(append_->shape(), dt, jd::format_type::undef);
  }

  jd::operator_desc op_desc(jd::kernel_kind::gather, jd::kernel_prop::forward_inference, jd::engine_kind::cpu, ts_descs,
                            attr_map);
  op_desc.set_binaryop_list(binaryops);
  jd::gather_desc gather_d(op_desc);
  gather_ = jd::gather(gather_d);
  if (!reshape_.empty() && !mul_.empty()) {
    vector<int64_t> dst_shape;
    int j = 0;
    int64_t mul_size = 1;
    for (int i = 0; i < mul_.size(); ++i) mul_size *= pre_dst_shape[mul_[i] - 1];
    for (int i = 0; i < pre_dst_shape.size(); ++i) {
      if (j < mul_.size() && i == (mul_[j] - 1)) {
        if (j == 0) dst_shape.push_back(mul_size);
        j++;
      } else {
        dst_shape.push_back(pre_dst_shape[i]);
      }
    }
    output[0]->set_shape(dst_shape);
  }
#endif
}

void GatherOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  rt_data_[0] = src_->data();
  rt_data_[1] = idx_->data();
  rt_data_[2] = dst_->data();
  if (binary_add_) {
    rt_data_[3] = append_->data();
  }
#ifdef WITH_SPARSELIB
  gather_.execute(rt_data_);
#endif
  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Gather);
}  // namespace executor
