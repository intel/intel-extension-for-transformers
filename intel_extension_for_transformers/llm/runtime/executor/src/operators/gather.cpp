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
using io = jd::exposed_enum::gather::io;
using dt = jd::data_type;
using ft = jd::format_type;
static const unordered_map<string, jd::data_type> type2sparsemem{
    {"fp32", jd::data_type::fp32}, {"s32", jd::data_type::s32}, {"fp16", jd::data_type::fp16},
    {"u8", jd::data_type::u8},     {"s8", jd::data_type::s8},   {"bf16", jd::data_type::bf16}};
#endif

GatherOperator::GatherOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf),
#ifdef WITH_SPARSELIB
      rt_data_(io::SIZE),
#endif
      keep_dims_(true) {
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
  iter = attrs_map.find("keep_dims");
  keep_dims_ = (iter != attrs_map.end() && iter->second != "true") ? false : true;
}

void GatherOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  dst_ = output[0];
  idx_ = input[0];
  src_ = input[1];
  if (binary_add_) append_ = input[2];
}

void GatherOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(src_->dtype());
}

void GatherOperator::DstShapeInfer(const vector<Tensor*>& input, const vector<Tensor*>& output) {
#ifdef WITH_SPARSELIB
  int64_t src_axis = stoi(src_axis_);
  int64_t idx_axis = stoi(idx_axis_);
  int64_t dst_axis_size = idx_->shape()[idx_axis];
  vector<int64_t> idx_shape = idx_->shape();
  vector<int64_t> src_shape = src_->shape();
  std::vector<int64_t> dst_shape;
  if (src_axis != 0 && idx_axis != 0) {
    LOG_IF(FATAL, src_axis != idx_axis) << "src_axis should equal to idx_axis when both of them are not zero";
    for (size_t i = 0; i < src_axis; i++) {
      LOG_IF(FATAL, src_shape[i] < idx_shape[i]) << "src shape less than idx on dim:" << i;
      dst_shape.push_back(idx_shape[i]);
    }
  } else {
    if (src_axis != 0) {
      for (size_t i = 0; i < src_axis; i++) {
        dst_shape.push_back(src_shape[i]);
      }
    } else {
      if (idx_axis != 0) {
        for (size_t i = 0; i < idx_axis; i++) {
          dst_shape.push_back(idx_shape[i]);
        }
      }
    }
  }
  dst_shape.push_back(dst_axis_size);
  size_t inner_size = 1;
  LOG_IF(FATAL, idx_axis != idx_shape.size() - 1)
      << "Not support gather in multi-dims now, idx_axis should be the last dim of idx";
  for (size_t i = src_axis + 1; i < src_shape.size(); i++) {
    inner_size *= src_shape[i];
    dst_shape.push_back(src_shape[i]);
  }
  dst_->set_shape(dst_shape);
#endif
}

void GatherOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // TODO(Yucheng): add Reshape
#ifdef WITH_SPARSELIB
  vector<int64_t> pre_dst_shape;
  if (!reshape_.empty()) {
    vector<int64_t> ref_shape;
    if (!reshape_dims_.empty()) ref_shape = input.back()->shape();
    pre_dst_shape = GetDstShape(reshape_, input[0]->size(), ref_shape, reshape_dims_);
    if (!mul_.empty()) {
      vector<int64_t> dst_shape;
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
    output[0]->set_shape(pre_dst_shape);
  } else {
    DstShapeInfer(input, output);
  }

  std::unordered_map<std::string, std::string> attr_map;
  attr_map["idx_axis"] = idx_axis_;
  attr_map["src_axis"] = src_axis_;
  std::vector<jd::tensor_desc> ts_descs(io::SIZE);
  jd::data_type dt = type2sparsemem.at(src_->dtype());
  std::vector<jd::binaryop_attr> binaryops;
  vector<int64_t> gather_dst_shape(dst_->shape());
  ts_descs[io::SRC] = {src_->shape(), dt, jd::plain_format(src_->shape().size())};
  ts_descs[io::IDX] = {idx_->shape(), dt::s32, jd::plain_format(idx_->shape().size())};
  ts_descs[io::DST] = {gather_dst_shape, dt, jd::plain_format(gather_dst_shape.size())};
  if (binary_add_) {
    LOG_IF(FATAL, append_->dtype() != "fp32") << "Gather only supports fp32 binary_add operation";
    attr_map["binaryop_list"] = "binary_add";
    binaryops.push_back({jd::binaryop_alg::add, dt});
    ts_descs[io::BINARY0] = {append_->shape(), dt, jd::plain_format(append_->shape().size())};
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

  if (!keep_dims_ && input[0]->shape() == vector<int64_t>({1}) && input[1]->shape().size() > 1) {
    vector<int64_t> dst_shape = output[0]->shape();
    auto axis = stoi(src_axis_);
    auto dim_val = dst_shape[axis];
    if (dim_val == 1) {
      dst_shape.erase(dst_shape.begin() + axis);
      output[0]->set_shape(dst_shape);
    } else {
      LOG(WARNING) << "Cannot squeeze dims at axis " << src_axis_ << ", which dim val is " << dim_val
                   << ", rather than 1";
    }
  }
#endif
}

void GatherOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
#ifdef WITH_SPARSELIB
  // 0. Alias variables part
  rt_data_[io::SRC] = src_->data();
  rt_data_[io::IDX] = idx_->data();
  rt_data_[io::DST] = dst_->data();
  if (binary_add_) rt_data_[io::BINARY0] = append_->data();
  gather_.execute(rt_data_);
  // 2. unref tensors
  this->unref_tensors(input);
#else
  LOG(FATAL) << "Gather operator relies on SparseLib!";
#endif
}

REGISTER_OPERATOR_CLASS(Gather);
}  // namespace executor
