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

#include "softmax_graph.hpp"

#include "common.hpp"

namespace executor {

SoftmaxGraphOperator::SoftmaxGraphOperator(const shared_ptr<OperatorConfig>& conf) :
                      Operator(conf), g_(dnnl::graph::engine::kind::cpu) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : -1;

  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
}

void SoftmaxGraphOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  if (output.size() > 1) {
    dst_min_ = output[1];
    dst_max_ = output[2];
  }
  switch (input_size) {
    case 1: {
      src_ = input[0];
      break;
    }
    case 3: {
      src_ = input[0];
      dst_min_ = input[1];
      dst_max_ = input[2];
      break;
    }
    default: {
      LOG(ERROR) << "Input size in Softmax is: " << input_size << ", not supported!";
    }
  }
}

void SoftmaxGraphOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(output_dtype_);
}

void SoftmaxGraphOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<int64_t> dst_shape = input[0]->shape();
  output[0]->set_shape(dst_shape);

  logical_tensor src_desc {0, data_type::f32, input[0]->shape(), layout_type::strided};
  logical_tensor dst_desc {1, data_type::f32, input[0]->shape(), layout_type::any};
  logical_inputs_.push_back(src_desc);
  logical_outputs_.push_back(dst_desc);

  dnnl::graph::op softmax_op(0, dnnl::graph::op::kind::SoftMax, {src_desc}, {dst_desc}, "softmax");
  softmax_op.set_attr("axis", static_cast<int64_t>(axis_));
  g_.add_op(softmax_op);

  auto partitions = g_.get_partitions();
  assert(partitions.size() == 1);
  partition_ = partitions[0];
  cp_ = partition_.compile(logical_inputs_, logical_outputs_, eng_);
}

void SoftmaxGraphOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  std::vector<dnnl::graph::tensor> inputs_ts, outputs_ts;
  inputs_ts.reserve(logical_inputs_.size());
  outputs_ts.reserve(logical_outputs_.size());
  for (int idx = 0; idx < logical_inputs_.size(); idx++) {
    inputs_ts.push_back(dnnl::graph::tensor {logical_inputs_[idx], eng_, input[idx]->mutable_data()});
  }
  for (int idx = 0; idx < logical_outputs_.size(); idx++) {
    outputs_ts.push_back(dnnl::graph::tensor {logical_outputs_[idx], eng_, output[idx]->mutable_data()});
  }
  cp_.execute(strm_, inputs_ts, outputs_ts);
  strm_.wait();
  this->unref_tensors(input);
}


REGISTER_OPERATOR_CLASS(SoftmaxGraph);
}  // namespace executor

