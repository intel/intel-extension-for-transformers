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

#include "inner_product_graph.hpp"

#include "common.hpp"

namespace executor {

InnerProductGraphOperator::InnerProductGraphOperator(const shared_ptr<OperatorConfig>& conf) :
                           Operator(conf), g_(dnnl::graph::engine::kind::cpu) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("src0_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src0_perm_, attrs_map["src0_perm"], ",");
    if (src0_perm_ == vector<int64_t>{1, 0})
      transpose_a_ = true;
  }
  iter = attrs_map.find("src1_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src1_perm_, attrs_map["src1_perm"], ",");
    if (src1_perm_ == vector<int64_t>{1, 0})
      transpose_b_ = false;
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("append_op");
  binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  gelu_erf_ = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  gelu_tanh_ = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;
  tanh_ = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  sigmoid_ = (iter != attrs_map.end() && iter->second == "sigmoid") ? true : false;
  relu_ = (iter != attrs_map.end() && iter->second == "relu") ? true : false;
  append_eltwise_ = (gelu_erf_ && !gelu_split_) || (gelu_tanh_ && !gelu_split_) || tanh_ || sigmoid_ || relu_;
  append_op_ = (iter != attrs_map.end()) ? iter->second : "";
  DLOG(INFO) << "append_op: " << append_op_;
}

void InnerProductGraphOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  if (output.size() > 1) {
    dst_min_ = output[1];
    dst_max_ = output[2];
  }
  switch (input_size) {
    case 2: {
      src0_ = input[0];
      src1_ = input[1];
      has_bias_ = false;
      break;
    }
    case 3: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = (append_sum_ || binary_add_) ? nullptr : input[2];
      has_bias_ = !(append_sum_ || binary_add_);
      break;
    }
    case 4: {
      src0_ = input[0];
      src1_ = input[1];
      bias_ = input[2];
      has_bias_ = true;
      break;
    }
    default: {
      LOG(ERROR) << "Input size in InnerProduct is: " << input_size << ", not supported!";
    }
  }
}

void InnerProductGraphOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(output_dtype_);
}

void InnerProductGraphOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_shape({input[0]->shape()[transpose_a_], input[1]->shape()[!transpose_b_]});

  logical_tensor ip_in_desc{0, data_type::f32, input[0]->shape(), layout_type::strided};
  logical_tensor ip_weight_desc{1, data_type::f32, input[1]->shape(), layout_type::strided, property_type::constant};
  logical_inputs_.push_back(ip_in_desc);
  logical_inputs_.push_back(ip_weight_desc);
  if (has_bias_) {
    logical_tensor ip_bias_desc{2, data_type::f32, input[2]->shape(), layout_type::strided, property_type::constant};
    logical_inputs_.push_back(ip_bias_desc);
  }
  logical_tensor ip_out_desc{3, data_type::f32, output[0]->shape(), layout_type::any};
  logical_outputs_.push_back(ip_out_desc);

  dnnl::graph::op ip_op(0, dnnl::graph::op::kind::MatMul, logical_inputs_, {ip_out_desc}, "matmul");
  ip_op.set_attr<bool>("transpose_a", transpose_a_);
  ip_op.set_attr<bool>("transpose_b", transpose_b_);
  g_.add_op(ip_op);

  // append
  if (append_sum_) {
    logical_tensor sum_in_desc{4, data_type::f32, input[has_bias_ ? 3 : 2]->shape(), layout_type::strided};
    logical_tensor sum_out_desc{5, data_type::f32, output[0]->shape(), layout_type::strided};
    dnnl::graph::op sum_op(1, dnnl::graph::op::kind::Add, {ip_out_desc, sum_in_desc}, {sum_out_desc}, "sum");
    logical_inputs_.push_back(sum_in_desc);
    logical_outputs_[0] = sum_out_desc;
    g_.add_op(sum_op);
  }

  if (tanh_) {
    logical_tensor elemente_out_desc{4, data_type::f32, output[0]->shape(), layout_type::strided};
    dnnl::graph::op tanh_op(1, dnnl::graph::op::kind::Tanh, {ip_out_desc}, {elemente_out_desc}, "tanh");
    logical_outputs_[0] = elemente_out_desc;
    g_.add_op(tanh_op);
  }

  if (gelu_erf_ && !gelu_split_) {
    logical_tensor elemente_out_desc{4, data_type::f32, output[0]->shape(), layout_type::strided};
    dnnl::graph::op gelu_op(1, dnnl::graph::op::kind::GELU, {ip_out_desc}, {elemente_out_desc}, "gelu");
    logical_outputs_[0] = elemente_out_desc;
    g_.add_op(gelu_op);
  }

  if (gelu_tanh_ && !gelu_split_) {
    LOG(WARNING) << "gelu_tanh_ is not supported by in onednn graph," <<
                 " and it is temporarily replaced by GELU.";
    logical_tensor elemente_out_desc{4, data_type::f32, output[0]->shape(), layout_type::strided};
    dnnl::graph::op gelu_op(1, dnnl::graph::op::kind::GELU, {ip_out_desc}, {elemente_out_desc}, "gelu");
    logical_outputs_[0] = elemente_out_desc;
    g_.add_op(gelu_op);
  }

  if (sigmoid_) {
    logical_tensor elemente_out_desc{4, data_type::f32, output[0]->shape(), layout_type::strided};
    dnnl::graph::op sigmoid_op(1, dnnl::graph::op::kind::Sigmoid, {ip_out_desc}, {elemente_out_desc}, "sigmoid");
    logical_outputs_[0] = elemente_out_desc;
    g_.add_op(sigmoid_op);
  }

  if (relu_) {
    logical_tensor elemente_out_desc{4, data_type::f32, output[0]->shape(), layout_type::strided};
    dnnl::graph::op relu_op(1, dnnl::graph::op::kind::ReLU, {ip_out_desc}, {elemente_out_desc}, "relu");
    logical_outputs_[0] = elemente_out_desc;
    g_.add_op(relu_op);
  }

  if (binary_add_) {
    LOG(ERROR) << "binary_add_ is not supported yet.";
  }

  auto partitions = g_.get_partitions();
  assert(partitions.size() == 1);
  partition_ = partitions[0];
  cp_ = partition_.compile(logical_inputs_, logical_outputs_, eng_);
}

void InnerProductGraphOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<dnnl::graph::tensor> inputs_ts, outputs_ts;
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


REGISTER_OPERATOR_CLASS(InnerProductGraph);
}  // namespace executor
