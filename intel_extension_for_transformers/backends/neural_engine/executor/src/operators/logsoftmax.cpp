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

#include "logsoftmax.hpp"

#include "common.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"s32", dnnl::memory::data_type::s32},
    {"fp16", dnnl::memory::data_type::f16}, {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},    {"bf16", dnnl::memory::data_type::bf16}};


LogSoftmaxOperator::LogSoftmaxOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : -1;

  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
}

void LogSoftmaxOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  output[0]->set_dtype("fp32");
}

void LogSoftmaxOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const memory::dims& src_shape_origin = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  vector<int64_t> dst_shape = src_shape_origin;

  // 1.3 Get tensor's adjusted strides
  vector<int64_t> src_stride = GetStrides(src_shape_origin);
  memory::dims dst_stride = src_stride;
  src_ = input[0];
  dst_ = output[0];

  // 1.4 Prepare memory descriptors
  memory::desc src_md(src_shape_origin, type2mem[src_->dtype()], src_stride);
  memory::desc dst_md(dst_shape, type2mem[dst_->dtype()], dst_stride);

  // 1.5 Set dst tensor shape
  output[0]->set_shape(dst_shape);

  //// Part2: Derive operator's format_any memory::desc and memory.
  // 2.1 Prepare format_any memory descriptors
  // 2.2 Prepare op descriptors
  if (axis_ == -1) {
    axis_ = src_shape_origin.size() - 1;
  }
  dnnl::logsoftmax_forward::desc logsoftmax_d(prop_kind::forward_inference, src_md, axis_);

  // 2.3 Prepare primitive descriptors (cached)
  dnnl::logsoftmax_forward::primitive_desc logsoftmax_pd(logsoftmax_d, eng_);

  // 2.4 Prepare primitive objects (cached)
  logsoftmax_p_ = dnnl::logsoftmax_forward(logsoftmax_pd);

  // 2.5 Prepare memory objects (cached)
  src_m_ = memory(src_md, eng_, DNNL_MEMORY_NONE);
  dst_m_ = memory(dst_md, eng_, DNNL_MEMORY_NONE);
}

void LogSoftmaxOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src_data = input[0]->data();
  void* dst_data = output[0]->mutable_data();

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_data), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  // 2. Reorder the data when the primitive memory and user memory are different
  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC] = src_m_;
  memory_args_[DNNL_ARG_DST] = dst_m_;

  // 4. Execute the primitive
  logsoftmax_p_.execute(s, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  // 6. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(LogSoftmax);
}  // namespace executor
