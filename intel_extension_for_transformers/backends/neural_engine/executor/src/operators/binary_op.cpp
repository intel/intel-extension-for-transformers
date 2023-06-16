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

#include "binary_op.hpp"

#include "common.hpp"

namespace executor {

// binary op doesn't support int32
static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32}, {"s32", dnnl::memory::data_type::f32},
    {"fp16", dnnl::memory::data_type::f16}, {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},    {"bf16", dnnl::memory::data_type::bf16}};

BinaryOpOperator::BinaryOpOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf), stream_(eng_), eng_(engine::kind::cpu, 0) {
  static unordered_map<string, algorithm> str2algo{
      {"add", algorithm::binary_add}, {"sub", algorithm::binary_sub}, {"mul", algorithm::binary_mul},
      {"div", algorithm::binary_div}, {"gt", algorithm::binary_gt},   {"ge", algorithm::binary_ge},
      {"lt", algorithm::binary_lt},   {"le", algorithm::binary_le},   {"eq", algorithm::binary_eq},
      {"ne", algorithm::binary_ne},   {"min", algorithm::binary_min}, {"max", algorithm::binary_max}};
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algo_ = str2algo[iter->second];
  } else {
    LOG(ERROR) << "Please provide algorithm for binary operator.";
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
}

// The ONEDNN binary primitive supports the following combinations of data types:
// src0/1 f32, bf16, f16, u8, s8, dst: f32, bf16, f16, u8, s8
// In-place mode requires the dst and src data types to be the same.
// Different data types will unavoidably lead to correctness issues.
void BinaryOpOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  LOG_IF(FATAL, output_dtype_ == "s32") << "Unsupported dst dtype s32...";
  if (input[0]->dtype() == "s32" || input[1]->dtype() == "s32") {
    LOG(WARNING) << "int32 isn't supported by dnnl, which will be cast to float32.";
  }
  if (output_dtype_.empty()) {
    output_dtype_ = input[0]->dtype();
  }
  output[0]->set_dtype(output_dtype_);
}

void BinaryOpOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  auto src0_shape = input[0]->shape();
  auto src1_shape = input[1]->shape();

  auto src0_shape_size = src0_shape.size();
  auto src1_shape_size = src1_shape.size();
  if (src0_shape_size < src1_shape_size) {
    int diff_len = src1_shape_size - src0_shape_size;
    vector<int64_t> padding(diff_len, 1);
    src0_shape.insert(src0_shape.begin(), padding.begin(), padding.end());
    input[0]->set_shape(src0_shape);
  } else if (src0_shape_size > src1_shape_size) {
    int diff_len = src0_shape_size - src1_shape_size;
    vector<int64_t> padding(diff_len, 1);
    src1_shape.insert(src1_shape.begin(), padding.begin(), padding.end());
    input[1]->set_shape(src1_shape);
  }
  vector<int64_t> out_shape;
  for (int i = 0; i < src1_shape.size(); i++) {
    if (src0_shape[i] != src1_shape[i] && src0_shape[i] != 1 && src1_shape[i] != 1) {
      LOG(ERROR) << "can not broadcast! " << i << "src 0 shape " << src0_shape[i] << "src1 shape" << src1_shape[i];
      return;
    }
    out_shape.push_back(max(src0_shape[i], src1_shape[i]));
  }
  output[0]->set_shape(out_shape);

  // Create src and dst memory descriptors.
  auto src_0_md = memory::desc(input[0]->shape(), type2mem[input[0]->dtype()], GetStrides(src0_shape));
  auto src_1_md = memory::desc(input[1]->shape(), type2mem[input[1]->dtype()], GetStrides(src1_shape));
  auto dst_md = memory::desc(output[0]->shape(), type2mem[output[0]->dtype()], GetStrides(out_shape));

  dnnl::primitive_attr binary_attr;
  dnnl::binary::desc binary_d(algo_, src_0_md, src_1_md, dst_md);
  auto binary_pd = dnnl::binary::primitive_desc(binary_d, binary_attr, eng_);

  // Create src memory objects.
  src_0_mem_ = memory(src_0_md, eng_, DNNL_MEMORY_NONE);
  src_1_mem_ = memory(src_1_md, eng_, DNNL_MEMORY_NONE);
  dst_mem_ = memory(dst_md, eng_, DNNL_MEMORY_NONE);

  // Create the primitive.
  binary_prim_ = dnnl::binary(binary_pd);
}

vector<vector<string>> BinaryOpOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // inplace input[0] -> output[0]
  if (input[0] != nullptr && input[0]->left_life() == 1 && input[0]->size() >= output[0]->size() &&
      input[0]->dtype() != "s32" && output[0]->dtype() == input[0]->dtype()) {
    inplace_pairs.emplace_back(vector<string>({input[0]->name(), output[0]->name()}));
  } else {
    // inplace input[1] -> output[0]
    if (input[1] != nullptr && input[1]->left_life() == 1 && input[1]->size() >= output[0]->size() &&
        input[1]->dtype() != "s32" && output[0]->dtype() == input[1]->dtype()) {
      inplace_pairs.emplace_back(vector<string>({input[1]->name(), output[0]->name()}));
    }
  }
  return inplace_pairs;
}

void BinaryOpOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<Tensor*> inputs;
  void* src0_fp32 = nullptr;
  void* src1_fp32 = nullptr;
  void* src0_data = const_cast<void*>(input[0]->data());
  void* src1_data = const_cast<void*>(input[1]->data());
  if (input[0]->dtype() == "s32") {
    size_t size = input[0]->size();
    src0_fp32 = new float[size];
    std::copy(static_cast<int32_t*>(src0_data), static_cast<int32_t*>(src0_data) + size,
              static_cast<float*>(src0_fp32));
    src0_data = src0_fp32;
  }
  if (input[1]->dtype() == "s32") {
    size_t size = input[1]->size();
    src1_fp32 = new float[size];
    std::copy(static_cast<int32_t*>(src1_data), static_cast<int32_t*>(src1_data) + size,
              static_cast<float*>(src1_fp32));
    src1_data = src1_fp32;
  }

  src_0_mem_.set_data_handle(src0_data, stream_);
  src_1_mem_.set_data_handle(src1_data, stream_);

  if (input[0] != nullptr && input[0]->left_life() == 1 && input[0]->size() >= output[0]->size() &&
      input[0]->dtype() != "s32" && output[0]->dtype() == input[0]->dtype() &&
      this->get_execution_mode() != ExecutionMode::DEBUG) {
    input[0]->unref_data(true);
    output[0]->set_data(src0_data);
    inputs.push_back(input[1]);
  } else if (input[1] != nullptr && input[1]->left_life() == 1 && input[1]->size() >= output[0]->size() &&
             input[1]->dtype() != "s32" && output[0]->dtype() == input[1]->dtype() &&
             this->get_execution_mode() != ExecutionMode::DEBUG) {
    input[1]->unref_data(true);
    output[0]->set_data(src1_data);
    inputs.push_back(input[0]);
  } else {
    inputs.push_back(input[0]);
    inputs.push_back(input[1]);
  }
  dst_mem_.set_data_handle(reinterpret_cast<void*>(output[0]->mutable_data()), stream_);

  std::unordered_map<int, memory> binary_args;
  binary_args.insert({DNNL_ARG_SRC_0, src_0_mem_});
  binary_args.insert({DNNL_ARG_SRC_1, src_1_mem_});
  binary_args.insert({DNNL_ARG_DST, dst_mem_});

  binary_prim_.execute(stream_, binary_args);
  stream_.wait();

  this->unref_tensors(inputs);
  if (src0_fp32) delete[] src0_fp32;
  if (src1_fp32) delete[] src1_fp32;
}

REGISTER_OPERATOR_CLASS(BinaryOp);
}  // namespace executor
