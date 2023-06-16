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

#include "binary_add.hpp"

#include "common.hpp"

namespace executor {

// binary op doesn't support int32
static unordered_map<string, dnnl::memory::data_type> type2mem{{"fp32", dnnl::memory::data_type::f32},
                                                               {"fp16", dnnl::memory::data_type::f16},
                                                               {"u8", dnnl::memory::data_type::u8},
                                                               {"s8", dnnl::memory::data_type::s8},
                                                               {"bf16", dnnl::memory::data_type::bf16}};

BinaryAddOperator::BinaryAddOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  static unordered_map<string, algorithm> str2algo{
      {"add", algorithm::binary_add}, {"sub", algorithm::binary_sub}, {"mul", algorithm::binary_mul},
      {"div", algorithm::binary_div}, {"gt", algorithm::binary_gt},   {"ge", algorithm::binary_ge},
      {"lt", algorithm::binary_lt},   {"le", algorithm::binary_le},   {"eq", algorithm::binary_eq},
      {"ne", algorithm::binary_ne},   {"min", algorithm::binary_min}, {"max", algorithm::binary_max}};
  auto iter = attrs_map.find("append_op");
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algo_ = str2algo[iter->second];
  } else {
    algo_ = algorithm::binary_add;
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
void BinaryAddOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (output_dtype_.empty()) {
    output_dtype_ = input[0]->dtype();
  }
  output[0]->set_dtype(output_dtype_);
}

void BinaryAddOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 1: Prepare op descriptor and set dst tensor shape.
  const memory::dims& src0_shape_origin = input[0]->shape();
  const memory::dims& src1_shape_origin = input[1]->shape();
  // check broadcast optimization
  if (src0_shape_origin.size() == src1_shape_origin.size() && src0_shape_origin.size() > 1) {
    bool tag = true;
    for (int i = 1; i < src0_shape_origin.size(); i++) {
      if (src0_shape_origin[i] != src1_shape_origin[i]) tag = false;
    }
    broadcast_ = tag;
  }

  dnnl::binary::desc binary_d;
  auto& dst_tensor_ptr = output[0];

  if (broadcast_) {
    binary_d = PrepareBroadcastBinaryDesc(src0_shape_origin, src1_shape_origin, input, output);
    dst_tensor_ptr->set_shape(GetBroadcastBinaryDstShape(src0_shape_origin, src1_shape_origin));
  } else {
    binary_d = PrepareStrideBinaryDesc(src0_shape_origin, src1_shape_origin, input, output);
    dst_tensor_ptr->set_shape(GetStrideBinaryDstShape(src0_shape_origin, src1_shape_origin));
  }

  // 2: Prepare primitive descriptors (cached)
  dnnl::primitive_attr attr;
  if (append_sum_) {
    dnnl::post_ops po;
    float beta = 1.0;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }
  binary_pd_ = dnnl::binary::primitive_desc(binary_d, attr, eng_);

  // 3: Prepare primitive objects (cached)
  binary_p_ = dnnl::binary(binary_pd_);
}

vector<vector<string>> BinaryAddOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // inplace input[0] -> output[0]
  if (!append_sum_ && input.size() == 2 && input[0] != nullptr && input[0]->left_life() == 1 &&
      input[0]->size() >= output[0]->size() && input[0]->dtype() == output[0]->dtype()) {
    inplace_pairs.emplace_back(vector<string>({input[0]->name(), output[0]->name()}));
  } else {
    // append_sum inplace input[2] -> output[0]
    if (append_sum_ && input.size() >= 3 && input[2] != nullptr && input[2]->size() >= output[0]->size() &&
        input[2]->dtype() == output[0]->dtype() && input[2]->left_life() == 1) {
      inplace_pairs.emplace_back(vector<string>({input[2]->name(), output[0]->name()}));
    }
  }
  return inplace_pairs;
}

void BinaryAddOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const auto& src0_data = input[0]->data();
  const auto& src1_data = input[1]->data();
  // when change data value please use mutable_data
  // Inplace Op: 1. append_sum. 2. non-append_sum
  Tensor* dst_ptr = output[0];
  vector<Tensor*> inputs(input);
  if (!append_sum_ && input.size() == 2 && input[0] != nullptr && input[0]->left_life() == 1 &&
      input[0]->size() >= dst_ptr->size() && input[0]->dtype() == output[0]->dtype() &&
      this->get_execution_mode() != ExecutionMode::DEBUG) {
    void* input_ptr = input[0]->mutable_data();
    input[0]->unref_data(true);
    dst_ptr->set_data(input_ptr);
    inputs = {input[1]};
  } else if (append_sum_ && input.size() >= 3 && input[2] != nullptr && input[2]->size() >= dst_ptr->size() &&
             input[2]->dtype() == output[0]->dtype()) {
    if (input[2]->left_life() == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
      void* input_ptr = input[2]->mutable_data();
      input[2]->unref_data(true);
      dst_ptr->set_data(input_ptr);
      inputs = {input[0], input[1]};
    } else {
      int data_size = input[2]->size();
      string data_type = input[2]->dtype();
      void* post_data_ptr = const_cast<void*>(input[2]->data());
      void* dst_data = dst_ptr->mutable_data();
      memcpy(dst_data, post_data_ptr, data_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  }
  auto dst_data = dst_ptr->mutable_data();

  void* inf_src0_data = const_cast<void*>(src0_data);
  void* inf_src1_data = const_cast<void*>(src1_data);

  if (broadcast_) {
    auto src0_shape = input[0]->shape();
    auto src1_shape = input[1]->shape();
    int src0_element_num = 1, src1_element_num = 1;
    for (int i = 0; i < src0_shape.size(); i++) {
      src0_element_num *= src0_shape[i];
      src1_element_num *= src1_shape[i];
    }
    if (src1_element_num > src0_element_num) {
      void* tmp_ptr = inf_src0_data;
      inf_src0_data = inf_src1_data;
      inf_src1_data = tmp_ptr;
    }
  }

  // 1. Prepare memory objects with data_ptr
  dnnl::stream s(eng_);
  user_src0_m_.set_data_handle(inf_src0_data, s);
  user_src1_m_.set_data_handle(inf_src1_data, s);
  user_dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

  memory any_src0_m = user_src0_m_;
  memory any_src1_m = user_src1_m_;
  memory any_dst_m = user_dst_m_;

  // 2. Reorder the data when the primitive memory and user memory are different
  if (binary_pd_.dst_desc() != user_dst_m_.get_desc()) {
    any_dst_m = memory(binary_pd_.dst_desc(), eng_);
  }

  // 3. Insert memory args
  memory_args_[DNNL_ARG_SRC_0] = any_src0_m;
  memory_args_[DNNL_ARG_SRC_1] = any_src1_m;
  memory_args_[DNNL_ARG_DST] = any_dst_m;

  // 4. Execute the primitive
  binary_p_.execute(s, memory_args_);

  // 5. Reorder the data of dst memory (When it is format_any)
  if (binary_pd_.dst_desc() != user_dst_m_.get_desc()) {
    dnnl::reorder(any_dst_m, user_dst_m_).execute(s, any_dst_m, user_dst_m_);
  }
  // 6. unref tensors
  this->unref_tensors(inputs);
}

dnnl::binary::desc BinaryAddOperator::PrepareBroadcastBinaryDesc(const memory::dims& src0_shape_origin,
                                                                 const memory::dims& src1_shape_origin,
                                                                 const vector<Tensor*>& input,
                                                                 const vector<Tensor*>& output) {
  memory::dims jit_pass_src0_shape = src0_shape_origin, jit_pass_src1_shape = src1_shape_origin;
  memory::format_tag dt_tag;
  memory::data_type src0_dt = type2mem[input[0]->dtype()];
  memory::data_type src1_dt = type2mem[input[1]->dtype()];
  if (src0_shape_origin[0] < src1_shape_origin[0]) {
    jit_pass_src0_shape = src1_shape_origin;
    jit_pass_src1_shape = src0_shape_origin;
    src0_dt = type2mem[input[1]->dtype()];
    src1_dt = type2mem[input[0]->dtype()];
  }

  // set dt_tag
  dt_tag = SetFormatTag(jit_pass_src0_shape.size());

  // dst_shape is same as jit_pass_src0_shape, all 3 tensor's dt_tag are same.
  memory::desc user_src0_md(jit_pass_src0_shape, src0_dt, dt_tag);
  memory::desc user_src1_md(jit_pass_src1_shape, src1_dt, dt_tag);
  memory::desc user_dst_md(jit_pass_src0_shape, type2mem[output[0]->dtype()], dt_tag);

  //  Prepare memory objects (cached)
  user_src0_m_ = memory(user_src0_md, eng_, DNNL_MEMORY_NONE);
  user_src1_m_ = memory(user_src1_md, eng_, DNNL_MEMORY_NONE);
  user_dst_m_ = memory(user_dst_md, eng_, DNNL_MEMORY_NONE);

  dnnl::binary::desc binary_d(algo_, user_src0_md, user_src1_md, user_dst_md);
  return binary_d;
}

// necessary assert checks have be done in PrepareBroadcastBinaryDesc
memory::dims BinaryAddOperator::GetBroadcastBinaryDstShape(const memory::dims& src0_shape_origin,
                                                           const memory::dims& src1_shape_origin) {
  for (int i = 0; i < src0_shape_origin.size(); i++)
    if (src0_shape_origin[i] < src1_shape_origin[i]) return src1_shape_origin;

  return src0_shape_origin;
}

dnnl::binary::desc BinaryAddOperator::PrepareStrideBinaryDesc(const memory::dims& src0_shape_origin,
                                                              const memory::dims& src1_shape_origin,
                                                              const vector<Tensor*>& input,
                                                              const vector<Tensor*>& output) {
  // 1 Get tensor's adjusted shapes
  auto dst_shape = GetStrideBinaryDstShape(src0_shape_origin, src1_shape_origin);

  // 2 Get tensor's adjusted strides
  memory::dims src0_stride = GetStrides(src0_shape_origin);
  memory::dims src1_stride = GetStrides(src1_shape_origin);
  memory::dims dst_stride = GetStrides(dst_shape);

  // 3 Prepare memory descriptors
  memory::desc user_src0_md(src0_shape_origin, type2mem[input[0]->dtype()], src0_stride);
  memory::desc user_src1_md(src1_shape_origin, type2mem[input[1]->dtype()], src1_stride);
  memory::desc user_dst_md(dst_shape, type2mem[output[0]->dtype()], dst_stride);

  // 4 Prepare format_any memory descriptors
  memory::desc any_dst_md(user_dst_md.dims(), user_dst_md.data_type(), memory::format_tag::any);

  // 5. Prepare memory objects (cached)
  user_src0_m_ = memory(user_src0_md, eng_, DNNL_MEMORY_NONE);
  user_src1_m_ = memory(user_src1_md, eng_, DNNL_MEMORY_NONE);
  user_dst_m_ = memory(user_dst_md, eng_, DNNL_MEMORY_NONE);

  // 6. Prepare op descriptors
  dnnl::binary::desc binary_d(algorithm::binary_add, user_src0_md, user_src1_md, any_dst_md);

  return binary_d;
}

memory::dims BinaryAddOperator::GetStrideBinaryDstShape(const memory::dims& src0_shape_origin,
                                                        const memory::dims& src1_shape_origin) {
  int64_t dsize = src0_shape_origin.size();
  memory::dims dst_shape(dsize, 0);
  for (int64_t i = 0; i < dsize; ++i) {
    dst_shape[i] = max(src0_shape_origin[i], src1_shape_origin[i]);
  }
  return dst_shape;
}

memory::format_tag BinaryAddOperator::SetFormatTag(int tensor_dim) {
  memory::format_tag tag = memory::format_tag::undef;
  switch (tensor_dim) {
    case 2:
      tag = memory::format_tag::ab;
      return tag;
    case 3:
      tag = memory::format_tag::abc;
      return tag;
    case 4:
      tag = memory::format_tag::abcd;
      return tag;
    default:
      assert(tensor_dim < 5 &&
             tensor_dim > 1);  // we only support up to 4D now, but we can support up to 10D tensor in future.
      return tag;
  }
}

REGISTER_OPERATOR_CLASS(BinaryAdd);
}  // namespace executor
