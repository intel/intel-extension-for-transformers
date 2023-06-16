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

#include "reorder.hpp"

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{{"fp32", dnnl::memory::data_type::f32},
                                                               {"s32", dnnl::memory::data_type::s32},
                                                               {"fp16", dnnl::memory::data_type::f16},
                                                               {"bf16", dnnl::memory::data_type::bf16},
                                                               {"u8", dnnl::memory::data_type::u8},
                                                               {"s8", dnnl::memory::data_type::s8},
                                                               {"bf16", dnnl::memory::data_type::bf16}};

ReorderOperator::ReorderOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();

  auto iter = attrs_map.find("src_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src_perm_, attrs_map["src_perm"], ",");
  }
  iter = attrs_map.find("dst_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_perm_, attrs_map["dst_perm"], ",");
  }
  iter = attrs_map.find("transpose_dims");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&transpose_dims_, attrs_map["transpose_dims"], ",");
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("append_op");
  append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
}

ReorderOperator::~ReorderOperator() {}

void ReorderOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  int input_size = input.size();
  dst_ = output[0];
  switch (input_size) {
    case 1: {
      src_ = input[0];
      break;
    }
    case 2: {
      src_ = input[0];
      post_ = input[1];
      break;
    }
    case 3: {
      src_ = input[0];
      src_min_ = input[1];
      src_max_ = input[2];
      break;
    }
    case 4: {
      src_ = input[0];
      src_min_ = input[1];
      src_max_ = input[2];
      post_ = input[3];
      break;
    }
  }
}

void ReorderOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  // reorder memory layout with same dtype by default
  if (output_dtype_.empty()) {
    output_dtype_ = src_->dtype();
  }
  dst_->set_dtype(output_dtype_);

  dnnl::primitive_attr attr;
  if (src_min_ != nullptr && src_max_ != nullptr) {
    const int ic_dim = src_min_->size() > 1 ? 0 | (1 << 1) : 0;
    vector<float> src_scales = GetScales(src_min_->data(), src_max_->data(), src_min_->size(), dst_->dtype());
    vector<int> zero_point = GetZeroPoints(src_min_->data(), src_scales, dst_->dtype());
    attr.set_output_scales(ic_dim, src_scales);
    attr.set_zero_points(DNNL_ARG_DST, ic_dim, zero_point);
  }

  if (append_sum_) {
    dnnl::post_ops po;
    float beta = 1.0;
    po.append_sum(beta);
    attr.set_post_ops(po);
  }
  attr_ = attr;
}

void ReorderOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const memory::dims& src_shape = src_->shape();
  memory::dims dst_shape_origin = src_shape;

  if (transpose_dims_.size()) {
    src_perm_.clear();
    dst_perm_.clear();
    auto dims = src_shape.size();
    for (int i = 0; i < dims; i++) {
      src_perm_.push_back(i);
      dst_perm_.push_back(i);
    }
    auto dim0 = transpose_dims_[0];
    auto dim1 = transpose_dims_[1];
    if (dim0 < 0) dim0 += dims;
    if (dim1 < 0) dim1 += dims;
    dst_perm_[dim0] = src_perm_[dim1];
    dst_perm_[dim1] = src_perm_[dim0];
  }

  vector<int64_t> dst_shape = GetShapes(dst_shape_origin, dst_perm_);

  memory::dims src_stride = GetStrides(src_shape, src_perm_);
  vector<int64_t> reverse_perm = ReversePerm(dst_perm_);
  vector<int64_t> dst_stride = GetStrides(dst_shape, reverse_perm);

  memory::desc src_md(src_shape, type2mem[src_->dtype()], src_stride);
  memory::desc dst_md(dst_shape_origin, type2mem[dst_->dtype()], dst_stride);

  dst_->set_shape(dst_shape);

  dnnl::reorder::primitive_desc reorder_pd(eng_, src_md, eng_, dst_md, attr_);
  reorder_prim_ = dnnl::reorder(reorder_pd);

  src_m_ = memory(src_md, eng_, DNNL_MEMORY_NONE);
  dst_m_ = memory(dst_md, eng_, DNNL_MEMORY_NONE);
}

vector<vector<string>> ReorderOperator::InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  vector<vector<string>> inplace_pairs;
  // skip inplace in debug mode
  if (this->get_execution_mode() == ExecutionMode::DEBUG) {
    return inplace_pairs;
  }
  // append_sum sum_tensor -> output[0]
  if (post_ != nullptr && post_->left_life() == 1) {
    inplace_pairs.emplace_back(vector<string>({post_->name(), output[0]->name()}));
  }
  return inplace_pairs;
}

// 2. inference kernel(for int8 and f32)
void ReorderOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (post_ != nullptr) {
    DLOG(INFO) << "reorder has post op " << post_->name();
    void* post_ptr = post_->mutable_data();
    if (post_->left_life() == 1 && this->get_execution_mode() != ExecutionMode::DEBUG) {
      post_->unref_data(true);
      dst_->set_data(post_ptr);
    } else {
      int data_size = post_->size();
      string data_type = post_->dtype();
      void* dst_data = dst_->mutable_data();
      memcpy(dst_data, post_ptr, data_size * type2bytes[data_type]);
      LOG(WARNING) << "post tensor will be used by multi node...";
    }
  }

  dnnl::stream s(eng_);
  src_m_.set_data_handle(const_cast<void*>(src_->data()), s);
  dst_m_.set_data_handle(reinterpret_cast<void*>(dst_->mutable_data()), s);
  reorder_prim_.execute(s, {{DNNL_ARG_SRC, src_m_}, {DNNL_ARG_DST, dst_m_}});
  this->unref_tensors(input);
}

void ReorderOperator::AdaptAttrs(const vector<Tensor*>& input, const vector<Tensor*>& output,
                                   const string& stage) {
  if (stage == "in") {
    // SparseLib 3D gemm - Reorder - Dense gemm
    if (input[0]->tensor_format() == TensorFormat::MmKMb &&
        src_perm_ == std::vector<int64_t>({0, 1}) && dst_perm_ == std::vector<int64_t>({1, 0})) {
      src_shape_origin_ = input[0]->shape();
      input[0]->set_shape({src_shape_origin_[1], src_shape_origin_[0] * src_shape_origin_[2]});
      adapt_attrs_ = true;
    }
  } else if (stage == "out") {
    if (adapt_attrs_) {
      input[0]->set_shape(src_shape_origin_);
      adapt_attrs_ = false;
    }
  } else {
    LOG(WARNING) << "Wrong stage parameter, should be in or out...";
  }
}

void ReorderOperator::AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output,
                                   const string& stage) {
  if (stage == "in") {
    // SparseLib 3D gemm - Reorder - Dense gemm
    if (input[0]->tensor_format() == TensorFormat::MmKMb &&
        src_perm_ == std::vector<int64_t>({0, 1}) && dst_perm_ == std::vector<int64_t>({1, 0})) {
      input[0]->reorder(input[0]->shape());
      input[0]->set_tensor_format(TensorFormat::KM);
      input[0]->set_shape({input[0]->shape()[0], input[0]->shape()[1] * input[0]->shape()[2]});
      output[0]->set_tensor_format(TensorFormat::MK);
      DLOG(INFO) << "Reorder src tensor from MmKMb to KM of operator " << name_;
    }
  } else if (stage == "out") {
    return;
  } else {
    LOG(WARNING) << "Wrong stage parameter, should be in or out...";
  }
}

REGISTER_OPERATOR_CLASS(Reorder);
}  // namespace executor
