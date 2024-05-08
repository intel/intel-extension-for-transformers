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

#include "slice.hpp"

#include "common.hpp"
#include "kernels/include/kernels/exposed_enum.hpp"

#ifdef _MSC_VER
#undef IN
#endif
namespace executor {
using io = jd::exposed_enum::slice::io;

static unordered_map<string, jd::data_type> type2sparsemem{
    {"fp32", jd::data_type::fp32}, {"s32", jd::data_type::s32}, {"fp16", jd::data_type::fp16},
    {"u8", jd::data_type::u8},     {"s8", jd::data_type::s8},   {"bf16", jd::data_type::bf16}};

template <typename T>
static void SliceData(const T* src_data, T* dst_data, const vector<int64_t>& src_shape,
                      const vector<int64_t>& dst_shape, const vector<int64_t>& starts, const vector<int64_t>& ends,
                      const vector<int64_t>& axes, const vector<int64_t>& steps) {
  int64_t src_size = 1;
  for (auto shape : src_shape) {
    src_size *= shape;
  }
  T* src_data_tmp = static_cast<T*>(malloc(src_size * sizeof(T)));
  T* dst_data_tmp = static_cast<T*>(malloc(src_size * sizeof(T)));
  memcpy(src_data_tmp, src_data, src_size * sizeof(T));
  memcpy(dst_data_tmp, src_data, src_size * sizeof(T));
  vector<int64_t> src_shape_tmp = src_shape;
  vector<int64_t> dst_shape_tmp = src_shape;
  for (int64_t i = 0; i < axes.size(); ++i) {
    dst_shape_tmp[axes[i]] = static_cast<int64_t>((ends[i] - starts[i] - 1) / steps[i]) + 1;
    int64_t IN = 1;
    int64_t IC = 1;
    int64_t IH = 1;
    int64_t ON = 1;
    int64_t OC = 1;
    int64_t OH = 1;
    int64_t step = steps[i];
    for (int64_t j = 0; j < axes[i]; ++j) {
      IN *= src_shape_tmp[j];
      ON *= dst_shape_tmp[j];
    }
    IC = src_shape_tmp[axes[i]];
    OC = dst_shape_tmp[axes[i]];
    for (int64_t j = axes[i] + 1; j < src_shape_tmp.size(); ++j) {
      IH *= src_shape_tmp[j];
      OH *= dst_shape_tmp[j];
    }
    int64_t start = starts[i] * IH;
#pragma omp parallel for
    for (int64_t on = 0; on < ON; ++on) {
#pragma omp simd
      for (int64_t oc = 0; oc < OC; ++oc) {
        memcpy(dst_data_tmp + on * OC * OH + oc * OH, src_data_tmp + start + on * IC * IH + (oc * step) * IH,
               OH * sizeof(T));
      }
    }
    memcpy(src_data_tmp, dst_data_tmp, ON * OC * OH * sizeof(T));
    src_shape_tmp = dst_shape_tmp;
  }
  int64_t dst_size = 1;
  for (auto shape : dst_shape) {
    dst_size *= shape;
  }
  memcpy(dst_data, dst_data_tmp, dst_size * sizeof(T));
  free(src_data_tmp);
  free(dst_data_tmp);
}

SliceOperator::SliceOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("starts");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&starts_, attrs_map["starts"], ",");
  }
  iter = attrs_map.find("ends");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&ends_, attrs_map["ends"], ",");
  }
  iter = attrs_map.find("axes");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&axes_, attrs_map["axes"], ",");
  }
  iter = attrs_map.find("steps");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&steps_, attrs_map["steps"], ",");
  }
  iter = attrs_map.find("ends_with_tensor");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&ends_with_tensor_, attrs_map["ends_with_tensor"], ",");
  }

  iter = attrs_map.find("starts_with_tensor");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&starts_with_tensor_, attrs_map["starts_with_tensor"], ",");
  }
}

void SliceOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  rt_data_.resize(2);
  output[0]->set_dtype(input[0]->dtype());
}

std::vector<int64_t> SliceOperator::GetIndicesFromTensor(const vector<Tensor*>& input, const int64_t& tensor_idx) {
  vector<int64_t> ret_indices;
  for (int t = 0; t < input[tensor_idx]->size(); ++t) {
    // executor kernels have no int64_t dtype implementation, just int
    // convert it to int64_t for indices collection.
    ret_indices.push_back(static_cast<int64_t>(*(static_cast<int*>(input[tensor_idx]->mutable_data()) + t)));
  }
  return ret_indices;
}

void SliceOperator::ClampIndices(int64_t* v, const int64_t& min, const int64_t& max) {
  if (*v < min) {
    *v = min;
  } else if (*v > max) {
    *v = max;
  } else {
    return;
  }
}

void SliceOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (!ends_with_tensor_.empty()) {
    int tensor_offset = 1;
    if (!starts_with_tensor_.empty()) {
      // for llama mode
      tensor_offset += 1;
    }
    if (ends_.empty()) {
      ends_.push_back(input[tensor_offset]->shape()[ends_with_tensor_[0]]);
    } else {
      ends_[0] = input[tensor_offset]->shape()[ends_with_tensor_[0]];
    }
  }
  if (!starts_with_tensor_.empty()) {
    if (starts_.empty()) {
      starts_.push_back(input[1]->shape()[starts_with_tensor_[0]]);
    } else {
      starts_[0] = input[1]->shape()[starts_with_tensor_[0]];
    }
    int offset = 1;
    if (input[1]->shape()[starts_with_tensor_[0]] == 0) {
      if (input.size() > 2) {
        offset = input[2]->shape()[1];
      } else {
        offset = 32;
      }
    }
    if (ends_with_tensor_.empty()) {
      if (ends_.empty()) {
        ends_.push_back(starts_[0] + offset);
      } else {
        ends_[0] = starts_[0] + offset;
      }
    }
  }
  const vector<int64_t>& src_shape = input[0]->shape();
  vector<int64_t> dst_shape = src_shape;
  int64_t tensor_idx = 1;
  if (starts_.empty()) {
    starts_ = GetIndicesFromTensor(input, tensor_idx);
    tensor_idx++;
  }
  if (ends_.empty()) {
    ends_ = GetIndicesFromTensor(input, tensor_idx);
    tensor_idx++;
  }
  // axes_ and steps_ are optional input tensors
  if (axes_.empty()) {
    if (tensor_idx <= input.size() - 1) {
      axes_ = GetIndicesFromTensor(input, tensor_idx);
      tensor_idx++;
    } else {
      for (int i = 0; i < src_shape.size(); ++i) axes_.push_back(i);
    }
  }
  if (steps_.empty()) {
    if (tensor_idx <= input.size() - 1) {
      steps_ = GetIndicesFromTensor(input, tensor_idx);
      tensor_idx++;
    } else {
      // default step is 1
      steps_ = vector<int64_t>(src_shape.size(), 1);
    }
  }
  for (int64_t i = 0; i < axes_.size(); ++i) {
    axes_[i] = axes_[i] < 0 ? src_shape.size() + axes_[i] : axes_[i];
    starts_[i] = starts_[i] < 0 ? src_shape[axes_[i]] + starts_[i] : starts_[i];
    ends_[i] = ends_[i] < 0 ? src_shape[axes_[i]] + ends_[i] : ends_[i];
    ends_[i] = ends_[i] > 99999 ? src_shape[axes_[i]] : ends_[i];
    // convert invalid inputs to valid values

    ClampIndices(&starts_[i], 0, dst_shape[axes_[i]]);
    ClampIndices(&ends_[i], 0, dst_shape[axes_[i]]);
    dst_shape[axes_[i]] = static_cast<int64_t>((ends_[i] - starts_[i] - 1) / steps_[i]) + 1;
  }
  output[0]->set_shape(dst_shape);
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["axis"] = std::to_string(axes_[0]);
  attr_map["begin"] = std::to_string(starts_[0]);
  attr_map["step"] = std::to_string(steps_[0]);
  if (steps_.size() == 1 && steps_[0] < 3) {
    std::vector<jd::tensor_desc> ts_descs(io::SIZE);
    jd::data_type dt = type2sparsemem[input[0]->dtype()];
    const auto& src_shape = input[0]->shape();
    ts_descs[io::SRC] = {src_shape, dt, jd::plain_format(src_shape.size())};
    ts_descs[io::DST] = {dst_shape, dt, jd::plain_format(dst_shape.size())};
    jd::operator_desc op_desc(jd::kernel_kind::slice, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                              ts_descs, attr_map);
    jd::slice_desc slice_d(op_desc);
    slice_ = jd::slice(slice_d);
  }
}

void SliceOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  if (steps_.size() == 1 && steps_[0] <= 2) {
    rt_data_[io::SRC] = input[0]->data();
    rt_data_[io::DST] = output[0]->data();
    slice_.execute(rt_data_);
  } else {
    Tensor* src = input[0];
    Tensor* dst = output[0];
    const vector<int64_t>& src_shape = src->shape();
    const vector<int64_t>& dst_shape = dst->shape();
    if (src->dtype() == "fp32") {
      const float* src_data = static_cast<const float*>(src->data());
      float* dst_data = static_cast<float*>(dst->mutable_data());
      SliceData<float>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
    } else if (src->dtype() == "s32") {
      const int32_t* src_data = static_cast<const int32_t*>(src->data());
      int32_t* dst_data = static_cast<int32_t*>(dst->mutable_data());
      SliceData<int32_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
    } else if (src->dtype() == "bf16") {
      const uint16_t* src_data = static_cast<const uint16_t*>(src->data());
      uint16_t* dst_data = static_cast<uint16_t*>(dst->mutable_data());
      SliceData<uint16_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
    } else if (src->dtype() == "u8") {
      const uint8_t* src_data = static_cast<const uint8_t*>(src->data());
      uint8_t* dst_data = static_cast<uint8_t*>(dst->mutable_data());
      SliceData<uint8_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
    } else if (src->dtype() == "s8") {
      const int8_t* src_data = static_cast<const int8_t*>(src->data());
      int8_t* dst_data = static_cast<int8_t*>(dst->mutable_data());
      SliceData<int8_t>(src_data, dst_data, src_shape, dst_shape, starts_, ends_, axes_, steps_);
    } else {
      LOG(ERROR) << "Dtype " << src->dtype() << "is not supported in slice op!";
    }
  }
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Slice);
}  // namespace executor
