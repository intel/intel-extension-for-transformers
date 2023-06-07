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

#include "quantize.hpp"

namespace executor {

QuantizeOperator::QuantizeOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();

  auto iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  iter = attrs_map.find("per_token");
  if (iter != attrs_map.end()) {
    per_token_ = attrs_map["per_token"] == "true" || attrs_map["per_token"] == "True";
  }
}

QuantizeOperator::~QuantizeOperator() {}

void QuantizeOperator::MapTensors(const vector<Tensor*>& input, const vector<Tensor*>& output) {
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
      src_min_ = input[1];
      src_max_ = input[2];
      break;
    }
  }
}

void QuantizeOperator::Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  MapTensors(input, output);
  dst_->set_dtype(output_dtype_);
  if (output_dtype_ == "u8" || output_dtype_ == "s8") {
    if (src_min_ != nullptr && src_max_ != nullptr) {
      scales_ = GetScales(src_min_->data(), src_max_->data(), src_min_->size(), dst_->dtype());
    } else {
      dst_min_->set_dtype("fp32");
      dst_max_->set_dtype("fp32");
      is_dynamic_ = true;
      DLOG(INFO) << name_ << " is dynamic !!!";
    }
  }
}

void QuantizeOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const vector<int64_t>& src_shape = src_->shape();
  dst_->set_shape(src_shape);
  if (is_dynamic_) {
    if (per_token_) {
      DLOG_IF(ERROR, output_dtype_ != "s8") << "per_token dynamic only support s8";
      std::unordered_map<std::string, std::string> op_attrs;
      dst_min_->set_shape({src_shape[0]});
      dst_max_->set_shape({src_shape[0]});
      mat_desc_ = {src_->shape(), src_->dtype() == "fp32" ? jd::data_type::fp32 : jd::data_type::bf16,
                   jd::format_type::undef};                                            // fp32
      dst_mat_desc_ = {dst_->shape(), jd::data_type::s8, jd::format_type::undef};      // s8
      scale_desc_ = {dst_max_->shape(), jd::data_type::fp32, jd::format_type::undef};  // fp32

      vector<jd::tensor_desc> ts_descs = {mat_desc_, dst_mat_desc_, scale_desc_};
      op_attrs["input_dt"] = src_->dtype();
      jd::operator_desc op_desc(jd::kernel_kind::dynamic_quant, jd::kernel_prop::forward_inference,
                                jd::engine_kind::cpu, ts_descs, op_attrs);
      jd::dynamic_quant_desc dynamic_quant_desc(op_desc);
      dynamic_quant_ = jd::dynamic_quant(dynamic_quant_desc);
    } else {
      dst_min_->set_shape({1});
      dst_max_->set_shape({1});
    }
  }
}

void QuantizeOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const void* src_data = src_->data();
  void* dst_data = dst_->mutable_data();
  if (is_dynamic_ && per_token_) {
    std::vector<const void*> runtime_data(3);
    runtime_data[io::SRC] = src_data;
    runtime_data[io::MAT_DST] = dst_data;
    runtime_data[io::SCALE_DST] = dst_max_->mutable_data();
    dynamic_quant_.execute(runtime_data);
  } else {
    const float* min_data = src_min_ != nullptr ? static_cast<const float*>(src_min_->data()) : nullptr;
    if (is_dynamic_) {
      if (src_->dtype() == "fp32") {
        runtime_minmax(reinterpret_cast<float*>(src_->mutable_data()), src_->size(),
                       reinterpret_cast<float*>(dst_min_->mutable_data()),
                       reinterpret_cast<float*>(dst_max_->mutable_data()));
        scales_ = GetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), dst_->dtype());
      } else {
        std::cout << "!!! BF16 quantize!!!" << std::endl;
        scales_.push_back(10.0f);
      }
      min_data = static_cast<const float*>(dst_min_->data());
      float* scale = reinterpret_cast<float*>(dst_max_->mutable_data());
      *scale = 1.0f / scales_[0];
    }
    if (min_data == nullptr) {
      if (dst_->dtype() == "u8") {
        LOG(ERROR) << "Neither choose dynamic quantization or passed min/max tensor for static ";
        return;
      }
    }
    // quantize
    if (src_data != nullptr && dst_data != nullptr) {
#if __AVX512F__
      if (src_->dtype() == "bf16") {
        if (dst_->dtype() == "s8") {
          Quantize_bf16_s8(src_->size(), src_data, scales_, dst_data);
          this->unref_tensors(input);
        } else if (dst_->dtype() == "u8") {
          Quantize_bf16_u8(src_->size(), src_data, min_data, scales_, dst_data);
          this->unref_tensors(input);
        }
        return;
      }
      if (dst_->dtype() == "u8") {
        Quantize_fp32_u8(src_->size(), src_data, min_data, scales_, dst_data);
      } else if (dst_->dtype() == "s8") {
        Quantize_fp32_s8(src_->size(), src_data, scales_, dst_data);
      } else {
        Quantize_fp32_bf16(src_->size(), src_data, scales_, dst_data);
      }
#else
      if (dst_->dtype() == "u8") {
        Quantize_u8(src_->size(), src_data, min_data, scales_, dst_data);
      } else {
        Quantize_others(src_->size(), dst_->dtype(), src_data, scales_, dst_data);
      }
#endif
    }
  }
  this->unref_tensors(input);
  return;
}

void QuantizeOperator::RuntimeMinmax() {
  // use onednn reduction calculate min/max
  memory::desc src_md(src_->shape(), memory::data_type::f32, GetStrides(src_->shape()));
  memory src_m(src_md, eng_);
  src_m.set_data_handle(src_->mutable_data());
  vector<int64_t> reduce_shape(dst_->shape().size(), 1);
  vector<int64_t> reduce_stride = GetStrides(reduce_shape);
  memory::desc dst_md(reduce_shape, memory::data_type::f32, reduce_stride);
  memory reduce_min(dst_md, eng_);
  memory reduce_max(dst_md, eng_);
  reduce_min.set_data_handle(dst_min_->mutable_data());
  reduce_max.set_data_handle(dst_max_->mutable_data());
  dnnl::reduction::desc reduce_min_d(algorithm::reduction_min, src_md, dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_min_pd(reduce_min_d, eng_);
  dnnl::reduction(reduce_min_pd).execute(eng_stream_, {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, reduce_min}});
  dnnl::reduction::desc reduce_max_d(algorithm::reduction_max, src_md, dst_md, 0.f, 0.f);
  dnnl::reduction::primitive_desc reduce_max_pd(reduce_max_d, eng_);
  dnnl::reduction(reduce_max_pd).execute(eng_stream_, {{DNNL_ARG_SRC, src_m}, {DNNL_ARG_DST, reduce_max}});
}
REGISTER_OPERATOR_CLASS(Quantize);
}  // namespace executor
