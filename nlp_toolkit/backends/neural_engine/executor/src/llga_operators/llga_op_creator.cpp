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

#include "llga_op_creator.hpp"

#include "common.hpp"

namespace executor {

vector<float> LLGAGetScales(const void* mins, const void* maxs, const int64_t size, const string& dtype) {
  const float* mins_p = static_cast<const float*>(mins);
  const float* maxs_p = static_cast<const float*>(maxs);

  vector<float> scales;
  if (dtype == "u8") {
    for (int i = 0; i < size; i++) {
      float max_sub_min = maxs_p[i] - mins_p[i];
      max_sub_min = max_sub_min < 1e-10 ? 1e-10 : max_sub_min;
      scales.emplace_back(255.f / max_sub_min);
    }
  } else if (dtype == "s8") {
    for (int i = 0; i < size; i++) {
      float abs_max = max(abs(maxs_p[i]), abs(mins_p[i]));
      abs_max = abs_max < 1e-10 ? 1e-10 : abs_max;
      scales.emplace_back(127.f / abs_max);
    }
  } else if (dtype == "fp32") {
    for (int i = 0; i < size; i++) {
      scales.emplace_back(1.f);
    }
  } else {
    LOG(ERROR) << "Can't suppport dst_dtype: " << dtype << " now!";
  }
  return scales;
}

void LLGAOPCreator::CreateWildcardOP(Model *model, OperatorConfig* op_conf, int index) {
  vector<logical_tensor> inputs, outputs;
  InitLogicalTensors(model, op_conf, &inputs, &outputs);

  llga_op wildcard_op(model->op_idx, llga_op::kind::Wildcard, inputs, outputs, "wildcard" + to_string(model->op_idx));
  model->AddLLGAOP(wildcard_op, index);
}

void LLGAOPCreator::InitLogicalTensors(Model *model, OperatorConfig* op_conf,
                                       vector<logical_tensor>* inputs, vector<logical_tensor>* outputs) {
  int output_size = op_conf->output_tensor_size();
  auto attrs_map = op_conf->attributes();

  int input_size = op_conf->input_tensor_size();
  for (int input_id = 0; input_id < input_size; ++input_id) {
    const string& tensor_name = op_conf->input_tensors(input_id)->name();
    inputs->push_back(model->GetLogicalTensor(tensor_name));
  }

  // TODO(lzw): delivery dtype if output dtype not specifced.
  string output_dtype_ = "";
  auto iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
  for (int output_id = 0; output_id < output_size; ++output_id) {
    // create logical tensor into name2lts
    data_type dtype = data_type::f32;
    if (output_dtype_ == "u8") {
      dtype = data_type::u8;
    } else if (output_dtype_ == "s8") {
      dtype = data_type::s8;
    } else if (output_dtype_ == "fp32") {
      dtype = data_type::f32;
    } else if (output_dtype_ == "") {
      dtype = inputs->at(0).get_data_type();
      if (dtype == data_type::u8 && op_conf->type() == "InnerProduct") {
        // TODO(lzw): remove this, append sum
        dtype = data_type::f32;
      }
    }
    logical_tensor dst_desc {model->desc_idx, dtype, layout_type::any};
    const string& tensor_name = op_conf->output_tensors(output_id)->name();
    model->AddLogicalTensor(tensor_name, dst_desc);
    outputs->push_back(dst_desc);
  }
}

bool LLGAOPCreator::CreateSoftmaxOp(Model *model, OperatorConfig* op_conf, int index) {
  auto attrs_map = op_conf->attributes();
  auto iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    string output_dtype_ = attrs_map["output_dtype"];
    if (output_dtype_ != "fp32") {
      return false;
    }
  }
  iter = attrs_map.find("axis");
  vector<logical_tensor> inputs, outputs;
  InitLogicalTensors(model, op_conf, &inputs, &outputs);
  llga_op softmax_op(model->op_idx, llga_op::kind::SoftMax, inputs, outputs, "softmax" + to_string(model->op_idx));
  iter = attrs_map.find("axis");
  int64_t axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : -1;
  softmax_op.set_attr("axis", static_cast<int64_t>(axis_));
  model->AddLLGAOP(softmax_op, index);
  return true;
}

int LLGAOPCreator::CreateInnerProductOpFp32(Model *model, const vector<logical_tensor> &inputs, int index,
                                            bool has_bias, bool transpose_a_, bool transpose_b_) {
  logical_tensor dst_desc {model->desc_idx, data_type::f32, layout_type::any};
  model->AddLogicalTensor(dst_desc);
  vector<logical_tensor> ip_inputs;
  ip_inputs.push_back(inputs[0]);
  ip_inputs.push_back(inputs[1]);
  if (has_bias)
    ip_inputs.push_back(inputs[2]);
  llga_op ip_op(model->op_idx, llga_op::kind::MatMul, ip_inputs, {dst_desc}, "matmul" + to_string(model->op_idx));
  ip_op.set_attr<bool>("transpose_a", transpose_a_);
  ip_op.set_attr<bool>("transpose_b", transpose_b_);
  model->AddLLGAOP(ip_op, index);
  return dst_desc.get_id();
}

int LLGAOPCreator::CreateInnerProductOpInt8(Model *model, const vector<logical_tensor> &inputs, int index,
                                            bool has_bias, bool transpose_a_, bool transpose_b_, bool append_sum) {
  auto src0_min_name = model->id2names_[inputs[has_bias+append_sum+2].get_id()];
  auto src0_min_ = model->tensors_[model->tensor_name_index_[src0_min_name]];
  auto src0_max_name = model->id2names_[inputs[has_bias+append_sum+3].get_id()];
  auto src0_max_ = model->tensors_[model->tensor_name_index_[src0_max_name]];
  auto src1_min_name = model->id2names_[inputs[has_bias+append_sum+4].get_id()];
  auto src1_min_ = model->tensors_[model->tensor_name_index_[src1_min_name]];
  auto src1_max_name = model->id2names_[inputs[has_bias+append_sum+5].get_id()];
  auto src1_max_ = model->tensors_[model->tensor_name_index_[src1_max_name]];
  vector<float> src0_scales = LLGAGetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), "u8");
  vector<float> src1_scales = LLGAGetScales(src1_min_->data(), src1_max_->data(), src1_min_->size(), "s8");

  auto weight_name = model->id2names_[inputs[1].get_id()];
  auto weight = model->tensors_[model->tensor_name_index_[weight_name]];
  int8_t* weight_data = reinterpret_cast<int8_t*>(weight->mutable_data());
  vector<float> sum_weight;
  if (transpose_b_) {
    for (int i = 0; i < weight->shape()[0]; i++) {
      float sum = 0;
      for (int j = 0; j < weight->shape()[1]; j++) {
        sum += static_cast<float>(weight_data[i * weight->shape()[1] + j]);
      }
      sum_weight.push_back(sum);
    }
  } else {
    for (int j = 0; j < weight->shape()[1]; j++) {
      float sum = 0;
      for (int i = 0; i < weight->shape()[0]; i++) {
        sum += static_cast<float>(weight_data[i * weight->shape()[1] + j]);
      }
      sum_weight.push_back(sum);
    }
  }

  float* min_data = reinterpret_cast<float*>(src0_min_->mutable_data());
  if (has_bias) {
    // convert int32 bias to float32 bias.
    auto bias_name = model->id2names_[inputs[2].get_id()];
    auto old_tensor = model->tensors_[model->tensor_name_index_[bias_name]];
    Tensor* new_tensor = new Tensor();  // TODO(lzw): release it.
    new_tensor->set_dtype("fp32");
    new_tensor->set_shape(old_tensor->shape());
    new_tensor->add_tensor_life(1);  // TODO(lzw): life?
    float* fp32_bias = reinterpret_cast<float*>(new_tensor->mutable_data());
    int32_t* s32_bias = reinterpret_cast<int32_t*>(old_tensor->mutable_data());
    for (int i = 0; i < new_tensor->size(); i++) {
      fp32_bias[i] = (s32_bias[i] - src0_scales[0] * (*min_data) * sum_weight[i]) / src1_scales[i] / src0_scales[0];
    }
    model->tensors_[model->tensor_name_index_[bias_name]] = new_tensor;
  }
  // dequantize
  logical_tensor in_desc {model->desc_idx, data_type::f32, layout_type::any};
  model->AddLogicalTensor(in_desc);
  logical_tensor w_desc {model->desc_idx, data_type::f32, layout_type::any};
  // logical_tensor w_desc {model->desc_idx, data_type::f32, inputs[1].get_dims(), layout_type::any,
                            // property_type::constant};
  model->AddLogicalTensor(w_desc);
  logical_tensor ip_out_desc {model->desc_idx, data_type::f32, layout_type::any};
  model->AddLogicalTensor(ip_out_desc);

  llga_op dequant_in{model->op_idx, llga_op::kind::Dequantize, {inputs[0]}, {in_desc},
                     "dequant_in" + to_string(model->op_idx)};
  dequant_in.set_attr<vector<float>>("scales", {1/src0_scales[0]});
  // float* min_data = (float*)src0_min_->mutable_data();
  dequant_in.set_attr<vector<int64_t>>("zps", {nearbyint(-(*min_data) * src0_scales[0])});
  // dequant_in.set_attr<vector<int64_t>>("zps", {0});
  dequant_in.set_attr<string>("qtype", "per_tensor");
  model->AddLLGAOP(dequant_in, index);

  // This is an alternative solution for zero points (float type), which is same to our calculation method.
  /*
  logical_tensor in_sub_desc {model->desc_idx, data_type::f32, layout_type::any};
  model->name2lts_[to_string(in_sub_desc.get_id())] = in_sub_desc;
  model->id2names_[in_sub_desc.get_id()] = to_string(in_sub_desc.get_id());
  model->opid2index_[model->op_idx] = index;
  llga_op add_op(model->op_idx, llga_op::kind::Add, {in_desc, inputs[has_bias+append_sum+2]}, {in_sub_desc},
                   "add" + to_string(model->op_idx));
  model->op_idx++;
  model->g_.add_op(add_op);
  */

  vector<int64_t> src1_zps;
  for (int i = 0; i < src1_scales.size(); i++) {
    src1_scales[i] = 1 / src1_scales[i];
    src1_zps.push_back(0);
  }
  llga_op dequant_w{model->op_idx, llga_op::kind::Dequantize, {inputs[1]}, {w_desc},
                    "dequant_w" + to_string(model->op_idx)};
  dequant_w.set_attr<vector<float>>("scales", src1_scales);
  dequant_w.set_attr<vector<int64_t>>("zps", src1_zps);
  if (src1_scales.size() == 1) {
    dequant_w.set_attr<string>("qtype", "per_tensor");
  } else {
    dequant_w.set_attr<string>("qtype", "per_channel");
    // dequant_w.set_attr<int64_t>("axis", 1);  // TODO(lzw): check this
  }
  model->AddLLGAOP(dequant_w, index);

  vector<logical_tensor> ip_inputs;
  ip_inputs.push_back(in_desc);
  ip_inputs.push_back(w_desc);
  if (has_bias)
    ip_inputs.push_back(inputs[2]);

  llga_op ip_op{model->op_idx, llga_op::kind::MatMul, ip_inputs, {ip_out_desc}, "matmul" + to_string(model->op_idx)};
  ip_op.set_attr<bool>("transpose_a", transpose_a_);
  ip_op.set_attr<bool>("transpose_b", transpose_b_);
  model->AddLLGAOP(ip_op, index);

  return ip_out_desc.get_id();
}

bool LLGAOPCreator::CreateInnerProductOp(Model *model, OperatorConfig* op_conf, int index) {
  // TODO(lzw): compile(append op) causes this part more complex. (remove compile part will clean here.)
  vector<logical_tensor> inputs, outputs;
  int input_size = op_conf->input_tensor_size();
  for (int input_id = 0; input_id < input_size; ++input_id) {
    const string& tensor_name = op_conf->input_tensors(input_id)->name();
    inputs.push_back(model->GetLogicalTensor(tensor_name));
  }

  bool transpose_a_ = false, transpose_b_ = true;
  vector<int64_t> src0_perm_;
  vector<int64_t> src1_perm_;
  string output_dtype_ = "fp32";
  auto attrs_map = op_conf->attributes();
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
  bool binary_add_ = (iter != attrs_map.end() && iter->second == "binary_add") ? true : false;
  bool append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;
  bool gelu_erf_ = (iter != attrs_map.end() && iter->second == "gelu_erf") ? true : false;
  bool gelu_tanh_ = (iter != attrs_map.end() && iter->second == "gelu_tanh") ? true : false;
  bool tanh_ = (iter != attrs_map.end() && iter->second == "tanh") ? true : false;
  bool sigmoid_ = (iter != attrs_map.end() && iter->second == "sigmoid") ? true : false;
  bool relu_ = (iter != attrs_map.end() && iter->second == "relu") ? true : false;
  bool gelu_split_ = false;
  bool append_eltwise_ = (gelu_erf_ && !gelu_split_) || (gelu_tanh_ && !gelu_split_) || tanh_ || sigmoid_ || relu_;
  string append_op_ = (iter != attrs_map.end()) ? iter->second : "";
  LOG(INFO) << "append_op: " << append_op_;

  bool has_bias = false;
  int last_lt_id = -1;
  if (input_size <= 4) {
    has_bias = input_size - 2 - (binary_add_ || append_sum_);  // TODO(lzw) check this
    last_lt_id = CreateInnerProductOpFp32(model, inputs, index, has_bias, transpose_a_, transpose_b_);
  } else {
    if (input_size <=7)
      has_bias = input_size - 6 - (binary_add_ || append_sum_);  // TODO(lzw) check this
    else
      has_bias = input_size - 8 - (binary_add_ || append_sum_);  // TODO(lzw) check this
    last_lt_id = CreateInnerProductOpInt8(model, inputs, index,
                                          has_bias, transpose_a_, transpose_b_, binary_add_ || append_sum_);
  }

  logical_tensor append_desc {model->desc_idx, data_type::f32, layout_type::any};
  model->AddLogicalTensor(append_desc);

  if (append_eltwise_) {
    if (tanh_) {
      llga_op tanh_op(model->op_idx, llga_op::kind::Tanh, {model->GetLogicalTensor(last_lt_id)}, {append_desc},
                      "tanh" + to_string(model->op_idx));
      model->AddLLGAOP(tanh_op, index);
    }

    if (gelu_erf_ && !gelu_split_) {
      llga_op gelu_op(model->op_idx, llga_op::kind::GELU, {model->GetLogicalTensor(last_lt_id)}, {append_desc},
                      "gelu" + to_string(model->op_idx));
      model->AddLLGAOP(gelu_op, index);
    }

    if (gelu_tanh_ && !gelu_split_) {
      LOG(WARNING) << "gelu_tanh_ is not supported by in onednn graph," <<
                  " and it is temporarily replaced by GELU.";
      llga_op gelu_op(model->op_idx, llga_op::kind::GELU, {model->GetLogicalTensor(last_lt_id)}, {append_desc},
                      "gelu" + to_string(model->op_idx));
      model->AddLLGAOP(gelu_op, index);
    }

    if (sigmoid_) {
      llga_op sigmoid_op(model->op_idx, llga_op::kind::Sigmoid, {model->GetLogicalTensor(last_lt_id)}, {append_desc},
                         "sigmoid" + to_string(model->op_idx));
      model->AddLLGAOP(sigmoid_op, index);
    }

    if (relu_) {
      llga_op relu_op(model->op_idx, llga_op::kind::ReLU, {model->GetLogicalTensor(last_lt_id)}, {append_desc},
                      "relu" + to_string(model->op_idx));
      model->AddLLGAOP(relu_op, index);
    }
    last_lt_id = append_desc.get_id();

  } else if (binary_add_ || append_sum_) {
    llga_op sum_op(model->op_idx, llga_op::kind::Add, {model->GetLogicalTensor(last_lt_id),
                   inputs[has_bias+2]}, {append_desc}, "sum" + to_string(model->op_idx));
    model->AddLLGAOP(sum_op, index);
    last_lt_id = append_desc.get_id();
  }
  if (output_dtype_ != "fp32") {
  // if (input_size > 4 && !append_sum_) {
    // add quantize op.
    auto dst_min_name = model->id2names_[inputs[has_bias+append_sum_+6].get_id()];
    auto dst_min_ = model->tensors_[model->tensor_name_index_[dst_min_name]];
    auto dst_max_name = model->id2names_[inputs[has_bias+append_sum_+7].get_id()];
    auto dst_max_ = model->tensors_[model->tensor_name_index_[dst_max_name]];
    vector<float> dst_scales;
    vector<int64_t> dst_zps;

    const string& tensor_name = op_conf->output_tensors(0)->name();
    data_type dtype = data_type::f32;
    if (output_dtype_ == "u8") {
      dtype = data_type::u8;
      dst_scales = LLGAGetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), "u8");
      float* min_data = reinterpret_cast<float*>(dst_min_->mutable_data());
      dst_zps.push_back(nearbyint(-(*min_data) * dst_scales[0]));
    } else if (output_dtype_ == "s8") {
      dst_scales = LLGAGetScales(dst_min_->data(), dst_max_->data(), dst_min_->size(), "s8");
      dtype = data_type::s8;
      dst_zps.push_back(0);
    }
    logical_tensor out_desc {model->desc_idx, dtype, layout_type::any};
    model->AddLogicalTensor(out_desc);
    llga_op quant_out{model->op_idx, llga_op::kind::Quantize, {model->GetLogicalTensor(last_lt_id)}, {out_desc},
                      "quant_out" + to_string(model->op_idx)};
    quant_out.set_attr<vector<float>>("scales", {1 / dst_scales[0]});
    quant_out.set_attr<vector<int64_t>>("zps", dst_zps);
    quant_out.set_attr<string>("qtype", "per_tensor");
    model->AddLLGAOP(quant_out, index);
    last_lt_id = out_desc.get_id();
  }

  const string& tensor_name = op_conf->output_tensors(0)->name();
  model->AddLogicalTensor(tensor_name, model->GetLogicalTensor(last_lt_id), last_lt_id);
  outputs.push_back(model->GetLogicalTensor(tensor_name));
  return true;
}

bool LLGAOPCreator::CreateQuantizeOp(Model *model, OperatorConfig* op_conf, int index) {
  vector<logical_tensor> inputs, outputs;
  InitLogicalTensors(model, op_conf, &inputs, &outputs);

  auto src0_min_name = model->id2names_[inputs[1].get_id()];
  auto src0_min_ = model->tensors_[model->tensor_name_index_[src0_min_name]];
  auto src0_max_name = model->id2names_[inputs[2].get_id()];
  auto src0_max_ = model->tensors_[model->tensor_name_index_[src0_max_name]];
  vector<float> src0_scales = LLGAGetScales(src0_min_->data(), src0_max_->data(), src0_min_->size(), "u8");

  float* min_data = reinterpret_cast<float*>(src0_min_->mutable_data());
  llga_op quantize_op(model->op_idx, llga_op::kind::Quantize, {inputs[0]}, outputs,
                      "quantize" + to_string(model->op_idx));
  quantize_op.set_attr<vector<float>>("scales", {1/src0_scales[0]});
  quantize_op.set_attr<vector<int64_t>>("zps", {nearbyint(- (*min_data) * src0_scales[0])});
  quantize_op.set_attr<string>("qtype", "per_tensor");
  model->AddLLGAOP(quantize_op, index);

  return true;
}

bool LLGAOPCreator::CreateBinaryAddOp(Model *model, OperatorConfig* op_conf, int index) {
  // TODO(lzw) boardcast
  vector<logical_tensor> inputs, outputs;
  InitLogicalTensors(model, op_conf, &inputs, &outputs);

  auto attrs_map = op_conf->attributes();
  auto iter = attrs_map.find("append_op");
  bool append_sum_ = (iter != attrs_map.end() && iter->second == "sum") ? true : false;

  if (!append_sum_) {
    llga_op sum1_op(model->op_idx, llga_op::kind::Add, {inputs[0], inputs[1]}, outputs,
                    "sum" + to_string(model->op_idx));
    model->AddLLGAOP(sum1_op, index);
  } else {
    logical_tensor out_desc {model->desc_idx, data_type::f32, layout_type::any};
    model->AddLogicalTensor(out_desc);
    llga_op sum1_op(model->op_idx, llga_op::kind::Add, {inputs[0], inputs[1]}, {out_desc},
                    "sum" + to_string(model->op_idx));
    model->AddLLGAOP(sum1_op, index);
    llga_op sum2_op(model->op_idx, llga_op::kind::Add, {inputs[2], out_desc}, outputs,
                    "sum" + to_string(model->op_idx));
    model->AddLLGAOP(sum2_op, index);
  }
  return true;
}

bool LLGAOPCreator::CreateLayerNormOp(Model *model, OperatorConfig* op_conf, int index) {
  vector<logical_tensor> inputs, outputs;
  InitLogicalTensors(model, op_conf, &inputs, &outputs);

  auto attrs_map = op_conf->attributes();
  float epsilon_ = StringToNum<float>(attrs_map["epsilon"]);

  llga_op layernorm_op(model->op_idx, llga_op::kind::LayerNorm, inputs, outputs,
                       "layernorm" + to_string(model->op_idx));
  layernorm_op.set_attr<float>("epsilon", epsilon_);
  layernorm_op.set_attr<bool>("keep_stats", false);
  model->AddLLGAOP(layernorm_op, index);

  return true;
}

bool LLGAOPCreator::CreateReshapeOp(Model *model, OperatorConfig* op_conf, int index) {
  vector<logical_tensor> inputs, outputs;
  InitLogicalTensors(model, op_conf, &inputs, &outputs);

  vector<int64_t> shape_;
  vector<int64_t> dims_;
  vector<int64_t> mul_;
  auto attrs_map = op_conf->attributes();
    auto iter = attrs_map.find("dst_shape");
  if (iter != attrs_map.end()) StringSplit<int64_t>(&shape_, attrs_map["dst_shape"], ",");
  iter = attrs_map.find("dims");
  if (iter != attrs_map.end()) StringSplit<int64_t>(&dims_, attrs_map["dims"], ",");
  iter = attrs_map.find("mul");
  if (iter != attrs_map.end()) StringSplit<int64_t>(&mul_, attrs_map["mul"], ",");

  if (dims_.size() > 0) return false;  // not supported by llga
  if (mul_.size() > 0) return false;   // not supported by llga
  llga_op reshape_op(model->op_idx, llga_op::kind::StaticReshape, inputs, outputs,
                     "reshape" + to_string(model->op_idx));
  reshape_op.set_attr<vector<int64_t>>("shape", shape_);
  reshape_op.set_attr<bool>("special_zero", true);
  model->AddLLGAOP(reshape_op, index);

  return true;
}

bool LLGAOPCreator::CreateMatmulOp(Model *model, OperatorConfig* op_conf, int index) {
  vector<logical_tensor> inputs, outputs;
  InitLogicalTensors(model, op_conf, &inputs, &outputs);

  vector<int64_t> src0_perm_;
  vector<int64_t> src1_perm_;
  vector<int64_t> dst_perm_;
  auto attrs_map = op_conf->attributes();

  auto iter = attrs_map.find("src0_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src0_perm_, attrs_map["src0_perm"], ",");
  }
  iter = attrs_map.find("src1_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&src1_perm_, attrs_map["src1_perm"], ",");
  }
  iter = attrs_map.find("dst_perm");
  if (iter != attrs_map.end()) {
    StringSplit<int64_t>(&dst_perm_, attrs_map["dst_perm"], ",");
  }

  if (!src0_perm_.empty()) return false;
  std::cout << "CreateMatmulOp..\n";
  if (!src1_perm_.empty() && !dst_perm_.empty()) {
    logical_tensor out_desc {model->desc_idx, data_type::f32, layout_type::any};
    std::cout << "in id: " << out_desc.get_id() << std::endl;
    model->AddLogicalTensor(out_desc);
    // TODO(lzw) transpose op won't fuse with following matmul, and single transpose is not supported by llga.
    llga_op trans_op(model->op_idx, llga_op::kind::StaticTranspose, {inputs[1]}, {out_desc},
                     "transpose1" + to_string(model->op_idx));
    trans_op.set_attr<vector<int64_t>>("order", src1_perm_);
    model->AddLLGAOP(trans_op, index);

    logical_tensor out2_desc {model->desc_idx, data_type::f32, layout_type::any};
    model->AddLogicalTensor(out2_desc);
    llga_op matmul_op(model->op_idx, llga_op::kind::MatMul, {inputs[0], out_desc}, {out2_desc},
                      "matmul" + to_string(model->op_idx));
    model->AddLLGAOP(matmul_op, index);

    llga_op trans2_op(model->op_idx, llga_op::kind::StaticTranspose, {out2_desc}, outputs,
                      "transpose2" + to_string(model->op_idx));
    trans2_op.set_attr<vector<int64_t>>("order", dst_perm_);
    model->AddLLGAOP(trans2_op, index);
    std::cout << "out2_desc id: " << out2_desc.get_id() << std::endl;
  } else {
    llga_op matmul_op(model->op_idx, llga_op::kind::MatMul, inputs, outputs,
                      "matmul" + to_string(model->op_idx));
    model->AddLLGAOP(matmul_op, index);
  }

  std::cout << "CreateMatmulOp..done\n";
  return true;
}

}  // namespace executor
