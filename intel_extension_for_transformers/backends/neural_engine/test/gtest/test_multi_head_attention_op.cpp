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

#include <map>
#include <string>

#include "../../include/common.hpp"
#include "../../include/conf.hpp"
#include "../../include/operators/multi_head_attention.hpp"
#include "gtest/gtest.h"
using executor::AttrConfig;
using executor::MemoryAllocator;
using executor::OperatorConfig;
using executor::Tensor;
using executor::TensorConfig;

struct OpArgs {
  vector<Tensor*> input;
  vector<Tensor*> output;
  shared_ptr<OperatorConfig> conf;
  bool is_dynamic;
};

struct TestParams {
  std::pair<OpArgs, Tensor*> args;
  bool expect_to_fail;
};

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::MultiHeadAttentionOperator multiheadattention(p.conf);
    int scales_num = 6;
    vector<void*> tmp_data(scales_num);
    vector<Tensor> tmp_tensor(scales_num);
    if (p.is_dynamic) {
      for (int i = p.input.size() - scales_num, j = 0; i < p.input.size(); i++, j++) {
        tmp_data[j] = p.input[i]->mutable_data();
        p.input[i]->unref_data(true);
        tmp_tensor[j].add_tensor_life(1);
        tmp_tensor[j].set_data(tmp_data[j]);
      }
    }
    multiheadattention.Prepare(p.input, p.output);
    multiheadattention.Reshape(p.input, p.output);
    if (p.is_dynamic) {
      for (int i = p.input.size() - scales_num, j = 0; i < p.input.size(); i++, j++) {
        tmp_tensor[j].unref_data(true);
        p.input[i]->set_data(tmp_data[j]);
      }
    }
    multiheadattention.Forward(p.input, p.output);
  } catch (const dnnl::error& e) {
    if (e.status != dnnl_status_t::dnnl_success && t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  } catch (const std::string& e) {
    if (e == "Windows" && t.expect_to_fail)
      return true;
    else
      return false;
  }
  if (!t.expect_to_fail) {
    bool is_equal;
    if (q->dtype() == "fp32") {
      is_equal = executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q->data(), q->size(), 0.1);
    } else if (q->dtype() == "s8") {
      is_equal = executor::CompareData<int8_t>(p.output[0]->data(), p.output[0]->size(), q->data(), q->size(), 13);
    } else if (q->dtype() == "u8") {
      is_equal = executor::CompareData<uint8_t>(p.output[0]->data(), p.output[0]->size(), q->data(), q->size(), 13);
    }
    return is_equal;
  }
  return false;
}

class MultiheadAttentionInt8Test : public testing::TestWithParam<TestParams> {
 protected:
  MultiheadAttentionInt8Test() {}
  ~MultiheadAttentionInt8Test() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MultiheadAttentionInt8Test, ) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

Tensor* get_fp32_dst(const shared_ptr<TensorConfig>& dst_tensor_config, vector<Tensor*> inputs, float output_scale) {
  using dnnl::matmul;
  using dnnl::memory;
  Tensor* dst_tensor = new Tensor(*dst_tensor_config);
  dst_tensor->add_tensor_life(1);
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream = dnnl::stream(engine);
  Tensor* q = inputs[0];
  Tensor* k = inputs[1];
  Tensor* v = inputs[2];
  Tensor* post = inputs[3];
  // qk_matmul
  dnnl::primitive_attr attr;
  dnnl::post_ops po;
  memory::desc binary_md;
  memory binary_mem;
  auto q_shape = executor::GetShapes(q->shape(), {0, 2, 1, 3});
  auto q_stride = executor::GetStrides(q->shape(), {0, 2, 1, 3});
  auto q_md = memory::desc(q_shape, dnnl::memory::data_type::f32, q_stride);
  auto q_mem = memory(q_md, engine, q->mutable_data());
  auto k_shape = executor::GetShapes(k->shape(), {0, 2, 3, 1});
  auto k_stride = executor::GetStrides(k->shape(), {0, 2, 3, 1});
  auto k_md = memory::desc(k_shape, dnnl::memory::data_type::f32, k_stride);
  auto k_mem = memory(k_md, engine, k->mutable_data());
  int bs = q_shape[0], head_num = q_shape[1], seq_len = q_shape[2];
  std::unique_ptr<float, void (*)(float*)> padding_seq(nullptr, [](float* p) {
    if (p) aligned_free(p);
  });
  if (post != nullptr) {
    binary_md =
        memory::desc({bs, head_num, seq_len, seq_len}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abcd);
    po.append_binary(dnnl::algorithm::binary_add, binary_md);
    padding_seq.reset(reinterpret_cast<float*>(aligned_alloc(64, sizeof(float) * bs * head_num * seq_len * seq_len)));
    int32_t* mask = reinterpret_cast<int32_t*>(post->mutable_data());
    for (int ibs = 0; ibs < bs; ibs++) {
      const auto curr_padding = padding_seq.get() + ibs * head_num * seq_len * seq_len;
      memset(curr_padding, 0, sizeof(float) * seq_len);
      for (int j = mask[ibs]; j < seq_len; j++) curr_padding[j] = -10000.f;
      for (int i = 1; i < seq_len; i++) memcpy(curr_padding + i * seq_len, curr_padding, sizeof(float) * seq_len);
      for (int ihn = 1; ihn < head_num; ihn++)
        memcpy(curr_padding + ihn * seq_len * seq_len, curr_padding, sizeof(float) * seq_len * seq_len);
    }
    binary_mem = memory(binary_md, engine, padding_seq.get());
  }
  attr.set_post_ops(po);
  attr.set_output_scales(0, {output_scale});
  auto qk_shape = q_shape;
  qk_shape[3] = k_shape[3];
  Tensor qk(nullptr, qk_shape, "fp32");
  qk.add_tensor_life(1);
  auto qk_md = memory::desc(qk_shape, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abcd);
  auto qk_mem = memory(qk_md, engine, qk.mutable_data());
  auto qk_matmul_d = matmul::desc(q_md, k_md, qk_md);
  auto qk_matmul_pd = matmul::primitive_desc(qk_matmul_d, attr, engine);
  auto qk_matmul_prim = matmul(qk_matmul_pd);
  std::unordered_map<int, memory> qk_matmul_args;
  qk_matmul_args.insert({DNNL_ARG_SRC, q_mem});
  qk_matmul_args.insert({DNNL_ARG_WEIGHTS, k_mem});
  if (post != nullptr) qk_matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, binary_mem});
  qk_matmul_args.insert({DNNL_ARG_DST, qk_mem});
  qk_matmul_prim.execute(engine_stream, qk_matmul_args);
  qk.set_name("qk");
  // qk.to_file();
  // softmax
  dnnl::softmax_forward::desc softmax_d(dnnl::prop_kind::forward_inference, qk_md, 3);
  dnnl::softmax_forward::primitive_desc softmax_pd(softmax_d, engine);
  dnnl::softmax_forward softmax_p = dnnl::softmax_forward(softmax_pd);
  Tensor a(nullptr, qk_shape, "fp32");
  a.set_name("a");
  a.add_tensor_life(1);
  auto a_mem = memory(qk_md, engine, a.mutable_data());
  std::unordered_map<int, memory> softmax_args;
  softmax_args[DNNL_ARG_SRC] = qk_mem;
  softmax_args[DNNL_ARG_DST] = a_mem;
  softmax_p.execute(engine_stream, softmax_args);
  // a.to_file();
  // av_matmul
  auto v_shape = executor::GetShapes(v->shape(), {0, 2, 1, 3});
  auto v_stride = executor::GetStrides(v->shape(), {0, 2, 1, 3});
  auto v_md = memory::desc(v_shape, dnnl::memory::data_type::f32, v_stride);
  auto v_mem = memory(v_md, engine, v->mutable_data());
  auto dst_shape = executor::GetShapes(dst_tensor->shape(), executor::ReversePerm({0, 2, 1, 3}));
  auto dst_stride = executor::GetStrides(dst_tensor->shape(), executor::ReversePerm({0, 2, 1, 3}));
  auto dst_md = memory::desc(dst_shape, dnnl::memory::data_type::f32, dst_stride);
  auto dst_mem = memory(dst_md, engine, dst_tensor->mutable_data());
  auto av_matmul_d = matmul::desc(qk_md, v_md, dst_md);
  auto av_matmul_pd = matmul::primitive_desc(av_matmul_d, engine);
  auto av_matmul_prim = matmul(av_matmul_pd);
  std::unordered_map<int, memory> av_matmul_args;
  av_matmul_args.insert({DNNL_ARG_SRC, a_mem});
  av_matmul_args.insert({DNNL_ARG_WEIGHTS, v_mem});
  av_matmul_args.insert({DNNL_ARG_DST, dst_mem});
  av_matmul_prim.execute(engine_stream, av_matmul_args);
  engine_stream.wait();
  qk.unref_data();
  a.unref_data();
  return dst_tensor;
}

Tensor* make_fp32_tensor_obj(const shared_ptr<TensorConfig>& a_tensor_config, float bound1 = -10, float bound2 = 10) {
  // step1: set shape
  Tensor* a_tensor = new Tensor(*a_tensor_config);
  // step2: set tensor life
  a_tensor->add_tensor_life(1);
  // step3: library buffer can only be obtained afterwards
  auto tensor_data = a_tensor->mutable_data();
  static int seed = 0;
  executor::InitVector(static_cast<float*>(tensor_data), a_tensor->size(), bound1, bound2, seed++);
  return a_tensor;
}

vector<Tensor*> quantize2int8_tensor_obj(const vector<shared_ptr<TensorConfig>>& tensor_configs,
                                         const float* origin_fp32_data, std::string quant_type, bool need_scale) {
  vector<Tensor*> tensors(3);
  for (int i = 0; i < 3; i++) {
    tensors[i] = new Tensor(*tensor_configs[i]);
    tensors[i]->add_tensor_life(1);
  }
  float* min_data = reinterpret_cast<float*>(tensors[1]->mutable_data());
  float* max_data = reinterpret_cast<float*>(tensors[2]->mutable_data());
  void* dst_data = tensors[0]->mutable_data();
  int batch_num, channel_num;
  if (tensors[0]->shape().size() > 2) {
    batch_num = tensors[0]->shape()[0] * tensors[0]->shape()[1];
    channel_num = tensors[0]->shape()[2] * tensors[0]->shape()[3];
  } else {
    batch_num = tensors[0]->shape()[0];
    channel_num = tensors[0]->shape()[1];
  }
  const auto src_dtype = tensors[0]->dtype();
  if (quant_type == "per_channel") {
    for (int y = 0; y < channel_num; y++) {
      min_data[y] = origin_fp32_data[y];
      max_data[y] = origin_fp32_data[y];
      for (int x = 1; x < batch_num; x++) {
        min_data[y] = std::min(min_data[y], origin_fp32_data[x * channel_num + y]);
        max_data[y] = std::max(max_data[y], origin_fp32_data[x * channel_num + y]);
      }
      vector<float> scales = executor::GetScales(min_data + y, max_data + y, 1, src_dtype);
      for (int x = 0; x < batch_num; x++)
        if (src_dtype == "u8") {
          int32_t data = nearbyint((origin_fp32_data[x * channel_num + y] - min_data[y]) * scales[0]);
          data = data < 0 ? 0 : data;
          data = data > 255 ? 255 : data;
          reinterpret_cast<uint8_t*>(dst_data)[x * channel_num + y] = static_cast<uint8_t>(data);
        } else if (src_dtype == "s8") {
          int32_t data = nearbyint(origin_fp32_data[x * channel_num + y] * scales[0]);
          data = data < -128 ? -128 : data;
          data = data > 127 ? 127 : data;
          reinterpret_cast<int8_t*>(dst_data)[x * channel_num + y] = static_cast<int8_t>(data);
        }
      if (need_scale) max_data[y] = 1.0 / scales[0];
    }
  } else if (quant_type == "per_tensor") {
    executor::runtime_minmax(origin_fp32_data, tensors[0]->size(), min_data, max_data);
    const auto scales = executor::GetScales(min_data, max_data, 1, src_dtype);
#if __AVX512F__
    executor::Quantize_avx512(tensors[0]->size(), src_dtype, origin_fp32_data, min_data, scales, dst_data);
#else
    executor::Quantize(tensors[0]->size(), src_dtype, origin_fp32_data, min_data, scales, dst_data);
#endif
    if (need_scale) *max_data = 1.0 / scales[0];
  } else if (quant_type == "per_token") {
    for (int x = 0; x < batch_num; x++) {
      executor::runtime_minmax(origin_fp32_data + x * channel_num, channel_num, min_data + x, max_data + x);
      vector<float> scales = executor::GetScales(min_data + x, max_data + x, 1, src_dtype);
#if __AVX512F__
      executor::Quantize_avx512(channel_num, src_dtype, origin_fp32_data + x * channel_num, min_data, scales,
                                reinterpret_cast<char*>(dst_data) + x * channel_num);
#else
      executor::Quantize(channel_num, src_dtype, origin_fp32_data + x * channel_num, min_data, scales,
                         reinterpret_cast<char*>(dst_data) + x * channel_num);
#endif
      if (need_scale) max_data[x] = 1.0 / scales[0];
    }
  } else {
    LOG(FATAL) << "Not support quantize type" << quant_type;
  }
  return tensors;
}

std::pair<OpArgs, Tensor*> GenerateInt8Case(const vector<vector<int64_t>>& input_shape,
                                            const vector<std::pair<float, float>>& bounds, float output_scale,
                                            bool is_dynamic) {
  std::shared_ptr<TensorConfig> QKV_config[3][4];
  Tensor* QKV[3][4];
  std::string QKV_buffer[3] = {"Q", "K", "V"};
  for (int i = 0; i < 3; i++) {
    vector<int64_t> scale_size;
    scale_size.push_back(is_dynamic ? input_shape[i][0] * input_shape[i][1] : 1);
    QKV_config[i][0] = std::make_shared<TensorConfig>(QKV_buffer[i] + "_fp32", input_shape[i], "fp32");
    QKV_config[i][1] = std::make_shared<TensorConfig>(QKV_buffer[i] + "_int8", input_shape[i], "s8");
    QKV_config[i][2] = std::make_shared<TensorConfig>(QKV_buffer[i] + "_min", scale_size, "fp32");
    QKV_config[i][3] = std::make_shared<TensorConfig>(QKV_buffer[i] + "_scale", scale_size, "fp32");
    QKV[i][0] = make_fp32_tensor_obj(QKV_config[i][0], bounds[i].first, bounds[i].second);
    // QKV[i][0]->to_file();
    auto tmp = quantize2int8_tensor_obj({QKV_config[i][1], QKV_config[i][2], QKV_config[i][3]},
                                        reinterpret_cast<const float*>(QKV[i][0]->data()),
                                        is_dynamic ? "per_token" : "per_tensor", is_dynamic);
    QKV[i][1] = tmp[0];
    QKV[i][2] = tmp[1];
    QKV[i][3] = tmp[2];
    // QKV[i][3]->print();
  }
  vector<shared_ptr<TensorConfig>> inputs_configs = {QKV_config[0][1], QKV_config[1][1], QKV_config[2][1],
                                                     QKV_config[0][2], QKV_config[0][3], QKV_config[1][2],
                                                     QKV_config[1][3], QKV_config[2][2], QKV_config[2][3]};
  vector<Tensor*> inputs = {QKV[0][1], QKV[1][1], QKV[2][1], QKV[0][2], QKV[0][3],
                            QKV[1][2], QKV[1][3], QKV[2][2], QKV[2][3]};

  // mask
  Tensor* mask = nullptr;
  std::shared_ptr<TensorConfig> mask_config;
  if (input_shape.size() > 3) {
    mask_config = std::make_shared<TensorConfig>("mask", input_shape[3], "int32");
    mask = new Tensor(*mask_config);
    mask->add_tensor_life(1);
    executor::InitVector(reinterpret_cast<int32_t*>(mask->mutable_data()), mask->size(), bounds[3].first,
                         bounds[3].second);
    inputs_configs.insert(inputs_configs.begin() + 3, mask_config);
    inputs.insert(inputs.begin() + 3, mask);
  }
  // get true fp32 result and calculate min/max
  vector<int64_t> dst_shape = input_shape[0];
  dst_shape[3] = input_shape[2][3];
  auto dst_fp32_config = std::make_shared<TensorConfig>("dst_fp32", dst_shape, "fp32");
  auto dst_int8_config = std::make_shared<TensorConfig>("dst_int8", dst_shape, is_dynamic ? "s8" : "u8");
  vector<int64_t> dst_scale_size;
  dst_scale_size.push_back(is_dynamic ? dst_shape[0] * dst_shape[1] : 1);
  auto dst_min_config = std::make_shared<TensorConfig>("dst_min", dst_scale_size, "fp32");
  auto dst_max_config = std::make_shared<TensorConfig>("dst_scale", dst_scale_size, "fp32");
  Tensor* dst_fp32 = get_fp32_dst(dst_fp32_config, {QKV[0][0], QKV[1][0], QKV[2][0], mask}, output_scale);
  // dst_fp32->to_file();
  auto dst_tensors = quantize2int8_tensor_obj({dst_int8_config, dst_min_config, dst_max_config},
                                              reinterpret_cast<const float*>(dst_fp32->data()),
                                              is_dynamic ? "per_token" : "per_tensor", is_dynamic);

  vector<shared_ptr<TensorConfig>> output_configs = {dst_int8_config};
  Tensor* dst = new Tensor();
  dst->set_name("dst");
  dst->add_tensor_life(1);
  vector<Tensor*> outputs = {dst};
  map<string, string> attr_map = {{"output_scale", std::to_string(output_scale)},
                                  {"Q_perm", "0,2,1,3"},
                                  {"K_perm", "0,2,3,1"},
                                  {"V_perm", "0,2,1,3"},
                                  {"dst_perm", "0,2,1,3"}};
  if (!is_dynamic) {
    auto qk_min_config = std::make_shared<TensorConfig>("qk_min", dst_scale_size, "fp32");
    auto qk_max_config = std::make_shared<TensorConfig>("qk_max", dst_scale_size, "fp32");
    inputs_configs.push_back(qk_min_config);
    inputs_configs.push_back(qk_max_config);
    inputs_configs.push_back(dst_min_config);
    inputs_configs.push_back(dst_max_config);
    Tensor* qk_min = make_fp32_tensor_obj(qk_min_config, 0, 0);
    Tensor* qk_max = make_fp32_tensor_obj(qk_max_config, 1, 1);
    inputs.push_back(qk_min);
    inputs.push_back(qk_max);
    inputs.push_back(dst_tensors[1]);
    inputs.push_back(dst_tensors[2]);
  } else {
    output_configs.push_back(dst_min_config);
    output_configs.push_back(dst_max_config);
    outputs.push_back(dst_tensors[1]);
    outputs.push_back(dst_tensors[2]);
  }
  // Step 1.1: Construct Operator config obj
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  auto op_config = std::make_shared<OperatorConfig>("multiheadattention", is_dynamic ? "s8" : "u8", inputs_configs,
                                                    output_configs, op_attr);

  // for (auto& i : inputs) i->print();
  // dst_tensors[0]->to_file();
  OpArgs op_args = {inputs, outputs, op_config, is_dynamic};
  return {op_args, dst_tensors[0]};
}

static auto CasesInt8 = []() {
  MemoryAllocator::InitStrategy();

  vector<TestParams> cases;

  // Config
  vector<int64_t> q_shape, k_shape, v_shape;
  vector<int64_t> mask_shape;

  {
    int bs = 2, seq = 32, hn = 8, hs = 40;
    q_shape = {bs, seq, hn, hs};
    k_shape = {bs, seq, hn, hs};
    v_shape = {bs, seq, hn, hs};
    cases.push_back({GenerateInt8Case({q_shape, k_shape, v_shape}, {{-6, 7}, {-5, 6}, {-4, 4}}, 0.15, true), false});
  }

  {
    int bs = 2, seq = 32, hn = 8, hs = 32;
    q_shape = {bs, seq, hn, hs};
    k_shape = {bs, seq, hn, hs};
    v_shape = {bs, seq, hn, hs};
    cases.push_back(
        {GenerateInt8Case({q_shape, k_shape, v_shape, {bs}}, {{-6, 7}, {-5, 6}, {-4, 4}, {1, 32}}, 0.15, false),
         false});
  }
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Engine, MultiheadAttentionInt8Test, CasesInt8());
