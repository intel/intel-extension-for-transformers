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
#include "../../include/operators/inner_product.hpp"
#include "gtest/gtest.h"
using executor::AttrConfig;
using executor::MemoryAllocator;
using executor::OperatorConfig;
using executor::Tensor;
using executor::TensorConfig;

struct OpArgs {
  std::vector<Tensor*> input;
  std::vector<Tensor*> output;
  shared_ptr<OperatorConfig> conf;
};

struct TestParams {
  std::pair<OpArgs, OpArgs> args;
  bool expect_to_fail;
};

template <typename T>
void prepare_sparse_data(T* vector_data, std::vector<int64_t> a_shape) {
  int64_t M = a_shape[0];
  int64_t K = a_shape[1];
  // Blocks zeros in the M dimension.
  int64_t BLOCK = 4;
  int64_t nums = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int64_t block_nums = nums / BLOCK;
  float sparse_ratio = 0.7;
  std::unordered_set<int64_t> zero_block_index;
  uint32_t seed = 123;
  std::srand(seed);
  while (zero_block_index.size() < block_nums * sparse_ratio) {
    zero_block_index.insert((std::rand() % (block_nums - 1)));
  }
  for (const auto& i : zero_block_index) {
    for (int j = 0; j < BLOCK; ++j) {
      int64_t zero_idx = i * BLOCK + j;
      int64_t zero_row = zero_idx / K;
      int64_t zero_col = zero_idx % K;
      // vector_data is (M, K). Block zeros is continuous in M-dim.
      vector_data[zero_col + zero_row * K] = 0;
    }
  }
}

template <typename T>
void transpose(T* old_data, T* new_data, int old_x, int old_y) {
  for (int x = 0; x < old_x; x++) {
    for (int y = 0; y < old_y; y++) new_data[y * old_x + x] = old_data[x * old_y + y];
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::InnerProductOperator inner_product(p.conf);
    inner_product.Prepare(p.input, p.output);
    inner_product.Reshape(p.input, p.output);
    inner_product.Forward(p.input, p.output);
  } catch (const dnnl::error& e) {
    if (e.status != dnnl_status_t::dnnl_success && t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    executor::InnerProductOperator inner_product(q.conf);
    inner_product.Prepare(q.input, q.output);
    inner_product.Reshape(q.input, q.output);
    inner_product.Forward(q.input, q.output);
    bool is_equal;
    if (q.conf->type() == "fp32") {
      float* true_data = new float[q.output[0]->size()];
      transpose(reinterpret_cast<float*>(q.output[0]->mutable_data()), reinterpret_cast<float*>(true_data),
                q.output[0]->shape()[0], q.output[0]->shape()[1]);
      is_equal =
          executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), true_data, q.output[0]->size(), 5e-3);
      delete[] true_data;
    } else if (q.conf->type() == "s8") {
      int8_t* true_data = new int8_t[q.output[0]->size()];
      transpose(reinterpret_cast<int8_t*>(q.output[0]->mutable_data()), reinterpret_cast<int8_t*>(true_data),
                q.output[0]->shape()[0], q.output[0]->shape()[1]);
      is_equal =
          executor::CompareData<int8_t>(p.output[0]->data(), p.output[0]->size(), true_data, q.output[0]->size(), 1);
      delete[] true_data;
    } else if (q.conf->type() == "u8") {
      uint8_t* true_data = new uint8_t[q.output[0]->size()];
      transpose(reinterpret_cast<uint8_t*>(q.output[0]->mutable_data()), reinterpret_cast<uint8_t*>(true_data),
                q.output[0]->shape()[0], q.output[0]->shape()[1]);
      is_equal =
          executor::CompareData<uint8_t>(p.output[0]->data(), p.output[0]->size(), true_data, q.output[0]->size(), 1);
      delete[] true_data;
    }
    return is_equal;
  }
  return false;
}

class InnerProductTest : public testing::TestWithParam<TestParams> {
 protected:
  InnerProductTest() {}
  ~InnerProductTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(InnerProductTest, TestPostfix) {
#if __AVX512VNNI__
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
#endif
}

Tensor* make_int32_bias_obj(const shared_ptr<TensorConfig>& bias_tensor_config, const float* origin_data,
                            Tensor* weight_fp32, Tensor* weight_min, Tensor* weight_max, Tensor* src_min,
                            Tensor* src_max) {
  Tensor* bias_tensor = new Tensor(*bias_tensor_config);
  bias_tensor->add_tensor_life(1);
  int32_t* bias_data = reinterpret_cast<int32_t*>(bias_tensor->mutable_data());
  vector<float> weight_scales = executor::GetScales(weight_min->data(), weight_max->data(), weight_min->size(), "s8");
  vector<float> src_scales = executor::GetScales(src_min->data(), src_max->data(), src_min->size(), "u8");

  const float zp = *reinterpret_cast<const float*>(src_min->data());
  const float* weight_data = reinterpret_cast<const float*>(weight_fp32->data());
  // #pragma omp parallel for
  for (int y = 0; y < weight_fp32->shape()[1]; y++) {
    float compensation = 0;
    for (int x = 0; x < weight_fp32->shape()[0]; x++) compensation += weight_data[x * weight_fp32->shape()[1] + y];
    bias_data[y] = (origin_data[y] + compensation * zp) * src_scales[0] * weight_scales[y];
  }
  return bias_tensor;
}

Tensor* get_fp32_dst(const shared_ptr<TensorConfig>& dst_tensor_config, vector<Tensor*> inputs) {
  using dnnl::matmul;
  using dnnl::memory;
  Tensor* dst_tensor = new Tensor(*dst_tensor_config);
  dst_tensor->add_tensor_life(1);
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream = dnnl::stream(engine);
  Tensor* src = inputs[0];
  Tensor* weight = inputs[1];
  Tensor* bias = inputs[2];
  Tensor* post = inputs[3];
  if (post != nullptr) {
    void* post_data = post->mutable_data();
    // post->unref_data(true);
    dst_tensor->set_data(post_data);
  }
  auto src_md = memory::desc(src->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto weights_md = memory::desc(weight->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto bias_md = memory::desc({1, bias->shape()[0]}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto dst_md = memory::desc(dst_tensor->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto src_mem = memory(src_md, engine, src->mutable_data());
  auto weights_mem = memory(weights_md, engine, weight->mutable_data());
  auto bias_mem = memory(bias_md, engine, bias->mutable_data());
  auto dst_mem = memory(dst_md, engine, dst_tensor->mutable_data());
  auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
  auto matmul_pd = matmul::primitive_desc(matmul_d, engine);
  auto matmul_prim = matmul(matmul_pd);
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});
  matmul_prim.execute(engine_stream, matmul_args);
  engine_stream.wait();
  return dst_tensor;
}

Tensor* make_fp32_tensor_obj(const shared_ptr<TensorConfig>& a_tensor_config, bool is_sparse = false) {
  // step1: set shape
  Tensor* a_tensor = new Tensor(*a_tensor_config);
  // step2: set tensor life
  a_tensor->add_tensor_life(1);
  // step3: library buffer can only be obtained afterwards
  auto tensor_data = a_tensor->mutable_data();
  executor::InitVector(static_cast<float*>(tensor_data), a_tensor->size());
  if (is_sparse) {
    float* fp32_ptr = static_cast<float*>(tensor_data);
    std::vector<int64_t> a_shape = a_tensor_config->shape();
    for (int i = 0; i < a_tensor->size(); ++i) {  // firstly remove zero.
      fp32_ptr[i] = (fp32_ptr[i] == 0) ? 1 : fp32_ptr[i];
    }
    prepare_sparse_data(fp32_ptr, a_shape);
  }
  return a_tensor;
}

vector<Tensor*> make_transposed_int8_tensor_obj(const vector<shared_ptr<TensorConfig>>& tensor_configs,
                                                const float* origin_fp32_data, bool per_channel = false) {
  vector<Tensor*> tensors(3);
  for (int i = 0; i < 3; i++) {
    tensors[i] = new Tensor(*tensor_configs[i]);
    tensors[i]->add_tensor_life(1);
  }
  float* transposed_data = new float[tensors[0]->size()];
  float* min_data = reinterpret_cast<float*>(tensors[1]->mutable_data());
  float* max_data = reinterpret_cast<float*>(tensors[2]->mutable_data());
  void* dst_data = tensors[0]->mutable_data();
  for (int y = 0; y < tensors[0]->shape()[0]; y++) {
    for (int x = 0; x < tensors[0]->shape()[1]; x++)
      transposed_data[y * tensors[0]->shape()[1] + x] = origin_fp32_data[x * tensors[0]->shape()[0] + y];
    if (per_channel) {
      executor::runtime_minmax(&transposed_data[y * tensors[0]->shape()[1]], tensors[0]->shape()[1], min_data + y,
                               max_data + y);
      if (tensors[0]->dtype() != "fp32") {
        vector<float> scales = executor::GetScales(min_data + y, max_data + y, 1, tensors[0]->dtype());
#if __AVX512F__
        executor::Quantize_avx512(tensors[0]->shape()[1], tensors[0]->dtype(),
                                  &transposed_data[y * tensors[0]->shape()[1]], min_data + y, scales,
                                  reinterpret_cast<char*>(dst_data) + y * tensors[0]->shape()[1]);
#else
        executor::Quantize(tensors[0]->shape()[1], tensors[0]->dtype(), &transposed_data[y * tensors[0]->shape()[1]],
                           min_data + y, scales, reinterpret_cast<char*>(dst_data) + y * tensors[0]->shape()[1]);
#endif
      }
    }
  }
  if (tensors[0]->dtype() == "fp32") {
    memcpy(dst_data, transposed_data, tensors[0]->size() * sizeof(float));
  } else if (!per_channel) {
    executor::runtime_minmax(transposed_data, tensors[0]->size(), min_data, max_data);
    vector<float> scales = executor::GetScales(min_data, max_data, 1, tensors[0]->dtype());
#if __AVX512F__
    executor::Quantize_avx512(tensors[0]->size(), tensors[0]->dtype(), transposed_data, min_data, scales, dst_data);
#else
    executor::Quantize(tensors[0]->size(), tensors[0]->dtype(), transposed_data, min_data, scales, dst_data);
#endif
  }
  delete[] transposed_data;
  return tensors;
}

OpArgs GenenrateCopies(const vector<shared_ptr<TensorConfig>>& old_configs, vector<Tensor*> old_tensors,
                       Tensor* old_dst, std::map<std::string, std::string> attr_map) {
  int tensor_number = old_configs.size();
  vector<shared_ptr<TensorConfig>> new_configs(tensor_number);
  vector<Tensor*> new_tensors(tensor_number);
  for (int i = 0; i < tensor_number; i++) {
    vector<int64_t> shape = old_tensors[i]->shape();
    if (shape.size() > 1) {
      new_configs[i] = std::make_shared<TensorConfig>(old_tensors[i]->name(), executor::GetShapes(shape, {1, 0}),
                                                      old_tensors[i]->dtype());
      new_tensors[i] = new Tensor(*new_configs[i]);
      new_tensors[i]->add_tensor_life(1);
      if (old_tensors[i]->dtype() == "fp32")
        transpose(reinterpret_cast<float*>(old_tensors[i]->mutable_data()),
                  reinterpret_cast<float*>(new_tensors[i]->mutable_data()), shape[0], shape[1]);
      else if (old_tensors[i]->dtype() == "u8")
        transpose(reinterpret_cast<uint8_t*>(old_tensors[i]->mutable_data()),
                  reinterpret_cast<uint8_t*>(new_tensors[i]->mutable_data()), shape[0], shape[1]);
      else if (old_tensors[i]->dtype() == "s8")
        transpose(reinterpret_cast<int8_t*>(old_tensors[i]->mutable_data()),
                  reinterpret_cast<int8_t*>(new_tensors[i]->mutable_data()), shape[0], shape[1]);
      else
        transpose(reinterpret_cast<int32_t*>(old_tensors[i]->mutable_data()),
                  reinterpret_cast<int32_t*>(new_tensors[i]->mutable_data()), shape[0], shape[1]);
    } else {
      new_configs[i] = std::make_shared<TensorConfig>(old_tensors[i]->name(), shape, old_tensors[i]->dtype());
      new_tensors[i] = new Tensor(*new_configs[i]);
      new_tensors[i]->add_tensor_life(1);
      memcpy(new_tensors[i]->mutable_data(), old_tensors[i]->mutable_data(),
             old_tensors[i]->size() * executor::type2bytes[old_tensors[i]->dtype()]);
    }
  }
  // exchange src&weight
  std::pair<int, int> swap_idx[3] = {
      {0, 1}, {tensor_number - 6, tensor_number - 4}, {tensor_number - 5, tensor_number - 3}};
  for (int i = 0; i < 3; i++) {
    std::swap(new_configs[swap_idx[i].first], new_configs[swap_idx[i].second]);
    std::swap(new_tensors[swap_idx[i].first], new_tensors[swap_idx[i].second]);
  }

  auto dst_config =
      std::make_shared<TensorConfig>(old_dst->name(), executor::GetShapes(old_dst->shape(), {1, 0}), old_dst->dtype());
  Tensor* dst = new Tensor(*dst_config);
  dst->add_tensor_life(1);
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config};

  attr_map["src1_perm"] = "1,0";
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  auto op_config =
      std::make_shared<OperatorConfig>("inner_product", old_dst->dtype(), new_configs, output_config, op_attr);
  return {new_tensors, {dst}, op_config};
}

std::pair<OpArgs, OpArgs> GenerateInt8Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           std::string output_type = "s8", std::string append_op = "") {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  const auto& src1_shape = input_shape[1];
  const auto& bias_shape = input_shape[2];
  std::vector<int64_t> dst_shape = {src0_shape[0], src1_shape[1]};
  auto weight_fp32_config = std::make_shared<TensorConfig>("weight", executor::GetShapes(src0_shape, {1, 0}), "fp32");
  auto weight_s8_config = std::make_shared<TensorConfig>("weight", src0_shape, "s8");
  auto weight_min_config = std::make_shared<TensorConfig>("weight_min", bias_shape, "fp32");
  auto weight_max_config = std::make_shared<TensorConfig>("weight_max", bias_shape, "fp32");
  Tensor* weight_fp32 = make_fp32_tensor_obj(weight_fp32_config, true);
  auto weight_tensors = make_transposed_int8_tensor_obj({weight_s8_config, weight_min_config, weight_max_config},
                                                        reinterpret_cast<const float*>(weight_fp32->data()), true);

  auto src_fp32_config = std::make_shared<TensorConfig>("src", executor::GetShapes(src1_shape, {1, 0}), "fp32");
  auto src_u8_config = std::make_shared<TensorConfig>("src", src1_shape, "u8");
  auto src_min_config = std::make_shared<TensorConfig>("src_min", vector<int64_t>({1}), "fp32");
  auto src_max_config = std::make_shared<TensorConfig>("src_max", vector<int64_t>({1}), "fp32");
  Tensor* src_fp32 = make_fp32_tensor_obj(src_fp32_config, false);
  auto src_tensors = make_transposed_int8_tensor_obj({src_u8_config, src_min_config, src_max_config},
                                                     reinterpret_cast<const float*>(src_fp32->data()), false);

  // bias should include compensation
  auto bias_fp32_config = std::make_shared<TensorConfig>("bias", bias_shape, "fp32");
  auto bias_int32_config = std::make_shared<TensorConfig>("bias", bias_shape, "s32");
  Tensor* bias_fp32 = make_fp32_tensor_obj(bias_fp32_config, false);
  Tensor* bias_int32 =
      make_int32_bias_obj(bias_int32_config, reinterpret_cast<const float*>(bias_fp32->data()), weight_fp32,
                          weight_tensors[1], weight_tensors[2], src_tensors[1], src_tensors[2]);

  auto post_fp32_config = std::make_shared<TensorConfig>("post", executor::GetShapes(dst_shape, {1, 0}), "fp32");
  auto post_config = std::make_shared<TensorConfig>("post", dst_shape, "fp32");
  Tensor* post_fp32 = nullptr;
  Tensor* post = nullptr;
  if (output_type == "fp32" && append_op == "sum") {
    post_fp32 = make_fp32_tensor_obj(post_fp32_config, false);
    post = new Tensor(*post_config);
    post->add_tensor_life(1);
    transpose(reinterpret_cast<float*>(post_fp32->mutable_data()), reinterpret_cast<float*>(post->mutable_data()),
              post_fp32->shape()[0], post_fp32->shape()[1]);
  }

  // get true fp32 result and calculate min/max
  auto dst_fp32_config = std::make_shared<TensorConfig>("dst", executor::GetShapes(dst_shape, {1, 0}), "fp32");
  auto dst_config = std::make_shared<TensorConfig>("dst", dst_shape, output_type);
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config};
  auto dst_min_config = std::make_shared<TensorConfig>("dst_min", vector<int64_t>({1}), "fp32");
  auto dst_max_config = std::make_shared<TensorConfig>("dst_max", vector<int64_t>({1}), "fp32");
  Tensor* dst_fp32 = get_fp32_dst(dst_fp32_config, {src_fp32, weight_fp32, bias_fp32, post});
  Tensor* dst = new Tensor(*dst_config);
  dst->add_tensor_life(1);
  Tensor* dst_min = new Tensor(*dst_min_config);
  dst_min->add_tensor_life(1);
  Tensor* dst_max = new Tensor(*dst_max_config);
  dst_max->add_tensor_life(1);
  executor::runtime_minmax(reinterpret_cast<float*>(dst_fp32->mutable_data()), dst_fp32->size(),
                           reinterpret_cast<float*>(dst_min->mutable_data()),
                           reinterpret_cast<float*>(dst_max->mutable_data()));

  std::vector<shared_ptr<TensorConfig>> inputs_configs = {weight_s8_config,  src_u8_config,     bias_int32_config,
                                                          weight_min_config, weight_max_config, src_min_config,
                                                          src_max_config,    dst_min_config,    dst_max_config};
  vector<Tensor*> inputs = {weight_tensors[0], src_tensors[0],    bias_int32,
                            weight_tensors[1], weight_tensors[2], src_tensors[1],
                            src_tensors[2],    dst_min,           dst_max};
  if (output_type == "fp32" && append_op == "sum") {
    inputs_configs.insert(inputs_configs.begin() + 3, post_config);
    inputs.insert(inputs.begin() + 3, post);
  }
  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"src0_perm", ""}, {"src1_perm", ""}, {"output_dtype", output_type}, {"append_op", append_op}};

  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  auto op_config =
      std::make_shared<OperatorConfig>("inner_product", output_type, inputs_configs, output_config, op_attr);

  OpArgs op_args = {inputs, {dst}, op_config};
  OpArgs op_args_copy = GenenrateCopies(inputs_configs, inputs, dst, attr_map);

  return {op_args, op_args_copy};
}

static auto CasesInt8 = []() {
  MemoryAllocator::InitStrategy();

  std::vector<TestParams> cases;

  // Config
  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;
  std::vector<int64_t> bias_shape;

  // case: sparse
  src0_shape = {16, 64};
  src1_shape = {64, 32};
  bias_shape = {16, 1};
  cases.push_back({GenerateInt8Case({src0_shape, src1_shape, bias_shape}, "s8", ""), false});

  // case: sparse
  src0_shape = {16, 64};
  src1_shape = {64, 64};
  bias_shape = {16, 1};
  cases.push_back({GenerateInt8Case({src0_shape, src1_shape, bias_shape}, "fp32", ""), false});
  // case: sparse
  src0_shape = {32, 128};
  src1_shape = {128, 64};
  bias_shape = {32, 1};
  cases.push_back({GenerateInt8Case({src0_shape, src1_shape, bias_shape}, "fp32", "sum"), false});
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, InnerProductTest, CasesInt8());
