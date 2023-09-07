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
#include "gtest/gtest.h"
#include "../../include/operators/binary_op.hpp"

using executor::AttrConfig;
using executor::MemoryAllocator;
using executor::OperatorConfig;
using executor::Tensor;
using executor::TensorConfig;
using executor::GetStrides;

struct OpArgs {
  std::vector<Tensor*> input;
  std::vector<Tensor*> output;
  shared_ptr<OperatorConfig> conf;
};

struct TestParams {
  std::pair<OpArgs, OpArgs> args;
  bool expect_to_fail;
};

void GetTrueData(const std::vector<Tensor*>& input, const std::vector<Tensor*>& output,
                 const shared_ptr<OperatorConfig>& conf) {
  auto attrs_map = conf->attributes();
  string algo;
  auto iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algo = iter->second;
  }

  auto src0_shape_ = input[0]->shape();
  auto src1_shape_ = input[1]->shape();

  auto src0_stride_ = GetStrides(src0_shape_);
  auto src1_stride_ = GetStrides(src1_shape_);

  if (src0_shape_.size() < src1_shape_.size()) {
    int diff_len = src1_shape_.size() - src0_shape_.size();
    for (int i = 0; i < diff_len; i++) {
      src0_shape_.insert(src0_shape_.begin(), 1);
      src0_stride_.insert(src0_stride_.begin(), 0);
    }
  } else if (src0_shape_.size() > src1_shape_.size()) {
    int diff_len = src0_shape_.size() - src1_shape_.size();
    for (int i = 0; i < diff_len; i++) {
      src1_shape_.insert(src1_shape_.begin(), 1);
      src1_stride_.insert(src1_stride_.begin(), 0);
    }
  }

  vector<int64_t> out_shape;
  for (int i = 0; i < src0_shape_.size(); i++) {
    if (src0_shape_[i] != src1_shape_[i] && src0_shape_[i] != 1 && src1_shape_[i] != 1) {
      LOG(ERROR) << "can not broadcast!";
    }
    out_shape.push_back(std::max(src0_shape_[i], src1_shape_[i]));
  }
  auto out_stride_ = GetStrides(out_shape);
  output[0]->set_shape(out_shape);

  auto src0_data = reinterpret_cast<const float*>(input[0]->data());
  auto src1_data = reinterpret_cast<const float*>(input[1]->data());
  auto dst_data = reinterpret_cast<float*>(output[0]->mutable_data());

  int size = output[0]->size();
  int dims = src0_shape_.size();

#pragma omp parallel for
  for (int out_idx = 0; out_idx < size; out_idx++) {
    vector<int64_t> coord(dims, 0);
    // 1. recover original coordinates
    int remain = 0;
    for (int s = 0; s < out_stride_.size(); s++) {
      if (s == 0) {
        coord[s] = static_cast<int>(out_idx / out_stride_[s]);
      } else {
        coord[s] = static_cast<int>(remain / out_stride_[s]);
      }
      remain = out_idx % out_stride_[s];
    }
    // 2. find corresponding index of src0/src1
    int src0_index = 0, src1_index = 0;
    for (int d = 0; d < dims; d++) {
      if (src0_shape_[d] != 1) {
        src0_index += coord[d] * src0_stride_[d];
      }
      if (src1_shape_[d] != 1) {
        src1_index += coord[d] * src1_stride_[d];
      }
    }

    // 3. do actual computation
    if (algo == "add") {
      dst_data[out_idx] = src0_data[src0_index] + src1_data[src1_index];
    } else if (algo == "sub") {
      dst_data[out_idx] = src0_data[src0_index] - src1_data[src1_index];
    } else if (algo == "mul") {
      dst_data[out_idx] = src0_data[src0_index] * src1_data[src1_index];
    } else if (algo == "div") {
      dst_data[out_idx] = src0_data[src0_index] / src1_data[src1_index];
    } else if (algo == "gt") {
      dst_data[out_idx] = src0_data[src0_index] > src1_data[src1_index];
    } else if (algo == "ge") {
      dst_data[out_idx] = src0_data[src0_index] >= src1_data[src1_index];
    } else if (algo == "lt") {
      dst_data[out_idx] = src0_data[src0_index] < src1_data[src1_index];
    } else if (algo == "le") {
      dst_data[out_idx] = src0_data[src0_index] <= src1_data[src1_index];
    } else if (algo == "eq") {
      dst_data[out_idx] = src0_data[src0_index] == src1_data[src1_index];
    } else if (algo == "ne") {
      dst_data[out_idx] = src0_data[src0_index] != src1_data[src1_index];
    } else if (algo == "min") {
      dst_data[out_idx] = std::min(src0_data[src0_index], src1_data[src1_index]);
    } else if (algo == "max") {
      dst_data[out_idx] = std::max(src0_data[src0_index], src1_data[src1_index]);
    }
  }
}

bool CheckResult(const TestParams& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    executor::BinaryOpOperator mul(p.conf);
    mul.Prepare(p.input, p.output);
    mul.Reshape(p.input, p.output);
    mul.Forward(p.input, p.output);
  } catch (const dnnl::error& e) {
    if (e.status != dnnl_status_t::dnnl_success && t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    GetTrueData(q.input, q.output, q.conf);
    // Should compare buffer with different addresses
    EXPECT_NE(p.output[0]->data(), q.output[0]->data());
    float eps = 1e-4;
    return executor::CompareData<float>(p.output[0]->data(), p.output[0]->size(), q.output[0]->data(),
                                        q.output[0]->size(), eps);
  }
  return false;
}

class MulTest : public testing::TestWithParam<TestParams> {
 protected:
  MulTest() {}
  ~MulTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MulTest, TestPostfix) {
  TestParams t = testing::TestWithParam<TestParams>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(const std::vector<std::vector<int64_t>>& input_shape,
                                           const string& algo) {
  // Step 1: Construct Tensor config ptr
  const auto& src0_shape = input_shape[0];
  shared_ptr<TensorConfig> src0_config = std::make_shared<TensorConfig>("src0", src0_shape);
  const auto& src1_shape = input_shape[1];
  shared_ptr<TensorConfig> src1_config = std::make_shared<TensorConfig>("src1", src1_shape);
  std::vector<shared_ptr<TensorConfig>> input_config = {src0_config, src1_config};
  std::vector<int64_t> dst_shape = {};
  shared_ptr<TensorConfig> dst_config = std::make_shared<TensorConfig>("dst", dst_shape);
  std::vector<shared_ptr<TensorConfig>> output_config = {dst_config};

  // Step 1.1: Construct Operator config obj
  std::map<std::string, std::string> attr_map;
  attr_map = {{"algorithm", algo}};
  shared_ptr<AttrConfig> op_attr = std::make_shared<AttrConfig>(attr_map);
  shared_ptr<OperatorConfig> op_config = std::make_shared<OperatorConfig>("binary_op", "BinaryOp",
                                         input_config, output_config, op_attr);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const shared_ptr<TensorConfig>& a_tensor_config) {
    // step1: set shape
    Tensor* a_tensor = new Tensor(*a_tensor_config);
    // step2: set tensor life
    a_tensor->add_tensor_life(1);
    // step3: library buffer can only be obtained afterwards
    auto tensor_data = a_tensor->mutable_data();
    executor::InitVector<float>(static_cast<float*>(tensor_data), a_tensor->size());

    Tensor* a_tensor_copy = new Tensor(*a_tensor_config);
    a_tensor_copy->add_tensor_life(1);
    auto tensor_data_copy = a_tensor_copy->mutable_data();
    memcpy(reinterpret_cast<void*>(tensor_data_copy), tensor_data, a_tensor_copy->size() * sizeof(float));
    return std::pair<Tensor*, Tensor*>{a_tensor, a_tensor_copy};
  };

  auto src0_tensors = make_tensor_obj(src0_config);
  auto src1_tensors = make_tensor_obj(src1_config);
  Tensor* dst_tensor = new Tensor(*dst_config);
  dst_tensor->add_tensor_life(1);
  Tensor* dst_tensor_copy = new Tensor(*dst_config);
  dst_tensor_copy->add_tensor_life(1);

  OpArgs op_args = {{src0_tensors.first, src1_tensors.first}, {dst_tensor}, op_config};
  OpArgs op_args_copy = {{src0_tensors.second, src1_tensors.second}, {dst_tensor_copy}, op_config};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::string memory_strategy = getenv("DIRECT_BUFFER") == NULL ? "cycle_buffer" : "direct_buffer";
  MemoryAllocator::SetStrategy(memory_strategy);
  std::vector<TestParams> cases;

  std::vector<int64_t> src0_shape;
  std::vector<int64_t> src1_shape;
  // case: 2D x 3D
  src0_shape = {16, 1024};
  src1_shape = {2, 16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "add")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "sub")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "mul")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "div")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "gt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ge")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "lt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "le")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "eq")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ne")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "min")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "max")});

  // case: 3D x 2D
  src0_shape = {2, 16, 1024};
  src1_shape = {16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "add")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "sub")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "mul")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "div")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "gt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ge")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "lt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "le")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "eq")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ne")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "min")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "max")});

  // case: 2D x 2D
  src0_shape = {16, 1024};
  src1_shape = {16, 1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "add")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "sub")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "mul")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "div")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "gt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ge")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "lt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "le")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "eq")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ne")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "min")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "max")});

  // case: 2D x 1D
  src0_shape = {16, 1024};
  src1_shape = {1024};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "add")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "sub")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "mul")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "div")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "gt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ge")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "lt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "le")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "eq")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ne")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "min")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "max")});

  // case: 2D x scalar
  src0_shape = {16, 1024};
  src1_shape = {1};
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "add")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "sub")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "mul")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "div")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "gt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ge")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "lt")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "le")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "eq")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "ne")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "min")});
  cases.push_back({GenerateFp32Case({src0_shape, src1_shape}, "max")});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, MulTest, CasesFp32());
