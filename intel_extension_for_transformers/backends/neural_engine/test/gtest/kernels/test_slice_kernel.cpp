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
#include <iostream>
#include "unit_test_utils.hpp"
#include "gtest/gtest.h"

namespace jd {

template <typename T>
void DataPrint(const T* data, std::vector<dim_t> shape, int stride = 0) {
  // print output file
  int outer = 1;
  if (stride == 0) stride = shape.back();
  for (long unsigned int i = 0; i < shape.size() - 1; i++) outer *= shape[i]; // NOLINT
  for (int i = 0; i < outer; i++) {
    for (dim_t j = 0; j < shape.back(); ++j) std::cout << (T)(data[i * stride + j]) << ",";
    std::cout << std::endl;
  }
}
template void DataPrint<int32_t>(const int32_t* data, std::vector<dim_t> shape, int stride);
template void DataPrint<float>(const float* data, std::vector<dim_t> shape, int stride);
template void DataPrint<int8_t>(const int8_t* data, std::vector<dim_t> shape, int stride);

struct OpArgs {
  std::vector<const void*> data;
  operator_desc conf;
};

struct test_params_t {
  std::pair<OpArgs, OpArgs> args;
  bool expect_to_fail;
  void memoryFree() {
    auto p = args.first;
    for (auto data : p.data) free(const_cast<void*>(data));
    auto q = args.second;
    for (auto data : q.data) free(const_cast<void*>(data));
  }
};

bool CheckResult(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  // DataPrint<float>(reinterpret_cast<const float*>(p.data[0]), p.conf.tensor_descs()[0].shape());
  // std::cout << "=======================\n";
  slice_desc slice_d(p.conf);
  slice slice_ker(slice_d);
  slice_ker.execute(p.data);
  // Should compare buffer with different addresses
  if (p.conf.tensor_descs()[1].dtype() == data_type::s8) {
    // DataPrint<int8_t>(reinterpret_cast<const int8_t*>(p.data[1]), p.conf.tensor_descs()[1].shape());
    // DataPrint<int8_t>(reinterpret_cast<const int8_t*>(q.data[1]), q.conf.tensor_descs()[1].shape());
    return compare_data<int8_t>(p.data[1], p.conf.tensor_descs()[1].size(), q.data[1], q.conf.tensor_descs()[1].size());
  } else if (p.conf.tensor_descs()[1].dtype() == data_type::fp32) {
    // DataPrint<float>(reinterpret_cast<const float*>(p.data[1]), p.conf.tensor_descs()[1].shape());
    // std::cout << "=======================\n";
    // DataPrint<float>(reinterpret_cast<const float*>(q.data[1]), q.conf.tensor_descs()[1].shape());
    return compare_data<float>(p.data[1], p.conf.tensor_descs()[1].size(), q.data[1], q.conf.tensor_descs()[1].size());
  } else {
    // DataPrint<float>(reinterpret_cast<const float*>(p.data[1]), p.conf.tensor_descs()[1].shape());
    // std::cout << "=======================\n";
    // DataPrint<float>(reinterpret_cast<const float*>(q.data[1]), q.conf.tensor_descs()[1].shape());
    return compare_data<uint16_t>(p.data[1], p.conf.tensor_descs()[1].size(), q.data[1],
                                  q.conf.tensor_descs()[1].size());
  }
}

class SliceOpTest : public testing::TestWithParam<test_params_t> {
 protected:
  SliceOpTest() {}
  ~SliceOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SliceOpTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(CheckResult(t));
  t.memoryFree();
}

std::pair<OpArgs, OpArgs> GenerateCase(std::vector<tensor_desc> const& ts_descs, int begin, int step, size_t axis) {
  data_type input_dt = ts_descs[0].dtype();
  auto src_shape = ts_descs[0].shape();
  auto dst_shape = ts_descs[1].shape();
  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["axis"] = std::to_string(axis);
  attr_map["begin"] = std::to_string(begin);
  attr_map["step"] = std::to_string(step);
  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const tensor_desc a_tensor_config) {
    void* tensor_data = sparselib_ut_memo(nullptr, a_tensor_config.size(), a_tensor_config.dtype(), memo_mode::MALLOC);
    void* tensor_data_copy =
        sparselib_ut_memo(nullptr, a_tensor_config.size(), a_tensor_config.dtype(), memo_mode::MALLOC);

    // init other tensor
    if (input_dt == data_type::s8 || input_dt == data_type::u8)
      init_vector(static_cast<int8_t*>(tensor_data), a_tensor_config.size());
    else if (input_dt == data_type::fp32)
      init_vector(static_cast<float*>(tensor_data), a_tensor_config.size());
    else
      init_vector(static_cast<uint16_t*>(tensor_data), a_tensor_config.size());
    memcpy(tensor_data_copy, tensor_data, a_tensor_config.size() * get_data_size(input_dt));

    return std::pair<void*, void*>{tensor_data, tensor_data_copy};
  };

  auto src_tensors = make_tensor_obj(ts_descs[0]);
  auto dst_data = reinterpret_cast<char*>(sparselib_ut_memo(nullptr, ts_descs[1].size(), input_dt, memo_mode::MALLOC));
  auto dst_data_copy =
      reinterpret_cast<char*>(sparselib_ut_memo(nullptr, ts_descs[1].size(), input_dt, memo_mode::MALLOC));
  std::vector<const void*> rt_data = {src_tensors.first, dst_data};
  std::vector<const void*> rt_data_copy = {src_tensors.second, dst_data_copy};

  auto src_data_copy = reinterpret_cast<char*>(src_tensors.second);
  auto src_shape_copy = src_shape;
  int outer_size = 1;
  int inner_size = 1;
  for (size_t i = 0; i < src_shape.size(); i++)
    if (i < axis)
      outer_size *= src_shape[i];
    else if (i > axis)
      inner_size *= src_shape[i];
#pragma omp parallel for
  for (int i = 0; i < outer_size; ++i) {
    if (step == 1) {
      memcpy(dst_data_copy + i * dst_shape[axis] * inner_size * get_data_size(input_dt),
             src_data_copy + (i * src_shape[axis] + begin) * inner_size * get_data_size(input_dt),
             get_data_size(input_dt) * inner_size * dst_shape[axis]);
    }
    if (step == 2) {
      for (int j = 0; j < dst_shape[axis]; j++) {
        memcpy(dst_data_copy + (i * dst_shape[axis] + j) * inner_size * get_data_size(input_dt),
               src_data_copy + (i * src_shape[axis] + begin + j * step) * inner_size * get_data_size(input_dt),
               get_data_size(input_dt) * inner_size);
      }
    }
  }
  operator_desc slice_d(kernel_kind::slice, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, attr_map);
  operator_desc slice_d_copy(kernel_kind::slice, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, attr_map);
  OpArgs op_args = {rt_data, slice_d};
  OpArgs op_args_copy = {rt_data_copy, slice_d_copy};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::vector<test_params_t> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> idx_shape;
  std::vector<int64_t> dst_shape;
  tensor_desc src, dst;

  for (data_type dt : {data_type::fp32, data_type::bf16, data_type::s8}) {
    src_shape = {1, 32, 16, 256};
    dst_shape = {1, 32, 16, 192};
    src = {src_shape, dt, jd::format_type::undef};
    dst = {dst_shape, dt, jd::format_type::undef};
    cases.push_back({GenerateCase({src, dst}, 0, 1, 3), false});

    src_shape = {1, 32, 16, 256};
    dst_shape = {1, 32, 16, 64};
    src = {src_shape, dt, jd::format_type::undef};
    dst = {dst_shape, dt, jd::format_type::undef};
    cases.push_back({GenerateCase({src, dst}, 2, 2, 3), false});

    src_shape = {1, 32, 16, 256};
    dst_shape = {1, 10, 16, 256};
    src = {src_shape, dt, jd::format_type::undef};
    dst = {dst_shape, dt, jd::format_type::undef};
    cases.push_back({GenerateCase({src, dst}, 2, 2, 1), false});

    src_shape = {1, 1, 2048, 2048};
    dst_shape = {1, 1, 32, 2048};
    src = {src_shape, dt, jd::format_type::undef};
    dst = {dst_shape, dt, jd::format_type::undef};
    cases.push_back({GenerateCase({src, dst}, 1, 1, 2), false});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_desc = tpi.param.args.first.conf.tensor_descs();
  auto& src_shape = tensor_desc[0].shape();
  auto& dst_shape = tensor_desc[1].shape();
  auto attrs_map = tpi.param.args.first.conf.attrs();

  auto add_dt_info = [&](const std::string& tensor_dt) {
    switch (tensor_desc[0].dtype()) {
      case data_type::s8:
        params.push_back(tensor_dt + "_s8");
        break;
      case data_type::fp32:
        params.push_back(tensor_dt + "_fp32");
        break;
      case data_type::bf16:
        params.push_back(tensor_dt + "_bf16");
        break;
      default:
        assert(false);
    }
  };

  add_dt_info("src0");
  for (auto&& i : src_shape) params.push_back(std::to_string(i));
  add_dt_info("dst");
  for (auto&& i : dst_shape) params.push_back(std::to_string(i));
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SliceOpTest, CasesFp32(), test_suffix);
}  // namespace jd
