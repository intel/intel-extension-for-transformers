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

#include "gtest/gtest.h"
#include "interface.hpp"
#include "kernels/exposed_enum.hpp"
#include "unit_test_utils.hpp"

namespace test {
using io = jd::exposed_enum::slice::io;
using dt = jd::data_type;

struct test_data_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data_kern;
  std::vector<const void*> rt_data_ref;
};

struct test_params_t {
  int axis;
  int begin;
  int step;
  dt input_dt;  // data type of src & dst
  std::vector<dim_t> src_dim;
  std::vector<dim_t> dst_dim;
  bool expect_to_fail;
};

bool CheckResult(const bool /*expect_to_fail*/, const test_data_t& d) {
  const auto& p = d.rt_data_kern;
  const auto& q = d.rt_data_ref;
  jd::slice_desc slice_d(d.op_desc);
  jd::slice slice_ker(slice_d);
  slice_ker.execute(p);

  // Should compare buffer with different addresses
  EXPECT_NE(p[io::DST], q[io::DST]);
  const auto dst_size = d.op_desc.tensor_descs()[io::DST].size();
  const auto dst_dtype = d.op_desc.tensor_descs()[io::DST].dtype();
  if (dst_dtype == dt::s8) {
    return compare_data<int8_t>(p[io::DST], dst_size, q[io::DST], dst_size);
  } else if (dst_dtype == dt::fp32) {
    return compare_data<float>(p[io::DST], dst_size, q[io::DST], dst_size);
  } else {
    return compare_data<uint16_t>(p[io::DST], dst_size, q[io::DST], dst_size);
  }
}

test_data_t gen_data(const test_params_t& p) {
  const jd::tensor_desc src_desc{p.src_dim, p.input_dt, jd::plain_format(p.src_dim.size())};
  const jd::tensor_desc dst_desc{p.dst_dim, p.input_dt, jd::plain_format(p.dst_dim.size())};

  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["axis"] = std::to_string(p.axis);
  attr_map["begin"] = std::to_string(p.begin);
  attr_map["step"] = std::to_string(p.step);

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const jd::tensor_desc a_tensor_config) {
    void* rt_data = sparselib_ut_memo(nullptr, a_tensor_config.size(), a_tensor_config.dtype(), memo_mode::MALLOC);
    void* rt_data_ref = sparselib_ut_memo(nullptr, a_tensor_config.size(), a_tensor_config.dtype(), memo_mode::MALLOC);

    // init other tensor
    if (p.input_dt == dt::s8 || p.input_dt == dt::u8) {
      init_vector(static_cast<int8_t*>(rt_data), a_tensor_config.size());
    } else if (p.input_dt == dt::fp32) {
      init_vector(static_cast<float*>(rt_data), a_tensor_config.size());
    } else if (p.input_dt == dt::bf16) {
      init_vector(static_cast<jd::bfloat16_t*>(rt_data), a_tensor_config.size());
    } else {
      SPARSE_LOG(FATAL) << "Unexpected dtype!";
    }
    memcpy(rt_data_ref, rt_data, a_tensor_config.size() * jd::type_size.at(p.input_dt));

    return std::pair<void*, void*>{rt_data, rt_data_ref};
  };

  const auto src_data = make_tensor_obj(src_desc);
  const auto dst_data =
      reinterpret_cast<char*>(sparselib_ut_memo(nullptr, dst_desc.size(), p.input_dt, memo_mode::MALLOC));
  const auto dst_data_ref =
      reinterpret_cast<char*>(sparselib_ut_memo(nullptr, dst_desc.size(), p.input_dt, memo_mode::MALLOC));
  const auto src_data_ref = reinterpret_cast<char*>(src_data.second);
  std::vector<const void*> rt_data(io::SIZE);
  rt_data[io::SRC] = src_data.first;
  rt_data[io::DST] = dst_data;
  std::vector<const void*> rt_data_ref(io::SIZE);
  rt_data_ref[io::SRC] = src_data.second;
  rt_data_ref[io::DST] = dst_data_ref;

  int outer_size = std::accumulate(p.src_dim.cbegin(), p.src_dim.cbegin() + p.axis, 1, std::multiplies<int>());
  int inner_size = std::accumulate(p.src_dim.cbegin() + p.axis + 1, p.src_dim.cend(), 1, std::multiplies<int>());

#pragma omp parallel for
  for (int i = 0; i < outer_size; ++i) {
    const auto dt_size = jd::type_size.at(p.input_dt);
    if (p.step == 1) {
      memcpy(dst_data_ref + i * p.dst_dim[p.axis] * inner_size * dt_size,
             src_data_ref + (i * p.src_dim[p.axis] + p.begin) * inner_size * dt_size,
             dt_size * inner_size * p.dst_dim[p.axis]);
    } else if (p.step == 2) {
      for (int j = 0; j < p.dst_dim[p.axis]; j++) {
        memcpy(dst_data_ref + (i * p.dst_dim[p.axis] + j) * inner_size * dt_size,
               src_data_ref + (i * p.src_dim[p.axis] + p.begin + j * p.step) * inner_size * dt_size,
               dt_size * inner_size);
      }
    } else {
      SPARSE_LOG(FATAL) << "Unimplemented step of slice: " << p.step;
    }
  }
  std::vector<jd::tensor_desc> ts_descs(io::SIZE);
  ts_descs[io::SRC] = src_desc;
  ts_descs[io::DST] = dst_desc;
  jd::operator_desc slice_d(jd::kernel_kind::slice, jd::kernel_prop::forward_inference, jd::engine_kind::cpu, ts_descs,
                            attr_map);

  return {slice_d, rt_data, rt_data_ref};
}

static auto CasesFp32 = []() {
  std::vector<test_params_t> cases;

  // Config
  std::vector<int64_t> src_shape, dst_shape;

  for (dt data_type : {jd::data_type::fp32, jd::data_type::bf16, jd::data_type::s8}) {
    src_shape = {1, 32, 16, 256};
    dst_shape = {1, 32, 16, 192};
    cases.push_back({3, 0, 1, data_type, src_shape, dst_shape, false});

    src_shape = {1, 32, 16, 256};
    dst_shape = {1, 32, 16, 64};
    cases.push_back({3, 2, 2, data_type, src_shape, dst_shape, false});

    src_shape = {1, 32, 16, 256};
    dst_shape = {1, 10, 16, 256};
    cases.push_back({1, 2, 2, data_type, src_shape, dst_shape, false});

    src_shape = {1, 1, 2048, 2048};
    dst_shape = {1, 1, 32, 2048};
    cases.push_back({2, 1, 1, data_type, src_shape, dst_shape, false});
  }
  return ::testing::ValuesIn(cases);
};

class SliceKernTest : public testing::TestWithParam<test_params_t> {
 protected:
  SliceKernTest() {}
  ~SliceKernTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SliceKernTest, ) {
  test_params_t p = testing::TestWithParam<test_params_t>::GetParam();
  const auto d = gen_data(p);
  EXPECT_TRUE(CheckResult(p.expect_to_fail, d));
  for (auto data : d.rt_data_kern) free(const_cast<void*>(data));
  for (auto data : d.rt_data_ref) free(const_cast<void*>(data));
}

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  const auto& p = tpi.param;
  std::vector<std::string> params;
  params.push_back("axis" + std::to_string(p.axis));
  params.push_back("begin" + std::to_string(p.begin));
  params.push_back("step" + std::to_string(p.step));
  params.push_back(jd::data_type_name.at(p.input_dt));
  params.push_back("src");
  for (auto&& i : p.src_dim) params.push_back(std::to_string(i));
  params.push_back("dst");
  for (auto&& i : p.dst_dim) params.push_back(std::to_string(i));
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SliceKernTest, CasesFp32(), test_suffix);
}  // namespace test
