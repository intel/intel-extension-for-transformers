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
using io = jd::exposed_enum::gather::io;
using dt = jd::data_type;
struct test_data_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data_kern;
  std::vector<const void*> rt_data_ref;
};

struct test_params_t {
  std::vector<dim_t> src_dim;
  std::vector<dim_t> idx_dim;
  std::vector<dim_t> dst_dim;
  std::vector<std::vector<dim_t>> binary_dim;
  dt input_dt;  // data type of src & dst & binary
  std::vector<std::string> append_ops;
  int src_axis; /* = 0 */
  int idx_axis; /* = 0 */
  bool expect_to_fail;
};

bool CheckResult(const bool expect_to_fail, const test_data_t& d) {
  const auto& p = d.rt_data_kern;
  const auto& q = d.rt_data_ref;

  try {
    jd::gather_desc gather_d(d.op_desc);
    jd::gather gather_ker(gather_d);
    gather_ker.execute(p);
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return expect_to_fail;
  }

  // Should compare buffer with different addresses
  EXPECT_NE(p[1], q[1]);
  const auto dst_size = d.op_desc.tensor_descs()[io::DST].size();
  const auto dst_dtype = d.op_desc.tensor_descs()[io::DST].dtype();
  if (dst_dtype == dt::s8) {
    return compare_data<int8_t>(p[io::DST], dst_size, q[io::DST], dst_size);
  } else if (dst_dtype == dt::fp32) {
    return compare_data<float>(p[io::DST], dst_size, q[io::DST], dst_size);
  } else if (dst_dtype == dt::bf16) {
    return compare_data<jd::bfloat16_t>(p[io::DST], dst_size, q[io::DST], dst_size);
  } else {
    SPARSE_LOG(ERROR) << "Unexpected dst type!";
    return false;
  }
}

template <typename T>
void binary_add(void* dst, void* append_src) {
  auto dst_T = reinterpret_cast<T*>(dst);
  auto src_T = reinterpret_cast<T*>(append_src);
  *dst_T = *dst_T + *src_T;
}

test_data_t gen_data(const test_params_t& p) {
  const jd::tensor_desc src_desc{p.src_dim, p.input_dt, jd::plain_format(p.src_dim.size())};
  const jd::tensor_desc idx_desc{p.idx_dim, dt::s32, jd::plain_format(p.idx_dim.size())};
  const jd::tensor_desc dst_desc{p.dst_dim, p.input_dt, jd::plain_format(p.dst_dim.size())};
  const auto bytes_dt = jd::type_size.at(p.input_dt);

  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["idx_axis"] = static_cast<char>(p.idx_axis + '0');
  attr_map["src_axis"] = static_cast<char>(p.src_axis + '0');

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const jd::tensor_desc& ts_desc, const bool is_idx) {
    void* tensor_data = sparselib_ut_memo(nullptr, ts_desc.size(), ts_desc.dtype(), memo_mode::MALLOC);
    void* tensor_data_copy = sparselib_ut_memo(nullptr, ts_desc.size(), ts_desc.dtype(), memo_mode::MALLOC);
    if (is_idx) {
      // init index tensor
      uint32_t seed = 123;
      std::srand(seed);
      for (int i = 0; i < ts_desc.size(); ++i) {
        int32_t index = (int32_t)(std::rand() % p.src_dim[p.src_axis]);
        memcpy((reinterpret_cast<char*>(tensor_data) + i * 4), &index, sizeof(int32_t));
      }
      memcpy(tensor_data_copy, tensor_data, ts_desc.size() * sizeof(int32_t));
    } else {
      // init other tensor
      if (p.input_dt == dt::s8 || p.input_dt == dt::u8)
        init_vector(static_cast<int8_t*>(tensor_data), ts_desc.size());
      else if (p.input_dt == dt::fp32)
        init_vector(static_cast<float*>(tensor_data), ts_desc.size());
      else
        init_vector(static_cast<uint16_t*>(tensor_data), ts_desc.size());
      memcpy(tensor_data_copy, tensor_data, ts_desc.size() * bytes_dt);
    }
    return std::pair<void*, void*>{tensor_data, tensor_data_copy};
  };
  auto src0_tensors = make_tensor_obj(src_desc, false);
  auto src1_tensors = make_tensor_obj(idx_desc, true);
  auto dst_data = reinterpret_cast<char*>(sparselib_ut_memo(nullptr, dst_desc.size(), p.input_dt, memo_mode::MALLOC));
  auto dst_data_copy =
      reinterpret_cast<char*>(sparselib_ut_memo(nullptr, dst_desc.size(), p.input_dt, memo_mode::MALLOC));
  std::vector<const void*> rt_data = {src0_tensors.first, src1_tensors.first, dst_data};
  std::vector<const void*> rt_data_copy = {src0_tensors.second, src1_tensors.second, dst_data_copy};
  std::vector<void*> append_vec_copys = {};

  std::vector<jd::binaryop_attr> binaryops;
  std::vector<jd::binaryop_attr> binaryops_copy;
  for (size_t k = 0; k < p.append_ops.size(); k++) {
    auto& append_op = p.append_ops[k];
    if (k == 0)
      attr_map["binaryop_list"] = "";
    else
      attr_map["binaryop_list"] += "_";
    if (append_op == "append_sum") {
      auto appends = make_tensor_obj({p.binary_dim[k], p.input_dt, jd::plain_format(p.binary_dim[k].size())}, false);
      rt_data.push_back(appends.first);
      rt_data_copy.push_back(appends.second);
      binaryops.push_back({jd::binaryop_alg::add, p.input_dt});
      binaryops_copy.push_back({jd::binaryop_alg::add, p.input_dt});
      append_vec_copys.push_back(appends.second);
      attr_map["binaryop_list"] += "add";
    } else {
      SPARSE_LOG(FATAL) << "Unimplemented binary op test!";
    }
  }
  auto src0_data_copy = reinterpret_cast<char*>(src0_tensors.second);
  auto src1_data_copy = reinterpret_cast<const int32_t*>(src1_tensors.second);
  auto src0_shape_copy = p.src_dim;
  int idx_size = 1;
  for (auto i : p.idx_dim) idx_size *= i;
  int src_size = 1;
  for (auto i : p.src_dim) src_size *= i;
  int outer_size = 1;
  for (int i = 0; i < p.src_axis; i++) outer_size *= p.src_dim[i];
  for (int i = 0; i < p.idx_axis; i++) outer_size *= p.idx_dim[i];
  int inner_size = 1;
  for (size_t i = p.src_axis + 1; i < p.src_dim.size(); i++) inner_size *= p.src_dim[i];

#pragma omp parallel for
  for (int i = 0; i < outer_size; ++i) {
    for (int j = 0; j < p.idx_dim[p.idx_axis]; ++j) {
      int indices_val = src1_data_copy[(i * p.idx_dim[p.idx_axis] + j) % idx_size];
      memcpy(dst_data_copy + (i * p.idx_dim[p.idx_axis] + j) * inner_size * bytes_dt,
             src0_data_copy + (i * p.src_dim[p.src_axis] + indices_val) * inner_size % src_size * bytes_dt,
             bytes_dt * inner_size);
    }
  }
  // TODO(yucheng/zhe): refactor here when postop-injector avaliable
#pragma omp parallel for collapse(2)
  for (int i = 0; i < p.dst_dim[0]; ++i) {
    for (int j = 0; j < p.dst_dim[1]; ++j) {
      for (size_t k = 0; k < p.append_ops.size(); k++) {
        if (p.append_ops[k] == "append_sum") {
          int broad_cast_i = i;
          if (p.binary_dim[k][0] == 1) broad_cast_i = 0;
          if (p.input_dt == dt::s8) {
            binary_add<int8_t>(
                dst_data_copy + (i * p.dst_dim[1] + j) * bytes_dt,
                reinterpret_cast<char*>(append_vec_copys[k]) + (broad_cast_i * p.dst_dim[1] + j) * bytes_dt);
          } else if (p.input_dt == dt::u8) {
            binary_add<uint8_t>(
                dst_data_copy + (i * p.dst_dim[1] + j) * bytes_dt,
                reinterpret_cast<char*>(append_vec_copys[k]) + (broad_cast_i * p.dst_dim[1] + j) * bytes_dt);
          } else if (p.input_dt == dt::fp32) {
            binary_add<float>(
                dst_data_copy + (i * p.dst_dim[1] + j) * bytes_dt,
                reinterpret_cast<char*>(append_vec_copys[k]) + (broad_cast_i * p.dst_dim[1] + j) * bytes_dt);
          }
        } else {
          SPARSE_LOG(FATAL) << "Unexpected binary operation!";
        }
      }
    }
  }
  std::vector<jd::tensor_desc> ts_descs(io::SIZE);
  ts_descs[io::SRC] = src_desc;
  ts_descs[io::IDX] = idx_desc;
  ts_descs[io::DST] = dst_desc;
  {
    int i = 0;
    for (const auto& dim : p.binary_dim)
      ts_descs[io::BINARY0 + (i++)] = {dim, p.input_dt, jd::plain_format(dim.size())};
  }
  jd::operator_desc gather_d(jd::kernel_kind::gather, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                             ts_descs, attr_map);
  gather_d.set_binaryop_list(binaryops);
  return {gather_d, rt_data, rt_data_copy};
}

static auto CasesFp32 = []() {
  std::vector<test_params_t> cases;

  // Config
  std::vector<int64_t> src, idx, dst, binary0, binary1;

  src = {4, 4, 1, 8};
  idx = {1};
  dst = {4, 1, 1, 8};
  cases.push_back({src, idx, dst, {}, dt::fp32, {}, 1, 0, false});

  src = {61, 76};
  idx = {1, 22, 32};
  dst = {1, 22, 32, 76};
  cases.push_back({src, idx, dst, {}, dt::fp32, {}, 0, 2, false});

  for (dt dt : {dt::fp32, dt::bf16, dt::s8}) {
    for (int inner_size : {1024, 1000}) {
      src = {30522, inner_size, 1, 1};
      idx = {256};
      dst = {idx[0]};
      for (size_t i = 1; i < src.size(); i++) dst.push_back(src[i]);
      binary0 = dst;
      binary1 = binary0;
      cases.push_back({src, idx, dst, {}, dt, {}, 0, 0, false});
      cases.push_back({src, idx, dst, {binary0}, dt, {"append_sum"}, 0, 0, false});
      cases.push_back({src, idx, dst, {binary0, binary1}, dt, {"append_sum", "append_sum"}, 0, 0, false});

      dst = {8, 32 * inner_size};
      binary0 = dst;
      binary1 = binary0;
      cases.push_back({src, idx, dst, {}, dt, {}, 0, 0, false});
      cases.push_back({src, idx, dst, {binary0}, dt, {"append_sum"}, 0, 0, false});
      cases.push_back({src, idx, dst, {binary0, binary1}, dt, {"append_sum", "append_sum"}, 0, 0, false});

      binary0 = {1, 32 * inner_size};
      binary1 = binary0;
      cases.push_back({src, idx, dst, {binary0}, dt, {"append_sum"}, 0, 0, false});
      cases.push_back({src, idx, dst, {binary0, binary0}, dt, {"append_sum", "append_sum"}, 0, 0, false});
    }
  }

  return ::testing::ValuesIn(cases);
};

class GatherKernTest : public testing::TestWithParam<test_params_t> {
 protected:
  GatherKernTest() {}
  ~GatherKernTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(GatherKernTest, ) {
  const auto& p = testing::TestWithParam<test_params_t>::GetParam();
  const auto d = gen_data(p);
  EXPECT_TRUE(CheckResult(p.expect_to_fail, d));
  for (auto data : d.rt_data_kern) free(const_cast<void*>(data));
  for (auto data : d.rt_data_ref) free(const_cast<void*>(data));
}

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  const auto& p = tpi.param;
  std::vector<std::string> params;
  params.push_back(jd::data_type_name.at(p.input_dt));
  params.push_back("src");
  for (auto&& i : p.src_dim) params.push_back(std::to_string(i));
  params.push_back("idx");
  for (auto&& i : p.idx_dim) params.push_back(std::to_string(i));
  params.push_back("dst");
  for (auto&& i : p.dst_dim) params.push_back(std::to_string(i));
  for (size_t i = 0; i < p.binary_dim.size(); i++) {
    params.push_back("add" + std::to_string(i));
    for (auto&& j : p.binary_dim[i]) params.push_back(std::to_string(j));
  }
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, GatherKernTest, CasesFp32(), test_suffix);
}  // namespace test
