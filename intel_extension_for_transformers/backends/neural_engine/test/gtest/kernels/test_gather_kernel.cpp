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

#include "unit_test_utils.hpp"
#include "gtest/gtest.h"

namespace jd {
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

  gather_desc gather_d(p.conf);
  gather gather_ker(gather_d);
  gather_ker.execute(p.data);

  // Should compare buffer with different addresses
  if (p.conf.tensor_descs()[2].dtype() == data_type::s8) {
    return compare_data<int8_t>(p.data[2], p.conf.tensor_descs()[2].size(), q.data[2], q.conf.tensor_descs()[2].size());
  } else if (p.conf.tensor_descs()[2].dtype() == data_type::fp32) {
    return compare_data<float>(p.data[2], p.conf.tensor_descs()[2].size(), q.data[2], q.conf.tensor_descs()[2].size());
  } else {
    return compare_data<uint16_t>(p.data[2], p.conf.tensor_descs()[2].size(), q.data[2],
                                  q.conf.tensor_descs()[2].size());
  }
}

class GatherOpTest : public testing::TestWithParam<test_params_t> {
 protected:
  GatherOpTest() {}
  ~GatherOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(GatherOpTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(CheckResult(t));
  t.memoryFree();
}

template <typename T>
void binary_add(void* dst, void* append_src) {
  auto dst_T = reinterpret_cast<T*>(dst);
  auto src_T = reinterpret_cast<T*>(append_src);
  *dst_T = *dst_T + *src_T;
}

std::pair<OpArgs, OpArgs> GenerateFp32Case(std::vector<tensor_desc> const& ts_descs,
                                           std::vector<std::string> append_ops = {}, int src_axis = 0,
                                           int idx_axis = 0) {
  data_type input_dt = ts_descs[0].dtype();
  auto src_shape = ts_descs[0].shape();
  auto idx_shape = ts_descs[1].shape();
  auto dst_shape = ts_descs[2].shape();
  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["idx_axis"] = static_cast<char>(idx_axis + '0');
  attr_map["src_axis"] = static_cast<char>(src_axis + '0');

  // Step 2: Construct Tensor ptr
  auto make_tensor_obj = [&](const tensor_desc a_tensor_config, const bool is_idx) {
    void* tensor_data = sparselib_ut_memo(nullptr, a_tensor_config.size(), a_tensor_config.dtype(), memo_mode::MALLOC);
    void* tensor_data_copy =
        sparselib_ut_memo(nullptr, a_tensor_config.size(), a_tensor_config.dtype(), memo_mode::MALLOC);
    if (is_idx) {
      // init index tensor
      uint32_t seed = 123;
      std::srand(seed);
      for (int i = 0; i < a_tensor_config.size(); ++i) {
        int32_t index = (int32_t)(std::rand() % src_shape[src_axis]);
        memcpy((reinterpret_cast<char*>(tensor_data) + i * 4), &index, sizeof(int32_t));
      }
      memcpy(tensor_data_copy, tensor_data, a_tensor_config.size() * sizeof(int32_t));
    } else {
      // init other tensor
      if (input_dt == data_type::s8 || input_dt == data_type::u8)
        init_vector(static_cast<int8_t*>(tensor_data), a_tensor_config.size());
      else if (input_dt == data_type::fp32)
        init_vector(static_cast<float*>(tensor_data), a_tensor_config.size());
      else
        init_vector(static_cast<uint16_t*>(tensor_data), a_tensor_config.size());
      memcpy(tensor_data_copy, tensor_data, a_tensor_config.size() * get_data_size(input_dt));
    }
    return std::pair<void*, void*>{tensor_data, tensor_data_copy};
  };

  auto src0_tensors = make_tensor_obj(ts_descs[0], false);
  auto src1_tensors = make_tensor_obj(ts_descs[1], true);
  auto dst_data = reinterpret_cast<char*>(sparselib_ut_memo(nullptr, ts_descs[2].size(), input_dt, memo_mode::MALLOC));
  auto dst_data_copy =
      reinterpret_cast<char*>(sparselib_ut_memo(nullptr, ts_descs[2].size(), input_dt, memo_mode::MALLOC));
  std::vector<const void*> rt_data = {src0_tensors.first, src1_tensors.first, dst_data};
  std::vector<const void*> rt_data_copy = {src0_tensors.second, src1_tensors.second, dst_data_copy};
  std::vector<void*> append_vec_copys = {};

  std::vector<binaryop_attr> binaryops;
  std::vector<binaryop_attr> binaryops_copy;
  for (size_t k = 0; k < append_ops.size(); k++) {
    auto& append_op = append_ops[k];
    if (k == 0)
      attr_map["binaryop_list"] = "";
    else
      attr_map["binaryop_list"] += "_";
    if (append_op == "append_sum") {
      auto appends = make_tensor_obj(ts_descs[3 + k], false);
      rt_data.push_back(appends.first);
      rt_data_copy.push_back(appends.second);
      binaryops.push_back({binaryop_alg::add, input_dt});
      binaryops_copy.push_back({binaryop_alg::add, input_dt});
      append_vec_copys.push_back(appends.second);
      attr_map["binaryop_list"] += "add";
    }
  }
  auto src0_data_copy = reinterpret_cast<char*>(src0_tensors.second);
  auto src1_data_copy = reinterpret_cast<const int32_t*>(src1_tensors.second);
  auto src0_shape_copy = src_shape;
  int idx_size = 1;
  for (auto i : idx_shape) idx_size *= i;
  int src_size = 1;
  for (auto i : src_shape) src_size *= i;
  int outer_size = 1;
  for (int i = 0; i < src_axis; i++) outer_size *= src_shape[i];
  for (int i = 0; i < idx_axis; i++) outer_size *= idx_shape[i];
  int inner_size = 1;
  for (int i = src_axis + 1; i < static_cast<int>(src_shape.size()); i++) inner_size *= src_shape[i];

#pragma omp parallel for
  for (int i = 0; i < outer_size; ++i) {
    for (int j = 0; j < idx_shape[idx_axis]; ++j) {
      int indices_val = src1_data_copy[(i * idx_shape[idx_axis] + j) % idx_size];
      memcpy(dst_data_copy + (i * idx_shape[idx_axis] + j) * inner_size * get_data_size(input_dt),
             src0_data_copy + (i * src_shape[src_axis] + indices_val) * inner_size % src_size * get_data_size(input_dt),
             get_data_size(input_dt) * inner_size);
    }
  }
  // TODO(yucheng/zhe): refactor here when postop-injector avaliable
#pragma omp parallel for
  for (int i = 0; i < dst_shape[0]; ++i) {
#pragma omp simd
    for (int j = 0; j < dst_shape[1]; ++j) {
      for (size_t k = 0; k < append_ops.size(); k++) {
        if (append_ops[k] == "append_sum") {
          int broad_cast_i = i;
          if (ts_descs[k + 3].shape()[0] == 1) broad_cast_i = 0;
          if (input_dt == data_type::s8) {
            binary_add<int8_t>(dst_data_copy + (i * dst_shape[1] + j) * get_data_size(input_dt),
                               reinterpret_cast<char*>(append_vec_copys[k]) +
                                   (broad_cast_i * dst_shape[1] + j) * get_data_size(input_dt));
          } else if (input_dt == data_type::u8) {
            binary_add<uint8_t>(dst_data_copy + (i * dst_shape[1] + j) * get_data_size(input_dt),
                                reinterpret_cast<char*>(append_vec_copys[k]) +
                                    (broad_cast_i * dst_shape[1] + j) * get_data_size(input_dt));
          } else if (input_dt == data_type::fp32) {
            binary_add<float>(dst_data_copy + (i * dst_shape[1] + j) * get_data_size(input_dt),
                              reinterpret_cast<char*>(append_vec_copys[k]) +
                                  (broad_cast_i * dst_shape[1] + j) * get_data_size(input_dt));
          }
        }
      }
    }
  }
  operator_desc gather_d(kernel_kind::gather, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, attr_map);
  operator_desc gather_d_copy(kernel_kind::gather, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                              attr_map);
  gather_d.set_binaryop_list(binaryops);
  gather_d_copy.set_binaryop_list(binaryops_copy);
  OpArgs op_args = {rt_data, gather_d};
  OpArgs op_args_copy = {rt_data_copy, gather_d_copy};

  return {op_args, op_args_copy};
}

static auto CasesFp32 = []() {
  std::vector<test_params_t> cases;

  // Config
  std::vector<int64_t> src_shape;
  std::vector<int64_t> idx_shape;
  std::vector<int64_t> dst_shape;
  tensor_desc src0, src1, dst, binary0, binary1;

  src_shape = {4, 4, 1, 8};
  idx_shape = {1};
  dst_shape = {4, 1, 1, 8};
  src0 = {src_shape, data_type::fp32, jd::format_type::undef};
  src1 = {idx_shape, data_type::s32, jd::format_type::undef};
  dst = {dst_shape, data_type::fp32, jd::format_type::undef};
  cases.push_back({GenerateFp32Case({src0, src1, dst}, {}, 1, 0), false});

  src_shape = {61, 76};
  idx_shape = {1, 22, 32};
  dst_shape = {1, 22, 32, 76};
  src0 = {src_shape, data_type::fp32, jd::format_type::undef};
  src1 = {idx_shape, data_type::s32, jd::format_type::undef};
  dst = {dst_shape, data_type::fp32, jd::format_type::undef};
  cases.push_back({GenerateFp32Case({src0, src1, dst}, {}, 0, 2), false});

  for (data_type dt : {data_type::fp32, data_type::bf16, data_type::s8}) {
    for (int inner_size : {1024, 1000}) {
      src_shape = {30522, inner_size, 1, 1};
      idx_shape = {256};
      dst_shape = {idx_shape[0]};
      for (size_t i = 1; i < src_shape.size(); i++) dst_shape.push_back(src_shape[i]);
      src0 = {src_shape, dt, jd::format_type::undef};
      src1 = {idx_shape, data_type::s32, jd::format_type::undef};
      dst = {dst_shape, dt, jd::format_type::undef};
      binary0 = dst;
      binary1 = binary0;
      cases.push_back({GenerateFp32Case({src0, src1, dst}, {}), false});
      cases.push_back({GenerateFp32Case({src0, src1, dst, binary0}, {"append_sum"}), false});
      cases.push_back({GenerateFp32Case({src0, src1, dst, binary0, binary1}, {"append_sum", "append_sum"}), false});

      dst_shape = {8, 32 * inner_size};
      dst = {dst_shape, dt, jd::format_type::undef};
      binary0 = dst;
      binary1 = binary0;
      cases.push_back({GenerateFp32Case({src0, src1, dst}, {}), false});
      cases.push_back({GenerateFp32Case({src0, src1, dst, binary0}, {"append_sum"}), false});
      cases.push_back({GenerateFp32Case({src0, src1, dst, binary0, binary1}, {"append_sum", "append_sum"}), false});

      binary0 = {{1, 32 * inner_size}, dt, jd::format_type::undef};
      binary1 = binary0;
      cases.push_back({GenerateFp32Case({src0, src1, dst, binary0}, {"append_sum"}), false});
      cases.push_back({GenerateFp32Case({src0, src1, dst, binary0, binary1}, {"append_sum", "append_sum"}), false});
    }
  }

  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_desc = tpi.param.args.first.conf.tensor_descs();
  auto& src_shape = tensor_desc[0].shape();
  auto& idx_shape = tensor_desc[1].shape();
  auto& dst_shape = tensor_desc[2].shape();
  auto attrs_map = tpi.param.args.first.conf.attrs();

  auto add_dt_info = [&](const std::string& tensor_dt) {
    switch (tensor_desc[0].dtype()) {
      case data_type::s8:
        params.push_back(tensor_dt + "_s8");
        break;
      case data_type::fp32:
        params.push_back(tensor_dt + "_fp32");
        break;
      case data_type::u8:
        params.push_back(tensor_dt + "_u8");
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
  add_dt_info("src1");
  for (auto&& i : idx_shape) params.push_back(std::to_string(i));
  add_dt_info("dst");
  for (auto&& i : dst_shape) params.push_back(std::to_string(i));
  for (size_t i = 3; i < tensor_desc.size(); i++) {
    add_dt_info("add" + std::to_string(i - 3));
    for (auto&& j : tensor_desc[i].shape()) params.push_back(std::to_string(j));
  }
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, GatherOpTest, CasesFp32(), test_suffix);
}  // namespace jd
