//  Copyright (c) 2022 Intel Corporation
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
#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "kernels/groupnorm_ref.hpp"

namespace jd {

using idx = exposed_enum::groupnorm::io;

struct op_args_t {
  operator_desc op_desc;
  std::shared_ptr<std::vector<bfloat16_t>> bf16_src;
  std::shared_ptr<std::vector<float>> fp32_src;
  std::shared_ptr<std::vector<bfloat16_t>> bf16_dst;
  std::shared_ptr<std::vector<float>> fp32_dst;
  std::shared_ptr<std::vector<float>> gamma;
  std::shared_ptr<std::vector<float>> beta;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  const auto& op_desc = p.op_desc;
  auto dt = op_desc.tensor_descs()[0].dtype();
  auto op_attr = op_desc.attrs();
  std::vector<const void*> data1(idx::SIZE, nullptr);
  std::vector<const void*> data2(idx::SIZE, nullptr);
  try {
    groupnorm_desc groupnorm_desc(op_desc);
    groupnorm groupnorm_ker(groupnorm_desc);

    if (dt == data_type::bf16) {
      data1[idx::SRC] = p.bf16_src->data();
      data1[idx::DST] = p.bf16_dst->data();
      data2[idx::SRC] = q.bf16_src->data();
      data2[idx::DST] = q.bf16_dst->data();
    } else {
      data1[idx::SRC] = p.fp32_src->data();
      data1[idx::DST] = p.fp32_dst->data();
      data2[idx::SRC] = q.fp32_src->data();
      data2[idx::DST] = q.fp32_dst->data();
    }

    std::shared_ptr<char> tmp_buf(reinterpret_cast<char*>(malloc(groupnorm_ker.get_workspace_size())),
                                  [](char* ptr) { free(ptr); });

    data1[idx::GAMMA] = p.gamma->data();
    data1[idx::BETA] = p.beta->data();
    data1[idx::WORKSPACE] = tmp_buf.get();
    data2[idx::GAMMA] = q.gamma->data();
    data2[idx::BETA] = q.beta->data();
    data2[idx::WORKSPACE] = tmp_buf.get();

    groupnorm_ker.execute(data1);
    std::shared_ptr<const kernel_desc_t> groupnorm_ref_desc;
    kernel_desc_t::create<groupnorm_ref_kd_t>(groupnorm_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> groupnorm_ref_ker;
    kernel_t::create<groupnorm_ref_k_t, groupnorm_ref_kd_t>(groupnorm_ref_ker, groupnorm_ref_desc);
    groupnorm_ref_ker->execute(data2);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }

  if (!t.expect_to_fail) {
    auto buf1 = data1[idx::DST];
    auto size = p.fp32_dst->size();
    auto buf2 = data2[idx::DST];
    bool ans = false;
    if (dt == data_type::bf16) {
      ans = compare_data<bfloat16_t>(buf1, size, buf2, size, 1e-2);
    } else {
      ans = compare_data<float>(buf1, size, buf2, size, 5e-3);
    }
    return ans;
  }
  return false;
}

class GroupNormKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  GroupNormKernelTest() {}
  virtual ~GroupNormKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(GroupNormKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs) {
  operator_desc groupnorm_desc(kernel_kind::groupnorm, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                               op_attrs);

  auto gen_data = [](auto type, int size, float bound1, float bound2, bool clear = false) {
    auto ptr = std::shared_ptr<std::vector<decltype(type)>>(new std::vector<decltype(type)>(size, 0));
    if (!clear) init_vector(ptr->data(), ptr->size(), bound1, bound2);
    return ptr;
  };

  auto src_size = ts_descs[idx::SRC].size();
  auto channel_size = ts_descs[idx::GAMMA].size();
  auto fp32_src = gen_data(static_cast<float>(1), src_size, 0.f, 5.f);
  auto gamma = gen_data(static_cast<float>(1), channel_size, 1.f, 5.f);
  auto beta = gen_data(static_cast<float>(1), channel_size, 1.f, 5.f);
  std::shared_ptr<std::vector<bfloat16_t>> bf16_src(new std::vector<bfloat16_t>(src_size, 0));
  cast_from_float_array<bfloat16_t>(*fp32_src, bf16_src->data(), src_size);
  auto fp32_dst = gen_data(static_cast<float>(1), src_size, 0.f, 0.f, true);
  auto bf16_dst = gen_data(static_cast<bfloat16_t>(1), src_size, 0.f, 0.f, true);
  auto correct_fp32_dst = gen_data(static_cast<float>(1), src_size, 0.f, 0.f, true);
  auto correct_bf16_dst = gen_data(static_cast<bfloat16_t>(1), src_size, 0.f, 0.f, true);

  op_args_t p = {groupnorm_desc, bf16_src, fp32_src, bf16_dst, fp32_dst, gamma, beta};
  op_args_t q = {groupnorm_desc, bf16_src, fp32_src, correct_bf16_dst, correct_fp32_dst, gamma, beta};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  std::vector<std::vector<int64_t>> problem_size = {{1, 8, 16, 16}, {1, 8, 64, 64}, {2, 8, 16, 16}, {2, 8, 64, 64}};
  for (auto&& shape : problem_size) {
    tensor_desc src_desc = {shape, jd::data_type::bf16, jd::format_type::abcd};
    tensor_desc dst_desc = {shape, jd::data_type::bf16, jd::format_type::abcd};
    tensor_desc gamma_desc = {{shape[1]}, jd::data_type::fp32, jd::format_type::a};
    tensor_desc beta_desc = {{shape[1]}, jd::data_type::fp32, jd::format_type::a};
    tensor_desc workspace_desc = {{}, jd::data_type::fp32, jd::format_type::a};
    cases.push_back({gen_case({src_desc, dst_desc, gamma_desc, beta_desc, workspace_desc},
                              {{"eps", "0"}, {"groups", "4"}}),
                     false});
  }

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(SparseLib, GroupNormKernelTest, case_func());
}  // namespace jd
