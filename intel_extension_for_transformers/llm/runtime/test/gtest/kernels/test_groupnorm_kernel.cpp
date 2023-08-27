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
#include "src/cpu/kernels/groupnorm_ref.hpp"
#include "kernels/exposed_enum.hpp"
#include "interface.hpp"

namespace test {

using idx = jd::exposed_enum::groupnorm::io;

struct op_args_t {
  jd::operator_desc op_desc;
  std::shared_ptr<std::vector<jd::bfloat16_t>> bf16_src;
  std::shared_ptr<std::vector<float>> fp32_src;
  std::shared_ptr<std::vector<jd::bfloat16_t>> bf16_dst;
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
  auto data_type = op_desc.tensor_descs()[0].dtype();
  auto op_attr = op_desc.attrs();
  std::vector<const void*> data1(idx::SIZE);
  std::vector<const void*> data2(idx::SIZE);
  try {
    jd::groupnorm_desc groupnorm_desc(op_desc);
    jd::groupnorm groupnorm_ker(groupnorm_desc);

    if (data_type == jd::data_type::bf16) {
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
    std::shared_ptr<const jd::kernel_desc_t> groupnorm_ref_desc;
    jd::kernel_desc_t::create<jd::groupnorm_ref_kd_t>(groupnorm_ref_desc, q.op_desc);
    std::shared_ptr<const jd::kernel_t> groupnorm_ref_ker;
    jd::kernel_t::create<jd::groupnorm_ref_k_t, jd::groupnorm_ref_kd_t>(groupnorm_ref_ker, groupnorm_ref_desc);
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
    if (data_type == jd::data_type::bf16) {
      ans = compare_data<jd::bfloat16_t>(buf1, size, buf2, size, 1e-2);
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

std::pair<op_args_t, op_args_t> gen_case(const std::vector<jd::tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<jd::postop_attr>& postop_attr = {}) {
  jd::operator_desc groupnorm_desc(jd::kernel_kind::groupnorm, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                                   ts_descs, op_attrs, postop_attr);

  auto gen_data = [](auto type, int size, float bound1, float bound2, bool clear = false) {
    auto pad_size = (size + 63) / 64 * 64;
    auto ptr = std::shared_ptr<std::vector<decltype(type)>>(new std::vector<decltype(type)>(pad_size));
    if (!clear) {
      init_vector(ptr->data(), size, bound1, bound2);
      init_vector(ptr->data() + size, pad_size - size, 0, 0);
    }
    return ptr;
  };

  auto src_size = ts_descs[idx::SRC].size();
  auto channel_size = ts_descs[idx::GAMMA].size();
  auto fp32_src = gen_data(static_cast<float>(1), src_size, 0.f, 5.f);
  auto gamma = gen_data(static_cast<float>(1), channel_size, 1.f, 5.f);
  auto beta = gen_data(static_cast<float>(1), channel_size, 1.f, 5.f);
  std::shared_ptr<std::vector<jd::bfloat16_t>> bf16_src(new std::vector<jd::bfloat16_t>(src_size));
  cast_from_float_array<jd::bfloat16_t>((*fp32_src).data(), reinterpret_cast<void*>(bf16_src->data()), src_size);
  auto fp32_dst = gen_data(static_cast<float>(1), src_size, 0.f, 0.f, true);
  auto bf16_dst = gen_data(static_cast<jd::bfloat16_t>(1.f), src_size, 0.f, 0.f, true);
  auto correct_fp32_dst = gen_data(static_cast<float>(1), src_size, 0.f, 0.f, true);
  auto correct_bf16_dst = gen_data(static_cast<jd::bfloat16_t>(1.f), src_size, 0.f, 0.f, true);

  op_args_t p = {groupnorm_desc, bf16_src, fp32_src, bf16_dst, fp32_dst, gamma, beta};
  op_args_t q = {groupnorm_desc, bf16_src, fp32_src, correct_bf16_dst, correct_fp32_dst, gamma, beta};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  std::vector<jd::data_type> dt_types = {jd::data_type::bf16, jd::data_type::fp32};
  std::vector<std::vector<dim_t>> problem_size = {{1, 8, 16, 16}, {1, 8, 64, 64}, {2, 8, 16, 16},
                                                  {2, 8, 64, 64}, {2, 8, 7, 7},   {2, 8, 111, 101}};
  jd::postop_attr swish_attr = {jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::swish, 2.f};
  for (auto&& dt : dt_types) {
    for (auto&& shape : problem_size) {
      jd::tensor_desc src_desc = {shape, dt, jd::format_type::abcd};
      jd::tensor_desc dst_desc = {shape, dt, jd::format_type::abcd};
      jd::tensor_desc gamma_desc = {{shape[1]}, jd::data_type::fp32, jd::format_type::a};
      jd::tensor_desc beta_desc = {{shape[1]}, jd::data_type::fp32, jd::format_type::a};
      jd::tensor_desc workspace_desc = {{}, jd::data_type::fp32, jd::format_type::a};
      cases.push_back({gen_case({src_desc, dst_desc, gamma_desc, beta_desc, workspace_desc},
                                {{"eps", "0.01"}, {"groups", "4"}}, {swish_attr}),
                       false});
    }
  }
  return ::testing::ValuesIn(cases);
};
std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  auto attrs = tpi.param.args.first.op_desc.attrs();
  auto dt = tpi.param.args.first.op_desc.tensor_dtypes()[0];
  const auto shapes = tpi.param.args.first.op_desc.tensor_shapes();
  std::vector<std::string> params;
  for (auto num : shapes[0]) {
    params.push_back(std::to_string(num));
  }
  switch (dt) {
    case jd::data_type::fp32:
      params.push_back("fp32");
      break;
    case jd::data_type::bf16:
      params.push_back("bf16");
      break;
    case jd::data_type::s8:
      params.push_back("s8");
      break;
    case jd::data_type::u8:
      params.push_back("u8");
      break;
    default:
      break;
  }
  return join_str(params, "_");
}
INSTANTIATE_TEST_SUITE_P(SparseLib, GroupNormKernelTest, case_func(), test_suffix);
}  // namespace test
