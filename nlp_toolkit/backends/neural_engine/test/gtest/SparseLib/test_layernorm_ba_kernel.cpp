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

namespace jd {
struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> data;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  auto tensor_desc = op_desc.tensor_descs();
  int row = tensor_desc[0].reduce_rows();
  int col = tensor_desc[0].shape().back();
  auto src_dt = tensor_desc[0].dtype();
  LOG_IF(FATAL, src_dt != data_type::fp32);
  auto dst_dt = tensor_desc[1].dtype();

  float* src = reinterpret_cast<float*>(const_cast<void*>(rt_data[0]));
  float* alpha = reinterpret_cast<float*>(const_cast<void*>(rt_data[2]));
  float* beta = reinterpret_cast<float*>(const_cast<void*>(rt_data[3]));

  void* dst_data = const_cast<void*>(rt_data[1]);
  auto dst_fp32 = static_cast<float*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);

  std::vector<float> v_mean, v_var;
  for (int i = 0; i < col; i++) {
    // calculate mean.
    float mean = 0;
    for (int j = 0; j < row; j++) mean += src[j * col + i];
    mean /= row;
    v_mean.push_back(mean);
    // calculate var
    float var = 0;
    for (int j = 0; j < row; j++) var += (src[j * col + i] - mean) * (src[j * col + i] - mean);
    var /= row;
    v_var.push_back(var);
    var += 1e-5;
    var = sqrt(var);
    var = 1 / var;
    // calculate layernorm.
    for (int j = 0; j < row; j++) {
      int dst_idx = j * col + i;
      float value = (src[dst_idx] - mean) * var;
      value = alpha[j] * value + beta[j];
      value = apply_postop_list(value, op_desc.apply_postops_list());
      if (dst_dt == data_type::fp32) {
        dst_fp32[dst_idx] = static_cast<float>(value);
      } else if (dst_dt == data_type::s8) {
        dst_s8[dst_idx] = static_cast<int8_t>(value);
      } else if (dst_dt == data_type::u8) {
        dst_u8[dst_idx] = static_cast<uint8_t>(value);
      }
    }
  }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  try {
    const auto& op_desc = p.op_desc;
    layernorm_ba_desc layernorm_ba_desc(op_desc);
    layernorm_ba layernorm_ba_ker(layernorm_ba_desc);
    layernorm_ba_ker.execute(p.data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }

  if (!t.expect_to_fail) {
    get_true_data(q.op_desc, q.data);
    auto buf1 = p.data[1];
    auto size1 = p.op_desc.tensor_descs()[1].size();
    auto buf2 = q.data[1];
    auto size2 = p.op_desc.tensor_descs()[1].size();
    auto dst_type = p.op_desc.tensor_descs()[1].dtype();
    EXPECT_NE(buf1, buf2);
    bool ans = false;
    if (dst_type == data_type::fp32) {
      ans = compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == data_type::u8) {
      ans = compare_data<uint8_t>(buf1, size1, buf2, size2, 1e-2);
    } else if (dst_type == data_type::s8) {
      ans = compare_data<int8_t>(buf1, size1, buf2, size2, 1e-2);
    }
    free(const_cast<void*>(p.data[0]));
    free(const_cast<void*>(p.data[1]));
    free(const_cast<void*>(q.data[0]));
    free(const_cast<void*>(q.data[1]));
    free(const_cast<void*>(q.data[2]));
    free(const_cast<void*>(q.data[3]));
    return ans;
  }
  free(const_cast<void*>(p.data[0]));
  free(const_cast<void*>(p.data[1]));
  free(const_cast<void*>(q.data[0]));
  free(const_cast<void*>(q.data[1]));
  free(const_cast<void*>(q.data[2]));
  free(const_cast<void*>(q.data[3]));
  return false;
}

class LayernormBaKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  LayernormBaKernelTest() {}
  virtual ~LayernormBaKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(LayernormBaKernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<postop_attr>& postop_attr = {}) {
  // malloc memory
  int row = ts_descs[0].reduce_rows();
  int col = ts_descs[0].shape().back();
  int num = row * col;
  void* src = nullptr;
  void* dst = nullptr;
  void* src_ref = nullptr;
  void* dst_ref = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();

  src = sparselib_ut_memo(src, num, in_dt, MALLOC, true);
  dst = sparselib_ut_memo(dst, num, out_dt, MALLOC, true);
  dst = sparselib_ut_memo(dst, num, out_dt, MEMSET);
  src_ref = sparselib_ut_memo(src_ref, num, in_dt, MALLOC, true);
  dst_ref = sparselib_ut_memo(dst_ref, num, out_dt, MALLOC, true);
  dst_ref = sparselib_ut_memo(dst_ref, num, out_dt, MEMSET);
  float* alpha = reinterpret_cast<float*>(aligned_alloc(64, row * sizeof(float)));
  float* beta = reinterpret_cast<float*>(aligned_alloc(64, row * sizeof(float)));

  // init alpha&beta
  for (int i = 0; i < row; i++) alpha[i] = 1 + rand_float_postfix();
  for (int i = 0; i < row; i++) beta[i] = 1 + rand_float_postfix();

  // init matrix.
  const unsigned int seed = 667095;
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      unsigned int seed_tmp = seed + i;
      float rand_val = rand_r(&seed_tmp) % 256 - 128 + rand_float_postfix();
      assign_val(src, in_dt, rand_val, i * col + j);
      assign_val(src_ref, in_dt, rand_val, i * col + j);
    }
  }

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;

  rt_data1.emplace_back(src);
  rt_data1.emplace_back(dst);
  rt_data1.emplace_back(alpha);
  rt_data1.emplace_back(beta);
  rt_data2.emplace_back(src_ref);
  rt_data2.emplace_back(dst_ref);
  rt_data2.push_back(alpha);
  rt_data2.push_back(beta);

  operator_desc layernorm_ba_desc(kernel_kind::layernorm_ba, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                                  op_attrs, postop_attr);

  op_args_t p = {layernorm_ba_desc, rt_data1};
  op_args_t q = {layernorm_ba_desc, rt_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  tensor_desc data_desc0 = {{768, 32}, jd::data_type::fp32, jd::format_type::ba};
  tensor_desc data_desc1 = {{768, 256}, jd::data_type::fp32, jd::format_type::ba};
  tensor_desc data_desc2 = {{128, 2, 128}, jd::data_type::fp32, jd::format_type::ba};
  tensor_desc data_desc3 = {{1024, 224}, jd::data_type::fp32, jd::format_type::ba};
  tensor_desc data_desc4 = {{768, 256}, jd::data_type::s8, jd::format_type::ba};
  tensor_desc affine_data_attr = {{}, jd::data_type::fp32, jd::format_type::ba};

  postop_attr s8_quantize = {data_type::s8,       postop_type::eltwise, postop_alg::quantize, rand_float_postfix(), 0,
                             rand_float_postfix()};
  postop_attr u8_quantize = {data_type::u8,       postop_type::eltwise, postop_alg::quantize, rand_float_postfix(), 0,
                             rand_float_postfix()};

  std::string tensor_shape0 = "728x32";
  std::string tensor_shape1 = "768x224";
  std::string tensor_shape2 = "256x128";
  std::string tensor_shape3 = "1024x256";
  std::string quantize_attrs8 = "s8quantize";
  std::string quantize_attru8 = "u8quantize";

  cases.push_back({gen_case({data_desc0, data_desc0, affine_data_attr}, {{"matrix_shape", tensor_shape0}}), false});
  cases.push_back({gen_case({data_desc1, data_desc1, affine_data_attr}, {{"matrix_shape", tensor_shape1}}), false});
  cases.push_back({gen_case({data_desc2, data_desc2, affine_data_attr}, {{"matrix_shape", tensor_shape2}}), false});
  cases.push_back({gen_case({data_desc3, data_desc3, affine_data_attr}, {{"matrix_shape", tensor_shape3}}), false});
  cases.push_back({gen_case({data_desc1, data_desc4, affine_data_attr},
                            {{"matrix_shape", tensor_shape0}, {"postop_list", quantize_attrs8}}, {s8_quantize}),
                   false});
  cases.push_back({gen_case({data_desc1, data_desc4, affine_data_attr},
                            {{"matrix_shape", tensor_shape0}, {"postop_list", quantize_attru8}}, {u8_quantize}),
                   false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, LayernormBaKernelTest, case_func());
}  // namespace jd
