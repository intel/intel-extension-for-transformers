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
#include "interface.hpp"
#include "benchmark_utils.hpp"

#define exp_ln_flt_max_f 0x42b17218
#define exp_ln_flt_min_f 0xc2aeac50
#define inf_float 0x7f800000
#define zmm_size 512

enum memo_mode { MALLOC, MEMSET };

unsigned short int fp32_2_bf16(float float_val) { return (*reinterpret_cast<unsigned int*>(&float_val)) >> 16; }

float bf16_2_fp32(unsigned short int bf16_val) {
  unsigned int ans = bf16_val << 16;
  return *reinterpret_cast<float*>(&ans);
}

int get_data_width(jd::data_type dtype) {
  int data_width = 0;
  switch (dtype) {
    case jd::data_type::fp32:
      data_width = 4;
      break;
    case jd::data_type::bf16:
      data_width = 2;
      break;
    default:
      throw std::runtime_error(std::string("sparselib_ut_malloc error:unsupport data type."));
      break;
  }
  return data_width;
}

void assign_val(void* ptr, jd::data_type dtype, int val, int idx) {
  switch (dtype) {
    case jd::data_type::fp32:
      *((float*)ptr + idx) = val;
      break;
    case jd::data_type::bf16:
      *((unsigned short int*)ptr + idx) = fp32_2_bf16(val);
      enum memo_mode { MALLOC, MEMSET };

      break;
    default:
      std::runtime_error(std::string("assign_val:unsupport this dtype."));
  }
}

void* sparselib_ut_memo(void* ptr, int num, jd::data_type dtype, memo_mode mode) {
  int data_width = get_data_width(dtype);
  switch (mode) {
    case MALLOC:
      ptr = malloc(num * data_width); /* code */
      break;
    case MEMSET:
      memset(ptr, 0, num * data_width);
      break;
    default:
      break;
  }
  return ptr;
}


namespace jd {
int num = 1;
struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> data;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rf_data) {
  unsigned int max = exp_ln_flt_max_f;
  unsigned int min = exp_ln_flt_min_f;
  float fmax = *reinterpret_cast<float*>(&max);
  float fmin = *reinterpret_cast<float*>(&min);

  float* src = (float*)rf_data[0];
  float* dst = (float*)rf_data[1];
  for (int i = 0; i < num; i++) {
    if (src[i] > fmax) {
      dst[i] = inf_float;
    } else if (src[i] < fmin) {
      dst[i] = 0;
    } else {
      dst[i] = expf(src[i]);
    }
  }
  return;
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;

  try {
    const auto& op_desc = p.op_desc;
    postop_desc postop_desc(op_desc);
    postop postop_kern(postop_desc);
    //postop_kern.execute(p.data);
    benchmarkOrExecute(&postop_kern, p.data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      std::cout << "kernel exception occurred." << std::endl;
      return false;
    }
  }

  if (!t.expect_to_fail) {
    get_true_data(q.op_desc, q.data);
    void* buf1;
    auto buf2 = q.data[1];
    auto dtype = p.op_desc.tensor_descs()[0].dtype();
    float err_rate;
    if (dtype == jd::data_type::fp32) {
      buf1 = const_cast<void*>(p.data[1]);
      err_rate = 1e-2;
    } else if (dtype == jd::data_type::bf16) {
      buf1 = reinterpret_cast<float*>(malloc(num * sizeof(float)));
      auto bf16_buf1 = const_cast<void*>(p.data[1]);
      for (int i = 0; i < num; i++) {
        *((float*)buf1 + i) = bf16_2_fp32(*((unsigned short int*)bf16_buf1 + i));
      }
      err_rate = 5;
    }

    EXPECT_NE(buf1, buf2);
    return compare_data<float>(buf1, num, buf2, num, err_rate);
  }
}

class PostopDefaultKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  PostopDefaultKernelTest() {}
  virtual ~PostopDefaultKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(PostopDefaultKernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const jd::kernel_kind ker_kind, const jd::kernel_prop ker_prop,
                                         const jd::engine_kind eng_kind, const std::vector<tensor_desc>& ts_descs) {
  std::unordered_map<std::string, std::string> op_attrs = {{"post_op", "exp"}};
  num = 1;
  for (auto&& i : ts_descs[0].shape()) num *= i;
  void* src;
  void* dst;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;
  auto dtype = ts_descs[0].dtype();
  src = sparselib_ut_memo(src, num, dtype, MALLOC);
  dst = sparselib_ut_memo(dst, num, dtype, MALLOC);
  sparselib_ut_memo(dst, num, dtype, MEMSET);
  float* src_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));
  float* dst_ref = reinterpret_cast<float*>(malloc(num * sizeof(float)));

  const unsigned int seed = 667095;
  memset(dst_ref, 0, num * sizeof(float));
  for (int i = 0; i < num; i++) {
    unsigned int seed_tmp = seed + i;
    int rand_val = rand_r(&seed_tmp) % 5;
    assign_val(src, dtype, rand_val, i);
    src_ref[i] = rand_val;
  }

  operator_desc postop_desc(ker_kind, ker_prop, eng_kind, ts_descs, op_attrs);
  std::vector<const void*> rf_data1;
  std::vector<const void*> rf_data2;

  rf_data1.emplace_back((void*)src);
  rf_data1.emplace_back((void*)dst);
  rf_data2.emplace_back((void*)src_ref);
  rf_data2.emplace_back((void*)dst_ref);

  op_args_t p = {postop_desc, rf_data1};
  op_args_t q = {postop_desc, rf_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  tensor_desc data0_desc = {{1024, 1024}, jd::data_type::fp32, jd::format_type::undef};
  tensor_desc data1_desc = {{1024, 1024}, jd::data_type::bf16, jd::format_type::undef};
  cases.push_back(
      {gen_case(kernel_kind::postop, kernel_prop::forward_inference, engine_kind::cpu, {data0_desc, data0_desc}), false});
  cases.push_back(
      {gen_case(kernel_kind::postop, kernel_prop::forward_inference, engine_kind::cpu, {data1_desc, data1_desc}), false});
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, PostopDefaultKernelTest, case_func());
}  // namespace jd
