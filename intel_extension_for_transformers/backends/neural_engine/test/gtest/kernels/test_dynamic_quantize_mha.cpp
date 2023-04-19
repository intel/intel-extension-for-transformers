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

#include <omp.h>

#include <exception>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "interface.hpp"
#include "kernels/dynamic_quantize_mha_ref.hpp"
#include "kernels/dynamic_quantize_mha_types.hpp"
#include "unit_test_utils.hpp"

namespace jd {

using dt = jd::data_type;
using ft = jd::format_type;
using io = ssd::dynamic_quantize_mha_io::io;

static std::mt19937 rand_gen(1);

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
  int nthr;  // 0 for not touching OMP_NUM_THREADS and using what set outside
  dim_t batch_size;
  dim_t head_num;
  dim_t sl_M;
  dim_t head_size;
  dim_t sl_N;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    std::shared_ptr<const kernel_desc_t> dynamic_quantize_mha_ref_desc;
    kernel_desc_t::create<dynamic_quantize_mha_ref_kd_t>(dynamic_quantize_mha_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> dynamic_quantize_mha_ref_kernel;
    kernel_t::create<dynamic_quantize_mha_ref_k_t, dynamic_quantize_mha_ref_kd_t>(dynamic_quantize_mha_ref_kernel,
                                                                                  dynamic_quantize_mha_ref_desc);
    const auto tmp_q = aligned_allocator_t<char>::allocate(dynamic_quantize_mha_ref_kernel->get_workspace_size());
    auto data_q = q.rt_data;
    data_q[io::TMP] = tmp_q;
    dynamic_quantize_mha_ref_kernel->execute(data_q);
    aligned_allocator_t<char>::deallocate(tmp_q);

    n_thread_t with_n_thread(p.nthr);
    dynamic_quantize_mha_desc dynamic_quantize_mha_desc(p.op_desc);
    dynamic_quantize_mha dynamic_quantize_mha_kernel(dynamic_quantize_mha_desc);
    const auto tmp_p = aligned_allocator_t<char>::allocate(dynamic_quantize_mha_ref_kernel->get_workspace_size());
    auto data_p = p.rt_data;
    dynamic_quantize_mha_kernel.execute(data_p);
    aligned_allocator_t<char>::deallocate(tmp_p);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    const auto dst = p.rt_data[io::DST];
    const auto dst_ref = q.rt_data[io::DST];
    const auto dst_scale = p.rt_data[io::DST_SCALE];
    const auto dst_scale_ref = q.rt_data[io::DST_SCALE];
    EXPECT_NE(dst, dst_ref);              // Should compare buffer with different addresses
    EXPECT_NE(dst_scale, dst_scale_ref);  // output s8
    const auto dst_size = p.batch_size * p.head_num * p.head_size * p.sl_M;
    const auto dst_scale_size = p.batch_size * p.sl_M;
    return compare_data<int8_t>(dst, dst_size, dst_ref, dst_size, 2e-2) &&
           compare_data<float>(dst_scale, dst_scale_size, dst_scale_ref, dst_scale_size, 1e-3);
  }
  return false;
}

class DynQuantMHAKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  DynQuantMHAKernelTest() {}
  virtual ~DynQuantMHAKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(DynQuantMHAKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
  for (auto data : {t.args.first.rt_data, t.args.second.rt_data})
    for (auto p : data)
      if (p != nullptr) delete[] reinterpret_cast<const char*>(p);
}

const void* make_data_obj(const tensor_desc desc, const float min_val, const float max_val) {
  int elem_num = std::accumulate(desc.shape().begin(), desc.shape().end(), 1, std::multiplies<dim_t>());
  const int bytes_size = elem_num * type_size[desc.dtype()];
  void* data_ptr = nullptr;
  if (min_val == 0 && max_val == 0) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    const auto seed = std::uniform_int_distribution<>()(rand_gen);
    if (desc.dtype() == dt::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, min_val, max_val, seed);
    } else if (desc.dtype() == dt::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, min_val, max_val, seed);
    } else if (desc.dtype() == dt::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, min_val, max_val, seed);
    } else if (desc.dtype() == dt::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, min_val, max_val, seed);
    } else {
      SPARSE_LOG(FATAL) << "Unexpected dt!";
    }
  }
  return data_ptr;
}
const void* copy_data_obj(const tensor_desc desc, const void* src) {
  int elem_num = std::accumulate(desc.shape().begin(), desc.shape().end(), 1, std::multiplies<dim_t>());
  const int bytes_size = elem_num * type_size[desc.dtype()];
  void* data_ptr = new uint8_t[bytes_size];
  memcpy(data_ptr, src, bytes_size);
  return data_ptr;
}

std::pair<op_args_t, op_args_t> gen_case(const dim_t batch_size, const dim_t head_size, const dim_t head_num,
                                         const dim_t sl_M, const dim_t sl_N, const bool dynamic_shape,
                                         const int nthr = 0,
                                         std::unordered_map<std::string, std::string> op_attrs = {}) {
  // Step 2: Configure tensor shape
  std::vector<tensor_desc> ts_descs(io::dynamic_quantize_mha_io_MAX + 1, {{}, dt::undef, ft::undef});
  ts_descs[io::Q] = {{batch_size, sl_M, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::K] = {{batch_size, sl_N, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::V] = {{batch_size, sl_N, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::DST] = {{batch_size, sl_M, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::MASK] = {{batch_size, sl_N}, dt::fp32, ft::ab};
  ts_descs[io::Q_SCALE] = {{batch_size, sl_M}, dt::fp32, ft::ab};
  ts_descs[io::K_SCALE] = {{batch_size, sl_N}, dt::fp32, ft::ab};
  ts_descs[io::V_SCALE] = {{batch_size, sl_N}, dt::fp32, ft::ab};
  ts_descs[io::DST_SCALE] = {{batch_size, sl_M}, dt::fp32, ft::ab};
  if (dynamic_shape) {
    const tensor_desc shape_ts_desc{{1}, dt::s32, ft::a};
    ts_descs[io::BATCH_SIZE] = shape_ts_desc;
    ts_descs[io::HEAD_NUM] = shape_ts_desc;
    ts_descs[io::HEAD_SIZE] = shape_ts_desc;
    ts_descs[io::M] = shape_ts_desc;
    ts_descs[io::N] = shape_ts_desc;
  }

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data(io::dynamic_quantize_mha_io_MAX + 1, nullptr);
  rt_data[io::Q] = make_data_obj(ts_descs[io::Q], -128, 127);
  rt_data[io::K] = make_data_obj(ts_descs[io::K], -128, 127);
  rt_data[io::V] = make_data_obj(ts_descs[io::V], -128, 127);
  rt_data[io::MASK] = make_data_obj(ts_descs[io::MASK], 0, 0);
  rt_data[io::Q_SCALE] = make_data_obj(ts_descs[io::Q_SCALE], 0.001, 0.01);
  rt_data[io::K_SCALE] = make_data_obj(ts_descs[io::K_SCALE], 0.001, 0.01);
  rt_data[io::V_SCALE] = make_data_obj(ts_descs[io::V_SCALE], 0.001, 0.01);
  rt_data[io::DST] = make_data_obj(ts_descs[io::DST], -128, 127);  // random dst and scale to be overwrite
  rt_data[io::DST_SCALE] = make_data_obj(ts_descs[io::DST_SCALE], INT32_MIN, INT32_MAX);
  for (int ibs = 0; ibs < batch_size; ibs++) {
    const auto batch_mask = reinterpret_cast<float*>(const_cast<void*>(rt_data[io::MASK])) + sl_N * ibs;
    const dim_t valid_sl = std::uniform_int_distribution<>(sl_N / 2, sl_N)(rand_gen);
    std::fill_n(batch_mask + valid_sl, sl_N - valid_sl, -1000);
  }
  if (dynamic_shape) {
    rt_data[io::BATCH_SIZE] = make_data_obj(ts_descs[io::BATCH_SIZE], batch_size, batch_size);
    rt_data[io::HEAD_NUM] = make_data_obj(ts_descs[io::HEAD_NUM], head_num, head_num);
    rt_data[io::HEAD_SIZE] = make_data_obj(ts_descs[io::HEAD_SIZE], head_size, head_size);
    rt_data[io::M] = make_data_obj(ts_descs[io::M], sl_M, sl_M);
    rt_data[io::N] = make_data_obj(ts_descs[io::N], sl_N, sl_N);
  }

  std::vector<const void*> rt_data_cpy(io::dynamic_quantize_mha_io_MAX + 1, nullptr);
  for (std::underlying_type<io>::type idx = 0; idx <= io::dynamic_quantize_mha_io_MAX; ++idx)
    if (rt_data[idx] != nullptr) rt_data_cpy[idx] = copy_data_obj(ts_descs[idx], rt_data[idx]);

  // hide shapes in ts_descs if use dynamic shape
  if (dynamic_shape)
    for (io idx : {io::Q, io::K, io::V, io::DST, io::MASK, io::Q_SCALE, io::K_SCALE, io::V_SCALE, io::DST_SCALE})
      ts_descs[idx] = {
          std::vector<dim_t>(ts_descs[idx].shape().size(), -1),
          ts_descs[idx].dtype(),
          ts_descs[idx].ftype(),
      };

  operator_desc op_desc(kernel_kind::dynamic_quantize_mha, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                        op_attrs);

  // Step 3: op_args_t testcase pair
  op_args_t op_args_p = {op_desc, rt_data, nthr, batch_size, head_num, sl_M, head_size, sl_N};
  op_args_t op_args_q = {op_desc, rt_data_cpy, nthr, batch_size, head_num, sl_M, head_size, sl_N};
  return {op_args_p, op_args_q};
}

static auto case_func = []() {
  google::InitGoogleLogging("DynQuantMHAKernelTest");
  std::vector<int> nthr_cases = {1, 3, 0};
  std::vector<test_params_t> cases;
  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);
    cases.push_back({gen_case(2, 32, 4, 64, 64, false, nthr, {})});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  const auto op_args = tpi.param.args.first;

  params.push_back("c" + std::to_string(static_cast<int>(op_args.nthr)));
  params.push_back(std::to_string(op_args.batch_size));
  params.push_back(std::to_string(op_args.head_num));
  params.push_back(std::to_string(op_args.sl_M));
  params.push_back(std::to_string(op_args.head_size));
  params.push_back(std::to_string(op_args.sl_N));
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, DynQuantMHAKernelTest, case_func(), test_suffix);
}  // namespace jd
