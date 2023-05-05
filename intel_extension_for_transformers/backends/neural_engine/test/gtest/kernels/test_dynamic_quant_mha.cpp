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
#include "kernels/mha_dense_ref.hpp"
#include "unit_test_utils.hpp"

namespace jd {
using dt = data_type;
using ft = format_type;
using io = exposed_enum::mha_dense::io;

static std::mt19937 rand_gen(1);

struct test_params_t {
  dim_t batch_size;
  dim_t head_num;
  dim_t sl_M;
  dim_t head_size;
  dim_t sl_N;
  bool dynamic_shape;
  int nthr;
  bool expect_to_fail;
};
struct test_data_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data_kern;
  std::vector<const void*> rt_data_ref;
};

bool check_result(const int nthr, const bool expect_to_fail, const test_data_t& d) {
  try {
    std::shared_ptr<const kernel_desc_t> dynamic_quant_mha_ref_desc;
    kernel_desc_t::create<mha_dense_ref_kd_t>(dynamic_quant_mha_ref_desc, d.op_desc);
    std::shared_ptr<const kernel_t> dynamic_quant_mha_ref_kernel;
    kernel_t::create<mha_dense_ref_k_t, mha_dense_ref_kd_t>(dynamic_quant_mha_ref_kernel, dynamic_quant_mha_ref_desc);
    const auto workspace_q = aligned_allocator_t<char>::allocate(dynamic_quant_mha_ref_kernel->get_workspace_size());
    auto data_q = d.rt_data_kern;
    data_q[io::WORKSPACE] = workspace_q;
    dynamic_quant_mha_ref_kernel->execute(data_q);
    aligned_allocator_t<char>::deallocate(workspace_q);

    n_thread_t with_n_thread(nthr);
    mha_dense_desc mha_dense_desc(d.op_desc);
    mha_dense dynq10n_mha_dense_kernel(mha_dense_desc);
    const auto workspace_p = aligned_allocator_t<char>::allocate(dynq10n_mha_dense_kernel.get_workspace_size());
    auto data_p = d.rt_data_ref;
    data_p[io::WORKSPACE] = workspace_p;
    dynq10n_mha_dense_kernel.execute(data_p);
    aligned_allocator_t<char>::deallocate(workspace_p);
  } catch (const std::exception& e) {
    if (expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!expect_to_fail) {
    const auto dst = d.rt_data_kern[io::DST];
    const auto dst_ref = d.rt_data_ref[io::DST];
    const auto dst_scale = d.rt_data_kern[io::DST_SCALE];
    const auto dst_scale_ref = d.rt_data_ref[io::DST_SCALE];
    EXPECT_NE(dst, dst_ref);              // Should compare buffer with different addresses
    EXPECT_NE(dst_scale, dst_scale_ref);  // output s8
    const auto dst_size = d.op_desc.tensor_descs()[io::DST].size();
    const auto dst_scale_size = d.op_desc.tensor_descs()[io::DST_SCALE].size();
    return compare_data<int8_t>(dst, dst_size, dst_ref, dst_size, 8e-2f) &&
           compare_data<float>(dst_scale, dst_scale_size, dst_scale_ref, dst_scale_size, 1e-3);
  }
  return false;
}

const void* make_data_obj(const tensor_desc desc, const float min_val, const float max_val) {
  int elem_num = std::accumulate(desc.shape().begin(), desc.shape().end(), dim_t{1}, std::multiplies<dim_t>());
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
  int elem_num = std::accumulate(desc.shape().begin(), desc.shape().end(), dim_t{1}, std::multiplies<dim_t>());
  const int bytes_size = elem_num * type_size[desc.dtype()];
  void* data_ptr = new uint8_t[bytes_size];
  memcpy(data_ptr, src, bytes_size);
  return data_ptr;
}

test_data_t gen_data(const dim_t batch_size, const dim_t head_num, const dim_t sl_M, const dim_t head_size,
                     const dim_t sl_N, const bool dynamic_shape, const int nthr) {
  n_thread_t with_n_thread(nthr);
  std::unordered_map<std::string, std::string> op_attrs{};
  op_attrs["approx_exp"] = "True";
  op_attrs["stable_softmax"] = "False";

  // Step 2: Configure tensor shape
  std::vector<tensor_desc> ts_descs(io::SIZE, {{}, dt::undef, ft::undef});
  ts_descs[io::SRC_Q] = {{batch_size, sl_M, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::SRC_K] = {{batch_size, sl_N, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::SRC_V] = {{batch_size, sl_N, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::DST] = {{batch_size, sl_M, head_num, head_size}, dt::s8, ft::abcd};
  ts_descs[io::BINARY_ADD] = {{batch_size, 1, 1, sl_N}, dt::fp32, ft::ab};
  ts_descs[io::ATT_SCALE] = {{1}, dt::fp32, ft::a};
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
  std::vector<const void*> rt_data(io::SIZE, nullptr);
  rt_data[io::SRC_Q] = make_data_obj(ts_descs[io::SRC_Q], -128, 127);
  rt_data[io::SRC_K] = make_data_obj(ts_descs[io::SRC_K], -128, 127);
  rt_data[io::SRC_V] = make_data_obj(ts_descs[io::SRC_V], -128, 127);
  rt_data[io::BINARY_ADD] = make_data_obj(ts_descs[io::BINARY_ADD], 0, 0);
  rt_data[io::ATT_SCALE] = make_data_obj(ts_descs[io::ATT_SCALE], 0, 0);
  reinterpret_cast<float*>(const_cast<void*>(rt_data[io::ATT_SCALE]))[0] = 1.f;
  rt_data[io::Q_SCALE] = make_data_obj(ts_descs[io::Q_SCALE], 0.001, 0.01);
  rt_data[io::K_SCALE] = make_data_obj(ts_descs[io::K_SCALE], 0.001, 0.01);
  rt_data[io::V_SCALE] = make_data_obj(ts_descs[io::V_SCALE], 0.001, 0.01);
  rt_data[io::DST] = make_data_obj(ts_descs[io::DST], -128, 127);  // random dst and scale to be overwrite
  rt_data[io::DST_SCALE] = make_data_obj(ts_descs[io::DST_SCALE], INT32_MIN, INT32_MAX);
  for (int ibs = 0; ibs < batch_size; ibs++) {
    const auto batch_mask = reinterpret_cast<float*>(const_cast<void*>(rt_data[io::BINARY_ADD])) + sl_N * ibs;
    const dim_t valid_sl = std::uniform_int_distribution<>(sl_N / 2, sl_N)(rand_gen);
    std::fill_n(batch_mask + valid_sl, sl_N - valid_sl, -1000.f);
  }
  if (dynamic_shape) {
    rt_data[io::BATCH_SIZE] = make_data_obj(ts_descs[io::BATCH_SIZE], batch_size, batch_size);
    rt_data[io::HEAD_NUM] = make_data_obj(ts_descs[io::HEAD_NUM], head_num, head_num);
    rt_data[io::HEAD_SIZE] = make_data_obj(ts_descs[io::HEAD_SIZE], head_size, head_size);
    rt_data[io::M] = make_data_obj(ts_descs[io::M], sl_M, sl_M);
    rt_data[io::N] = make_data_obj(ts_descs[io::N], sl_N, sl_N);
  }

  std::vector<const void*> rt_data_cpy(io::SIZE, nullptr);
  for (std::underlying_type<io>::type idx = 0; idx < io::SIZE; ++idx)
    if (rt_data[idx] != nullptr) rt_data_cpy[idx] = copy_data_obj(ts_descs[idx], rt_data[idx]);

  // hide shapes in ts_descs if use dynamic shape
  if (dynamic_shape)
    for (io idx : {io::SRC_Q, io::SRC_K, io::SRC_V, io::DST, io::BINARY_ADD, io::ATT_SCALE, io::Q_SCALE, io::K_SCALE,
                   io::V_SCALE, io::DST_SCALE})
      ts_descs[idx] = {
          std::vector<dim_t>(ts_descs[idx].shape().size(), -1),
          ts_descs[idx].dtype(),
          ts_descs[idx].ftype(),
      };
  operator_desc op_desc(kernel_kind::mha_dense, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, op_attrs);

  return {op_desc, rt_data, rt_data_cpy};
}

static auto case_func = []() {
  google::InitGoogleLogging("DynQuantMHAKernTest");
  std::vector<test_params_t> cases;
  std::vector<int> nthr_cases = {1, 3, 0};
  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);
    cases.push_back({2, 4, 48, 32, 48, false, nthr});
  }
  cases.push_back({2, 4, 48, 40, 48, false, 0});
  cases.push_back({2, 4, 64, 32, 77, false, 0});
  cases.push_back({2, 4, 64, 40, 77, false, 0});
  cases.push_back({2, 4, 1024, 80, 77, false, 0});
  cases.push_back({2, 4, 256, 160, 77, false, 0});
  cases.push_back({1, 1, 256, 160, 256, false, 0});
  if (isa_available(amx_int8)) cases.push_back({1, 1, 4096, 40, 4096, false, 0});
  return ::testing::ValuesIn(cases);
};

class DynQuantMHAKernTest : public testing::TestWithParam<test_params_t> {
 protected:
  DynQuantMHAKernTest() {}
  virtual ~DynQuantMHAKernTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(DynQuantMHAKernTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  const auto d = gen_data(t.batch_size, t.head_num, t.sl_M, t.head_size, t.sl_N, t.dynamic_shape, t.nthr);
  EXPECT_TRUE(check_result(t.nthr, t.expect_to_fail, d));
  for (auto data : {d.rt_data_kern, d.rt_data_ref})
    for (auto p : data)
      if (p != nullptr) delete[] reinterpret_cast<const char*>(p);
}
static std::string test_suffix(const testing::TestParamInfo<test_params_t>& tpi) {
  auto&& p = tpi.param;
  std::vector<std::string> params_str;
  params_str.push_back("c" + std::to_string(p.nthr));
  params_str.push_back(std::to_string(p.batch_size));
  params_str.push_back(std::to_string(p.head_num));
  params_str.push_back(std::to_string(p.sl_M));
  params_str.push_back(std::to_string(p.head_size));
  params_str.push_back(std::to_string(p.sl_N));
  return join_str(params_str, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, DynQuantMHAKernTest, case_func(), test_suffix);
}  // namespace jd
