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
#include "src/cpu/kernels/mha_dense_ref.hpp"
#include "unit_test_utils.hpp"

namespace test {
using io = jd::exposed_enum::mha_dense::io;

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
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data_kern;
  std::vector<const void*> rt_data_ref;
};

bool check_result(const int nthr, const bool expect_to_fail, const test_data_t& d) {
  try {
    std::shared_ptr<const jd::kernel_desc_t> dynamic_quant_mha_ref_desc;
    jd::kernel_desc_t::create<jd::mha_dense_ref_kd_t>(dynamic_quant_mha_ref_desc, d.op_desc);
    std::shared_ptr<const jd::kernel_t> dynamic_quant_mha_ref_kernel;
    jd::kernel_t::create<jd::mha_dense_ref_k_t, jd::mha_dense_ref_kd_t>(dynamic_quant_mha_ref_kernel,
                                                                        dynamic_quant_mha_ref_desc);
    const auto workspace_q = aligned_allocator_t<char>::allocate(dynamic_quant_mha_ref_kernel->get_workspace_size());
    auto data_q = d.rt_data_kern;
    data_q[io::WORKSPACE] = workspace_q;
    dynamic_quant_mha_ref_kernel->execute(data_q);
    aligned_allocator_t<char>::deallocate(workspace_q);

    n_thread_t with_n_thread(nthr);
    jd::mha_dense_desc mha_dense_desc(d.op_desc);
    jd::mha_dense dynq10n_mha_dense_kernel(mha_dense_desc);
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

test_data_t gen_data(const dim_t batch_size, const dim_t head_num, const dim_t sl_M, const dim_t head_size,
                     const dim_t sl_N, const bool dynamic_shape, const int nthr) {
  n_thread_t with_n_thread(nthr);
  std::unordered_map<std::string, std::string> op_attrs{};
  op_attrs["approx_exp"] = "True";
  op_attrs["stable_softmax"] = "False";

  // Step 2: Configure tensor shape
  std::vector<jd::tensor_desc> ts_descs(io::SIZE, {{}, jd::data_type::undef, jd::format_type::undef});
  ts_descs[io::SRC_Q] = {{batch_size, sl_M, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::SRC_K] = {{batch_size, sl_N, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::SRC_V] = {{batch_size, sl_N, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::DST] = {{batch_size, sl_M, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::BINARY_ADD] = {{batch_size, 1, 1, sl_N}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::ATT_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::Q_SCALE] = {{batch_size, sl_M}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::K_SCALE] = {{batch_size, sl_N}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::V_SCALE] = {{batch_size, sl_N}, jd::data_type::fp32, jd::format_type::ab};
  ts_descs[io::DST_SCALE] = {{batch_size, sl_M}, jd::data_type::fp32, jd::format_type::ab};
  if (dynamic_shape) {
    const jd::tensor_desc shape_ts_desc{{1}, jd::data_type::s32, jd::format_type::a};
    ts_descs[io::BATCH_SIZE] = shape_ts_desc;
    ts_descs[io::HEAD_NUM] = shape_ts_desc;
    ts_descs[io::HEAD_SIZE] = shape_ts_desc;
    ts_descs[io::M] = shape_ts_desc;
    ts_descs[io::N] = shape_ts_desc;
  }
  // Step 2: Construct runtime data
  std::vector<const void*> rt_data(io::SIZE, nullptr);
  std::vector<const void*> rt_data_cpy(io::SIZE, nullptr);
  std::vector<std::pair<const void*, const void*>> data_pairs(io::SIZE, {nullptr, nullptr});
  data_pairs[io::SRC_Q] = make_data_obj(ts_descs[io::SRC_Q], false, {-128, 127}, 0.f, jd::format_type::uncoded, nullptr,
                                        std::uniform_int_distribution<>()(rand_gen));
  data_pairs[io::SRC_K] = make_data_obj(ts_descs[io::SRC_K], false, {-128, 127}, 0.f, jd::format_type::uncoded, nullptr,
                                        std::uniform_int_distribution<>()(rand_gen));
  data_pairs[io::SRC_V] = make_data_obj(ts_descs[io::SRC_V], false, {-128, 127}, 0.f, jd::format_type::uncoded, nullptr,
                                        std::uniform_int_distribution<>()(rand_gen));
  data_pairs[io::BINARY_ADD] = make_data_obj(ts_descs[io::BINARY_ADD], true, {0, 0}, 0.f, jd::format_type::uncoded,
                                             nullptr, std::uniform_int_distribution<>()(rand_gen));
  for (int ibs = 0; ibs < batch_size; ibs++) {
    const auto batch_mask = reinterpret_cast<float*>(const_cast<void*>(data_pairs[io::BINARY_ADD].first)) + sl_N * ibs;
    const auto batch_mask_cpy =
        reinterpret_cast<float*>(const_cast<void*>(data_pairs[io::BINARY_ADD].second)) + sl_N * ibs;
    const dim_t valid_sl = std::uniform_int_distribution<>(sl_N / 2, sl_N)(rand_gen);
    std::fill_n(batch_mask + valid_sl, sl_N - valid_sl, -1000.f);
    std::fill_n(batch_mask_cpy + valid_sl, sl_N - valid_sl, -1000.f);
  }
  data_pairs[io::ATT_SCALE] = make_data_obj(ts_descs[io::ATT_SCALE], true, {0, 0}, 0.f, jd::format_type::uncoded,
                                            nullptr, std::uniform_int_distribution<>()(rand_gen));
  reinterpret_cast<float*>(const_cast<void*>(data_pairs[io::ATT_SCALE].first))[0] = 1.f;
  reinterpret_cast<float*>(const_cast<void*>(data_pairs[io::ATT_SCALE].second))[0] = 1.f;
  data_pairs[io::Q_SCALE] = make_data_obj(ts_descs[io::Q_SCALE], false, {0.001, 0.01}, 0.f, jd::format_type::uncoded,
                                          nullptr, std::uniform_int_distribution<>()(rand_gen));
  data_pairs[io::K_SCALE] = make_data_obj(ts_descs[io::K_SCALE], false, {0.001, 0.01}, 0.f, jd::format_type::uncoded,
                                          nullptr, std::uniform_int_distribution<>()(rand_gen));
  data_pairs[io::V_SCALE] = make_data_obj(ts_descs[io::V_SCALE], false, {0.001, 0.01}, 0.f, jd::format_type::uncoded,
                                          nullptr, std::uniform_int_distribution<>()(rand_gen));
  data_pairs[io::DST] = make_data_obj(ts_descs[io::DST], false, {-128, 127});
  data_pairs[io::DST_SCALE] = make_data_obj(ts_descs[io::DST_SCALE], false, {-128, 127}, 0.f, jd::format_type::uncoded,
                                            nullptr, std::uniform_int_distribution<>()(rand_gen));

  if (dynamic_shape) {
    data_pairs[io::BATCH_SIZE] =
        make_data_obj(ts_descs[io::BATCH_SIZE], false, {static_cast<float>(batch_size), static_cast<float>(batch_size)},
                      0.f, jd::format_type::uncoded, nullptr, std::uniform_int_distribution<>()(rand_gen));
    data_pairs[io::HEAD_NUM] =
        make_data_obj(ts_descs[io::HEAD_NUM], false, {static_cast<float>(head_num), static_cast<float>(head_num)}, 0.f,
                      jd::format_type::uncoded, nullptr, std::uniform_int_distribution<>()(rand_gen));
    data_pairs[io::HEAD_SIZE] =
        make_data_obj(ts_descs[io::HEAD_SIZE], false, {static_cast<float>(head_size), static_cast<float>(head_size)},
                      0.f, jd::format_type::uncoded, nullptr, std::uniform_int_distribution<>()(rand_gen));
    data_pairs[io::M] = make_data_obj(ts_descs[io::M], false, {static_cast<float>(sl_M), static_cast<float>(sl_M)}, 0.f,
                                      jd::format_type::uncoded, nullptr, std::uniform_int_distribution<>()(rand_gen));
    data_pairs[io::N] = make_data_obj(ts_descs[io::N], false, {static_cast<float>(sl_N), static_cast<float>(sl_N)}, 0.f,
                                      jd::format_type::uncoded, nullptr, std::uniform_int_distribution<>()(rand_gen));
  }
  for (std::underlying_type<io>::type idx = 0; idx < io::SIZE; ++idx) {
    rt_data[idx] = data_pairs[idx].first;
    rt_data_cpy[idx] = data_pairs[idx].second;
  }

  // hide shapes in ts_descs if use dynamic shape
  if (dynamic_shape)
    for (io idx : {io::SRC_Q, io::SRC_K, io::SRC_V, io::DST, io::BINARY_ADD, io::ATT_SCALE, io::Q_SCALE, io::K_SCALE,
                   io::V_SCALE, io::DST_SCALE})
      ts_descs[idx] = {
          std::vector<dim_t>(ts_descs[idx].shape().size(), -1),
          ts_descs[idx].dtype(),
          ts_descs[idx].ftype(),
      };
  jd::operator_desc op_desc(jd::kernel_kind::mha_dense, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);

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
  cases.push_back({1, 1, 4096, 40, 4096, false, 0});
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
}  // namespace test
