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
#include <cmath>

#include "gtest/gtest.h"
#include "src/cpu/kernels/mha_dense_ref.hpp"
#include "unit_test_utils.hpp"
#include "interface.hpp"

namespace test {
using io = jd::exposed_enum::mha_dense::io;

struct test_params_t {
  dim_t bs;
  dim_t sl_m;
  dim_t sl_n;
  dim_t head_num;
  dim_t head_size;
  bool has_pmask;
  bool has_badd;
  bool stable_softmax;
  int nthr;
  bool expect_to_fail;
};

struct test_data_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data_kern;
  std::vector<const void*> rt_data_ref;
};

static std::mt19937 rand_gen(1);

inline static std::string TestParam2str(testing::TestParamInfo<test_params_t> tpi) {
  auto&& p = tpi.param;
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(tpi.param.nthr));
  params.push_back(std::to_string(p.bs));                      // bs
  params.push_back(std::to_string(p.sl_m));                    // sl_m
  params.push_back(std::to_string(p.sl_n));                    // sl_n
  params.push_back(std::to_string(p.head_num));                // head_num
  params.push_back(std::to_string(p.head_size));               // head_size
  if (p.has_pmask) params.push_back("pmask");                  // has_pmask
  if (p.has_badd) params.push_back("badd");                    // has_badd
  params.push_back(p.stable_softmax ? "stable" : "unstable");  // stable_softmax
  return join_str(params, "_");
}

bool check_result(const int nthr, const bool expect_to_fail, const test_data_t& d) {
  try {
    std::shared_ptr<const jd::kernel_desc_t> mha_dense_ref_desc;
    jd::kernel_desc_t::create<jd::mha_dense_ref_kd_t>(mha_dense_ref_desc, d.op_desc);
    std::shared_ptr<const jd::kernel_t> mha_dense_ref_kernel;
    jd::kernel_t::create<jd::mha_dense_ref_k_t, jd::mha_dense_ref_kd_t>(mha_dense_ref_kernel, mha_dense_ref_desc);
    mha_dense_ref_kernel->execute(d.rt_data_ref);

    n_thread_t with_n_thread(nthr);
    jd::mha_dense_desc mha_dense_desc(d.op_desc);
    jd::mha_dense mha_dense_kernel(mha_dense_desc);
    const auto tmp_p = std::shared_ptr<char>(aligned_allocator_t<char>::allocate(mha_dense_kernel.get_workspace_size()),
                                             [](char* ptr) { aligned_allocator_t<char>::deallocate(ptr); });
    auto data_p(d.rt_data_kern);
    data_p[io::WORKSPACE] = tmp_p.get();
    mha_dense_kernel.execute(data_p);
  } catch (const std::exception& e) {
    SPARSE_LOG(ERROR) << e.what();
    return expect_to_fail;
  }

  if (!expect_to_fail) {
    auto buf1 = d.rt_data_kern[io::DST];
    auto dst_size = d.op_desc.tensor_descs()[io::DST].size();
    auto buf2 = d.rt_data_ref[io::DST];
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    switch (d.op_desc.tensor_descs()[io::DST].dtype()) {
      case jd::data_type::bf16:
        return compare_data<jd::bfloat16_t>(buf1, dst_size, buf2, dst_size, 1e-2);
      default:
        SPARSE_LOG(ERROR) << "Unexpected dst type";
    }
  }
  return false;
}
test_data_t gen_data(const test_params_t& p) {
  n_thread_t with_nthr(p.nthr);
  std::vector<jd::tensor_desc> ts_descs(io::SIZE, jd::tensor_desc{});
  ts_descs[io::SRC_Q] = {{p.bs, p.sl_m, p.head_num, p.head_size}, jd::data_type::bf16, jd::format_type::abcd};
  ts_descs[io::SRC_K] = {{p.bs, p.sl_n, p.head_num, p.head_size}, jd::data_type::bf16, jd::format_type::abcd};
  ts_descs[io::SRC_V] = {{p.bs, p.sl_n, p.head_num, p.head_size}, jd::data_type::bf16, jd::format_type::abcd};
  ts_descs[io::DST] = {{p.bs, p.sl_m, p.head_num, p.head_size}, jd::data_type::bf16, jd::format_type::abcd};
  ts_descs[io::ATT_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  // TODO(Yi): enable broadcasting
  if (p.has_badd) ts_descs[io::BINARY_ADD] = {{1, 1, p.sl_m, p.sl_n}, jd::data_type::fp32, jd::format_type::abcd};
  if (p.has_pmask) ts_descs[io::MASK] = {{p.bs}, jd::data_type::s32, jd::format_type::a};

  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["approx_exp"] = "True";
  attr_map["stable_softmax"] = p.stable_softmax ? "True" : "False";

  // Step 2: Construct Tensor ptr
  const float att_scale_val = 1.f / std::sqrt(p.sl_n);
  const std::pair<const void*, const void*> empty_tensor_data{nullptr, nullptr};
  auto Qs = make_data_obj(ts_descs[io::SRC_Q], false, {-1, 1});
  auto Ks = make_data_obj(ts_descs[io::SRC_K], false, {-1, 1});
  auto Vs = make_data_obj(ts_descs[io::SRC_V], false, {-1, 1});
  auto dsts = make_data_obj(ts_descs[io::DST], true);
  auto att_scales = make_data_obj(ts_descs[io::ATT_SCALE], false, {att_scale_val, att_scale_val});
  auto badds = p.has_badd ? make_data_obj(ts_descs[io::BINARY_ADD], false, {-1.f, 1.f}) : empty_tensor_data;
  auto pmasks =
      p.has_pmask ? make_data_obj(ts_descs[io::MASK], false, {1, static_cast<float>(p.sl_n)}) : empty_tensor_data;

  std::vector<const void*> data_p(io::SIZE, nullptr);
  data_p[io::SRC_Q] = Qs.first;
  data_p[io::SRC_K] = Ks.first;
  data_p[io::SRC_V] = Vs.first;
  data_p[io::DST] = dsts.first;
  data_p[io::ATT_SCALE] = att_scales.first;
  if (p.has_badd) data_p[io::BINARY_ADD] = badds.first;
  if (p.has_pmask) data_p[io::MASK] = pmasks.first;

  std::vector<const void*> data_q(io::SIZE, nullptr);
  data_q[io::SRC_Q] = Qs.second;
  data_q[io::SRC_K] = Ks.second;
  data_q[io::SRC_V] = Vs.second;
  data_q[io::DST] = dsts.second;
  data_q[io::ATT_SCALE] = att_scales.second;
  if (p.has_badd) data_q[io::BINARY_ADD] = badds.second;
  if (p.has_pmask) data_q[io::MASK] = pmasks.second;

  jd::operator_desc op_desc(jd::kernel_kind::mha_dense, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, attr_map);
  return {op_desc, data_p, data_q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  // case param: bs sl_m sl_n head_num head_size has_pmask has_badd stable_softmax nthr expect_to_fail
  cases.push_back(test_params_t{1, 64, 64, 1, 32, false, true, false, 1, false});
  cases.push_back(test_params_t{2, 64, 64, 1, 32, false, true, false, 1, false});
  cases.push_back(test_params_t{2, 1024, 1024, 1, 40, false, true, false, 0, false});
  cases.push_back(test_params_t{2, 1024, 1024, 1, 80, false, true, false, 0, false});
  cases.push_back(test_params_t{2, 256, 256, 1, 160, false, true, false, 0, false});

  cases.push_back(test_params_t{1, 64, 32, 1, 32, false, true, false, 1, false});
  cases.push_back(test_params_t{1, 64, 33, 1, 32, false, true, false, 1, false});
  cases.push_back(test_params_t{1, 64, 61, 1, 32, false, true, false, 1, false});
  cases.push_back(test_params_t{1, 1, 61, 1, 32, false, true, false, 1, false});
  cases.push_back(test_params_t{1, 1, 61, 1, 32, true, true, false, 1, false});
  cases.push_back(test_params_t{1, 1, 35, 1, 64, true, true, false, 1, false});
  cases.push_back(test_params_t{2, 1, 42, 1, 64, false, true, false, 1, false});
  cases.push_back(test_params_t{1, 64, 33, 1, 32, true, true, false, 3, false});
  cases.push_back(test_params_t{1, 64, 33, 1, 32, true, true, false, 0, false});

  // stable diffusion cases
  cases.push_back(test_params_t{2, 1024, 77, 1, 40, false, false, false, 0, false});
  cases.push_back(test_params_t{2, 1024, 77, 1, 80, false, false, false, 0, false});
  cases.push_back(test_params_t{2, 256, 77, 1, 160, false, false, false, 0, false});
  cases.push_back(test_params_t{2, 1024, 77, 1, 40, false, false, true, 0, false});
  cases.push_back(test_params_t{2, 1024, 77, 1, 80, false, false, true, 0, false});
  cases.push_back(test_params_t{2, 256, 77, 1, 160, false, false, true, 0, false});

  cases.push_back(test_params_t{1, 256, 9216, 1, 64, false, false, true, 0, false});

  // gpt neox cases
  cases.push_back(test_params_t{4, 1, 32, 64, 96, false, false, true, 0, false});
  cases.push_back(test_params_t{4, 1, 33, 64, 96, false, false, true, 0, false});

  return ::testing::ValuesIn(cases);
};

class MhaDenseBf16KernTest : public testing::TestWithParam<test_params_t> {
 protected:
  MhaDenseBf16KernTest() {}
  ~MhaDenseBf16KernTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MhaDenseBf16KernTest, ) {
  const test_params_t& p = testing::TestWithParam<test_params_t>::GetParam();
  const auto d = gen_data(p);
  EXPECT_TRUE(check_result(p.nthr, p.expect_to_fail, d));

  for (auto data : {d.rt_data_kern, d.rt_data_ref})
    for (auto p : data)
      if (p != nullptr) delete[] reinterpret_cast<const char*>(p);
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseBf16KernTest, case_func(), TestParam2str);
}  // namespace test
