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
  int badd_dim /* = 0*/;
  jd::data_type dt_dst /* = jd::data_type::u8*/;
  jd::format_type ft_kv /* = jd::format_type::u8*/;
  int nthr;
  bool expect_to_fail;
};

struct test_data_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data_kern;
  std::vector<const void*> rt_data_ref;
};

static std::mt19937 rand_gen(1);

inline static std::string TestParam2str(const testing::TestParamInfo<test_params_t>& tpi) {
  auto&& p = tpi.param;
  std::vector<std::string> params_str;
  params_str.push_back("c" + std::to_string(p.nthr));
  params_str.push_back(std::to_string(p.bs));                    // bs
  params_str.push_back(std::to_string(p.sl_m));                  // sl_m
  params_str.push_back(std::to_string(p.sl_n));                  // sl_n
  params_str.push_back(std::to_string(p.head_num));              // head_num
  params_str.push_back(std::to_string(p.head_size));             // head_size
  params_str.push_back("badddim" + std::to_string(p.badd_dim));  // badddim
  params_str.push_back(jd::data_type_name.at(p.dt_dst) + std::string{"dst"});
  params_str.push_back(jd::format_type_name.at(p.ft_kv));  // kv_ft
  return join_str(params_str, "_");
}

bool check_result(const int nthr, const bool expect_to_fail, const test_data_t& d) {
  try {
    n_thread_t with_n_thread(nthr);
    jd::mha_dense_desc mha_dense_desc(d.op_desc);
    jd::mha_dense mha_dense_kernel(mha_dense_desc);
    const auto tmp_p = std::shared_ptr<char>(aligned_allocator_t<char>::allocate(mha_dense_kernel.get_workspace_size()),
                                             [](char* ptr) { aligned_allocator_t<char>::deallocate(ptr); });
    auto data_p = d.rt_data_kern;
    data_p[io::WORKSPACE] = tmp_p.get();
    mha_dense_kernel.execute(data_p);

    std::shared_ptr<const jd::kernel_desc_t> mha_dense_ref_desc;
    jd::kernel_desc_t::create<jd::mha_dense_ref_kd_t>(mha_dense_ref_desc, d.op_desc);
    std::shared_ptr<const jd::kernel_t> mha_dense_ref_kernel;
    jd::kernel_t::create<jd::mha_dense_ref_k_t, jd::mha_dense_ref_kd_t>(mha_dense_ref_kernel, mha_dense_ref_desc);
    mha_dense_ref_kernel->execute(d.rt_data_ref);
  } catch (const std::exception& e) {
    SPARSE_LOG(ERROR) << e.what();
    return expect_to_fail;
  }

  if (!expect_to_fail) {
    auto buf1 = d.rt_data_kern[io::DST];
    auto size1 = d.op_desc.tensor_descs()[io::DST].size();
    auto buf2 = d.rt_data_ref[io::DST];
    auto size2 = d.op_desc.tensor_descs()[io::DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    switch (d.op_desc.tensor_descs()[io::DST].dtype()) {
      case jd::data_type::fp32:
        return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
      case jd::data_type::s32:
        return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
      case jd::data_type::u8:
        return compare_data<uint8_t>(buf1, size1, buf2, size2, 4e-3);
      case jd::data_type::s8:
        return compare_data<int8_t>(buf1, size1, buf2, size2, 4e-3);
      case jd::data_type::bf16:
        return compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 1e-1);  // TODO(Yi): can we use samller tolerance?
      default:
        SPARSE_LOG(ERROR) << "Unexpected dst type";
    }
  }
  return false;
}

std::pair<const void*, const void*> make_tensor_obj(const jd::tensor_desc& ts_desc, float min_value, float max_value) {
  dim_t elem_num = std::accumulate(ts_desc.shape().begin(), ts_desc.shape().end(), 1LL, std::multiplies<dim_t>());
  int bytes_size = elem_num * jd::type_size[ts_desc.dtype()];
  void* data_ptr = nullptr;
  if (min_value == 0.f && max_value == 0.f) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    const auto seed = std::uniform_int_distribution<>()(rand_gen);
    if (ts_desc.dtype() == jd::data_type::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, min_value, max_value, seed);
    } else if (ts_desc.dtype() == jd::data_type::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, min_value, max_value, seed);
    } else if (ts_desc.dtype() == jd::data_type::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, min_value, max_value, seed);
    } else if (ts_desc.dtype() == jd::data_type::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, min_value, max_value, seed);
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
std::pair<const void*, const void*> make_tensor_obj(const jd::tensor_desc& ts_desc) {
  return make_tensor_obj(ts_desc, -10, 10);
}
std::pair<const void*, const void*> make_tensor_obj(const jd::tensor_desc& ts_desc, float value) {
  return make_tensor_obj(ts_desc, value, value);
}

test_data_t gen_data(const dim_t bs, const dim_t sl_m, const dim_t sl_n, const dim_t head_num,
                     const dim_t head_size, int badd_dim = 0, const jd::data_type dt_dst = jd::data_type::u8,
                     const jd::format_type kv_ft = jd::format_type::abcd) {
  std::vector<dim_t> badd_fullshape = {bs, head_num, sl_m, sl_n};
  std::vector<jd::tensor_desc> ts_descs(io::SIZE, jd::tensor_desc{});
  ts_descs[io::SRC_Q] = {{bs, sl_m, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::SRC_K] = {{bs, sl_n, head_num, head_size}, jd::data_type::s8, kv_ft};
  ts_descs[io::SRC_V] = {{bs, sl_n, head_num, head_size}, jd::data_type::s8, kv_ft};
  ts_descs[io::MASK] = {{bs}, jd::data_type::s32, jd::format_type::a};
  ts_descs[io::DST] = {{bs, sl_m, head_num, head_size}, dt_dst, jd::format_type::abcd};
  if (badd_dim > 0) {
    SPARSE_LOG_IF(FATAL, badd_dim > 4) << "Unsupported binary add dimention";
    ts_descs[io::BINARY_ADD] = {std::vector<dim_t>(badd_fullshape.cend() - badd_dim, badd_fullshape.cend()),
                                jd::data_type::fp32, jd::plain_format(badd_dim)};
  }
  ts_descs[io::ATT_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::Q_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::K_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::V_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::SRC_DST_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::SRC_DST_ZP] = {{1}, jd::data_type::fp32, jd::format_type::a};

  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["approx_exp"] = "True";
  attr_map["stable_softmax"] = "True";
  attr_map["softmax_rescale"] = std::to_string(float{UINT8_MAX});  // TODO(Yi): Use 255?

  // Step 2: Construct Tensor ptr
  auto Qs = make_tensor_obj(ts_descs[io::SRC_Q]);
  auto Ks = make_tensor_obj(ts_descs[io::SRC_K]);
  auto Vs = make_tensor_obj(ts_descs[io::SRC_V]);
  auto masks = make_tensor_obj(ts_descs[io::MASK], 1, sl_n);
  auto dsts = make_tensor_obj(ts_descs[io::DST], 0);

  auto badds = badd_dim > 0 ? make_tensor_obj(ts_descs[io::BINARY_ADD], -1.f, 1.f)
                            : std::pair<const void*, const void*>{nullptr, nullptr};

  auto att_scales = make_tensor_obj(ts_descs[io::ATT_SCALE], 1.f);  // TODO(Yi): 1/sqrt
  auto q_scales = make_tensor_obj(ts_descs[io::Q_SCALE], 1.1f);
  auto k_scales = make_tensor_obj(ts_descs[io::K_SCALE], 0.9f);
  auto v_scales = make_tensor_obj(ts_descs[io::V_SCALE], 1.2f);
  auto dst_scales = make_tensor_obj(ts_descs[io::SRC_DST_SCALE], 1.2f);
  auto dst_zps = make_tensor_obj(ts_descs[io::SRC_DST_ZP], 110);

  std::vector<const void*> data_p(io::SIZE, nullptr);
  data_p[io::SRC_Q] = Qs.first;
  data_p[io::SRC_K] = Ks.first;
  data_p[io::SRC_V] = Vs.first;
  data_p[io::MASK] = masks.first;
  data_p[io::DST] = dsts.first;
  data_p[io::BINARY_ADD] = badds.first;
  data_p[io::ATT_SCALE] = att_scales.first;
  data_p[io::Q_SCALE] = q_scales.first;
  data_p[io::K_SCALE] = k_scales.first;
  data_p[io::V_SCALE] = v_scales.first;
  data_p[io::SRC_DST_SCALE] = dst_scales.first;
  data_p[io::SRC_DST_ZP] = dst_zps.first;

  std::vector<const void*> data_q(io::SIZE, nullptr);
  data_q[io::SRC_Q] = Qs.second;
  data_q[io::SRC_K] = Ks.second;
  data_q[io::SRC_V] = Vs.second;
  data_q[io::MASK] = masks.second;
  data_q[io::DST] = dsts.second;
  data_q[io::BINARY_ADD] = badds.second;
  data_q[io::ATT_SCALE] = att_scales.second;
  data_q[io::Q_SCALE] = q_scales.second;
  data_q[io::K_SCALE] = k_scales.second;
  data_q[io::V_SCALE] = v_scales.second;
  data_q[io::SRC_DST_SCALE] = dst_scales.second;
  data_q[io::SRC_DST_ZP] = dst_zps.second;

  jd::operator_desc op_desc(jd::kernel_kind::mha_dense, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, attr_map);

  return {op_desc, data_p, data_q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  // gencase: bs seqlen head_num head_size

  cases.push_back({1, 64, 64, 1, 64, 0, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({2, 64, 64, 1, 32, 0, jd::data_type::u8, jd::format_type::abcd, 1, false});

  // acbd
  cases.push_back({2, 64, 64, 1, 32, 0, jd::data_type::u8, jd::format_type::acbd, 1, false});

  // headsize 256
  cases.push_back({1, 64, 64, 1, 256, 0, jd::data_type::u8, jd::format_type::abcd, 1, false});

  // binary add
  cases.push_back({3, 64, 64, 2, 256, 1, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({3, 64, 64, 2, 256, 2, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({3, 64, 64, 2, 256, 3, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({3, 64, 64, 2, 256, 4, jd::data_type::u8, jd::format_type::abcd, 1, false});

  // dt_dst
  cases.push_back({1, 64, 64, 1, 256, 0, jd::data_type::bf16, jd::format_type::abcd, 1, false});

  // seqlen 2k
  cases.push_back({1, 2048, 2048, 1, 32, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});
  cases.push_back({1, 2041, 2041, 1, 32, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});
  cases.push_back({1, 512, 512, 1, 256, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});

  // head_size = 64 / 32
  cases.push_back({4, 384, 384, 16, 64, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});
  cases.push_back({4, 384, 384, 16, 32, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});

  // a variety of seqlen
  for (int seq_len : {32, 33, 47, 48, 49, 63, 64, 65, 128, 384})
    cases.push_back({12, seq_len, seq_len, 4, 32, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});

  // kv-cache
  for (int sl_n : {32, 33, 37, 63, 64}) {
    cases.push_back({4, 1, sl_n, 16, 256, 2, jd::data_type::u8, jd::format_type::acbd, 0, false});
    cases.push_back({4, 1, sl_n, 16, 256, 2, jd::data_type::bf16, jd::format_type::acbd, 0, false});
  }

  return ::testing::ValuesIn(cases);
};

class MhaDenseKernTest : public testing::TestWithParam<test_params_t> {
 protected:
  MhaDenseKernTest() {}
  ~MhaDenseKernTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MhaDenseKernTest, ) {
  const test_params_t& t = testing::TestWithParam<test_params_t>::GetParam();
  const auto d = gen_data(t.bs, t.sl_m, t.sl_n, t.head_num, t.head_size, t.badd_dim, t.dt_dst);
  EXPECT_TRUE(check_result(t.nthr, t.expect_to_fail, d));

  for (auto data : {d.rt_data_kern, d.rt_data_ref})
    for (auto p : data)
      if (p != nullptr) delete[] reinterpret_cast<const char*>(p);
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseKernTest, case_func(), TestParam2str);
}  // namespace test
