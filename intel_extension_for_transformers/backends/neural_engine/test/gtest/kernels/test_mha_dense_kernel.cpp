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

#include "amx_utils.hpp"
#include "cpu_isa.hpp"
#include "gtest/gtest.h"
#include "kernels/mha_dense_ref.hpp"
#include "unit_test_utils.hpp"

namespace jd {
using dt = data_type;
struct OpArgs {
  std::vector<const void*> rt_data;
  operator_desc op_desc;
};

struct test_params_t {
  std::pair<OpArgs, OpArgs> args;
  int nthr;
  bool expect_to_fail;
};

inline static std::string TestParam2str(testing::TestParamInfo<test_params_t> tpi) {
  const auto& op_desc = tpi.param.args.first.op_desc;
  const auto& ts_desc = op_desc.tensor_descs();
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(tpi.param.nthr));
  params.push_back(std::to_string(ts_desc[mha_dense_io::SRC_Q].shape()[0]));                       // bs
  params.push_back(std::to_string(ts_desc[mha_dense_io::SRC_Q].shape()[1]));                       // seqlen
  params.push_back(std::to_string(ts_desc[mha_dense_io::SRC_Q].shape()[2]));                       // head_num
  params.push_back(std::to_string(ts_desc[mha_dense_io::SRC_Q].shape()[3]));                       // head_size
  params.push_back("badddim" + std::to_string(ts_desc[mha_dense_io::BINARY_ADD].shape().size()));  // badddim
  params.push_back(num2id(op_desc.attrs().at("QK_rescale")));
  params.push_back(num2id(op_desc.attrs().at("softmax_rescale")));
  params.push_back(num2id(op_desc.attrs().at("QKV_rescale")));
  params.push_back(num2id(op_desc.attrs().at("QKV_dstzp")));
  return join_str(params, "_");
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    n_thread_t with_n_thread(t.nthr);
    mha_dense_desc mha_dense_desc(p.op_desc);
    mha_dense mha_dense_kernel(mha_dense_desc);
    const auto tmp_p = std::shared_ptr<char>(aligned_allocator_t<char>::allocate(mha_dense_kernel.get_workspace_size()),
                                             [](char* ptr) { aligned_allocator_t<char>::deallocate(ptr); });
    auto data_p = p.rt_data;
    data_p[mha_dense_io::WORKSPACE] = tmp_p.get();
    mha_dense_kernel.execute(data_p);

    std::shared_ptr<const kernel_desc_t> mha_dense_ref_desc;
    kernel_desc_t::create<mha_dense_ref_kd_t>(mha_dense_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> mha_dense_ref_kernel;
    kernel_t::create<mha_dense_ref_k_t, mha_dense_ref_kd_t>(mha_dense_ref_kernel, mha_dense_ref_desc);
    mha_dense_ref_kernel->execute(q.rt_data);
  } catch (const std::exception& e) {
    SPARSE_LOG(ERROR) << e.what();
    return t.expect_to_fail;
  }

  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[mha_dense_io::DST];
    auto size1 = p.op_desc.tensor_descs()[mha_dense_io::DST].size();
    auto buf2 = q.rt_data[mha_dense_io::DST];
    auto size2 = q.op_desc.tensor_descs()[mha_dense_io::DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    switch (p.op_desc.tensor_descs()[mha_dense_io::DST].dtype()) {
      case dt::fp32:
        return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
      case dt::s32:
        return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
      case dt::u8:
        return compare_data<uint8_t>(buf1, size1, buf2, size2, 4e-3);
      case dt::s8:
        return compare_data<int8_t>(buf1, size1, buf2, size2, 4e-3);
      default:
        SPARSE_LOG(ERROR) << "Unexpected dst type";
    }
  }
  return false;
}

class MhaDenseOpTest : public testing::TestWithParam<test_params_t> {
 protected:
  MhaDenseOpTest() {}
  ~MhaDenseOpTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MhaDenseOpTest, ) {
  const test_params_t& t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));

  for (auto op_args : {t.args.first, t.args.second})
    for (auto rt_data : op_args.rt_data) {
      char* data = reinterpret_cast<char*>(const_cast<void*>(rt_data));
      delete[] data;
    }
}

std::pair<const void*, const void*> make_tensor_obj(const tensor_desc& ts_desc, float min_value = -10,
                                                    float max_value = 10) {
  int64_t elem_num = std::accumulate(ts_desc.shape().begin(), ts_desc.shape().end(), 1LL, std::multiplies<int64_t>());
  int bytes_size = elem_num * type_size[ts_desc.dtype()];
  void* data_ptr = nullptr;
  if (min_value == 0.f && max_value == 0.f) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    if (ts_desc.dtype() == dt::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == dt::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == dt::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == dt::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, min_value, max_value);
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<OpArgs, OpArgs> gen_case(  //
    const dim_t bs, const dim_t seqlen, const dim_t head_num, const dim_t head_size, float QK_rescale,
    float softmax_rescale, float QKV_rescale, int QKV_dstzp, int badd_dim = 0) {
  std::vector<dim_t> badd_fullshape = {bs, head_num, seqlen, seqlen};
  std::vector<tensor_desc> ts_descs(mha_dense_io::mha_dense_io_MAX + 1,
                                    tensor_desc{{}, data_type::undef, format_type::undef});
  ts_descs[mha_dense_io::SRC_Q] = {{bs, seqlen, head_num, head_size}, data_type::s8, format_type::undef};
  ts_descs[mha_dense_io::SRC_K] = {{bs, seqlen, head_num, head_size}, data_type::s8, format_type::undef};
  ts_descs[mha_dense_io::SRC_V] = {{bs, seqlen, head_num, head_size}, data_type::s8, format_type::undef};
  ts_descs[mha_dense_io::MASK] = {{bs}, data_type::s32, format_type::undef};
  ts_descs[mha_dense_io::DST] = {{bs, seqlen, head_num, head_size}, data_type::u8, format_type::undef};
  if (badd_dim > 0) {
    SPARSE_LOG_IF(FATAL, badd_dim > 4) << "Unsupported binary add dimention";
    ts_descs[mha_dense_io::BINARY_ADD] = {std::vector<dim_t>(badd_fullshape.cend() - badd_dim, badd_fullshape.cend()),
                                          data_type::fp32, format_type::a};
  }

  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["QK_rescale"] = std::to_string(QK_rescale);
  attr_map["softmax_rescale"] = std::to_string(softmax_rescale);
  attr_map["QKV_rescale"] = std::to_string(QKV_rescale);
  attr_map["QKV_dstzp"] = std::to_string(QKV_dstzp);

  // Step 2: Construct Tensor ptr
  auto Qs = make_tensor_obj(ts_descs[mha_dense_io::SRC_Q]);
  auto Ks = make_tensor_obj(ts_descs[mha_dense_io::SRC_K]);
  auto Vs = make_tensor_obj(ts_descs[mha_dense_io::SRC_V]);
  auto min_sl = std::max(dim_t(16), seqlen / 4);
  auto masks = make_tensor_obj(ts_descs[mha_dense_io::MASK], min_sl, seqlen);
  auto dsts = make_tensor_obj(ts_descs[mha_dense_io::DST], 0, 0);

  auto badds = badd_dim > 0 ? make_tensor_obj(ts_descs[mha_dense_io::BINARY_ADD], -1.f, 1.f)
                            : std::pair<const void*, const void*>{nullptr, nullptr};

  std::vector<const void*> data_p(mha_dense_io::mha_dense_io_MAX + 1, nullptr);
  data_p[mha_dense_io::SRC_Q] = Qs.first;
  data_p[mha_dense_io::SRC_K] = Ks.first;
  data_p[mha_dense_io::SRC_V] = Vs.first;
  data_p[mha_dense_io::MASK] = masks.first;
  data_p[mha_dense_io::DST] = dsts.first;
  data_p[mha_dense_io::BINARY_ADD] = badds.first;

  std::vector<const void*> data_q(mha_dense_io::mha_dense_io_MAX + 1, nullptr);
  data_q[mha_dense_io::SRC_Q] = Qs.second;
  data_q[mha_dense_io::SRC_K] = Ks.second;
  data_q[mha_dense_io::SRC_V] = Vs.second;
  data_q[mha_dense_io::MASK] = masks.second;
  data_q[mha_dense_io::DST] = dsts.second;
  data_q[mha_dense_io::BINARY_ADD] = badds.second;

  operator_desc op_desc(kernel_kind::mha_dense, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, attr_map);
  attr_map["approx_exp"] = "True";
  operator_desc op_desc2(kernel_kind::mha_dense, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, attr_map);

  OpArgs op_args_p = {data_p, op_desc};
  OpArgs op_args_q = {data_q, op_desc2};

  return {op_args_p, op_args_q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  // gencase: bs seqlen head_num head_size

  cases.push_back({gen_case(1, 64, 1, 64, 1.1f, 255.f, 5e-2f, 110), 1, false});
  cases.push_back({gen_case(2, 64, 1, 32, 1.1f, 255.f, 5e-2f, 110), 1, false});

  // headsize 256
  cases.push_back({gen_case(1, 64, 1, 256, 1.1f, 255.f, 5e-2f, 110), 1, false});

  // binary add
  cases.push_back({gen_case(3, 64, 2, 256, 1.1f, 255.f, 5e-2f, 110, 1), 1, false});
  cases.push_back({gen_case(3, 64, 2, 256, 1.1f, 255.f, 5e-2f, 110, 2), 1, false});
  cases.push_back({gen_case(3, 64, 2, 256, 1.1f, 255.f, 5e-2f, 110, 3), 1, false});
  cases.push_back({gen_case(3, 64, 2, 256, 1.1f, 255.f, 5e-2f, 110, 4), 1, false});

  // seqlen 2k
  cases.push_back({gen_case(1, 2048, 1, 32, 1.1f, 255.f, 5e-2f, 110), 1, false});
  cases.push_back({gen_case(1, 2041, 1, 32, 1.1f, 255.f, 5e-2f, 110), 1, false});
  cases.push_back({gen_case(1, 512, 1, 256, 1.1f, 255.f, 5e-2f, 110), 1, false});

  // head_size = 64 / 32
  cases.push_back({gen_case(4, 384, 16, 64, 1.1f, 255.f, 5e-2f, 110), 1, false});
  cases.push_back({gen_case(4, 384, 16, 32, 1.1f, 255.f, 5e-2f, 110), 1, false});

  // a variety of seqlen
  for (int seq_len : {32, 33, 47, 48, 49, 63, 64, 65, 128, 384})
    cases.push_back({gen_case(12, seq_len, 4, 32, 1.1f, 255.f, 5e-2f, 110), 1, false});

  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(SparseLib, MhaDenseOpTest, case_func(), TestParam2str);
}  // namespace jd
