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
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "interface.hpp"
#include "unit_test_utils.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/sparse_data.hpp"

namespace test {
struct op_args_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data;
  float sparsity;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const jd::operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& wei_desc = ts_descs[0];
  const auto& src_desc = ts_descs[1];
  const auto& dst_desc = ts_descs[3];
  int N = wei_desc.shape()[0];
  int K = wei_desc.shape()[1];
  int NUM_M = src_desc.shape()[0];
  int M_MICRO = src_desc.shape()[2];
  const auto& dst_dt = dst_desc.dtype();
  auto attrs_map = op_desc.attrs();

  // runtime data alias
  const auto wei_data = reinterpret_cast<const jd::bfloat16_t*>(rt_data[0]);
  const auto src_data = reinterpret_cast<const jd::bfloat16_t*>(rt_data[1]);
  const auto bia_data = reinterpret_cast<const float*>(rt_data[2]);
  void* dst_data = const_cast<void*>(rt_data[3]);

  std::vector<float> float_dst_data(N * NUM_M * M_MICRO, 0);
  jd::bfloat16_t* bf_dst_data = reinterpret_cast<jd::bfloat16_t*>(dst_data);
  float* fp_dst_data = reinterpret_cast<float*>(dst_data);

  // Computing the kernel
  for (int num_m = 0; num_m < NUM_M; ++num_m) {
#pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
      for (int m = 0; m < M_MICRO; ++m) {
        for (int k = 0; k < K; ++k) {
          float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] +=
              static_cast<float>(jd::bfloat16_t(wei_data[n * K + k])) *
              static_cast<float>(jd::bfloat16_t(src_data[num_m * K * M_MICRO + k * M_MICRO + m]));
        }
        float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] += bia_data[n];
        float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] =
            apply_postop_list(float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m], op_desc.apply_postops_list());
        if (dst_dt == jd::data_type::bf16) {
          bf_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] =
              jd::bfloat16_t(float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m]);
        } else {
          fp_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] = float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m];
        }
      }
    }
  }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  jd::sparse_matmul* spmm_kern = nullptr;
  try {
    const auto& op_desc = p.op_desc;
    jd::sparse_matmul_desc spmm_desc(op_desc);
    spmm_kern = new jd::sparse_matmul(spmm_desc);
    spmm_kern->execute(p.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (spmm_kern != nullptr) {
    auto attrs_map = p.op_desc.attrs();
    const auto sparse_addr = str_to_num<uint64_t>(attrs_map["sparse_ptr"]);
    const auto all_bsr_data = reinterpret_cast<std::vector<jd::bsr_data_t<jd::bfloat16_t>*>*>(sparse_addr);
    for (auto sparse_data : *all_bsr_data) {
      delete sparse_data;
    }
    delete all_bsr_data;
    delete spmm_kern;
  }
  if (!t.expect_to_fail) {
    get_true_data(q.op_desc, q.rt_data);
    auto buf1 = p.rt_data[3];
    auto size1 = p.op_desc.tensor_descs()[3].size();
    auto buf2 = q.rt_data[3];
    auto size2 = q.op_desc.tensor_descs()[3].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[3].dtype();
    if (dst_type == jd::data_type::bf16) {
      return compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 3e-2);
    }
    return compare_data<float>(buf1, size1, buf2, size2, 3e-2);
  }
  return false;
}

class SpmmAMXX16KernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  SpmmAMXX16KernelTest() {}
  virtual ~SpmmAMXX16KernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SpmmAMXX16KernelTest, TestPostfix) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
  for (auto iter : t.args.first.rt_data) {
    char* data = reinterpret_cast<char*>(const_cast<void*>(iter));
    delete[] data;
  }
  for (auto iter : t.args.second.rt_data) {
    char* data = reinterpret_cast<char*>(const_cast<void*>(iter));
    delete[] data;
  }
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, float sparsity, dim_t micro_bs = 64,
                                         dim_t micro_oc = -1, bool bf16_out = true,
                                         std::vector<jd::postop_alg> postop_algs = {}) {
  std::unordered_map<std::string, std::string> op_attrs;
  // Step 1: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  jd::tensor_desc wei_desc = {{N, K}, jd::data_type::bf16, jd::format_type::bsr};
  jd::tensor_desc src_desc = {{M / micro_bs, K, micro_bs}, jd::data_type::bf16, jd::format_type::abc};
  jd::tensor_desc bia_desc = {{N, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{M / micro_bs, N, micro_bs}, jd::data_type::fp32, jd::format_type::abc};
  if (bf16_out) {
    dst_desc = {{M / micro_bs, N, micro_bs}, jd::data_type::bf16, jd::format_type::abc};
  }
  std::vector<jd::tensor_desc> ts_descs = {wei_desc, src_desc, bia_desc, dst_desc};
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    dim_t rows = ts_descs[index].shape()[0];
    dim_t cols = ts_descs[index].shape()[1];
    if (index == 1 || index == 3) {
      rows = ts_descs[index].shape()[1];
      cols = ts_descs[index].shape()[0] * ts_descs[index].shape()[2];
    }
    bool is_clear = (index == 3) ? true : false;
    if (index > 0) {
      sparsity = 0.f;
    }
    auto data_pair = make_data_obj({rows, cols}, ts_descs[index].dtype(), is_clear, {-.5, .5}, sparsity);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  // Step 2: sparse data encoding
  if (micro_oc == -1) {
    micro_oc = N;
  }
  volatile auto sparse_ptr = jd::spns::reorder_to_bsr_amx<jd::bfloat16_t, 32>(N, K, micro_oc, rt_data1[0]);
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, jd::postop_alg alg) { return str_lists + '_' + jd::postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(jd::postop_alg_name[postop_algs[0]]), accu_op);
  }
  std::vector<jd::postop_attr> apply_postops_list;
  std::for_each(postop_algs.begin(), postop_algs.end(), [&apply_postops_list](jd::postop_alg alg) {
    return apply_postops_list.push_back({jd::data_type::bf16, jd::postop_type::eltwise, alg});
  });

  op_attrs["micro_oc"] = std::to_string(micro_oc);
  jd::operator_desc an_op_desc(jd::kernel_kind::sparse_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                               ts_descs, op_attrs, apply_postops_list);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1, sparsity};
  op_args_t op_args_copy = {an_op_desc, rt_data2, sparsity};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  std::vector<std::vector<jd::postop_alg>> postop_lists = {
      {},
      {jd::postop_alg::gelu},
      {jd::postop_alg::exp},
      {jd::postop_alg::gelu, jd::postop_alg::exp},
  };

  /* minimal case */
  cases.push_back({gen_case(64, 32, 16, .9f, 64, -1, false)});

  /* BERT-LARGE case */
  // To save time we only test post ops for BERT cases
  for (std::vector<jd::postop_alg> algs : postop_lists) {
    cases.push_back({gen_case(128, 768, 768, .9f, 64, -1, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, 384, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, 192, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 128, -1, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, -1, false, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, 384, false, algs)});
  }

  /* DLRM case */
  cases.push_back({gen_case(4096, 1024, 1024, .9f, 64, -1, true)});
  cases.push_back({gen_case(4096, 1024, 1024, .9f, 128, -1, true)});
  cases.push_back({gen_case(4096, 1024, 1024, .9f, 64, 512, true)});
  cases.push_back({gen_case(4096, 1024, 1024, .9f, 64, 256, true)});

  cases.push_back({gen_case(4096, 512, 512, .9f, 64, -1, true)});
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_desc = tpi.param.args.first.op_desc.tensor_descs();
  auto attrs_map = tpi.param.args.first.op_desc.attrs();
  params.push_back("sp" + std::to_string(static_cast<int>(tpi.param.args.first.sparsity * 100)));
  params.push_back(std::to_string(tensor_desc[jd::ssd::SRC].shape()[0]));
  params.push_back(std::to_string(tensor_desc[jd::ssd::SRC].shape()[1]));
  params.push_back(std::to_string(tensor_desc[jd::ssd::WEI].shape()[0]));
  params.push_back(std::to_string(tensor_desc[jd::ssd::SRC].shape()[2]));  // micro bs
  params.push_back(attrs_map["micro_oc"]);
  params.push_back(std::to_string(tensor_desc[jd::ssd::DST].dtype() == jd::data_type::bf16));
  if (!attrs_map["postop_list"].empty()) params.push_back(attrs_map["postop_list"]);
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SpmmAMXX16KernelTest, case_func(), test_suffix);
}  // namespace test
