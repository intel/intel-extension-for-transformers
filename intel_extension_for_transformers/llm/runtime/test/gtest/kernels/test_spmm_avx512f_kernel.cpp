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

#include <omp.h>
#include <glog/logging.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <exception>

#include "interface.hpp"
#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/sparse_data.hpp"

namespace test {
struct op_args_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data;
  float sparsity;  // sparsity of weight matrix; for testcase labeling
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const jd::operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& wei_desc = ts_descs[jd::ssd::WEI];
  const auto& src_desc = ts_descs[jd::ssd::SRC];
  const auto& bias_desc = ts_descs[jd::ssd::BIAS];
  int dims = wei_desc.shape().size();
  int M = src_desc.shape()[0];
  int K = wei_desc.shape()[0];
  int N = wei_desc.shape()[1];
  bool has_bias = !bias_desc.shape().empty();
  auto attrs_map = op_desc.attrs();
  std::vector<dim_t> left_stride = {K, 1};
  std::vector<dim_t> right_stride = {N, 1};
  std::vector<dim_t> dst_stride = {N, 1};

  // runtime data alias
  const auto left_fp32 = static_cast<const float*>(rt_data[jd::ssd::SRC]);
  const auto right_fp32 = static_cast<const float*>(rt_data[jd::ssd::WEI]);
  const auto bias_fp32 = static_cast<const float*>(rt_data[jd::ssd::BIAS]);
  auto dst_fp32 = static_cast<float*>(const_cast<void*>(rt_data[jd::ssd::DST]));

  // Computing the kernel
  SPARSE_LOG_IF(FATAL, dims != 2) << "Weight must be 2D!";
  for (int i = 0; i < M; ++i) {
#pragma omp parallel for
    for (int j = 0; j < N; ++j) {
      float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
#pragma omp simd
      for (int k = 0; k < K; ++k) {
        auto left_k = left_fp32[i * left_stride[0] + k * left_stride[1]];
        auto right_k = right_fp32[k * right_stride[0] + j * right_stride[1]];
        value += left_k * right_k;
      }

      // Accumulate bias or post sum
      if (has_bias) {
        value += bias_fp32[j];
      }

      value = apply_postop_list(value, op_desc.apply_postops_list());

      dst_fp32[i * dst_stride[0] + j * dst_stride[1]] = static_cast<float>(value);
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
    const uint64_t& sparse_addr = str_to_num<uint64_t>(attrs_map["sparse_ptr"]);
    auto sparse_data_ptr = reinterpret_cast<jd::bsc_data_t<float>*>(sparse_addr);
    delete sparse_data_ptr;
    delete spmm_kern;
  }
  if (!t.expect_to_fail) {
    get_true_data(q.op_desc, q.rt_data);
    auto buf1 = p.rt_data[jd::ssd::DST];
    auto size1 = p.op_desc.tensor_descs()[jd::ssd::DST].size();
    auto buf2 = q.rt_data[jd::ssd::DST];
    auto size2 = q.op_desc.tensor_descs()[jd::ssd::DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  }
  return false;
}

class SpmmAVX512FKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  SpmmAVX512FKernelTest() {}
  virtual ~SpmmAVX512FKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SpmmAVX512FKernelTest, ) {
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

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, float sparsity,
                                         std::vector<jd::postop_alg> postop_algs = {}) {
  // Step 1: Construct operator config
  std::unordered_map<std::string, std::string> op_attrs = {};

  // Step 2: Construct runtime data
  jd::tensor_desc src_desc = {{M, K}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc wei_desc = {{K, N}, jd::data_type::fp32, jd::format_type::bsc};
  jd::tensor_desc bia_desc = {{N, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{M, N}, jd::data_type::fp32, jd::format_type::abc};
  std::vector<jd::tensor_desc> ts_descs = {wei_desc, src_desc, bia_desc, dst_desc};

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  for (size_t i = 0; i < ts_descs.size(); ++i) {
    bool is_clear = i == jd::ssd::DST || i == jd::ssd::BIAS;
    std::vector<float> ranges = (i == jd::ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    float data_sparsity = (i == jd::ssd::WEI) ? sparsity : 0;
    auto data_pair =
        make_data_obj(ts_descs[i].shape(), ts_descs[i].dtype(), is_clear, ranges, data_sparsity, ts_descs[i].ftype());
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  std::vector<const float*> rt_data_ins;
  for (auto p : rt_data1) rt_data_ins.push_back(static_cast<const float*>(p));

  // Step 3: sparse data encoding
  jd::bsc_data_t<float> bsc_obj =
      jd::spns::tobsc<float>(K, N, 1, 16, static_cast<const float*>(rt_data1[jd::ssd::WEI]));
  auto sparse_ptr = new jd::bsc_data_t<float>(bsc_obj);  // Will be deleted in `check_result`
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, jd::postop_alg alg) { return str_lists + '_' + jd::postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(jd::postop_alg_name[postop_algs[0]]), accu_op);
  }
  std::vector<jd::postop_attr> apply_postops_list;
  std::for_each(postop_algs.begin(), postop_algs.end(), [&apply_postops_list](jd::postop_alg alg) {
    return apply_postops_list.push_back({jd::data_type::fp32, jd::postop_type::eltwise, alg});
  });
  jd::operator_desc an_op_desc(jd::kernel_kind::sparse_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                               ts_descs, op_attrs, apply_postops_list);

  // Step 4: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1, sparsity};
  op_args_t op_args_copy = {an_op_desc, rt_data2, sparsity};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  // Config

  std::vector<std::vector<jd::postop_alg>> postop_lists = {
      {},
      {jd::postop_alg::gelu},
      {jd::postop_alg::exp},
      {jd::postop_alg::gelu, jd::postop_alg::exp},
  };

  for (std::vector<jd::postop_alg> algs : postop_lists) {
    cases.push_back({gen_case(128, 256, 256, .7f, algs)});
    cases.push_back({gen_case(384, 256, 256, .7f, algs)});
    cases.push_back({gen_case(128, 1024, 256, .7f, algs)});
    cases.push_back({gen_case(384, 1024, 256, .7f, algs)});
    cases.push_back({gen_case(128, 256, 1024, .7f, algs)});
    cases.push_back({gen_case(384, 256, 1024, .7f, algs)});
    cases.push_back({gen_case(128, 768, 768, .7f, algs)});
    cases.push_back({gen_case(384, 768, 768, .7f, algs)});
    cases.push_back({gen_case(128, 3072, 768, .7f, algs)});
    cases.push_back({gen_case(384, 3072, 768, .7f, algs)});
    cases.push_back({gen_case(128, 768, 3072, .7f, algs)});
    cases.push_back({gen_case(384, 768, 3072, .7f, algs)});
    cases.push_back({gen_case(128, 1024, 1024, .7f, algs)});
    cases.push_back({gen_case(384, 1024, 1024, .7f, algs)});
    cases.push_back({gen_case(128, 4096, 1024, .7f, algs)});
    cases.push_back({gen_case(384, 4096, 1024, .7f, algs)});
    cases.push_back({gen_case(128, 1024, 4096, .7f, algs)});
    cases.push_back({gen_case(384, 1024, 4096, .7f, algs)});
  }

  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_desc = tpi.param.args.first.op_desc.tensor_descs();
  auto attrs_map = tpi.param.args.first.op_desc.attrs();
  params.push_back("sp" + std::to_string(static_cast<int>(tpi.param.args.first.sparsity * 100)));
  params.push_back(std::to_string(tensor_desc[jd::ssd::SRC].shape()[0]));
  params.push_back(std::to_string(tensor_desc[jd::ssd::SRC].shape()[1]));
  params.push_back(std::to_string(tensor_desc[jd::ssd::WEI].shape()[1]));
  if (!attrs_map["postop_list"].empty()) params.push_back(attrs_map["postop_list"]);
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SpmmAVX512FKernelTest, case_func(), test_suffix);
}  // namespace test
