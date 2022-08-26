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
#include "unit_test_utils.hpp"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
  float sparsity;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rt_data) {
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
  const auto wei_data = static_cast<const bfloat16_t*>(rt_data[0]);
  const auto src_data = static_cast<const bfloat16_t*>(rt_data[1]);
  const auto bia_data = static_cast<const float*>(rt_data[2]);
  void* dst_data = const_cast<void*>(rt_data[3]);

  std::vector<float> float_dst_data(N * NUM_M * M_MICRO, 0);
  bfloat16_t* bf_dst_data = static_cast<bfloat16_t*>(dst_data);
  float* fp_dst_data = static_cast<float*>(dst_data);

  // Computing the kernel
  for (int num_m = 0; num_m < NUM_M; ++num_m) {
#pragma omp parallel for
    for (int n = 0; n < N; ++n) {
#pragma omp parallel for
      for (int m = 0; m < M_MICRO; ++m) {
        for (int k = 0; k < K; ++k) {
          float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] +=
              make_fp32(wei_data[n * K + k]) * make_fp32(src_data[num_m * K * M_MICRO + k * M_MICRO + m]);
        }
        float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] += bia_data[n];
        float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] =
            apply_postop_list(float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m], op_desc.apply_postops_list());
        if (dst_dt == dt::bf16) {
          bf_dst_data[num_m * N * M_MICRO + n * M_MICRO + m] =
              make_bf16(float_dst_data[num_m * N * M_MICRO + n * M_MICRO + m]);
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
  sparse_matmul* spmm_kern = nullptr;
  try {
    const auto& op_desc = p.op_desc;
    sparse_matmul_desc spmm_desc(op_desc);
    spmm_kern = new sparse_matmul(spmm_desc);
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
    auto sparse_data_ptr = reinterpret_cast<void*>(sparse_addr);
    delete sparse_data_ptr;
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
    float eps = 5e-3;
    const auto& dst_type = p.op_desc.tensor_descs()[3].dtype();
    if (dst_type == dt::bf16) {
      eps = 1.0;
      return compare_data<bfloat16_t>(buf1, size1, buf2, size2, eps);
    }
    return compare_data<float>(buf1, size1, buf2, size2, eps);
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
}

template <typename T>
void prepare_sparse_data(T* weight, dim_t N, dim_t K, dim_t n_blksize, dim_t k_blksize, float ratio) {
  uint32_t seed = 9527;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      weight[n * K + k] = make_bf16(rand_r(&seed) % 10 + 1);
    }
  }
  // sparsify a_mat
  for (int nb = 0; nb < N / n_blksize; ++nb) {
    for (int kb = 0; kb < K / k_blksize; ++kb) {
      bool fill_zero = rand_r(&seed) % 100 <= (dim_t)(ratio * 100);
      if (fill_zero) {
        for (int n = 0; n < n_blksize; ++n) {
          for (int k = 0; k < k_blksize; ++k) {
            weight[(nb * n_blksize + n) * K + kb * k_blksize + k] = make_bf16(0);
          }
        }
      }
    }
  }
}

std::pair<const void*, const void*> make_data_obj(const dt& tensor_dt, dim_t rows, dim_t cols, dim_t index,
                                                  float sparsity = 0.f, const std::vector<float>& ranges = {-1, 1}) {
  dim_t elem_num = rows * cols;
  dim_t bytes_size = elem_num * type_size[tensor_dt];
  void* data_ptr = nullptr;
  switch (index) {
    case 0: {  // prepare wei
      data_ptr = new bfloat16_t[elem_num];
      bfloat16_t* bf16_ptr = static_cast<bfloat16_t*>(data_ptr);
      prepare_sparse_data<bfloat16_t>(bf16_ptr, rows, cols, 16, 1, sparsity);
      break;
    }
    case 1: {  // prepare src
      data_ptr = new bfloat16_t[elem_num];
      bfloat16_t* bf16_ptr = static_cast<bfloat16_t*>(data_ptr);
      init_vector(bf16_ptr, elem_num, ranges[0], ranges[1]);
      break;
    }
    case 2: {  // prepare bias
      data_ptr = new float[elem_num];
      float* fp32_ptr = static_cast<float*>(data_ptr);
      init_vector(fp32_ptr, elem_num, ranges[0], ranges[1]);
      break;
    }
    case 3: {  // prepare dst
      if (tensor_dt == dt::bf16) {
        data_ptr = new bfloat16_t[elem_num];
        memset(data_ptr, 0, bytes_size);
      } else {
        data_ptr = new float[elem_num];
        memset(data_ptr, 0, bytes_size);
      }
      break;
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, float sparsity, dim_t micro_bs = 64,
                                         dim_t micro_oc = -1, bool bf16_out = true,
                                         std::vector<postop_alg> postop_algs = {}) {
  std::unordered_map<std::string, std::string> op_attrs;
  // Step 1: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  tensor_desc wei_desc = {{N, K}, dt::bf16, ft::bsr};
  tensor_desc src_desc = {{M / micro_bs, K, micro_bs}, dt::bf16, ft::abc};
  tensor_desc bia_desc = {{N, 1}, dt::fp32, ft::ab};
  tensor_desc dst_desc = {{M / micro_bs, N, micro_bs}, dt::fp32, ft::abc};
  if (bf16_out) {
    dst_desc = {{M / micro_bs, N, micro_bs}, dt::bf16, ft::abc};
  }
  std::vector<tensor_desc> ts_descs = {wei_desc, src_desc, bia_desc, dst_desc};
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    dim_t rows = ts_descs[index].shape()[0];
    dim_t cols = ts_descs[index].shape()[1];
    if (index == 1 || index == 3) {
      rows = ts_descs[index].shape()[1];
      cols = ts_descs[index].shape()[0] * ts_descs[index].shape()[2];
    }
    auto data_pair = make_data_obj(ts_descs[index].dtype(), rows, cols, index, sparsity);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  // Step 2: sparse data encoding
  if (micro_oc == -1) {
    micro_oc = N;
  }
  volatile auto sparse_ptr = spns::reorder_to_bsr_amx<bfloat16_t, 32>(N, K, micro_oc, rt_data1[0]);
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, postop_alg alg) { return str_lists + '_' + postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(postop_alg_name[postop_algs[0]]), accu_op);
  }
  std::vector<postop_attr> apply_postops_list;
  std::for_each(postop_algs.begin(), postop_algs.end(), [&apply_postops_list](postop_alg alg) {
    return apply_postops_list.push_back({data_type::bf16, postop_type::eltwise, alg});
  });

  op_attrs["micro_oc"] = std::to_string(micro_oc);
  operator_desc an_op_desc(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                           op_attrs);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1, sparsity};
  op_args_t op_args_copy = {an_op_desc, rt_data2, sparsity};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;

  std::vector<std::vector<postop_alg>> postop_lists = {
      {},
      {postop_alg::gelu},
      {postop_alg::exp},
      {postop_alg::gelu, postop_alg::exp},
  };

  /* minimal case */
  cases.push_back({gen_case(64, 32, 16, .9f, 64, -1, false)});

  /* BERT-LARGE case */
  // To save time we only test post ops for BERT cases
  for (std::vector<postop_alg> algs : postop_lists) {
    cases.push_back({gen_case(128, 768, 768, .9f, 64, -1, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, 384, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, 192, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 128, -1, true, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, -1, false, algs)});
    cases.push_back({gen_case(128, 768, 768, .9f, 64, 384, false, algs)});
  }

  /* DLRM case */
  cases.push_back({gen_case(32768, 1024, 1024, .9f, 64, -1, true)});
  cases.push_back({gen_case(32768, 1024, 1024, .9f, 128, -1, true)});
  cases.push_back({gen_case(32768, 1024, 1024, .9f, 64, 512, true)});
  cases.push_back({gen_case(32768, 1024, 1024, .9f, 64, 256, true)});

  cases.push_back({gen_case(32768, 512, 512, .9f, 64, -1, true)});
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_desc = tpi.param.args.first.op_desc.tensor_descs();
  auto attrs_map = tpi.param.args.first.op_desc.attrs();
  params.push_back("sp" + std::to_string(static_cast<int>(tpi.param.args.first.sparsity * 100)));
  params.push_back(std::to_string(tensor_desc[ssd::SRC].shape()[0]));
  params.push_back(std::to_string(tensor_desc[ssd::SRC].shape()[1]));
  params.push_back(std::to_string(tensor_desc[ssd::WEI].shape()[0]));
  params.push_back(std::to_string(tensor_desc[ssd::SRC].shape()[2]));  // micro bs
  params.push_back(attrs_map["micro_oc"]);
  params.push_back(std::to_string(tensor_desc[ssd::DST].dtype() == dt::bf16));
  if (!attrs_map["postop_list"].empty()) params.push_back(attrs_map["postop_list"]);
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SpmmAMXX16KernelTest, case_func(), test_suffix);
}  // namespace jd
