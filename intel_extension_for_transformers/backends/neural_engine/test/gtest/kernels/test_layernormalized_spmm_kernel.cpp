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
#include "unit_test_utils.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/layernormalized_spmm_ref.hpp"

namespace jd {
struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> data;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  const auto& op_desc = p.op_desc;
  auto op_attr = op_desc.attrs();

  try {
    layernormalized_spmm_desc layernormalized_spmm_desc(op_desc);
    layernormalized_spmm layernormalized_spmm_ker(layernormalized_spmm_desc);
    layernormalized_spmm_ker.execute(p.data);

    std::shared_ptr<const kernel_desc_t> layernormalized_spmm_ref_desc;
    kernel_desc_t::create<layernormalized_spmm_ref_kd_t>(layernormalized_spmm_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> layernormalized_spmm_ref_ker;
    kernel_t::create<layernormalized_spmm_ref_k_t, layernormalized_spmm_ref_kd_t>(layernormalized_spmm_ref_ker,
                                                                                  layernormalized_spmm_ref_desc);
    layernormalized_spmm_ref_ker->execute(q.data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  auto free_memory = [&]() {
    for (auto&& i : p.data) {
      free(const_cast<void*>(i));
    }
    for (auto&& i : q.data) {
      free(const_cast<void*>(i));
    }

    auto op_desc = t.args.first.op_desc;
    auto op_attrs = op_desc.attrs();
    const uint64_t data_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
    bsr_data_t<int8_t>* bsr_data = reinterpret_cast<bsr_data_t<int8_t>*>(data_addr);
    delete bsr_data;
  };
  if (!t.expect_to_fail) {
    auto buf1 = p.data[8];
    auto size = p.op_desc.tensor_descs()[3].size();
    auto buf2 = q.data[8];
    auto dst_type = p.op_desc.tensor_descs()[3].dtype();
    EXPECT_NE(buf1, buf2);
    bool ans = false;
    if (dst_type == data_type::fp32) {
      ans = compare_data<float>(buf1, size, buf2, size, 5e-3);
      if (op_attr["split_output"] == "true" && ans) {
        auto buf3 = q.data[11];
        auto buf4 = p.data[11];
        if (op_desc.apply_postops_list().back().dt == data_type::s8) {
          ans = compare_data<int8_t>(buf4, size, buf3, size, 1e-2);
        } else {
          ans = compare_data<uint8_t>(buf4, size, buf3, size, 1e-2);
        }
      }
    } else if (dst_type == data_type::u8) {
      ans = compare_data<uint8_t>(buf1, size, buf2, size, 1e-2);
    } else if (dst_type == data_type::s8) {
      ans = compare_data<int8_t>(buf1, size, buf2, size, 1e-2);
    }
    free_memory();
    return ans;
  }
  free_memory();
  return true;
}

class LayernormalizedSpmmKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  LayernormalizedSpmmKernelTest() {}
  virtual ~LayernormalizedSpmmKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(LayernormalizedSpmmKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

template <typename T>
void prepare_sparse_data(T* vector_data, dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, float sparsity,
                         uint32_t* seed = nullptr) {
  uint32_t default_seed = 123;
  if (seed == nullptr) seed = &default_seed;
  std::srand(default_seed);
  for (int i = 0; i < rows; i += blk_row) {
    for (int j = 0; j < cols; j += blk_col) {
      bool fill_zero = std::rand() % 100 <= (sparsity * 100);
      if (fill_zero) {
        for (int bi = i; bi < i + blk_row; ++bi) {
          for (int bj = j; bj < j + blk_col; ++bj) {
            vector_data[bi * cols + bj] = 0;
          }
        }
      }
    }
  }
}

std::pair<const void*, const void*> make_data_obj(const std::vector<int64_t>& a_shape, const data_type& a_dt,
                                                  bool is_clear = false, float sparsity = 0.f,
                                                  const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;

  if (is_clear) {
    data_ptr = malloc(bytes_size);
    memset(data_ptr, 0, bytes_size);
  } else {
    if (a_dt == data_type::fp32) {
      data_ptr = malloc(elem_num * sizeof(float));
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == data_type::s32) {
      data_ptr = malloc(elem_num * sizeof(int32_t));
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == data_type::u8) {
      data_ptr = malloc(elem_num * sizeof(uint8_t));
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == data_type::s8) {
      data_ptr = malloc(elem_num * sizeof(int8_t));
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
      if (sparsity != 0.f) {
        int8_t* s8_ptr = static_cast<int8_t*>(data_ptr);
        prepare_sparse_data(s8_ptr, a_shape[0], a_shape[1], 4, 1, sparsity);
      }
    }
  }

  void* data_ptr_copy = malloc(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

void auto_blocking(int& BM, int BN, const int M, const int N) {  // NOLINT
  if (BM > M) {
    BM = M;
  } else if (BM <= 0) {  // try to get optimized block size
    int cores = omp_get_num_procs();
    const int blocks_n = N / BN;

    BM = ceil_div(M, ceil_div(cores, blocks_n));
    int TILE_SIZE_M = 4;
    BM = ceil_div(BM, TILE_SIZE_M) * TILE_SIZE_M;  // round to a multiple of 4
  }
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs, float sparsity,
                                         const std::vector<postop_attr>& postop_attr = {}) {
  bool append_sum = (op_attrs["append_sum"] == "true");
  auto wei_desc = ts_descs[0];
  auto src_desc = ts_descs[1];
  int M = wei_desc.shape()[0];
  int K = wei_desc.shape()[1];
  int num_mbs = src_desc.shape().size() == 2 ? 1 : src_desc.shape()[0];
  int micro_bs = src_desc.shape().size() == 2 ? src_desc.shape()[1] : src_desc.shape()[2];
  int N = num_mbs * micro_bs;

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  std::vector<tensor_desc> spmm_ts_descs = ts_descs;
  tensor_desc mean_desc = {{num_mbs, micro_bs}, data_type::fp32, format_type::a};
  tensor_desc var_desc = {{num_mbs, micro_bs}, data_type::fp32, format_type::a};
  spmm_ts_descs.push_back(mean_desc);
  spmm_ts_descs.push_back(var_desc);

  int tensor_num = spmm_ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = spmm_ts_descs[index];
    bool is_clear = (index == ssd::DST && !append_sum);
    float data_sparsity = (index == ssd::WEI) ? sparsity : 0;
    auto ranges = (index == ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, data_sparsity, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  auto sparse_ptr = new bsr_data_t<int8_t>(
      spns::reorder_to_bsr_group<int8_t, 4>(M, K, 4, 1, static_cast<const int8_t*>(rt_data1[ssd::WEI])));
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));

  // gen workspace
  float* workspace = nullptr;
  float* workspace_ref = nullptr;

  int BM = str_to_num<int>(op_attrs["micro_oc"]);
  auto_blocking(BM, micro_bs, M, N);
  workspace = reinterpret_cast<float*>(malloc(2 * ceil_div(M, BM) * N * sizeof(float)));
  workspace_ref = reinterpret_cast<float*>(malloc(2 * ceil_div(M, BM) * N * sizeof(float)));
  rt_data1.push_back(workspace);
  rt_data2.push_back(workspace_ref);

  // gen lnorm dst, lnorm alpha, lnorm beta, lnorm dst2
  void* lnorm_dst = nullptr;
  void* lnorm_dst2 = nullptr;
  void* lnorm_dst_ref = nullptr;
  void* lnorm_dst2_ref = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  // set rand seed
  unsigned int seed = 456;
  srand(seed);

  auto out_dt = data_type::fp32;

  lnorm_dst = sparselib_ut_memo(lnorm_dst, num_mbs * micro_bs * M, out_dt, MALLOC);
  lnorm_dst = sparselib_ut_memo(lnorm_dst, num_mbs * micro_bs * M, out_dt, MEMSET);
  lnorm_dst_ref = sparselib_ut_memo(lnorm_dst_ref, num_mbs * micro_bs * M, out_dt, MALLOC);
  lnorm_dst_ref = sparselib_ut_memo(lnorm_dst_ref, num_mbs * micro_bs * M, out_dt, MEMSET);
  rt_data1.push_back(lnorm_dst);
  rt_data2.push_back(lnorm_dst_ref);

  float* alpha = reinterpret_cast<float*>(malloc(M * sizeof(float)));
  float* beta = reinterpret_cast<float*>(malloc(M * sizeof(float)));
  float* alpha_ref = reinterpret_cast<float*>(malloc(M * sizeof(float)));
  float* beta_ref = reinterpret_cast<float*>(malloc(M * sizeof(float)));

  // init alpha&beta
  for (int i = 0; i < M; i++) {
    alpha[i] = 1 + rand_float_postfix();
    alpha_ref[i] = alpha[i];
  }
  for (int i = 0; i < M; i++) {
    beta[i] = 1 + rand_float_postfix();
    beta_ref[i] = beta[i];
  }
  rt_data1.push_back(alpha);
  rt_data1.push_back(beta);
  rt_data2.push_back(alpha_ref);
  rt_data2.push_back(beta_ref);

  if (op_attrs["split_output"] == "true") {
    lnorm_dst2 = sparselib_ut_memo(lnorm_dst2, num_mbs * micro_bs * M, data_type::s8, MALLOC);
    lnorm_dst2_ref = sparselib_ut_memo(lnorm_dst2_ref, num_mbs * micro_bs * M, data_type::s8, MALLOC);
    rt_data1.push_back(lnorm_dst2);
    rt_data2.push_back(lnorm_dst2_ref);
  }

  operator_desc op_desc(kernel_kind::layernormalized_spmm, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                        op_attrs, postop_attr);
  op_args_t p = {op_desc, rt_data1};
  op_args_t q = {op_desc, rt_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  tensor_desc wei_desc = {{1024, 1024}, jd::data_type::s8, jd::format_type::bsr};
  tensor_desc src_desc = {{4, 1024, 384}, jd::data_type::u8, jd::format_type::ab};
  tensor_desc src_desc_2D = {{1024, 1536}, jd::data_type::u8, jd::format_type::ab};
  tensor_desc bia_desc = {{1024, 1}, jd::data_type::s32, jd::format_type::ab};
  // dst_desc only support fp32 now, can supoort int8 in the future. If dt(dst_desc) == int8, split_output feature can't
  // enable.
  tensor_desc dst_desc = {{4, 1024, 384}, jd::data_type::fp32, jd::format_type::ab};
  tensor_desc dst_desc_2D = {{1024, 1536}, jd::data_type::fp32, jd::format_type::ab};
  tensor_desc scales_desc = {{1024, 1}, jd::data_type::fp32, jd::format_type::ab};
  postop_attr u8_quantize = {data_type::u8,       postop_type::eltwise, postop_alg::quantize, rand_float_postfix(), 0,
                             rand_float_postfix()};

  cases.push_back({gen_case({wei_desc, src_desc, bia_desc, dst_desc, scales_desc},
                            {{"split_output", "true"}, {"append_sum", "true"}}, 0.9, {u8_quantize}),
                   false});
  cases.push_back({gen_case({wei_desc, src_desc_2D, bia_desc, dst_desc_2D, scales_desc},
                            {{"split_output", "true"}, {"append_sum", "true"}}, 0.9, {u8_quantize}),
                   false});
  cases.push_back(
      {gen_case({wei_desc, src_desc, bia_desc, dst_desc, scales_desc}, {{"append_sum", "true"}}, 0.9), false});
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(SparseLib, LayernormalizedSpmmKernelTest, case_func());
}  // namespace jd
