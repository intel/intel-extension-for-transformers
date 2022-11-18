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
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <exception>
#include "interface.hpp"
#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "kernels/spmm_types.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
  float sparisty;  // sparsity of weight matrix; for testcase labeling
  int nthr;        // 0 for not touching OMP_NUM_THREADS and using what set outside
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

void get_true_data(const operator_desc& op_desc, const std::vector<const void*>& rt_data) {
  // shape configure alias
  const auto& ts_descs = op_desc.tensor_descs();
  const auto& wei_shape = ts_descs[ssd::WEI].shape();
  const auto& wei_type = ts_descs[ssd::WEI].dtype();
  const auto& src_type = ts_descs[ssd::SRC].dtype();
  const auto& src_shape = ts_descs[ssd::SRC].shape();
  const auto& dst_type = ts_descs[ssd::DST].dtype();
  const auto& dst_shape = ts_descs[ssd::DST].shape();
  assert((src_shape.size() == 2 || src_shape.size() == 3) && src_shape.size() == dst_shape.size());

  int oc = wei_shape[0];
  int ic = wei_shape[1];
  int micro_bs = src_shape.back();
  int num_mbs = src_shape.size() == 2 ? 1 : src_shape[0];

  const auto& left_dt = wei_type;
  const auto& right_dt = src_type;
  const auto& dst_dt = dst_type;
  bool has_bias = !ts_descs[ssd::BIAS].shape().empty();
  auto attrs_map = op_desc.attrs();
  bool append_sum = (attrs_map["append_sum"] == "true");
  std::vector<dim_t> left_stride = {ic, 1};
  std::vector<dim_t> right_stride = {micro_bs * ic, micro_bs, 1};
  std::vector<dim_t> dst_stride = {micro_bs * oc, micro_bs, 1};

  // runtime data alias
  const auto left_data = rt_data[ssd::WEI];
  const auto right_data = rt_data[ssd::SRC];
  const auto bias_data = static_cast<const int32_t*>(rt_data[ssd::BIAS]);
  auto dst_data = const_cast<void*>(rt_data[ssd::DST]);
  const auto scales_data = static_cast<const float*>(rt_data[ssd::SCALES]);

  // buffer data
  auto left_fp32 = static_cast<const float*>(left_data);  // ptr alias
  auto left_u8 = static_cast<const uint8_t*>(left_data);
  auto left_s8 = static_cast<const int8_t*>(left_data);

  auto right_fp32 = static_cast<const float*>(right_data);  // ptr alias
  auto right_u8 = static_cast<const uint8_t*>(right_data);
  auto right_s8 = static_cast<const int8_t*>(right_data);

  auto dst_fp32 = static_cast<float*>(dst_data);  // ptr alias
  auto dst_s32 = static_cast<int32_t*>(dst_data);
  auto dst_s8 = static_cast<int8_t*>(dst_data);
  auto dst_u8 = static_cast<uint8_t*>(dst_data);

  // TODO(zhe1wang): add per channel support for post-op;
  auto postop_list = op_desc.apply_postops_list();

// Computing the kernel
#pragma omp parallel for collapse(3)
  for (dim_t idx_mbs = 0; idx_mbs < num_mbs; ++idx_mbs) {
    for (dim_t i = 0; i < oc; ++i) {
      for (dim_t j = 0; j < micro_bs; ++j) {
        float value = 0;  // Consistent with the actual precision (float or double) of cpu instructions.
#pragma omp simd
        for (dim_t k = 0; k < ic; ++k) {
          dim_t l_idx = i * left_stride[0] + k * left_stride[1];
          dim_t r_idx = idx_mbs * right_stride[0] + k * right_stride[1] + j * right_stride[2];
          auto left_k = (left_dt == dt::fp32)
                            ? left_fp32[l_idx]
                            : ((left_dt == dt::u8) ? left_u8[l_idx] : ((left_dt == dt::s8) ? left_s8[l_idx] : 0));
          auto right_k = (right_dt == dt::fp32)
                             ? right_fp32[r_idx]
                             : ((right_dt == dt::u8) ? right_u8[r_idx] : ((right_dt == dt::s8) ? right_s8[r_idx] : 0));
          value += left_k * right_k;
        }

        // Accumulate bias or post sum
        if (has_bias) {
          value += bias_data[i];
        }
        dim_t dst_idx = idx_mbs * dst_stride[0] + i * dst_stride[1] + j * dst_stride[2];
        dim_t scale_idx = i;
        if (dst_dt != dt::s32) {
          value = value * scales_data[scale_idx];
        }
        if (append_sum) {
          value += dst_fp32[dst_idx];
        }
        value = apply_postop_list(value, postop_list);
        // Quantize dst data
        if (dst_dt == dt::fp32) {
          dst_fp32[dst_idx] = static_cast<float>(value);
        } else if (dst_dt == dt::s32) {
          dst_s32[dst_idx] = static_cast<int32_t>(value);
        } else if (dst_dt == dt::s8) {
          dst_s8[dst_idx] = static_cast<int8_t>(value);
        } else if (dst_dt == dt::u8) {
          dst_u8[dst_idx] = static_cast<uint8_t>(value);
        }
      }
    }
  }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    n_thread_t with_n_thread(p.nthr);
    const auto& op_desc = p.op_desc;
    sparse_matmul_desc spmm_desc(op_desc);
    sparse_matmul spmm_kern(spmm_desc);
    spmm_kern.execute(p.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    get_true_data(q.op_desc, q.rt_data);
    auto buf1 = p.rt_data[ssd::DST];
    auto size1 = p.op_desc.tensor_descs()[ssd::DST].size();
    auto buf2 = q.rt_data[ssd::DST];
    auto size2 = q.op_desc.tensor_descs()[ssd::DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[ssd::DST].dtype();
    if (dst_type == dt::fp32) {
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::s32) {
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
    } else if (dst_type == dt::s8) {
      return compare_data<int8_t>(buf1, size1, buf2, size2, 8e-3);
    }
  }
  return false;
}

class SpmmVNNIKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  SpmmVNNIKernelTest() {}
  virtual ~SpmmVNNIKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SpmmVNNIKernelTest, ) {
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
  auto op_desc = t.args.first.op_desc;
  auto op_attrs = op_desc.attrs();
  const uint64_t data_addr = str_to_num<uint64_t>(op_attrs["sparse_ptr"]);
  bsr_data_t<int8_t>* bsr_data = reinterpret_cast<bsr_data_t<int8_t>*>(data_addr);
  delete bsr_data;
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

std::pair<const void*, const void*> make_data_obj(const std::vector<int64_t>& a_shape, const dt& a_dt,
                                                  bool is_clear = false, float sparsity = 0.f,
                                                  const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    if (a_dt == dt::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
      if (sparsity != 0.f) {
        int8_t* s8_ptr = static_cast<int8_t*>(data_ptr);
        prepare_sparse_data(s8_ptr, a_shape[0], a_shape[1], 4, 1, sparsity);
      }
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::vector<float> make_output_scale(dim_t size, const std::vector<float>& ranges = {0, 10}) {
  std::vector<float> output_scale(size, 0);
  init_vector(output_scale.data(), size, ranges[0], ranges[1]);
  return output_scale;
}

std::vector<float> make_output_zo(dim_t size, const std::vector<float>& ranges = {-100, -1}) {
  std::vector<float> output_zo(size, 0);
  init_vector(output_zo.data(), size, ranges[0], ranges[1]);
  return output_zo;
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, float sparsity, dim_t micro_bs = -1, int nthr = 0,
                                         jd::data_type dt_dst = dt::s8,
                                         std::unordered_map<std::string, std::string> op_attrs = {},
                                         std::vector<postop_alg> postop_algs = {}) {
  bool append_sum = (op_attrs["append_sum"] == "true");
  LOG_IF(FATAL, append_sum && dt_dst != dt::fp32) << "append_sum must be applied with fp32 dst type";
  micro_bs = micro_bs <= 0 ? N : micro_bs;
  LOG_IF(FATAL, N % micro_bs != 0) << "micro_bs must be a multiple of N";
  dim_t num_mbs = N / micro_bs;

  // Step 1: Construct runtime data
  tensor_desc wei_desc = {{M, K}, dt::s8, ft::bsr};
  tensor_desc src_desc = {{num_mbs, K, micro_bs}, dt::u8, ft::ab};
  tensor_desc bia_desc = {{M, 1}, dt::s32, ft::ab};
  tensor_desc dst_desc = {{num_mbs, M, micro_bs}, dt_dst, ft::ab};
  tensor_desc scales_desc = {{M, 1}, dt::fp32, ft::ab};
  std::vector<tensor_desc> ts_descs = {wei_desc, src_desc, bia_desc, dst_desc, scales_desc};

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == ssd::DST && !append_sum);
    float data_sparsity = (index == ssd::WEI) ? sparsity : 0;
    auto ranges = (index == ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, data_sparsity, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  // TODO(zhe1wang): add per channel support for post-op;
  float scale = make_output_scale(1)[0];
  float zero_point = make_output_zo(1)[0];

  // Step 2: sparse data encoding
  auto sparse_ptr = new bsr_data_t<int8_t>(
      spns::reorder_to_bsr_group<int8_t, 4>(M, K, 4, 1, static_cast<const int8_t*>(rt_data1[ssd::WEI])));
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  std::vector<postop_attr> apply_postops_list;
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, postop_alg alg) { return str_lists + '_' + postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(postop_alg_name[postop_algs[0]]), accu_op);
    for (auto& alg : postop_algs) {
      postop_attr attr(dt::fp32, postop_type::eltwise, alg, 0.0, 0.0, scale);
      apply_postops_list.push_back(attr);
    }
  }
  if (dt_dst == dt::s8 || dt_dst == dt::u8) {
    postop_attr attr(dt_dst, postop_type::eltwise, postop_alg::quantize, zero_point, 0.0, scale);
    apply_postops_list.push_back(attr);
  }
  operator_desc an_op_desc(kernel_kind::sparse_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                           op_attrs, apply_postops_list);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1, sparsity, nthr};
  op_args_t op_args_copy = {an_op_desc, rt_data2, sparsity, nthr};

  return {op_args, op_args_copy};
}

std::string level_to_string(const ssd::subfunc_level& l) { return std::to_string(static_cast<uint8_t>(l)); }

static auto case_func = []() {
  std::vector<std::vector<dim_t>> bert_sizes = {
      // mini
      {256, 256, 128},
      {256, 256, 384},
      {256, 1024, 128},
      {256, 1024, 384},
      {1024, 256, 128},
      {1024, 256, 384},
      // base
      {768, 768, 128},
      {768, 768, 384},
      {768, 3072, 128},
      {768, 3072, 384},
      {3072, 768, 128},
      {3072, 768, 384},
      // large
      {1024, 1024, 128},
      {1024, 1024, 384},
      {1024, 4096, 128},
      {1024, 4096, 384},
      {4096, 1024, 128},
      {4096, 1024, 384},
  };

  std::vector<std::vector<postop_alg>> postop_lists = {
      {},
      {postop_alg::gelu},
  };

  google::InitGoogleLogging("SpmmVNNIKernelTest");
  std::vector<int> nthr_cases;
  bool use_benchmark =
      getenv("SPARSE_LIB_USE_BENCHMARK") != nullptr && strcmp(getenv("SPARSE_LIB_USE_BENCHMARK"), "1") == 0;
  if (use_benchmark  // number cores is decided by numactl in benchmarking
      || (getenv(OMP_NUM_THREADS) != nullptr && strlen(getenv(OMP_NUM_THREADS)) != 0)) {
    nthr_cases = {0};
  } else {  // OMP_NUM_THREAD;S is set outside
    nthr_cases = {1, 2, 3, 4, 0};
  }

  std::vector<test_params_t> cases;

  for (auto& algs : postop_lists) {
    for (auto bert_size : bert_sizes) {
      cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, dt::u8, {}, algs)});
      cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, dt::s8, {}, algs)});
      cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, dt::fp32, {}, algs)});
      cases.push_back(
          {gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, dt::fp32, {{"append_sum", "true"}}, algs)});
    }
  }

  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);

    if (!use_benchmark) {
      // Append sum with super high sparsity
      cases.push_back({gen_case(32, 32, 128, .99f, -1, nthr, dt::s8)});
      cases.push_back({gen_case(32, 32, 128, .99f, -1, nthr, dt::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 128, 1.0f, -1, nthr, dt::fp32, {{"append_sum", "true"}})});

      // Append sum with small batch size
      cases.push_back({gen_case(32, 32, 32, .7f, -1, nthr, dt::s8)});
      cases.push_back({gen_case(32, 32, 32, .7f, -1, nthr, dt::fp32)});
      cases.push_back({gen_case(32, 32, 32, .7f, -1, nthr, dt::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 16, .7f, -1, nthr, dt::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 48, .7f, -1, nthr, dt::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 224, .7f, -1, nthr, dt::fp32, {{"append_sum", "true"}})});

      // Test blocking
      cases.push_back({gen_case(256, 1024, 384, .7f, 64, nthr, dt::s8)});
      cases.push_back({gen_case(256, 1024, 384, .7f, -1, nthr, dt::s8, {{"micro_oc", "128"}})});
      cases.push_back({gen_case(256, 1024, 384, .7f, 64, nthr, dt::s8, {{"micro_oc", "128"}})});

      // Test subfunc_level
      cases.push_back(
          {gen_case(32, 32, 128, .7f, -1, nthr, dt::fp32, {{"sub_func", level_to_string(ssd::subfunc_level::none)}})});
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, dt::fp32,
                                {{"sub_func", level_to_string(ssd::subfunc_level::non_kdims)}})});
      cases.push_back(
          {gen_case(32, 32, 128, .7f, -1, nthr, dt::fp32, {{"sub_func", level_to_string(ssd::subfunc_level::kdims)}})});

      // case: sparse: s8xu8+s32=s8, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, dt::s8)});
      // case: sparse: s8xu8+s32=fp32, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, dt::fp32)});
      // case: sparse: s8xu8+s32+append_fp32=fp32, weight(M, K) * activation(K, N) + bias(M, 1) + append(M, N) =
      // dst(M, N)
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, dt::fp32, {{"append_sum", "true"}})});
    }

    // multiple cores with multiple batches
    int bs = omp_get_max_threads();
    if (bs != 1 && use_benchmark) {
      // without 3d input
      for (auto bert_size : bert_sizes) {
        cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2] * bs, .9f, -1, nthr, dt::s8)});
      }
      // with 3d input
      for (auto bert_size : bert_sizes) {
        cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2] * bs, .9f, bert_size[2], nthr, dt::s8)});
      }
    }
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_desc = tpi.param.args.first.op_desc.tensor_descs();
  auto& wei_shape = tensor_desc[ssd::WEI].shape();
  auto& src_shape = tensor_desc[ssd::SRC].shape();
  auto attrs_map = tpi.param.args.first.op_desc.attrs();
  dim_t oc = wei_shape[0];
  dim_t ic = wei_shape[1];
  dim_t bs = std::accumulate(src_shape.begin(), src_shape.end(), 1, std::multiplies<dim_t>()) / ic;
  dim_t micro_bs = src_shape.back();

  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back("sp" + std::to_string(static_cast<int>(tpi.param.args.first.sparisty * 100)));
  params.push_back(std::to_string(oc));
  params.push_back(std::to_string(ic));
  params.push_back(std::to_string(bs));
  switch (tensor_desc[ssd::DST].dtype()) {
    case dt::s8:
      params.push_back("s8");
      break;
    case dt::fp32:
      params.push_back("fp32");
      break;
    case dt::u8:
      params.push_back("u8");
      break;
    default:
      assert(false);
  }
  if (attrs_map["micro_oc"] != "") params.push_back("moc" + attrs_map["micro_oc"]);
  if (micro_bs != bs) params.push_back("mbs" + std::to_string(micro_bs));
  if (attrs_map["sub_func"] != "") params.push_back("sfunc" + attrs_map["sub_func"]);
  if (attrs_map["append_sum"] != "") {
    params.push_back(attrs_map["append_sum"]);
  }
  if (attrs_map["postop_list"] != "") params.push_back(attrs_map["postop_list"]);
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SpmmVNNIKernelTest, case_func(), test_suffix);
}  // namespace jd
