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
#include "src/cpu/kernels/spmm_ref.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"
#define WORKSPACE

namespace test {
struct op_args_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data;
  float sparisty;  // sparsity of weight matrix; for testcase labeling
  int nthr;        // 0 for not touching OMP_NUM_THREADS and using what set outside
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

bool check_result(const test_params_t& t) {
  bool result = false;
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    n_thread_t with_n_thread(p.nthr);
    jd::sparse_matmul_desc spmm_desc(p.op_desc);
    jd::sparse_matmul spmm_kern(spmm_desc);
    spmm_kern.execute(p.rt_data);

    std::shared_ptr<const jd::kernel_desc_t> spmm_ref_desc;
    jd::kernel_desc_t::create<jd::spmm_ref_kd_t>(spmm_ref_desc, q.op_desc);
    std::shared_ptr<const jd::kernel_t> spmm_ref_kernel;
    jd::kernel_t::create<jd::spmm_ref_k_t, jd::spmm_ref_kd_t>(spmm_ref_kernel, spmm_ref_desc);
    spmm_ref_kernel->execute(q.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[jd::ssd::DST];
    auto size1 = p.op_desc.tensor_descs()[jd::ssd::DST].size();
    auto buf2 = q.rt_data[jd::ssd::DST];
    auto size2 = q.op_desc.tensor_descs()[jd::ssd::DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[jd::ssd::DST].dtype();
    if (dst_type == jd::data_type::fp32) {
      result = compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == jd::data_type::s32) {
      result = compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == jd::data_type::u8) {
      result = compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
    } else if (dst_type == jd::data_type::s8) {
      result = compare_data<int8_t>(buf1, size1, buf2, size2, 8e-3);
    }
    if (p.rt_data.size() > jd::ssd::DST_M2) {
      // Check M1
      buf1 = p.rt_data[jd::ssd::DST_M1];
      size1 = p.op_desc.tensor_descs()[jd::ssd::DST_M1].size();
      buf2 = q.rt_data[jd::ssd::DST_M1];
      size2 = q.op_desc.tensor_descs()[jd::ssd::DST_M1].size();
      // Should compare buffer with different addresses
      EXPECT_NE(buf1, buf2);
      result &= compare_data<float>(buf1, size1, buf2, size2, 5e-3);

      // Check M2
      buf1 = p.rt_data[jd::ssd::DST_M2];
      size1 = p.op_desc.tensor_descs()[jd::ssd::DST_M2].size();
      buf2 = q.rt_data[jd::ssd::DST_M2];
      size2 = q.op_desc.tensor_descs()[jd::ssd::DST_M2].size();
      // Should compare buffer with different addresses
      EXPECT_NE(buf1, buf2);
      result &= compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    }
  }
  return result;
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
  jd::bsr_data_t<int8_t>* bsr_data = reinterpret_cast<jd::bsr_data_t<int8_t>*>(data_addr);
  delete bsr_data;
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

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, float sparsity, dim_t micro_bs = -1,
                                         int nthr = 0, jd::data_type dt_dst = jd::data_type::s8,
                                         std::unordered_map<std::string, std::string> op_attrs = {},
                                         std::vector<jd::postop_alg> postop_algs = {}) {
  bool append_sum = (op_attrs["append_sum"] == "true");
  bool mean_var = (op_attrs["welford"] == "true");
  LOG_IF(FATAL, append_sum && dt_dst != jd::data_type::fp32) << "append_sum must be applied with fp32 dst type";
  micro_bs = micro_bs <= 0 ? N : micro_bs;
  LOG_IF(FATAL, N % micro_bs != 0) << "micro_bs must be a multiple of N";
  dim_t num_mbs = N / micro_bs;

  // Step 1: Construct runtime data
  jd::tensor_desc wei_desc = {{M, K}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc src_desc = {{num_mbs, K, micro_bs}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc bia_desc = {{M, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{num_mbs, M, micro_bs}, dt_dst, jd::format_type::ab};
  jd::tensor_desc scales_desc = {{M, 1}, jd::data_type::fp32, jd::format_type::ab};
  std::vector<jd::tensor_desc> ts_descs = {wei_desc, src_desc, bia_desc, dst_desc, scales_desc};
  if (dt_dst == jd::data_type::fp32 && mean_var == true) {
    jd::tensor_desc mean_desc = {{num_mbs, micro_bs}, jd::data_type::fp32, jd::format_type::a};
    jd::tensor_desc var_desc = {{num_mbs, micro_bs}, jd::data_type::fp32, jd::format_type::a};
    ts_descs.push_back(mean_desc);
    ts_descs.push_back(var_desc);
#ifdef WORKSPACE
    jd::tensor_desc workspace_desc = {{M * 2, N}, jd::data_type::fp32, jd::format_type::ab};
    ts_descs.push_back(workspace_desc);
#endif
  }

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;

  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == jd::ssd::DST && !append_sum);
    float data_sparsity = (index == jd::ssd::WEI) ? sparsity : 0;
    auto ranges = (index == jd::ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges, data_sparsity);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  // TODO(zhe1wang): add per channel support for post-op;
  float scale = make_output_scale(1)[0];
  float zero_point = make_output_zo(1)[0];

  // Step 2: sparse data encoding
  auto sparse_ptr = new jd::bsr_data_t<int8_t>(
      jd::spns::reorder_to_bsr_group<int8_t, 4>(M, K, 4, 1, static_cast<const int8_t*>(rt_data1[jd::ssd::WEI])));
  op_attrs["sparse_ptr"] = std::to_string(reinterpret_cast<uint64_t>(sparse_ptr));
  std::vector<jd::postop_attr> apply_postops_list;
  if (postop_algs.size()) {
    auto accu_op = [](std::string str_lists, jd::postop_alg alg) { return str_lists + '_' + jd::postop_alg_name[alg]; };
    op_attrs["postop_list"] = std::accumulate(postop_algs.begin() + 1, postop_algs.end(),
                                              std::string(jd::postop_alg_name[postop_algs[0]]), accu_op);
    for (auto& alg : postop_algs) {
      jd::postop_attr attr(jd::data_type::fp32, jd::postop_type::eltwise, alg, 0.0, 0.0, scale);
      apply_postops_list.push_back(attr);
    }
  }
  if (dt_dst == jd::data_type::s8 || dt_dst == jd::data_type::u8) {
    jd::postop_attr attr(dt_dst, jd::postop_type::eltwise, jd::postop_alg::quantize, zero_point, 0.0, scale);
    apply_postops_list.push_back(attr);
  }
  jd::operator_desc an_op_desc(jd::kernel_kind::sparse_matmul, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                               ts_descs, op_attrs, apply_postops_list);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {an_op_desc, rt_data1, sparsity, nthr};
  op_args_t op_args_copy = {an_op_desc, rt_data2, sparsity, nthr};

  return {op_args, op_args_copy};
}

std::string level_to_string(const jd::ssd::subfunc_level& l) { return std::to_string(static_cast<uint8_t>(l)); }

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

  std::vector<std::vector<jd::postop_alg>> postop_lists = {
      {},
      {jd::postop_alg::gelu},
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
      cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, jd::data_type::u8, {}, algs)});
      cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, jd::data_type::s8, {}, algs)});
      cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, jd::data_type::fp32, {}, algs)});
      cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2], .9f, -1, 0, jd::data_type::fp32,
                                {{"append_sum", "true"}}, algs)});
    }
  }

  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);

    if (!use_benchmark) {
      // Append sum with super high sparsity
      cases.push_back({gen_case(32, 32, 128, .99f, -1, nthr, jd::data_type::s8)});
      cases.push_back({gen_case(32, 32, 128, .99f, -1, nthr, jd::data_type::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 128, 1.0f, -1, nthr, jd::data_type::fp32, {{"append_sum", "true"}})});

      // Append sum with small batch size
      cases.push_back({gen_case(32, 32, 32, .7f, -1, nthr, jd::data_type::s8)});
      cases.push_back({gen_case(32, 32, 32, .7f, -1, nthr, jd::data_type::fp32)});
      cases.push_back({gen_case(32, 32, 32, .7f, -1, nthr, jd::data_type::fp32,
                                {{"append_sum", "true"}, {"welford", "true"}}, {})});
      cases.push_back({gen_case(256, 1024, 384, .7f, -1, nthr, jd::data_type::fp32,
                                {{"append_sum", "true"}, {"welford", "true"}}, {})});
      cases.push_back({gen_case(256, 1024, 1536, .7f, 384, nthr, jd::data_type::fp32,
                                {{"append_sum", "true"}, {"welford", "true"}}, {})});
      cases.push_back({gen_case(1024, 1024, 1536, .7f, 384, nthr, jd::data_type::fp32,
                                {{"append_sum", "true"}, {"welford", "true"}}, {})});

      cases.push_back({gen_case(32, 32, 32, .7f, -1, nthr, jd::data_type::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 16, .7f, -1, nthr, jd::data_type::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 48, .7f, -1, nthr, jd::data_type::fp32, {{"append_sum", "true"}})});
      cases.push_back({gen_case(32, 32, 224, .7f, -1, nthr, jd::data_type::fp32, {{"append_sum", "true"}})});

      // Test blocking
      cases.push_back({gen_case(256, 1024, 384, .7f, 64, nthr, jd::data_type::s8)});
      cases.push_back({gen_case(256, 1024, 384, .7f, -1, nthr, jd::data_type::s8, {{"micro_oc", "128"}})});
      cases.push_back({gen_case(256, 1024, 384, .7f, 64, nthr, jd::data_type::s8, {{"micro_oc", "128"}})});

      // Test subfunc_level
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, jd::data_type::fp32,
                                {{"sub_func", level_to_string(jd::ssd::subfunc_level::none)}})});
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, jd::data_type::fp32,
                                {{"sub_func", level_to_string(jd::ssd::subfunc_level::non_kdims)}})});
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, jd::data_type::fp32,
                                {{"sub_func", level_to_string(jd::ssd::subfunc_level::kdims)}})});

      // case: sparse: s8xu8+s32=s8, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, jd::data_type::s8)});
      // case: sparse: s8xu8+s32=fp32, weight(M, K) * activation(K, N) + bias(M, 1) = dst(M, N)
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, jd::data_type::fp32)});
      // case: sparse: s8xu8+s32+append_fp32=fp32, weight(M, K) * activation(K, N) + bias(M, 1) + append(M, N) =
      // dst(M, N)
      cases.push_back({gen_case(32, 32, 128, .7f, -1, nthr, jd::data_type::fp32, {{"append_sum", "true"}})});
    }

    // multiple cores with multiple batches
    int bs = omp_get_max_threads();
    if (bs != 1 && use_benchmark) {
      // without 3d input
      for (auto bert_size : bert_sizes) {
        cases.push_back({gen_case(bert_size[0], bert_size[1], bert_size[2] * bs, .9f, -1, nthr, jd::data_type::s8)});
      }
      // with 3d input
      for (auto bert_size : bert_sizes) {
        cases.push_back(
            {gen_case(bert_size[0], bert_size[1], bert_size[2] * bs, .9f, bert_size[2], nthr, jd::data_type::s8)});
      }
    }
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_descs = tpi.param.args.first.op_desc.tensor_descs();
  auto& wei_shape = tensor_descs[jd::ssd::WEI].shape();
  auto& src_shape = tensor_descs[jd::ssd::SRC].shape();
  auto attrs_map = tpi.param.args.first.op_desc.attrs();
  dim_t oc = wei_shape[0];
  dim_t ic = wei_shape[1];
  dim_t bs = std::accumulate(src_shape.begin(), src_shape.end(), dim_t{1}, std::multiplies<dim_t>()) / ic;
  dim_t micro_bs = src_shape.back();

  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back("sp" + std::to_string(static_cast<int>(tpi.param.args.first.sparisty * 100)));
  params.push_back(std::to_string(oc));
  params.push_back(std::to_string(ic));
  params.push_back(std::to_string(bs));
  switch (tensor_descs[jd::ssd::DST].dtype()) {
    case jd::data_type::s8:
      params.push_back("s8");
      break;
    case jd::data_type::fp32:
      params.push_back("fp32");
      break;
    case jd::data_type::u8:
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
  if (tensor_descs.size() == jd::ssd::DST_M2 + 1 || tensor_descs.size() == jd::ssd::WORK_SPACE + 1)
    params.push_back("mean_var");
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SpmmVNNIKernelTest, case_func(), test_suffix);
}  // namespace test
