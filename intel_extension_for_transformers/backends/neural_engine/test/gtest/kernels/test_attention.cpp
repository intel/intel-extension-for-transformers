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
#include "kernels/attention_types.hpp"
#include "kernels/spmm_types.hpp"
#include "kernels/matmul_types.hpp"
#include "kernels/attention_ref.hpp"

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

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    n_thread_t with_n_thread(p.nthr);
    attention_desc attention_desc(p.op_desc);
    attention attention_kernel(attention_desc);
    attention_kernel.execute(p.rt_data);

    std::shared_ptr<const kernel_desc_t> attention_ref_desc;
    kernel_desc_t::create<attention_ref_kd_t>(attention_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> attention_ref_kernel;
    kernel_t::create<attention_ref_k_t, attention_ref_kd_t>(attention_ref_kernel, attention_ref_desc);
    attention_ref_kernel->execute(q.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[attention_io::MERGE_DST];
    auto size1 = p.op_desc.tensor_descs()[attention_io::MERGE_DST].size();
    auto buf2 = q.rt_data[attention_io::MERGE_DST];
    auto size2 = q.op_desc.tensor_descs()[attention_io::MERGE_DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[attention_io::MERGE_DST].dtype();
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

class SpmmAttentionKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  SpmmAttentionKernelTest() {}
  virtual ~SpmmAttentionKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(SpmmAttentionKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
  // free memory
  const auto& op_desc = t.args.first.op_desc;
  auto op_attrs = op_desc.attrs();

  for (auto rt_data : {t.args.first.rt_data, t.args.second.rt_data}) {
    for (auto idx : {attention_io::MERGE_SRC, attention_io::MERGE_DST, attention_io::Q_K_SRC2})
      delete[] reinterpret_cast<const uint8_t*>(rt_data[idx]);
    delete reinterpret_cast<const float*>(rt_data[attention_io::QK_V_OUTPUT_SCALES]);
    delete reinterpret_cast<const float*>(rt_data[attention_io::QK_V_OUTPUT_ZERO_POINT]);
  }

  for (std::string ptr_field : {"q_weight_ptr", "k_weight_ptr", "v_weight_ptr", "q_bias_ptr", "k_bias_ptr",
                                "v_bias_ptr", "q_scales_ptr", "k_scales_ptr", "v_scales_ptr", "sparse_ptr"})
    delete[] reinterpret_cast<uint8_t*>(str_to_num<uint64_t>(op_attrs[ptr_field]));
}

template <typename T>
void prepare_sparse_data(T* vector_data, dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, float sparsity,
                         uint32_t* seed = nullptr) {
  uint32_t default_seed = 123;
  if (seed == nullptr) seed = &default_seed;
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

const void* make_data_obj(const std::vector<int64_t>& a_shape, const dt& a_dt, const void* src_data = nullptr,
                          bool is_clear = false, float sparsity = 0.f, const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<size_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else if (src_data != nullptr) {
    data_ptr = new uint8_t[bytes_size];
    memcpy(data_ptr, src_data, bytes_size);
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
  return data_ptr;
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

std::pair<op_args_t, op_args_t> gen_case(const dim_t head_num, const dim_t head_size, const dim_t batch_size,
                                         const dim_t seq_len, const float sparsity, const int nthr = 0,
                                         const jd::data_type dt_dst = dt::u8,
                                         std::unordered_map<std::string, std::string> op_attrs = {}) {
  // Step 1: Construct runtime data for equivalent merged spmm
  const dim_t ip_chanel = head_num * head_size;  // channel width for any of the three linear layer
  std::vector<tensor_desc> ts_descs_qkv_ip = {
      {{ip_chanel * 3, ip_chanel}, dt::s8, ft::bsr},            // WEI
      {{batch_size, ip_chanel, seq_len}, dt::u8, ft::ab},       // SRC
      {{ip_chanel * 3, 1}, dt::s32, ft::ab},                    // BIAS
      {{ip_chanel * 3, batch_size * seq_len}, dt_dst, ft::ab},  // DST
      {{ip_chanel * 3, 1}, dt::fp32, ft::ab},                   // SCALE
  };
  std::vector<const void*> rt_data_qkv_ip(ts_descs_qkv_ip.size(), nullptr);
  for (size_t index = 0; index < ts_descs_qkv_ip.size(); ++index) {
    auto& tsd = ts_descs_qkv_ip[index];
    bool is_clear = index == ssd::DST;
    float data_sparsity = (index == ssd::WEI) ? sparsity : 0;
    auto ranges = (index == ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    const void* data_addr = make_data_obj(tsd.shape(), tsd.dtype(), nullptr, is_clear, data_sparsity, ranges);
    rt_data_qkv_ip[index] = data_addr;
  }

  tensor_desc attention_src_desc = {{batch_size, ip_chanel, seq_len}, dt::u8, ft::ab};
  tensor_desc attention_dst_desc = {{head_num, head_size, batch_size, seq_len}, dt::u8, ft::ab};
  tensor_desc q_weight_desc = {{ip_chanel, ip_chanel}, dt::s8, ft::bsr};
  tensor_desc k_weight_desc = {{ip_chanel, ip_chanel}, dt::s8, ft::bsr};
  tensor_desc v_weight_desc = {{ip_chanel, ip_chanel}, dt::s8, ft::bsr};
  tensor_desc q_bias_desc = {{ip_chanel, 1}, dt::s32, ft::ab};
  tensor_desc k_bias_desc = {{ip_chanel, 1}, dt::s32, ft::ab};
  tensor_desc v_bias_desc = {{ip_chanel, 1}, dt::s32, ft::ab};
  tensor_desc q_scales_desc = {{ip_chanel, 1}, dt::fp32, ft::ab};
  tensor_desc k_scales_desc = {{ip_chanel, 1}, dt::fp32, ft::ab};
  tensor_desc v_scales_desc = {{ip_chanel, 1}, dt::fp32, ft::ab};
  tensor_desc reshape_desc = {{batch_size, seq_len}, dt::fp32, ft::ab};
  tensor_desc q_k_src2_desc = {{batch_size, head_num, seq_len, seq_len}, dt::fp32, ft::ab};
  tensor_desc q_k_scales_desc = {{}, dt::undef, ft::undef};  // currently pass by static value
  tensor_desc qk_v_scales_desc = {{1}, dt::fp32, ft::a};
  tensor_desc qk_v_zp_desc = {{1}, dt::fp32, ft::a};

  std::vector<tensor_desc> ts_descs = {attention_src_desc, attention_dst_desc, q_weight_desc, k_weight_desc,
                                       v_weight_desc,      q_bias_desc,        k_bias_desc,   v_bias_desc,
                                       q_scales_desc,      k_scales_desc,      v_scales_desc, reshape_desc,
                                       q_k_src2_desc,      q_k_scales_desc,    qk_v_zp_desc,  qk_v_scales_desc};

  std::vector<std::vector<dim_t>> ts_shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_shapes.begin(), [](tensor_desc d) { return d.shape(); });
  std::vector<dt> ts_types(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_types.begin(), [](tensor_desc d) { return d.dtype(); });

  std::vector<const void*> rt_data_p(attention_io::QK_V_OUTPUT_SCALES + 1, nullptr);
  std::vector<const void*> rt_data_q(attention_io::QK_V_OUTPUT_SCALES + 1, nullptr);

  rt_data_p[attention_io::MERGE_SRC] =
      make_data_obj(ts_shapes[attention_io::MERGE_SRC], ts_types[attention_io::MERGE_SRC], rt_data_qkv_ip[ssd::SRC]);
  rt_data_q[attention_io::MERGE_SRC] =
      make_data_obj(ts_shapes[attention_io::MERGE_SRC], ts_types[attention_io::MERGE_SRC], rt_data_qkv_ip[ssd::SRC]);

  rt_data_p[attention_io::MERGE_DST] =
      make_data_obj(ts_shapes[attention_io::MERGE_DST], ts_types[attention_io::MERGE_DST], nullptr, true);
  rt_data_q[attention_io::MERGE_DST] =
      make_data_obj(ts_shapes[attention_io::MERGE_DST], ts_types[attention_io::MERGE_DST], nullptr, true);

  // for binary add of QxK matmul
  rt_data_p[attention_io::Q_K_SRC2] =
      make_data_obj(ts_shapes[attention_io::Q_K_SRC2], ts_types[attention_io::Q_K_SRC2], nullptr);
  rt_data_q[attention_io::Q_K_SRC2] = make_data_obj(ts_shapes[attention_io::Q_K_SRC2], ts_types[attention_io::Q_K_SRC2],
                                                    rt_data_p[attention_io::Q_K_SRC2]);

  // for output scale & zp of QKxV matmul
  rt_data_p[attention_io::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_q[attention_io::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_p[attention_io::QK_V_OUTPUT_SCALES] = new float(.003f);
  rt_data_q[attention_io::QK_V_OUTPUT_SCALES] = new float(.003f);

  // Merge weight
  const size_t wei_bytes =
      ts_descs[attention_io::Q_WEIGHT].size() * type_size[ts_descs[attention_io::Q_WEIGHT].dtype()];
  char* q_weight_addr = new char[wei_bytes];
  char* k_weight_addr = new char[wei_bytes];
  char* v_weight_addr = new char[wei_bytes];
  const char* rt_data_qkv_ip_wei = static_cast<const char*>(rt_data_qkv_ip[ssd::WEI]);
  memcpy(q_weight_addr, rt_data_qkv_ip_wei, wei_bytes);
  memcpy(k_weight_addr, rt_data_qkv_ip_wei + wei_bytes, wei_bytes);
  memcpy(v_weight_addr, rt_data_qkv_ip_wei + wei_bytes * 2, wei_bytes);
  op_attrs["q_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_weight_addr));
  op_attrs["k_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_weight_addr));
  op_attrs["v_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_weight_addr));

  // Merge bias
  const size_t bias_bytes = ts_descs[attention_io::Q_BIAS].size() * type_size[ts_descs[attention_io::Q_BIAS].dtype()];
  char* q_bias_addr = new char[bias_bytes];
  char* k_bias_addr = new char[bias_bytes];
  char* v_bias_addr = new char[bias_bytes];
  const char* rt_data_qkv_ip_bias = static_cast<const char*>(rt_data_qkv_ip[ssd::BIAS]);
  memcpy(q_bias_addr, rt_data_qkv_ip_bias, bias_bytes);
  memcpy(k_bias_addr, rt_data_qkv_ip_bias + bias_bytes, bias_bytes);
  memcpy(v_bias_addr, rt_data_qkv_ip_bias + bias_bytes * 2, bias_bytes);
  op_attrs["q_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_bias_addr));
  op_attrs["k_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_bias_addr));
  op_attrs["v_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_bias_addr));

  // Merge scales
  const size_t scale_bytes =
      ts_descs[attention_io::Q_SCALES].size() * type_size[ts_descs[attention_io::Q_SCALES].dtype()];
  char* q_scales_addr = new char[scale_bytes];
  char* k_scales_addr = new char[scale_bytes];
  char* v_scales_addr = new char[scale_bytes];
  const char* rt_data_qkv_ip_scale = static_cast<const char*>(rt_data_qkv_ip[ssd::SCALES]);
  memcpy(q_scales_addr, rt_data_qkv_ip_scale, scale_bytes);
  memcpy(k_scales_addr, rt_data_qkv_ip_scale + scale_bytes, scale_bytes);
  memcpy(v_scales_addr, rt_data_qkv_ip_scale + scale_bytes * 2, scale_bytes);
  op_attrs["q_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_scales_addr));
  op_attrs["k_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_scales_addr));
  op_attrs["v_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_scales_addr));

  operator_desc op_desc(kernel_kind::attention, kernel_prop::forward_inference, engine_kind::cpu, ts_descs, op_attrs);

  // Step 3: op_args_t testcase pair
  op_args_t op_args_p = {op_desc, rt_data_p, sparsity, nthr};
  op_args_t op_args_q = {op_desc, rt_data_q, sparsity, nthr};

  for (size_t index = 0; index < ts_descs_qkv_ip.size(); ++index) {
    delete[] reinterpret_cast<const char*>(rt_data_qkv_ip[index]);
  }

  return {op_args_p, op_args_q};
}

static auto case_func = []() {
  google::InitGoogleLogging("SpmmAttentionKernelTest");
  std::vector<int> nthr_cases = {1, 2, 3, 4, 0};
  std::vector<test_params_t> cases;
  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);
    cases.push_back({gen_case(1, 32, 1, 32, .85f, nthr, dt::u8,
                              {
                                  {"out_scale", "0.125"},
                                  {"scr0_scale", "1"},
                                  {"scr1_scale", "1"},
                                  {"softmax_in_zero_point", "140"},
                                  {"softmax_in_scale", "500"},
                                  {"softmax_out_zero_point", "0"},
                                  {"softmax_out_scale", "0.00324144"},
                              })});
    cases.push_back({gen_case(16, 64, 2, 128, .85f, nthr, dt::u8,
                              {
                                  {"out_scale", "0.125"},
                                  {"scr0_scale", "1"},
                                  {"scr1_scale", "1"},
                                  {"softmax_in_zero_point", "140"},
                                  {"softmax_in_scale", "500"},
                                  {"softmax_out_zero_point", "0"},
                                  {"softmax_out_scale", "0.00324144"},
                              })});
    cases.push_back({gen_case(4, 64, 2, 128, .85f, nthr, dt::u8,
                              {
                                  {"out_scale", "0.125"},
                                  {"scr0_scale", "1"},
                                  {"scr1_scale", "1"},
                                  {"softmax_in_zero_point", "140"},
                                  {"softmax_in_scale", "500"},
                                  {"softmax_out_zero_point", "0"},
                                  {"softmax_out_scale", "0.00324144"},
                              })});
    cases.push_back({gen_case(12, 64, 2, 128, .85f, nthr, dt::u8,
                              {
                                  {"out_scale", "0.125"},
                                  {"scr0_scale", "1"},
                                  {"scr1_scale", "1"},
                                  {"softmax_in_zero_point", "140"},
                                  {"softmax_in_scale", "500"},
                                  {"softmax_out_zero_point", "0"},
                                  {"softmax_out_scale", "0.00324144"},
                              })});
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
  switch (tensor_desc[attention_io::MERGE_DST].dtype()) {
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

INSTANTIATE_TEST_SUITE_P(SparseLib, SpmmAttentionKernelTest, case_func(), test_suffix);
}  // namespace jd
