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
#include <exception>
#include "gtest/gtest.h"

#include "interface.hpp"
#include "kernels/attention_types.hpp"
#include "kernels/matmul_types.hpp"
#include "kernels/spmm_types.hpp"
#include "src/cpu/kernels/attention_ref.hpp"
#include "unit_test_utils.hpp"

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
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    n_thread_t with_n_thread(p.nthr);
    jd::attention_desc attention_desc(p.op_desc);
    jd::attention attention_kernel(attention_desc);
    attention_kernel.execute(p.rt_data);

    std::shared_ptr<const jd::kernel_desc_t> attention_ref_desc;
    jd::kernel_desc_t::create<jd::attention_ref_kd_t>(attention_ref_desc, q.op_desc);
    std::shared_ptr<const jd::kernel_t> attention_ref_kernel;
    jd::kernel_t::create<jd::attention_ref_k_t, jd::attention_ref_kd_t>(attention_ref_kernel, attention_ref_desc);
    attention_ref_kernel->execute(q.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[jd::attention_io::MERGE_DST];
    auto size1 = p.op_desc.tensor_descs()[jd::attention_io::MERGE_DST].size();
    auto buf2 = q.rt_data[jd::attention_io::MERGE_DST];
    auto size2 = q.op_desc.tensor_descs()[jd::attention_io::MERGE_DST].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[jd::attention_io::MERGE_DST].dtype();
    if (dst_type == jd::data_type::fp32) {
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == jd::data_type::s32) {
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == jd::data_type::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 8e-3);
    } else if (dst_type == jd::data_type::s8) {
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
    for (auto idx : {jd::attention_io::MERGE_SRC, jd::attention_io::MERGE_DST, jd::attention_io::Q_K_SRC2})
      delete[] reinterpret_cast<const uint8_t*>(rt_data[idx]);
    delete reinterpret_cast<const float*>(rt_data[jd::attention_io::QK_V_OUTPUT_SCALES]);
    delete reinterpret_cast<const float*>(rt_data[jd::attention_io::QK_V_OUTPUT_ZERO_POINT]);
  }

  for (std::string ptr_field : {"q_weight_ptr", "k_weight_ptr", "v_weight_ptr", "q_bias_ptr", "k_bias_ptr",
                                "v_bias_ptr", "q_scales_ptr", "k_scales_ptr", "v_scales_ptr", "sparse_ptr"})
    delete[] reinterpret_cast<uint8_t*>(str_to_num<uint64_t>(op_attrs[ptr_field]));
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
                                         const jd::data_type dt_dst = jd::data_type::u8,
                                         std::unordered_map<std::string, std::string> op_attrs = {}) {
  // Step 1: Construct runtime data for equivalent merged spmm
  const dim_t ip_chanel = head_num * head_size;  // channel width for any of the three linear layer
  std::vector<jd::tensor_desc> ts_descs_qkv_ip = {
      {{ip_chanel * 3, ip_chanel}, jd::data_type::s8, jd::format_type::bsr},       // WEI
      {{batch_size, ip_chanel, seq_len}, jd::data_type::u8, jd::format_type::ab},  // SRC
      {{ip_chanel * 3, 1}, jd::data_type::s32, jd::format_type::ab},               // BIAS
      {{ip_chanel * 3, batch_size * seq_len}, dt_dst, jd::format_type::ab},        // DST
      {{ip_chanel * 3, 1}, jd::data_type::fp32, jd::format_type::ab},              // SCALE
  };
  std::vector<const void*> rt_data_qkv_ip(ts_descs_qkv_ip.size(), nullptr);
  for (size_t index = 0; index < ts_descs_qkv_ip.size(); ++index) {
    auto& tsd = ts_descs_qkv_ip[index];
    bool is_clear = index == jd::ssd::DST;
    float data_sparsity = (index == jd::ssd::WEI) ? sparsity : 0;
    auto ranges = (index == jd::ssd::SCALES) ? std::vector<float>{0, 1} : std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges, data_sparsity);
    rt_data_qkv_ip[index] = data_pair.first;
    if (data_pair.second != nullptr) {
      delete[] reinterpret_cast<const char*>(data_pair.second);
    }
  }

  jd::tensor_desc attention_src_desc = {{batch_size, ip_chanel, seq_len}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc attention_dst_desc = {
      {head_num, head_size, batch_size, seq_len}, jd::data_type::u8, jd::format_type::ab};
  jd::tensor_desc q_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc k_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc v_weight_desc = {{ip_chanel, ip_chanel}, jd::data_type::s8, jd::format_type::bsr};
  jd::tensor_desc q_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc k_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc v_bias_desc = {{ip_chanel, 1}, jd::data_type::s32, jd::format_type::ab};
  jd::tensor_desc q_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc k_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc v_scales_desc = {{ip_chanel, 1}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc reshape_desc = {{batch_size, seq_len}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc q_k_src2_desc = {{batch_size, head_num, seq_len, seq_len}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc q_k_scales_desc = {
      {}, jd::data_type::undef, jd::format_type::undef};  // currently pass by static value
  jd::tensor_desc qk_v_scales_desc = {{1}, jd::data_type::fp32, jd::format_type::a};
  jd::tensor_desc qk_v_zp_desc = {{1}, jd::data_type::fp32, jd::format_type::a};

  std::vector<jd::tensor_desc> ts_descs = {attention_src_desc, attention_dst_desc, q_weight_desc, k_weight_desc,
                                           v_weight_desc,      q_bias_desc,        k_bias_desc,   v_bias_desc,
                                           q_scales_desc,      k_scales_desc,      v_scales_desc, reshape_desc,
                                           q_k_src2_desc,      q_k_scales_desc,    qk_v_zp_desc,  qk_v_scales_desc};

  std::vector<std::vector<dim_t>> ts_shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_shapes.begin(), [](jd::tensor_desc d) { return d.shape(); });
  std::vector<jd::data_type> ts_types(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), ts_types.begin(), [](jd::tensor_desc d) { return d.dtype(); });

  std::vector<const void*> rt_data_p(jd::attention_io::QK_V_OUTPUT_SCALES + 1, nullptr);
  std::vector<const void*> rt_data_q(jd::attention_io::QK_V_OUTPUT_SCALES + 1, nullptr);

  auto data_pair = make_data_obj(ts_shapes[jd::attention_io::MERGE_SRC], ts_types[jd::attention_io::MERGE_SRC], false,
                                 {-10, 10}, 0.f, jd::format_type::uncoded, rt_data_qkv_ip[jd::ssd::SRC]);
  rt_data_p[jd::attention_io::MERGE_SRC] = data_pair.first;
  rt_data_q[jd::attention_io::MERGE_SRC] = data_pair.second;

  data_pair = make_data_obj(ts_shapes[jd::attention_io::MERGE_DST], ts_types[jd::attention_io::MERGE_DST], true);
  rt_data_p[jd::attention_io::MERGE_DST] = data_pair.first;
  rt_data_q[jd::attention_io::MERGE_DST] = data_pair.second;

  // for binary add of QxK matmul
  data_pair = make_data_obj(ts_shapes[jd::attention_io::Q_K_SRC2], ts_types[jd::attention_io::Q_K_SRC2]);
  rt_data_p[jd::attention_io::Q_K_SRC2] = data_pair.first;
  rt_data_q[jd::attention_io::Q_K_SRC2] = data_pair.second;

  // for output scale & zp of QKxV matmul
  rt_data_p[jd::attention_io::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_q[jd::attention_io::QK_V_OUTPUT_ZERO_POINT] = new float(113);
  rt_data_p[jd::attention_io::QK_V_OUTPUT_SCALES] = new float(.003f);
  rt_data_q[jd::attention_io::QK_V_OUTPUT_SCALES] = new float(.003f);

  // Merge weight
  const size_t wei_bytes =
      ts_descs[jd::attention_io::Q_WEIGHT].size() * jd::type_size[ts_descs[jd::attention_io::Q_WEIGHT].dtype()];
  char* q_weight_addr = new char[wei_bytes];
  char* k_weight_addr = new char[wei_bytes];
  char* v_weight_addr = new char[wei_bytes];
  const char* rt_data_qkv_ip_wei = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::WEI]);
  memcpy(q_weight_addr, rt_data_qkv_ip_wei, wei_bytes);
  memcpy(k_weight_addr, rt_data_qkv_ip_wei + wei_bytes, wei_bytes);
  memcpy(v_weight_addr, rt_data_qkv_ip_wei + wei_bytes * 2, wei_bytes);
  op_attrs["q_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_weight_addr));
  op_attrs["k_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_weight_addr));
  op_attrs["v_weight_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_weight_addr));

  // Merge bias
  const size_t bias_bytes =
      ts_descs[jd::attention_io::Q_BIAS].size() * jd::type_size[ts_descs[jd::attention_io::Q_BIAS].dtype()];
  char* q_bias_addr = new char[bias_bytes];
  char* k_bias_addr = new char[bias_bytes];
  char* v_bias_addr = new char[bias_bytes];
  const char* rt_data_qkv_ip_bias = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::BIAS]);
  memcpy(q_bias_addr, rt_data_qkv_ip_bias, bias_bytes);
  memcpy(k_bias_addr, rt_data_qkv_ip_bias + bias_bytes, bias_bytes);
  memcpy(v_bias_addr, rt_data_qkv_ip_bias + bias_bytes * 2, bias_bytes);
  op_attrs["q_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_bias_addr));
  op_attrs["k_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_bias_addr));
  op_attrs["v_bias_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_bias_addr));

  // Merge scales
  const size_t scale_bytes =
      ts_descs[jd::attention_io::Q_SCALES].size() * jd::type_size[ts_descs[jd::attention_io::Q_SCALES].dtype()];
  char* q_scales_addr = new char[scale_bytes];
  char* k_scales_addr = new char[scale_bytes];
  char* v_scales_addr = new char[scale_bytes];
  const char* rt_data_qkv_ip_scale = static_cast<const char*>(rt_data_qkv_ip[jd::ssd::SCALES]);
  memcpy(q_scales_addr, rt_data_qkv_ip_scale, scale_bytes);
  memcpy(k_scales_addr, rt_data_qkv_ip_scale + scale_bytes, scale_bytes);
  memcpy(v_scales_addr, rt_data_qkv_ip_scale + scale_bytes * 2, scale_bytes);
  op_attrs["q_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(q_scales_addr));
  op_attrs["k_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(k_scales_addr));
  op_attrs["v_scales_ptr"] = std::to_string(reinterpret_cast<uint64_t>(v_scales_addr));

  jd::operator_desc op_desc(jd::kernel_kind::attention, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);

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
    cases.push_back({gen_case(1, 32, 1, 32, .85f, nthr, jd::data_type::u8,
                              {
                                  {"out_scale", "0.125"},
                                  {"scr0_scale", "1"},
                                  {"scr1_scale", "1"},
                                  {"softmax_in_zero_point", "140"},
                                  {"softmax_in_scale", "500"},
                                  {"softmax_out_zero_point", "0"},
                                  {"softmax_out_scale", "0.00324144"},
                              })});
    cases.push_back({gen_case(16, 64, 2, 128, .85f, nthr, jd::data_type::u8,
                              {
                                  {"out_scale", "0.125"},
                                  {"scr0_scale", "1"},
                                  {"scr1_scale", "1"},
                                  {"softmax_in_zero_point", "140"},
                                  {"softmax_in_scale", "500"},
                                  {"softmax_out_zero_point", "0"},
                                  {"softmax_out_scale", "0.00324144"},
                              })});
    cases.push_back({gen_case(4, 64, 2, 128, .85f, nthr, jd::data_type::u8,
                              {
                                  {"out_scale", "0.125"},
                                  {"scr0_scale", "1"},
                                  {"scr1_scale", "1"},
                                  {"softmax_in_zero_point", "140"},
                                  {"softmax_in_scale", "500"},
                                  {"softmax_out_zero_point", "0"},
                                  {"softmax_out_scale", "0.00324144"},
                              })});
    cases.push_back({gen_case(12, 64, 2, 128, .85f, nthr, jd::data_type::u8,
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
  auto& wei_shape = tensor_desc[jd::ssd::WEI].shape();
  auto& src_shape = tensor_desc[jd::ssd::SRC].shape();
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
  switch (tensor_desc[jd::attention_io::MERGE_DST].dtype()) {
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
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, SpmmAttentionKernelTest, case_func(), test_suffix);
}  // namespace test
