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
#include "src/cpu/kernels/layernorm_ba_ref.hpp"
#include "interface.hpp"
namespace test {
struct op_args_t {
  jd::operator_desc op_desc;
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
    jd::layernorm_ba_desc layernorm_ba_desc(op_desc);
    jd::layernorm_ba layernorm_ba_ker(layernorm_ba_desc);
    layernorm_ba_ker.execute(p.data);

    std::shared_ptr<const jd::kernel_desc_t> lnorm_ba_ref_desc;
    jd::kernel_desc_t::create<jd::layernorm_ba_ref_kd_t>(lnorm_ba_ref_desc, q.op_desc);
    std::shared_ptr<const jd::kernel_t> lnorm_ref_ker;
    jd::kernel_t::create<jd::layernorm_ba_ref_k_t, jd::layernorm_ba_ref_kd_t>(lnorm_ref_ker, lnorm_ba_ref_desc);
    lnorm_ref_ker->execute(q.data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  auto binary_list = p.op_desc.get_binaryop_list();
  auto free_memory = [&] {
    free(const_cast<void*>(p.data[0]));
    free(const_cast<void*>(p.data[1]));
    free(const_cast<void*>(q.data[0]));
    free(const_cast<void*>(q.data[1]));
    free(const_cast<void*>(q.data[2]));
    free(const_cast<void*>(q.data[3]));
    if (op_attr["spec_type"] == "direct") {
      free(const_cast<void*>(q.data[4]));
      free(const_cast<void*>(q.data[5]));
    }
    if (op_attr["split_output"] == "true") {
      free(const_cast<void*>(p.data[p.data.size() - 1]));
      free(const_cast<void*>(q.data[q.data.size() - 1]));
    }
    for (auto&& i : binary_list) {
      if (i.static_addr) free(i.static_addr);
      if (i.scale) free(i.scale);
      if (i.zp) free(i.zp);
    }
  };

  if (!t.expect_to_fail) {
    auto buf1 = p.data[1];
    auto size1 = p.op_desc.tensor_descs()[1].size();
    auto buf2 = q.data[1];
    auto size2 = p.op_desc.tensor_descs()[1].size();
    auto dst_type = p.op_desc.tensor_descs()[1].dtype();
    EXPECT_NE(buf1, buf2);
    bool ans = false;
    if (dst_type == jd::data_type::fp32) {
      ans = compare_data<float>(buf1, size1, buf2, size2, 5e-3);
      if (op_attr["split_output"] == "true" && ans) {
        auto buf3 = q.data[6];
        auto buf4 = p.data[6];
        if (op_desc.apply_postops_list().back().dt == jd::data_type::s8)
          ans = compare_data<int8_t>(buf4, size1, buf3, size1, 1e-2);
        else
          ans = compare_data<uint8_t>(buf4, size1, buf3, size1, 1e-2);
      }
    } else if (dst_type == jd::data_type::u8) {
      ans = compare_data<uint8_t>(buf1, size1, buf2, size2, 1e-2);
    } else if (dst_type == jd::data_type::s8) {
      ans = compare_data<int8_t>(buf1, size1, buf2, size2, 1e-2);
    }
    free_memory();
    return ans;
  }
  free_memory();
  return false;
}

class LayernormBaKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  LayernormBaKernelTest() {}
  virtual ~LayernormBaKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(LayernormBaKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<jd::tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<jd::postop_attr>& postop_attr = {},
                                         bool per_channel_quant = false) {
  // malloc memory
  auto input_tensor_desc = ts_descs[0].shape();
  int row = input_tensor_desc.size() == 3 ? input_tensor_desc[1] : input_tensor_desc[0];
  int col = ts_descs[0].shape().back();
  int batch = input_tensor_desc.size() == 3 ? input_tensor_desc[0] : 1;
  int num = batch * row * col;
  void* src = nullptr;
  void* dst = nullptr;
  void* dst2 = nullptr;
  void* dst2_ref = nullptr;
  void* src_ref = nullptr;
  void* dst_ref = nullptr;
  memo_mode MALLOC = memo_mode::MALLOC;
  memo_mode MEMSET = memo_mode::MEMSET;

  // set rand seed
  unsigned int seed = 456;
  srand(seed);

  auto in_dt = ts_descs[0].dtype();
  auto out_dt = ts_descs[1].dtype();

  src = sparselib_ut_memo(src, num, in_dt, MALLOC);
  dst = sparselib_ut_memo(dst, num, out_dt, MALLOC);
  dst = sparselib_ut_memo(dst, num, out_dt, MEMSET);
  src_ref = sparselib_ut_memo(src_ref, num, in_dt, MALLOC);
  dst_ref = sparselib_ut_memo(dst_ref, num, out_dt, MALLOC);
  dst_ref = sparselib_ut_memo(dst_ref, num, out_dt, MEMSET);
  if (op_attrs["split_output"] == "true") {
    dst2 = sparselib_ut_memo(dst2, num, jd::data_type::s8, MALLOC);
    dst2_ref = sparselib_ut_memo(dst2_ref, num, jd::data_type::s8, MALLOC);
  }
  float* alpha = reinterpret_cast<float*>(malloc(row * sizeof(float)));
  float* beta = reinterpret_cast<float*>(malloc(row * sizeof(float)));
  float* mean = nullptr;
  float* var = nullptr;
  if (op_attrs["spec_type"] == "direct") {
    mean = reinterpret_cast<float*>(malloc(batch * col * sizeof(float)));
    var = reinterpret_cast<float*>(malloc(batch * col * sizeof(float)));
    for (int i = 0; i < batch * col; i++) {
      mean[i] = std::rand() % 256 - 128 + rand_float_postfix();    // NOLINT
      var[i] = std::abs(std::rand() % 10 + rand_float_postfix());  // NOLINT
    }
  }

  // init alpha&beta
  for (int i = 0; i < row; i++) alpha[i] = 1 + rand_float_postfix();
  for (int i = 0; i < row; i++) beta[i] = 1 + rand_float_postfix();

  // init matrix.
  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        float rand_val = std::rand() % 256 - 128 + rand_float_postfix();  // NOLINT
        assign_val(src, in_dt, rand_val, k * row * col + i * col + j);
        assign_val(src_ref, in_dt, rand_val, k * row * col + i * col + j);
      }
    }
  }
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;

  rt_data1.emplace_back(src);
  rt_data1.emplace_back(dst);
  rt_data1.emplace_back(alpha);
  rt_data1.emplace_back(beta);
  rt_data2.emplace_back(src_ref);
  rt_data2.emplace_back(dst_ref);
  rt_data2.push_back(alpha);
  rt_data2.push_back(beta);
  if (op_attrs["spec_type"] == "direct") {
    rt_data1.push_back(mean);
    rt_data1.push_back(var);
    rt_data2.push_back(mean);
    rt_data2.push_back(var);
  }
  if (op_attrs["split_output"] == "true") {
    rt_data1.push_back(dst2);
    rt_data2.push_back(dst2_ref);
  }

  if (per_channel_quant) op_attrs["binaryop_list"] = "u8_perchannel_quant";

  jd::operator_desc layernorm_ba_desc(jd::kernel_kind::layernorm_ba, jd::kernel_prop::forward_inference,
                                      jd::engine_kind::cpu, ts_descs, op_attrs, postop_attr);

  // init per_channel quant factor
  if (per_channel_quant) {
    jd::binaryop_attr u8_per_channel_quantize = {jd::binaryop_alg::per_channel_quant, jd::data_type::u8};
    std::vector<jd::binaryop_attr> binaryop_list = {u8_per_channel_quantize};
    float* scale = reinterpret_cast<float*>(malloc(row * sizeof(float)));
    float* zp = reinterpret_cast<float*>(malloc(row * sizeof(float)));
    for (int i = 0; i < row; i++) {
      scale[i] = 1 + rand_float_postfix();
      zp[i] = rand() % 10;  // NOLINT
      scale[i] /= 1;
    }
    auto per_chan_quant_attr = binaryop_list.front();
    per_chan_quant_attr.set_scale(scale);
    per_chan_quant_attr.set_zp(zp);
    binaryop_list[0] = per_chan_quant_attr;
    layernorm_ba_desc.set_binaryop_list(binaryop_list);
  }
  op_args_t p = {layernorm_ba_desc, rt_data1};
  op_args_t q = {layernorm_ba_desc, rt_data2};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  jd::tensor_desc data_desc0 = {{768, 300}, jd::data_type::fp32, jd::format_type::ba};
  jd::tensor_desc data_desc1 = {{768, 256}, jd::data_type::fp32, jd::format_type::ba};
  jd::tensor_desc data_desc2 = {{8, 768, 256}, jd::data_type::fp32, jd::format_type::ba};
  jd::tensor_desc data_desc3 = {{1024, 256}, jd::data_type::fp32, jd::format_type::ba};
  jd::tensor_desc data_desc4 = {{768, 256}, jd::data_type::s8, jd::format_type::ba};
  jd::tensor_desc data_desc5 = {{1024, 1536}, jd::data_type::fp32, jd::format_type::ba};
  jd::tensor_desc data_desc6 = {{1024, 1536}, jd::data_type::u8, jd::format_type::ba};

  jd::postop_attr s8_quantize = {
      jd::data_type::s8,   jd::postop_type::eltwise, jd::postop_alg::quantize, rand_float_postfix(), 0,
      rand_float_postfix()};
  jd::postop_attr u8_quantize = {
      jd::data_type::u8,   jd::postop_type::eltwise, jd::postop_alg::quantize, rand_float_postfix(), 0,
      rand_float_postfix()};

  std::string tensor_shape0 = "728x300";
  std::string tensor_shape1 = "768x256";
  std::string tensor_shape2 = "8x768x256";
  std::string tensor_shape3 = "1024x256";
  std::string tensor_shape4 = "1024x1536";
  std::string quantize_attrs8 = "s8quantize";
  std::string quantize_attru8 = "u8quantize";

  cases.push_back(
      {gen_case({data_desc1, data_desc1}, {{"matrix_shape", tensor_shape1}, {"spec_type", "normal"}}), false});
  cases.push_back(
      {gen_case({data_desc2, data_desc2}, {{"matrix_shape", tensor_shape2}, {"spec_type", "normal"}}), false});
  cases.push_back(
      {gen_case({data_desc3, data_desc3}, {{"matrix_shape", tensor_shape3}, {"spec_type", "normal"}}), false});
  cases.push_back(
      {gen_case({data_desc1, data_desc4},
                {{"matrix_shape", tensor_shape1}, {"postop_list", quantize_attrs8}, {"spec_type", "normal"}},
                {s8_quantize}),
       false});
  cases.push_back(
      {gen_case({data_desc1, data_desc4}, {{"matrix_shape", tensor_shape1}, {"spec_type", "normal"}}, {}, true),
       false});

  cases.push_back({gen_case({data_desc1, data_desc1},
                            {{"matrix_shape", tensor_shape1},
                             {"postop_list", quantize_attru8},
                             {"spec_type", "direct"},
                             {"split_output", "true"}},
                            {u8_quantize}),
                   false});

  cases.push_back(
      {gen_case({data_desc5, data_desc6},
                {{"matrix_shape", tensor_shape1}, {"postop_list", quantize_attru8}, {"spec_type", "direct"}},
                {u8_quantize}),
       false});

  cases.push_back({gen_case({data_desc2, data_desc2},
                            {{"matrix_shape", tensor_shape2},
                             {"postop_list", quantize_attru8},
                             {"spec_type", "direct"},
                             {"split_output", "true"}},
                            {u8_quantize}),
                   false});

  cases.push_back(
      {gen_case({data_desc0, data_desc0}, {{"matrix_shape", tensor_shape0}, {"spec_type", "direct"}}), false});
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto tensor_desc = tpi.param.args.first.op_desc.tensor_descs();
  auto& tensor_shape = tensor_desc[0].shape();
  auto attrs_map = tpi.param.args.first.op_desc.attrs();
  params.push_back("shape");
  for (auto&& i : tensor_shape) params.push_back(std::to_string(i));

  auto add_dt_info = [&](jd::data_type data_type, const std::string& tensor_dt) {
    switch (data_type) {
      case jd::data_type::s8:
        params.push_back(tensor_dt + "_s8");
        break;
      case jd::data_type::fp32:
        params.push_back(tensor_dt + "_fp32");
        break;
      case jd::data_type::u8:
        params.push_back(tensor_dt + "_u8");
        break;
      default:
        assert(false);
    }
  };

  add_dt_info(tensor_desc[0].dtype(), "indt");
  add_dt_info(tensor_desc[1].dtype(), "outdt");
  params.push_back(attrs_map["spec_type"]);
  if (attrs_map["postop_list"] != "") params.push_back(attrs_map["postop_list"]);
  if (attrs_map["binaryop_list"] != "") params.push_back(attrs_map["binaryop_list"]);
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, LayernormBaKernelTest, case_func(), test_suffix);
}  // namespace test
