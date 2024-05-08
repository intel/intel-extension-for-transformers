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
#include "interface.hpp"
#include "src/cpu/kernels/dynamic_quant_matmul_ref.hpp"
#include "unit_test_utils.hpp"

namespace test {

using io = jd::exposed_enum::dynamic_quant_matmul::io;

struct op_args_t {
  jd::operator_desc op_desc;
  std::shared_ptr<std::vector<int8_t>> activation;
  std::shared_ptr<std::vector<int8_t>> reordered_weight;
  std::shared_ptr<std::vector<int8_t>> dst;
  std::shared_ptr<std::vector<float>> fp32_dst;
  std::shared_ptr<std::vector<float>> scale_a;
  std::shared_ptr<std::vector<float>> scale_w;
  std::shared_ptr<std::vector<float>> bias;
  std::shared_ptr<std::vector<float>> scale_dst;
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  const auto& op_desc = p.op_desc;
  std::vector<const void*> data1, data2;
  auto dst_dt = op_desc.tensor_descs()[io::DST].dtype();
  try {
    jd::dynamic_quant_matmul_desc dynamic_quant_matmul_desc(op_desc);
    jd::dynamic_quant_matmul dynamic_quant_matmul_ker(dynamic_quant_matmul_desc);
    std::shared_ptr<char> tmp_buf(reinterpret_cast<char*>(malloc(dynamic_quant_matmul_ker.get_workspace_size())),
                                  [](char* ptr) { free(ptr); });

    data1 = {p.activation->data(), p.reordered_weight->data(), p.dst->data(), p.scale_a->data(),
             p.scale_w->data(),    p.scale_dst->data(),        tmp_buf.get(), p.bias->data()};

    data2 = {q.activation->data(), q.reordered_weight->data(), q.dst->data(), q.scale_a->data(),
             q.scale_w->data(),    q.scale_dst->data(),        tmp_buf.get(), q.bias->data()};
    if (dst_dt == jd::data_type::fp32) {
      data1[io::DST] = p.fp32_dst->data();
      data2[io::DST] = q.fp32_dst->data();
    }
    dynamic_quant_matmul_ker.execute(data1);
    std::shared_ptr<const jd::kernel_desc_t> dynamic_quant_matmul_ref_desc;
    jd::kernel_desc_t::create<jd::dynamic_quant_matmul_ref_kd_t>(dynamic_quant_matmul_ref_desc, q.op_desc);
    std::shared_ptr<const jd::kernel_t> dynamic_quant_matmul_ref_ker;
    jd::kernel_t::create<jd::dynamic_quant_matmul_ref_k_t, jd::dynamic_quant_matmul_ref_kd_t>(
        dynamic_quant_matmul_ref_ker, dynamic_quant_matmul_ref_desc);
    dynamic_quant_matmul_ref_ker->execute(data2);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }

  if (!t.expect_to_fail) {
    auto buf1 = data1[io::DST];
    auto size = p.dst->size();
    auto buf2 = data2[io::DST];
    bool ans1 = false;
    switch (dst_dt) {
      case jd::data_type::fp32:
        ans1 = compare_data<float>(buf1, size, buf2, size, 5e-3);
        return ans1;
      default:
        ans1 = compare_data<int8_t>(buf1, size, buf2, size, 1e-2);
        break;
    }
    auto buf3 = data1[io::SCALE_DST];
    auto size2 = p.scale_dst->size();
    auto buf4 = data2[io::SCALE_DST];
    auto ans2 = compare_data<float>(buf3, size2, buf4, size2, 5e-3);
    return ans1 && ans2;
  }
  return false;
}

class DynamicQuantMatmulKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  DynamicQuantMatmulKernelTest() {}
  virtual ~DynamicQuantMatmulKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(DynamicQuantMatmulKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
}

void reorder_stage(std::vector<int8_t>* src, std::vector<int8_t>* dst, int k, int n, int pad_n) {
#pragma omp parallel for
  for (int k_loop = 0; k_loop < k / 4; k_loop++) {
    for (int n_loop = 0; n_loop < n; n_loop++) {
      (*dst)[k_loop * pad_n * 4 + n_loop * 4] = (*src)[k_loop * n * 4 + n_loop];
      (*dst)[k_loop * pad_n * 4 + n_loop * 4 + 1] = (*src)[k_loop * n * 4 + n + n_loop];
      (*dst)[k_loop * pad_n * 4 + n_loop * 4 + 2] = (*src)[k_loop * n * 4 + 2 * n + n_loop];
      (*dst)[k_loop * pad_n * 4 + n_loop * 4 + 3] = (*src)[k_loop * n * 4 + 3 * n + n_loop];
    }
  }
}

std::shared_ptr<std::vector<int8_t>> transpose_amx_tileKx64_reorder_buf(int8_t* src, int k, int pad_n) {
  int tile_k = 64;
  while (k % tile_k != 0) tile_k -= 4;
  auto contain_block_row = k / tile_k;
  auto contain_block_col = pad_n / 16;
  auto block_size = tile_k * 16;
  std::shared_ptr<std::vector<int8_t>> trans_reorder_buf(new std::vector<int8_t>);
  auto trans_reorder_buf_ptr = trans_reorder_buf.get();
  trans_reorder_buf_ptr->resize(k * pad_n);
  std::fill(trans_reorder_buf_ptr->begin(), trans_reorder_buf_ptr->end(), 0);
#pragma omp parallel for
  for (int k_loop = 0; k_loop < contain_block_row; k_loop++)
    for (int n_loop = 0; n_loop < contain_block_col; n_loop++)
      for (int i = 0; i < tile_k / 4; i++)
        for (int j = 0; j < 64; j++)
          (*trans_reorder_buf_ptr)[n_loop * contain_block_row * block_size + k_loop * 64 + i * contain_block_row * 64 +
                                   j] = src[k_loop * contain_block_col * block_size + n_loop * 64 + i * pad_n * 4 + j];
  return trans_reorder_buf;
}

std::pair<op_args_t, op_args_t> gen_case(const std::vector<jd::tensor_desc>& ts_descs,
                                         std::unordered_map<std::string, std::string> op_attrs,
                                         const std::vector<jd::postop_attr>& postop_attr = {}) {
  auto activation_shape = ts_descs[io::ACTIVATION].shape();
  auto weight_shape = ts_descs[io::WEIGHT].shape();
  int b = activation_shape[0], m = activation_shape[1], k = weight_shape[0], n = weight_shape[1];
  int pad_n = ceil_div(n, 16) * 16;

  auto gen_data = [](auto type, int size, float bound1, float bound2, bool clear = false) {
    auto ptr = std::shared_ptr<std::vector<decltype(type)>>(new std::vector<decltype(type)>(size, 0));
    if (!clear) init_vector(ptr->data(), ptr->size(), bound1, bound2);
    return ptr;
  };

  auto activation = gen_data(static_cast<int8_t>(1), b * m * k, 0.f, 10.f);
  auto weight = gen_data(static_cast<int8_t>(1), k * n, 0.f, 10.f);
  auto dst_mat = gen_data(static_cast<int8_t>(1), b * m * n, 0.f, 0.f, true);
  auto fp32_dst_mat = gen_data(static_cast<float>(1), b * m * n, 10.f, 20.f);
  auto scale_a = gen_data(static_cast<float>(1), b * m, 0.f, 1.f);
  auto scale_w = gen_data(static_cast<float>(1), n, 0.f, 1.f);
  auto bias = gen_data(static_cast<float>(1), n, 10.f, 20.f);
  auto scale_dst = gen_data(static_cast<float>(1), b * m, 0.f, 0.f, true);
  auto correct_scale_dst = gen_data(static_cast<float>(1), b * m, 0.f, 0.f, true);
  auto correct_dst_mat = gen_data(static_cast<int8_t>(1), b * m * n, 0.f, 0.f, true);
  auto reorder_buf = gen_data(static_cast<int8_t>(1), k * pad_n, 0.f, 0.f, true);
  auto correct_fp32_dst_mat = std::shared_ptr<std::vector<float>>(new std::vector<float>(b * m * n));
  memcpy(correct_fp32_dst_mat->data(), fp32_dst_mat->data(), b * m * n * sizeof(float));

  reorder_stage(weight.get(), reorder_buf.get(), k, n, pad_n);
  auto trans_reorder_wei = transpose_amx_tileKx64_reorder_buf(reorder_buf.get()->data(), k, pad_n);
  jd::operator_desc dynamic_quant_matmul_desc(jd::kernel_kind::dynamic_quant_matmul, jd::kernel_prop::forward_inference,
                                              jd::engine_kind::cpu, ts_descs, op_attrs, postop_attr);

  op_args_t p = {dynamic_quant_matmul_desc,
                 activation,
                 trans_reorder_wei,
                 dst_mat,
                 fp32_dst_mat,
                 scale_a,
                 scale_w,
                 bias,
                 scale_dst};
  op_args_t q = {dynamic_quant_matmul_desc, activation, trans_reorder_wei, correct_dst_mat,
                 correct_fp32_dst_mat,      scale_a,    scale_w,           bias,
                 correct_scale_dst};
  return {p, q};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  std::vector<std::vector<dim_t>> shapes = {{512, 1280, 1280}, {512, 1280, 10240}, {77, 768, 1024}};

  std::vector<dim_t> batchs = {1, 2};
  std::vector<jd::data_type> dt_types = {jd::data_type::s8, jd::data_type::fp32};
  jd::postop_attr swish_attr = {jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::swish, 2.f};
  for (auto&& batch : batchs) {
    for (auto&& shape : shapes) {
      for (auto&& data_type : dt_types) {
        jd::tensor_desc activation_desc = {{batch, shape[0], shape[2]}, jd::data_type::s8, jd::format_type::abc};
        jd::tensor_desc weight_desc = {{shape[2], shape[1]}, jd::data_type::s8, jd::format_type::ab};
        jd::tensor_desc dst_desc = {{batch, shape[0], shape[1]}, data_type, jd::format_type::abc};
        jd::tensor_desc sclae_a_desc = {{batch, shape[0]}, jd::data_type::fp32, jd::format_type::ab};
        jd::tensor_desc scale_w_desc = {{shape[1]}, jd::data_type::fp32, jd::format_type::a};
        jd::tensor_desc scale_dst_desc = {{batch, shape[0]}, jd::data_type::fp32, jd::format_type::ab};
        jd::tensor_desc workspace_desc = {{}, jd::data_type::undef, jd::format_type::a};
        jd::tensor_desc bias_desc = {{shape[1]}, jd::data_type::fp32, jd::format_type::a};
        std::unordered_map<std::string, std::string> op_attrs = {{"large_wei_threshold", "0.8"}};
        if (data_type == jd::data_type::fp32) op_attrs["append_sum"] = true;
        cases.push_back({gen_case({activation_desc, weight_desc, dst_desc, sclae_a_desc, scale_w_desc, scale_dst_desc,
                                   workspace_desc, bias_desc},
                                  op_attrs, {swish_attr}),
                         false});
      }
    }
  }
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(SparseLib, DynamicQuantMatmulKernelTest, case_func());
}  // namespace test
