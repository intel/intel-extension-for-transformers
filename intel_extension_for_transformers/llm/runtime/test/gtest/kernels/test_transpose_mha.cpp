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
#include "gtest/gtest.h"
#include "kernels/transpose_mha_types.hpp"
#include "unit_test_utils.hpp"
#include "interface.hpp"

namespace test {
using io = jd::ssd::transpose_mha_io::io;

struct op_args_t {
  jd::operator_desc op_desc;
  std::vector<const void*> data;
};

struct head_info {
  int head_size;
  int head_num;
};

struct test_params_t {
  std::pair<op_args_t, const void*> args;
  bool expect_to_fail;
};

template <typename _T>
static void matrix_transpose(_T* mat, size_t rows, size_t cols, _T* tmat) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      tmat[j * rows + i] = mat[i * cols + j];
    }
  }
}

template <typename _TDST, typename _TSRC>
static _TDST convert(_TSRC _src) {
  return static_cast<_TDST>(_src);
}

static void matf32_quantize_perlayer_s8(float* srcmat, int8_t* tarmat, int size, float* scale) {
  float minval = std::numeric_limits<float>::max();
  float maxval = std::numeric_limits<float>::min();
  for (int i = 0; i < size; i++) {
    if (srcmat[i] < minval) {
      minval = srcmat[i];
    }
    if (srcmat[i] > maxval) {
      maxval = srcmat[i];
    }
  }
  *scale = std::max(abs(minval), abs(maxval)) / 127;
  for (int i = 0; i < size; i++) {
    tarmat[i] = convert<char>(srcmat[i] / scale[0]);
  }
}

static void matf32_quantize_perlayer_u8(float* srcmat, unsigned char* tarmat, int size, float* scale, int* zero_point) {
  float minval = std::numeric_limits<float>::max();
  float maxval = std::numeric_limits<float>::min();
  for (int i = 0; i < size; i++) {
    if (srcmat[i] < minval) {
      minval = srcmat[i];
    }
    if (srcmat[i] > maxval) {
      maxval = srcmat[i];
    }
  }
  if (minval > 0.f) {
    minval = 0.f;
  }
  *scale = (maxval - minval) / 255;
  *zero_point = convert<int>(-minval / scale[0]);
  for (int i = 0; i < size; i++) {
    tarmat[i] = convert<unsigned char>((srcmat[i] - minval) / scale[0]);
  }
}

template <typename T1, typename T2>
static void ref_mm_row_NN_f32(T1* matA, T2* matB, float* matC, float* matD, int m, int n, int k, float alpha,
                              float beta) {
  int NBlock = 128;
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; i += NBlock) {
    for (int j = 0; j < m; j++) {
      int remainn = i + NBlock <= n ? NBlock : n - i;
      for (int ii = 0; ii < remainn; ii++) {
        auto tmp = 0.f;
        for (int ik = 0; ik < k; ik++) {
          float v1 = matA[ik + j * k];
          float v2 = matB[ik * n + i + ii];
          tmp += v1 * v2;
        }
        tmp = tmp * alpha;
        if (beta != 0.f) {
          tmp += matD[(i + ii) + j * n] * beta;
        }
        matC[(i + ii) + j * n] = tmp;
      }
    }
  }
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto dst_ref = t.args.second;

  try {
    const auto& op_desc = p.op_desc;
    jd::transpose_mha_desc transpose_mha_desc(op_desc);
    jd::transpose_mha transpose_mha_ker(transpose_mha_desc);
    transpose_mha_ker.execute(p.data);
  } catch (const std::exception& e) {
    SPARSE_LOG(WARNING) << e.what();
    return t.expect_to_fail;
  }

  if (!t.expect_to_fail) {
    auto buf1 = p.data[io::DST];
    auto size = p.op_desc.tensor_descs()[io::DST].size();
    EXPECT_NE(buf1, dst_ref);
    return compare_data<uint8_t>(buf1, size, dst_ref, size, 8e-3);
  }
  return false;
}

class TransposeAttentionTest : public testing::TestWithParam<test_params_t> {
 protected:
  TransposeAttentionTest() {}
  virtual ~TransposeAttentionTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(TransposeAttentionTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));

  for (int i = 0; i < io::transpose_mha_io_MAX + 1; ++i) {
    if (i < 6)
      aligned_allocator_t<char>::deallocate(const_cast<void*>(t.args.first.data[i]));
    else
      delete reinterpret_cast<int32_t*>(const_cast<void*>(t.args.first.data[i]));
  }
  aligned_allocator_t<uint8_t>::deallocate(const_cast<void*>(t.args.second));
}

std::pair<op_args_t, uint8_t*> gen_case(const std::vector<jd::tensor_desc>& ts_descs,
                                        std::unordered_map<std::string, std::string> op_attrs) {
  // malloc memory
  auto spmm_shape = ts_descs.front().shape();
  int batch = spmm_shape[0], b = spmm_shape[1], k = spmm_shape[2], m = spmm_shape[3], n = spmm_shape[3];
  const int aSize = batch * b * k * m;  // k
  const int bSize = batch * b * k * n;  // q
  const int cSize = batch * m;          // mask
  const int dSize = batch * b * k * m;  // v
  const int eSize = batch * b * k * n;  // dst
  const auto matA = aligned_allocator_t<int8_t>::allocate(aSize);
  const auto matAfp32 = aligned_allocator_t<float>::allocate(aSize);
  const auto matB = aligned_allocator_t<int8_t>::allocate(bSize);
  const auto matBfp32 = aligned_allocator_t<float>::allocate(bSize);
  const auto matC = aligned_allocator_t<float>::allocate(cSize);
  const auto matD = aligned_allocator_t<int8_t>::allocate(dSize);
  const auto matDfp32 = aligned_allocator_t<float>::allocate(dSize);
  const auto matE = aligned_allocator_t<uint8_t>::allocate(eSize);
  const auto ref_matE = aligned_allocator_t<uint8_t>::allocate(eSize);
  const auto matEfp32 = aligned_allocator_t<float>::allocate(eSize);

  const auto trans_matA = aligned_allocator_t<int8_t>::allocate(aSize);
  const auto exp_out = aligned_allocator_t<float>::allocate(batch * b * m * n);
  const auto sumexp_out = aligned_allocator_t<float>::allocate(batch * b * n, true);
  const auto softmax_u8 = aligned_allocator_t<uint8_t>::allocate(batch * b * m * n);

  // set rand seed
  srand(0);

  // init matrix.
  init_vector(matAfp32, aSize, -.5f, .5f);
  init_vector(matBfp32, bSize, 0.f, 1.f);
  init_vector(matC, cSize, -.5f, .5f);
  init_vector(matDfp32, dSize, -.005f, .995f);

  float scaleA, scaleB, scaleD;
  matf32_quantize_perlayer_s8(matAfp32, matA, aSize, &scaleA);
  matf32_quantize_perlayer_s8(matBfp32, matB, bSize, &scaleB);
  matf32_quantize_perlayer_s8(matDfp32, matD, dSize, &scaleD);
#pragma omp parallel for  // calculate
  for (int i = 0; i < batch * b; i++) {
    matrix_transpose(matA + i * m * k, k, m, trans_matA + i * m * k);
    ref_mm_row_NN_f32(trans_matA + i * m * k, matB + i * n * k, exp_out + i * m * n, exp_out, m, n, k, scaleA * scaleB,
                      0.f);
    int _batch = i / b;
    // int _b = i % b;
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < n; k++) {
        exp_out[i * m * n + j * n + k] += matC[_batch * m + j];
      }
    }
  }
  for (int i = 0; i < batch * b * m * n; i++) {
    exp_out[i] = expf(exp_out[i]);
  }
  for (int i = 0; i < batch * b; i++) {
    for (int j = 0; j < m; j++) {
      for (int in = 0; in < n; in++) {
        sumexp_out[i * n + in] += exp_out[i * m * n + j * n + in];
      }
    }
  }
  for (int i = 0; i < batch * b; i++) {
    for (int j = 0; j < m; j++) {
      for (int in = 0; in < n; in++) {
        exp_out[i * m * n + j * n + in] = exp_out[i * m * n + j * n + in] / sumexp_out[i * n + in];
        softmax_u8[i * m * n + j * n + in] =
            get_quantize(exp_out[i * m * n + j * n + in], 0, (1 / 255.f), jd::data_type::u8);
      }
    }
  }
  for (int i = 0; i < batch * b; i++) {
    ref_mm_row_NN_f32(matD + i * k * m, softmax_u8 + i * m * n, matEfp32 + i * k * n, NULL, k, n, m,
                      scaleD * (1 / 255.f), 0.f);
  }
  float scaleE;
  int zpE;
  matf32_quantize_perlayer_u8(matEfp32, ref_matE, eSize, &scaleE, &zpE);
  std::vector<const void*> rt_data(io::transpose_mha_io_MAX + 1);

  rt_data[io::SRC_K] = matA;
  rt_data[io::SRC_Q] = matB;
  rt_data[io::MASK] = matC;
  rt_data[io::SRC_V] = matD;
  rt_data[io::DST] = matE;
  rt_data[io::TMP2M] = aligned_allocator_t<uint8_t>::allocate(omp_get_max_threads() * (1 << 21));
  rt_data[io::SL_PAD] = new int(m);
  rt_data[io::BATCH] = new int(batch);
  rt_data[io::HEAD_NUM] = new int(b);
  rt_data[io::HEAD_SIZE] = new int(k);
  rt_data[io::SEQ_LEN] = new int(m);
  rt_data[io::SCALE_Q] = new float(scaleB);
  rt_data[io::SCALE_K] = new float(scaleA);
  rt_data[io::SCALE_V] = new float(scaleD);
  rt_data[io::SCALE_DST] = new float(scaleE);
  rt_data[io::ZP_DST] = new int(zpE);

  aligned_allocator_t<float>::deallocate(matAfp32);
  aligned_allocator_t<float>::deallocate(matBfp32);
  aligned_allocator_t<float>::deallocate(matDfp32);
  aligned_allocator_t<float>::deallocate(matEfp32);
  aligned_allocator_t<int8_t>::deallocate(trans_matA);
  aligned_allocator_t<float>::deallocate(exp_out);
  aligned_allocator_t<float>::deallocate(sumexp_out);
  aligned_allocator_t<uint8_t>::deallocate(softmax_u8);

  jd::operator_desc trans_attention_desc(jd::kernel_kind::transpose_mha, jd::kernel_prop::forward_inference,
                                         jd::engine_kind::cpu, ts_descs, op_attrs);

  return {{trans_attention_desc, rt_data}, ref_matE};
}

static auto case_func = []() {
  std::vector<test_params_t> cases;
  std::vector<head_info> head_infos = {{32, 4}, {64, 16}};
  std::vector<int> batchs = {4};

  // assume seq_pad == seq_len
  std::vector<int> seq_lens = {32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384};
  for (auto&& head_info : head_infos) {
    for (auto&& batch : batchs) {
      for (auto&& seq_len : seq_lens) {
        // we assign the real shape of the K-matrix to the K_desc only for get the problem size when we gen the input &
        // true-result data. In real mha kernel usage, pls make the shape field in jd::tensor_desc EMPTY.
        jd::tensor_desc K_desc = {
            {batch, head_info.head_num, head_info.head_size, seq_len}, jd::data_type::s8, jd::format_type::undef};
        jd::tensor_desc Q_desc = {{}, jd::data_type::s8, jd::format_type::undef};
        jd::tensor_desc mask_desc = {{}, jd::data_type::fp32, jd::format_type::undef};
        jd::tensor_desc V_desc = {{}, jd::data_type::s8, jd::format_type::undef};
        jd::tensor_desc ret_desc = {{}, jd::data_type::u8, jd::format_type::undef};
        std::unordered_map<std::string, std::string> op_attrs;
        // we set the op_attrs only for gen the test_suffix in uts, in real mha kernel usage, user don't need to set
        // op_attrs, even they set, the op_attrs will not participate the kernel-hashing stage.
        op_attrs["seq_pad"] = std::to_string(seq_len);
        op_attrs["batch"] = std::to_string(batch);
        op_attrs["head_num"] = std::to_string(head_info.head_num);
        op_attrs["k"] = std::to_string(head_info.head_size);
        op_attrs["seq_len"] = std::to_string(seq_len);
        op_attrs["impl"] = "vnni_b";  // TODO(Yi): find A better way to integrate vnni_b
        cases.push_back({gen_case({K_desc, Q_desc, mask_desc, V_desc, ret_desc}, op_attrs)});
      }
    }
  }

  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  std::vector<std::string> params;
  auto attrs_map = tpi.param.args.first.op_desc.attrs();
  params.push_back("batch");
  params.push_back(attrs_map["batch"]);
  params.push_back("b");
  params.push_back(attrs_map["head_num"]);
  params.push_back("k");
  params.push_back(attrs_map["k"]);
  params.push_back("m");
  params.push_back(attrs_map["seq_len"]);
  params.push_back("n");
  params.push_back(attrs_map["seq_len"]);
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, TransposeAttentionTest, case_func(), test_suffix);
}  // namespace test
