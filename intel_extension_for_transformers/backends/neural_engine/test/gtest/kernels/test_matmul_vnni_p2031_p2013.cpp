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
#include "kernels/matmul_types.hpp"
#include "kernels/matmul_ref.hpp"
#include "jit_domain/jit_seq_cpy_2x8x8.hpp"
#include "jit_domain/jit_seq_cpy_48x4.hpp"
#include "jit_domain/jit_matmul_vnni_8xkx48.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;

// test seq_cpy_2x8x8
struct seqcpyA_param_t {
  dim_t M;
  dim_t N;
  uint8_t offset;
};
inline static std::string seqcpyATestParam2str(testing::TestParamInfo<seqcpyA_param_t> tpi) {
  return std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N) + "_off" + std::to_string(tpi.param.offset);
}
class MMVNNIP2031P2013SEQCPYATest : public testing::TestWithParam<seqcpyA_param_t> {
 protected:
  MMVNNIP2031P2013SEQCPYATest() {}
  virtual ~MMVNNIP2031P2013SEQCPYATest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(MMVNNIP2031P2013SEQCPYATest, ) {
  seqcpyA_param_t t = testing::TestWithParam<seqcpyA_param_t>::GetParam();
  const int M = t.M, N = t.N;
  const int ld_src = N;
  if (M % 8 != 0) {
    SPARSE_LOG(WARNING) << "M must be a multiple of 8";
    return;
  }
  size_t dst_size = M * (ceil_div(N, 8) * 8);
  int8_t* src = new int8_t[M * N];
  for (auto i = 0; i < M * N; ++i) src[i] = i % UINT8_MAX - INT8_MIN;
  uint8_t* dst = reinterpret_cast<uint8_t*>(aligned_alloc(64, dst_size));
  memset(dst, 3, dst_size);

  // get true data
  uint8_t* dst_ref = new uint8_t[dst_size];
  memset(dst_ref, 3, dst_size);
  for (auto j = 0; j < N; j += 8)
    for (auto i = 0; i < M; i += 4)
      for (auto ii = 0; ii < 4; ++ii)
        for (auto jj = 0; jj < 8; ++jj)
          dst_ref[j * M + i * 8 + jj * 4 + ii] =
              (i + ii >= M || j + jj >= N) ? 0 : (src[(i + ii) * ld_src + j + jj] + t.offset);

  // run kernel
  auto jit_ker = new jit_seq_cpy_2x8x8({M, N, ld_src, t.offset});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  for (dim_t i = 0; i < M; i += 8) {
    jit_seq_cpy_2x8x8::rt_data_t rt_data{
        src + i * N,
        dst + i * 8,
    };
    (*jit_ker)(&rt_data);
  }

  ASSERT_TRUE(compare_data<uint8_t>(dst, dst_size, dst_ref, dst_size, 0));

  delete[] src;
  delete[] dst_ref;
  free(dst);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MMVNNIP2031P2013SEQCPYATest,
                         ::testing::ValuesIn(std::vector<seqcpyA_param_t>{
                             {64, 64, 0},
                             {64, 1, 0},
                             {64, 15, 0},
                             {64, 63, 0},
                             {64, 65, 0},
                             {64, 64, 128},
                             {64, 1, 128},
                             {64, 15, 128},
                             {64, 63, 128},
                             {64, 65, 131},
                         }),
                         seqcpyATestParam2str);

// test seq_cpy_48x4

struct seqcpyB_param_t {
  dim_t M;
  dim_t N;
  bool sum;
};
inline static std::string seqcpyBTestParam2str(testing::TestParamInfo<seqcpyB_param_t> tpi) {
  std::string repr = std::to_string(tpi.param.M) + "_" + std::to_string(tpi.param.N);
  if (tpi.param.sum) repr += "_sum";
  return repr;
}
class MMVNNIP2031P2013SEQCPYBTest : public testing::TestWithParam<seqcpyB_param_t> {
 protected:
  MMVNNIP2031P2013SEQCPYBTest() {}
  virtual ~MMVNNIP2031P2013SEQCPYBTest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(MMVNNIP2031P2013SEQCPYBTest, ) {
  seqcpyB_param_t t = testing::TestWithParam<seqcpyB_param_t>::GetParam();
  const int M = t.M, N = t.N;
  const int ld_src = N;
  const int pad_n = pad_to(N, 48);
  const int max_sum_size = pad_n * 16;  // up to 16 threads
  if (M % 4 != 0) {
    SPARSE_LOG(WARNING) << "M must be a multiple of 4! Test case skipped.";
    return;
  }
  const size_t dst_size = M * (ceil_div(N, 48) * 48);
  uint8_t* const src = new uint8_t[M * N];
  for (auto i = 0; i < M * N; ++i) src[i] = i % UINT8_MAX;

  // get true data
  uint8_t* const dst_ref = new uint8_t[dst_size];
  memset(dst_ref, 0, dst_size);
#pragma omp parallel for collapse(3)
  for (auto j = 0; j < N; j += 48)
    for (auto i = 0; i < M; i += 4)
      for (auto ii = 0; ii < 4; ++ii)
#pragma omp simd
        for (auto jj = 0; jj < 48; ++jj) {
          auto value = (i + ii >= M || j + jj >= N) ? 0 : src[(i + ii) * ld_src + j + jj];
          dst_ref[j * M + i * 48 + jj * 4 + ii] = value;
        }

  // run kernel
  const n_thread_t with_n_thread(3);
  uint8_t* const dst = aligned_allocator_t<uint8_t>::allocate(dst_size, true);
  int32_t* const dst_sum = aligned_allocator_t<int32_t>::allocate(max_sum_size);
  memset(dst_sum, -1, max_sum_size * sizeof(int32_t));
  auto jit_ker = new jit_seq_cpy_48x4({M, N, ld_src, t.sum, true});
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());
  int real_num_threads = 0;
  bool sum_append = false;
#pragma omp parallel for firstprivate(sum_append)
  for (dim_t i = 0; i < M; i += 4) {
    jit_seq_cpy_48x4 ::rt_data_t rt_data{
        src + i * N,
        dst + i * 48,
        dst_sum + pad_n * omp_get_thread_num(),
        sum_append,
    };
    (*jit_ker)(&rt_data);
    sum_append = true;
    if (i == 0) real_num_threads = omp_get_num_threads();
  }
  real_num_threads = std::min(real_num_threads, ceil_div(M, 4));

  ASSERT_TRUE(compare_data<uint8_t>(dst, dst_size, dst_ref, dst_size, 0));

  if (t.sum) {
    // collaps sum
    for (dim_t i = 1; i < real_num_threads; ++i) {
#pragma omp simd
      for (dim_t j = 0; j < N; ++j) {
        dst_sum[j] += dst_sum[i * pad_n + j];
      }
    }

    //  sum true data
    int32_t* dst_sum_ref = aligned_allocator_t<int32_t>::allocate(pad_n, true);
#pragma omp parallel for
    for (auto j = 0; j < N; ++j)
      for (auto i = 0; i < M; ++i) {
        dst_sum_ref[j] += src[i * N + j];
      }

    ASSERT_TRUE(compare_data<int32_t>(dst_sum, N, dst_sum_ref, N, 0));
    aligned_allocator_t<int32_t>::deallocate(dst_sum_ref);
  }

  delete[] src;
  delete[] dst_ref;
  aligned_allocator_t<uint8_t>::deallocate(dst);
  aligned_allocator_t<int32_t>::deallocate(dst_sum);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(SparseLib, MMVNNIP2031P2013SEQCPYBTest,
                         ::testing::ValuesIn(std::vector<seqcpyB_param_t>{
                             {4, 48},
                             {128, 768, true},
                             {64, 48, true},
                             {4, 1, true},
                             {64, 47, true},
                             {64, 49, true},
                             {64, 65, true},
                         }),
                         seqcpyBTestParam2str);

// test mm_8xkx48
struct mm_param_t {
  dim_t K;
  dim_t ld_dst;
  uint8_t bias_lshift;
  bool binary_add;
  std::vector<postop_attr> postop_list;
};

inline static std::string mmTestParam2str(testing::TestParamInfo<mm_param_t> tpi) {
  std::vector<std::string> params;
  params.push_back(std::to_string(tpi.param.K));
  params.push_back(std::to_string(tpi.param.ld_dst));
  params.push_back("lsh" + std::to_string(tpi.param.bias_lshift));
  if (tpi.param.binary_add) params.push_back("badd");
  for (postop_attr& p_attr : tpi.param.postop_list) {
    params.push_back("post" + std::string(postop_alg_name.at(p_attr.op_alg)));
    params.push_back(std::string(data_type_name.at(p_attr.dt)));
    params.push_back(num2id(p_attr.alpha));
    params.push_back(num2id(p_attr.beta));
    params.push_back(num2id(p_attr.scale));
  }
  return join_str(params, "_");
}
class MMVNNIP2031P2013MMTileTest : public testing::TestWithParam<mm_param_t> {
 protected:
  MMVNNIP2031P2013MMTileTest() {}
  virtual ~MMVNNIP2031P2013MMTileTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

template <typename dt_dst>
inline void exec_mmtile(jit_matmul_vnni_8xkx48_t* ker, const uint8_t* src0, const int8_t* src1, const int32_t* bias,
                        const float* src_b0, dt_dst* dst) {
  jit_matmul_vnni_8xkx48_t::rt_data_t<dt_dst> rt_data;
  rt_data.src0 = src0;
  rt_data.src1 = src1;
  rt_data.bias = bias;
  rt_data.src_b0 = src_b0;
  rt_data.dst = dst;
  (*ker)(&rt_data);
}

TEST_P(MMVNNIP2031P2013MMTileTest, ) {
  const mm_param_t t = testing::TestWithParam<mm_param_t>::GetParam();
  const data_type dst_type = t.postop_list.size() != 0 ? t.postop_list.back().dt : data_type::fp32;
  const dim_t M = 8, K = t.K, N = 48;
  const size_t dst_buf_size = M * t.ld_dst;
  const size_t dst_buf_bytes = dst_buf_size * type_size.at(dst_type);
  SPARSE_LOG_IF(FATAL, t.ld_dst < N) << "ld_dst must be gretaer then N";
  uint8_t* const src0 = new uint8_t[M * K];
  init_vector(src0, M * K, 0, 10);
  int8_t* const src1 = new int8_t[K * N];
  init_vector(src1, N * K, -10, 10);
  float* src_b0 = nullptr;
  if (t.binary_add) {
    src_b0 = new float[dst_buf_size];
    init_vector(src_b0, dst_buf_size, -10, 10);
  }

  int32_t* const bias = new int32_t[N];
  init_vector(bias, N, 0, 100);
  float scale[1];
  init_vector(scale, 1);

  // get true data
  void* const dst0_ref = new char[dst_buf_bytes];
  memset(dst0_ref, 3, dst_buf_bytes);  // something special to check if the kernel keep the additional memory untouched
  const auto dst0_ref_u8 = reinterpret_cast<uint8_t*>(dst0_ref);
  const auto dst0_ref_s8 = reinterpret_cast<int8_t*>(dst0_ref);
  const auto dst0_ref_f32 = reinterpret_cast<float*>(dst0_ref);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float value = -(bias[j] << t.bias_lshift);
      for (int k = 0; k < K; ++k) {
        float l = src0[i * 4 + k % 4 + (k / 4) * 4 * M];
        float r = src1[j * 4 + k % 4 + (k / 4) * 4 * N];
        value += l * r;
      }
      value *= scale[0];
      if (t.binary_add) value += src_b0[i * t.ld_dst + j];
      value = apply_postop_list(value, t.postop_list);
      switch (dst_type) {
        case data_type::u8:
          dst0_ref_u8[i * t.ld_dst + j] = static_cast<uint8_t>(value);
          break;
        case data_type::s8:
          dst0_ref_s8[i * t.ld_dst + j] = static_cast<int8_t>(value);
          break;
        case data_type::fp32:
          dst0_ref_f32[i * t.ld_dst + j] = value;
          break;
        default:
          SPARSE_LOG(FATAL) << "Unexpected dst type!";
          break;
      }
    }
  }

  // run kernel
  void* const dst0 = new char[dst_buf_bytes];
  memset(dst0, 3, dst_buf_bytes);  // something special to check if the kernel keep the additional memory untouched
  auto jit_ker = new jit_matmul_vnni_8xkx48_t(jit_matmul_vnni_8xkx48_t::param_t{
      t.K,
      t.ld_dst,
      scale[0],
      t.bias_lshift,
      t.binary_add,
      dst_type,
      t.postop_list,
      M,
      N,
  });
  ASSERT_NE(jit_ker, nullptr);
  ASSERT_TRUE(jit_ker->create_kernel());

  switch (dst_type) {
    case data_type::u8:
      exec_mmtile<uint8_t>(jit_ker, src0, src1, bias, src_b0, reinterpret_cast<uint8_t*>(dst0));
      ASSERT_TRUE(compare_data<uint8_t>(dst0, dst_buf_size, dst0_ref, dst_buf_size, 8e-3));
      break;
    case data_type::s8:
      exec_mmtile<int8_t>(jit_ker, src0, src1, bias, src_b0, reinterpret_cast<int8_t*>(dst0));
      ASSERT_TRUE(compare_data<int8_t>(dst0, dst_buf_size, dst0_ref, dst_buf_size, 8e-3));
      break;
    case data_type::fp32:
      exec_mmtile<float>(jit_ker, src0, src1, bias, src_b0, reinterpret_cast<float*>(dst0));
      ASSERT_TRUE(compare_data<float>(dst0, dst_buf_size, dst0_ref, dst_buf_size, 8e-3));
      break;
    default:
      SPARSE_LOG(FATAL) << "Unexpected dst type!";
      break;
  }
  delete[] src0;
  delete[] src1;
  if (src_b0 != nullptr) delete[] src_b0;
  delete[] bias;
  delete[] reinterpret_cast<char*>(dst0);
  delete[] reinterpret_cast<char*>(dst0_ref);
  delete jit_ker;
}
INSTANTIATE_TEST_SUITE_P(
    SparseLib, MMVNNIP2031P2013MMTileTest,
    ::testing::ValuesIn(std::vector<mm_param_t>{
        {16, 48, 2, false, {}},                                                                   // basic
        {64, 64, 7, true, {}},                                                                    // ld_dst != N
        {64, 65, 7, true, {}},                                                                    // ld_dst != N
        {64, 64, 7, true, {{dt::u8, postop_type::eltwise, postop_alg::quantize, 0.0, 0.0, 10}}},  // post quant s8
        {64, 64, 7, true, {{dt::u8, postop_type::eltwise, postop_alg::quantize, 128, 0.0, 10}}},  // post quant u8
    }),
    mmTestParam2str);

// test trmm vnni
struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
  int nthr;  // 0 for not touching OMP_NUM_THREADS and using what set outside
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
    const auto& op_desc = p.op_desc;
    transpose_matmul_desc kernel_desc(op_desc);
    transpose_matmul kernel(kernel_desc);
    kernel.execute(p.rt_data);

    std::shared_ptr<const kernel_desc_t> ker_ref_desc;
    kernel_desc_t::create<matmul_ref_kd_t>(ker_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> attention_ref_kernel;
    kernel_t::create<matmul_ref_k_t, matmul_ref_kd_t>(attention_ref_kernel, ker_ref_desc);
    attention_ref_kernel->execute(q.rt_data);
  } catch (const std::exception& e) {
    return t.expect_to_fail;
  }
  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[ssd::DST0];
    auto size1 = p.op_desc.tensor_descs()[ssd::DST0].size();
    auto buf2 = q.rt_data[ssd::DST0];
    auto size2 = q.op_desc.tensor_descs()[ssd::DST0].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[ssd::DST0].dtype();
    if (dst_type == dt::fp32) {
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::s32) {
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 0);
    } else if (dst_type == dt::s8) {
      return compare_data<int8_t>(buf1, size1, buf2, size2, 0);
    }
  }
  return false;
}

class MMVNNIP2031P2013KernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  MMVNNIP2031P2013KernelTest() {}
  virtual ~MMVNNIP2031P2013KernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MMVNNIP2031P2013KernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
  for (auto op_args : {t.args.first, t.args.second})
    for (auto rt_data : op_args.rt_data) {
      char* data = reinterpret_cast<char*>(const_cast<void*>(rt_data));
      delete[] data;
    }
}

std::pair<const void*, const void*> make_data_obj(const std::vector<int64_t>& a_shape, const dt& a_dt,
                                                  bool is_clear = false, const std::vector<float>& ranges = {-10, 10}) {
  if (a_shape.size() == 0) return {nullptr, nullptr};
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<dim_t>());
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
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, dim_t bs0, dim_t bs1, int nthr = 0,
                                         bool has_binary_add = true,
                                         std::unordered_map<std::string, std::string> attrs = {},
                                         std::vector<postop_attr> post_ops = {}) {
  // Step 1: Construct operator config
  const data_type dst_type = post_ops.size() != 0 ? post_ops.back().dt : data_type::fp32;

  const tensor_desc src0_desc{{bs1, K, bs0, M}, dt::s8, ft::ab};
  const tensor_desc src1_desc{{bs1, K, bs0, N}, dt::s8, ft::ab};
  const tensor_desc dst_desc{{bs0, bs1, M, N}, dst_type, ft::ab};
  const tensor_desc src2_desc = has_binary_add ? tensor_desc{{bs0, bs1, M, N}, dt::fp32, ft::ab} : tensor_desc();
  const std::vector<tensor_desc> ts_descs{src0_desc, src1_desc, dst_desc, src2_desc};

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    const bool is_clear = (index == ssd::DST0);
    const auto ranges = std::vector<float>{-10, 10};
    const auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  operator_desc op_desc(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                        attrs, post_ops);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {op_desc, rt_data1, nthr};
  op_args_t op_args_copy = {op_desc, rt_data2, nthr};

  return {op_args, op_args_copy};
}

static auto case_func = []() {
  google::InitGoogleLogging("MMVNNIP2031P2013KernelTest");
  std::vector<int> nthr_cases = {1, 2, 3, 4, 0};

  std::vector<test_params_t> cases;

  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);

    cases.push_back({gen_case(96, 64, 96, 1, 1, nthr, false,
                              {
                                  {"src0_scale", "8"},
                                  {"src1_scale", "8"},
                              })});
    cases.push_back({gen_case(16, 64, 16, 1, 1, nthr, false,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "8"},
                                  {"src1_scale", "8"},
                              })});
    cases.push_back({gen_case(64, 64, 64, 1, 1, nthr, false,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "8"},
                                  {"src1_scale", "8"},
                              })});
    cases.push_back({gen_case(128, 64, 128, 3, 12, nthr, true,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "12.3"},
                                  {"src1_scale", "11.2"},
                              })});
    cases.push_back({gen_case(128, 64, 128, 3, 12, nthr, true,
                              std::unordered_map<std::string, std::string>{
                                  {"src0_scale", "12.3"},
                                  {"src1_scale", "11.2"},
                                  {"out_scale", "0.125"},
                              },
                              std::vector<postop_attr>{
                                  postop_attr{data_type::s8, postop_type::eltwise, postop_alg::quantize, 0, 0, 0.0001},
                              })});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  auto& descs = tpi.param.args.first.op_desc.tensor_descs();
  auto attrs = tpi.param.args.first.op_desc.attrs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });

  const dim_t bs0 = shapes[ssd::DST0][0];
  const dim_t bs1 = shapes[ssd::DST0][1];
  const dim_t M = shapes[ssd::SRC0][3];  // aka src0_perm_shape[2]
  const dim_t K = shapes[ssd::SRC0][1];  // aka src0_perm_shape[3]
  const dim_t N = shapes[ssd::SRC1][3];  // aka src1_perm_shape[3]
  const bool has_binary_add = shapes[ssd::SRC2].size() != 0;
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back(std::to_string(bs0));
  params.push_back(std::to_string(bs1));
  params.push_back(std::to_string(M));
  params.push_back(std::to_string(K));
  params.push_back(std::to_string(N));
  if (has_binary_add) params.push_back("badd");
  for (const postop_attr& p_attr : tpi.param.args.first.op_desc.apply_postops_list()) {
    params.push_back("post" + std::string(postop_alg_name.at(p_attr.op_alg)));
    params.push_back(std::string(data_type_name.at(p_attr.dt)));
    params.push_back(num2id(p_attr.alpha));
    params.push_back(num2id(p_attr.beta));
    params.push_back(num2id(p_attr.scale));
  }
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMVNNIP2031P2013KernelTest, case_func(), test_suffix);
}  // namespace jd
