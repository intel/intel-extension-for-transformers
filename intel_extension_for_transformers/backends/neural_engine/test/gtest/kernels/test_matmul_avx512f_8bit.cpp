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

#include "gtest/gtest.h"
#include "unit_test_utils.hpp"
#include "kernels/matmul_types.hpp"
#include "interface.hpp"
#include "kernels/matmul_ref.hpp"
#include "utils.hpp"
#include "kernels/fp8.hpp"
#include "cpu_parallel.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace jd {
using dt = jd::data_type;
using ft = jd::format_type;
using io = ssd::matmul_io::io;

struct op_args_t {
  operator_desc op_desc;
  std::vector<const void*> rt_data;
  int nthr;  // 0 for not touching OMP_NUM_THREADS and using what set outside
};

struct test_params_t {
  std::pair<op_args_t, op_args_t> args;
  bool expect_to_fail;
};

int8_t to_8bit(float fp32, data_type type) {
  if (type == dt::s8) {
    return fp32_to_int8(fp32);
  } else if (type == dt::f8_e4m3) {
    return float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(fp32);
  } else if (type == dt::f8_e5m2) {
    return float8_base<FloatEncoding::E5M2>::convert_float_to_fp8(fp32);
  } else {
    return 0;
  }
}
void reference(int8_t* srcptr, int8_t* dstptr, data_type type, int row, int col, int rowpad, int colpad, int srcstride,
               int dststride) {
  int srcld = srcstride / 2;
  auto sptr = reinterpret_cast<int8_t*>(srcptr);
  auto dptr = (dstptr);
  int NTile = 16;
  for (int irow = 0; irow < rowpad; irow += NTile) {
    for (int icol = 0; icol < colpad; icol += 1) {
      for (int iin = 0; iin < NTile; iin++) {
        if (irow + iin < row) {
          if (icol < col) {
            *(dptr + irow * dststride + icol * NTile + iin) = *(sptr + (irow + iin) * srcld + icol);
          } else {
            *(dptr + irow * dststride + icol * NTile + iin) = to_8bit(static_cast<float>(0), type);
          }
        } else {
          *(dptr + irow * dststride + icol * NTile + iin) = to_8bit(static_cast<float>(0), type);
        }
      }
    }
  }
}

int8_t* pack(int8_t* input, int n, int k, data_type type, int ncores) {
  int ldb = k;
  int npad = pad_to(n, 16);
  int kpad = pad_to(k, 1);
  int8_t* output = new int8_t[npad * kpad];
  Parallel2DRowMajor _para;
  _para.update(npad, kpad, 16, 1, ncores);
#pragma omp parallel
  {
    int tidx = omp_get_thread_num();
    int colidx, rowidx, rowsize, colsize;
    _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      int rowremain = remainsize(rowidx, n, rowsize);
      int colremain = remainsize(colidx, k, colsize);
      reference(input + rowidx * ldb + colidx, output + rowidx * kpad + colidx * 16, type, rowremain, colremain,
                rowsize, colsize, k * sizeof(bfloat16_t), kpad);
    }
  }
  return output;
}

bool check_result(const test_params_t& t) {
  const auto& p = t.args.first;
  const auto& q = t.args.second;
  try {
    n_thread_t with_n_thread(p.nthr);
    const auto& op_desc = p.op_desc;
    transpose_matmul_desc kernel_desc(op_desc);
    transpose_matmul kernel(kernel_desc);
    kernel.execute(p.rt_data);

    std::shared_ptr<const kernel_desc_t> matmul_ref_desc;
    kernel_desc_t::create<matmul_ref_kd_t>(matmul_ref_desc, q.op_desc);
    std::shared_ptr<const kernel_t> matmul_ref_ref_ker;
    kernel_t::create<matmul_ref_k_t, matmul_ref_kd_t>(matmul_ref_ref_ker, matmul_ref_desc);
    matmul_ref_ref_ker->execute(q.rt_data);
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  if (!t.expect_to_fail) {
    auto buf1 = p.rt_data[io::DST0];
    auto size1 = p.op_desc.tensor_descs()[io::DST0].size();
    auto buf2 = q.rt_data[io::DST0];
    auto size2 = q.op_desc.tensor_descs()[io::DST0].size();
    // Should compare buffer with different addresses
    EXPECT_NE(buf1, buf2);
    const auto& dst_type = p.op_desc.tensor_descs()[io::DST0].dtype();
    if (dst_type == dt::fp32) {
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::s32) {
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == dt::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 1);
    } else if (dst_type == dt::s8) {
      return compare_data<int8_t>(buf1, size1, buf2, size2, 1);
    } else if (dst_type == dt::bf16) {
      return compare_data<bfloat16_t>(buf1, size1, buf2, size2, 0.1);
    }
  }
  return false;
}

class MMAVX512FP8KernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  MMAVX512FP8KernelTest() {}
  virtual ~MMAVX512FP8KernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(MMAVX512FP8KernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));
  for (auto op_args : {t.args.first, t.args.second}) {
    for (auto rt_data : op_args.rt_data) {
      char* data = reinterpret_cast<char*>(const_cast<void*>(rt_data));
      delete[] data;
    }
    // auto tensor = op_args.op_desc.tensor_descs()[io::SRC1];
    auto attr = op_args.op_desc.attrs();
    if (attr["weight_8bit"] != "") {
      int8_t* weight = reinterpret_cast<int8_t*>(str_to_num<intptr_t>(attr["weight_8bit"]));
      delete[] weight;
    }
  }
}

std::pair<const void*, const void*> make_data_obj(const std::vector<int64_t>& a_shape, const dt& a_dt,
                                                  bool is_clear = false, const std::vector<float>& ranges = {-10, 10}) {
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
    } else if (a_dt == dt::bf16) {
      data_ptr = new bfloat16_t[elem_num];
      init_vector(static_cast<bfloat16_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::f8_e4m3) {
      data_ptr = new float8_t[elem_num];
      float8_t* fp8 = reinterpret_cast<float8_t*>(data_ptr);
      float* fp32 = new float[elem_num];
      init_vector(fp32, elem_num, ranges[0], ranges[1]);
      for (int i = 0; i < elem_num; i++) {
        fp8[i] = float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(fp32[i]);
      }
      delete[] fp32;
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, data_type src1_dtype = dt::bf16, int nthr = 0,
                                         std::unordered_map<std::string, std::string> attrs = {},
                                         std::vector<postop_attr> postop_attr = {}) {
  // Step 1: Construct operator config
  tensor_desc src0_desc = {{M, K}, dt::bf16, ft::ab};
  tensor_desc src1_desc = {{N, K}, src1_dtype, ft::ab};
  tensor_desc dst_desc = {{M, N}, dt::bf16, ft::ab};
  tensor_desc bias_desc = {{N}, dt::bf16, ft::ab};
  std::vector<tensor_desc> ts_descs = {src0_desc, src1_desc, dst_desc, bias_desc};

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == io::DST0);
    auto ranges = std::vector<float>{-2, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  attrs["thread_nums"] = std::to_string(nthr);
  std::unordered_map<std::string, std::string> attrs1 = attrs;
  std::unordered_map<std::string, std::string> attrs2 = attrs;

  if (src1_dtype == dt::bf16) {
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(new int8_t[N * K]));
    attrs1["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data1[io::SRC1]));
  } else {
    int ncores = str_to_num<intptr_t>(attrs["thread_nums"]);
    data_type type = data_type::f8_e4m3;

    for (auto& it : data_type_name) {
      if (it.second == attrs["weight_type"]) {
        type = it.first;
        break;
      }
    }
    int8_t* weight_8bit = pack(reinterpret_cast<int8_t*>(const_cast<void*>(rt_data1[io::SRC1])), N, K, type, ncores);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  }
  operator_desc op_desc1(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                         attrs1, postop_attr);
  operator_desc op_desc2(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                         attrs2, postop_attr);

  op_args_t op_args = {op_desc1, rt_data1, nthr};
  op_args_t op_args_copy = {op_desc2, rt_data2, nthr};
  return {op_args, op_args_copy};
}

static auto case_func = []() {
  google::InitGoogleLogging("MMAVX512FP8KernelTest");
  std::vector<int> nthr_cases = {1, 2, 3, 4};
  std::vector<test_params_t> cases;
  postop_attr fp32_swish_attr{data_type::fp32, postop_type::eltwise, postop_alg::swish, 2};
  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);
    cases.push_back({gen_case(4, 2, 16, dt::f8_e4m3, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 4096, dt::f8_e4m3, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 16384, dt::f8_e4m3, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 4096, dt::s8, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 16384, dt::s8, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back(
        {gen_case(4, 4096, 4096, dt::bf16, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}, {"weight_type", "f8_e4m3"}})});

    cases.push_back({gen_case(4, 4096, 4096, dt::bf16, nthr,
                              {{"alpha", "1.f"},
                               {"beta", "1.f"},
                               {"thread_nums", std::to_string(nthr)},
                               {"weight_type", "f8_e4m3"},
                               {"postop_list", "fp32_swish"}},
                              {fp32_swish_attr})});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  auto& descs = tpi.param.args.first.op_desc.tensor_descs();
  auto attrs = tpi.param.args.first.op_desc.attrs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });

  const dim_t M = shapes[io::SRC0][0];
  const dim_t K = shapes[io::SRC0][1];
  const dim_t N = shapes[io::SRC1][0];
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back(std::to_string(M));
  params.push_back(std::to_string(K));
  params.push_back(std::to_string(N));
  params.push_back(data_type_name.at(descs[io::SRC1].dtype()));
  if (attrs["alpha"] != "" && str_to_num<float>(attrs["alpha"]) != 1.f)
    params.push_back(std::string("alpha") + num2id(attrs["alpha"]));
  if (attrs["weight_type"] != "") {
    params.push_back(attrs["weight_type"]);
  }
  if (attrs["postop_list"] != "") {
    params.push_back(attrs["postop_list"]);
  }
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMAVX512FP8KernelTest, case_func(), test_suffix);
}  // namespace jd
