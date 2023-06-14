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
#include "unit_test_utils.hpp"
#include "kernels/matmul_types.hpp"
#include "interface.hpp"
#include "src/cpu/kernels/matmul_ref.hpp"
#include "kernels/data_pack.hpp"
#include "engine_factory.hpp"

#define OMP_NUM_THREADS "OMP_NUM_THREADS"

namespace test {
using io = jd::ssd::matmul_io::io;

struct op_args_t {
  jd::operator_desc op_desc;
  std::vector<const void*> rt_data;
  jd::exec_context_t context;
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
    std::shared_ptr<const jd::kernel_desc_t> matmul_ref_desc;
    jd::kernel_desc_t::create<jd::matmul_ref_kd_t>(matmul_ref_desc, q.op_desc);
    std::shared_ptr<const jd::kernel_t> matmul_ref_ref_ker;
    jd::kernel_t::create<jd::matmul_ref_k_t, jd::matmul_ref_kd_t>(matmul_ref_ref_ker, matmul_ref_desc);
    matmul_ref_ref_ker->execute(q.rt_data);

    n_thread_t with_n_thread(p.nthr);
    static jd::engine_factory factory;
    const jd::engine_t* cpu_engine = factory.create(jd::engine_kind::cpu, jd::runtime_kind::undef);
    std::shared_ptr<jd::kernel_t> matmul_kernel;
    jd::stream_t* stream = nullptr;
    cpu_engine->create_stream(reinterpret_cast<jd::stream_t**>(&stream));
    cpu_engine->create_kernel(p.op_desc, matmul_kernel, stream);
    matmul_kernel->execute(p.context);
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
    if (dst_type == jd::data_type::fp32) {
      return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == jd::data_type::s32) {
      return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
    } else if (dst_type == jd::data_type::u8) {
      return compare_data<uint8_t>(buf1, size1, buf2, size2, 1);
    } else if (dst_type == jd::data_type::s8) {
      return compare_data<int8_t>(buf1, size1, buf2, size2, 1);
    } else if (dst_type == jd::data_type::bf16) {
      return compare_data<jd::bfloat16_t>(buf1, size1, buf2, size2, 5e-2);
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
    for (auto input : op_args.context.inputs()) {
      delete input;
    }
    for (auto output : op_args.context.outputs()) {
      delete output;
    }
    for (auto rt_data : op_args.rt_data) {
      char* data = reinterpret_cast<char*>(const_cast<void*>(rt_data));
      delete[] data;
    }

    auto attr = op_args.op_desc.attrs();
    if (attr["weight_8bit"] != "") {
      int8_t* weight = reinterpret_cast<int8_t*>(str_to_num<intptr_t>(attr["weight_8bit"]));
      delete[] weight;
    }
  }
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N, jd::data_type src1_dtype = jd::data_type::bf16,
                                         int nthr = 0, std::unordered_map<std::string, std::string> attrs = {},
                                         std::vector<jd::postop_attr> postop_attr = {}) {
  // Step 1: Construct operator config
  jd::tensor_desc src0_desc = {{M, K}, jd::data_type::bf16, jd::format_type::ab};
  jd::tensor_desc src1_desc = {{N, K}, src1_dtype, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{M, N}, jd::data_type::bf16, jd::format_type::ab};
  jd::tensor_desc bias_desc = {{N}, jd::data_type::bf16, jd::format_type::a};
  jd::tensor_desc scale_desc = {{N}, jd::data_type::fp32, jd::format_type::a};
  jd::tensor_desc zp0_desc = {{1}, jd::data_type::fp32, jd::format_type::a};
  jd::tensor_desc append_sum_desc = {{M, N}, jd::data_type::bf16, jd::format_type::a};
  std::vector<jd::tensor_desc> ts_descs = {src0_desc, src1_desc, dst_desc, bias_desc, scale_desc, zp0_desc};
  if (attrs["append_sum"] != "") {
    ts_descs.push_back(append_sum_desc);
  }

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;

  static jd::engine_factory factory;
  const jd::engine_t* cpu_engine = factory.create(jd::engine_kind::cpu, jd::runtime_kind::undef);
  jd::stream_t* stream = nullptr;
  cpu_engine->create_stream(reinterpret_cast<jd::stream_t**>(&stream));

  jd::exec_context_t contexts[2] = {jd::exec_context_t(stream), jd::exec_context_t(stream)};
  for (size_t index = 0; index < ts_descs.size(); ++index) {
    auto& tsd = ts_descs[index];
    auto ranges = (index == io::ZP0) ? std::vector<float>{0.f, 0.f} : std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), false, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
    for (size_t i = 0; i < 2; i++) {
      jd::memory_storage_t* mem;
      cpu_engine->create_memory_storage(&mem);
      if (i == 0) {
        mem->set_handle(const_cast<void*>(data_pair.first));
      } else {
        mem->set_handle(const_cast<void*>(data_pair.second));
      }
      if (index == io::SRC0) {
        contexts[i].set_dynamic_shape({M});
      }
      if (index == io::DST0) {
        contexts[i].add_output(mem);
      } else {
        contexts[i].add_input(mem);
      }
    }
  }
  attrs["thread_nums"] = std::to_string(nthr);
  std::unordered_map<std::string, std::string> attrs1 = attrs;
  std::unordered_map<std::string, std::string> attrs2 = attrs;

  if (src1_dtype == jd::data_type::bf16) {
    attrs1["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data1[io::SRC1]));
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(new uint8_t[N * K]));
    attrs2["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data2[io::SRC1]));
    attrs2["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(new uint8_t[N * K]));
  } else if (src1_dtype == jd::data_type::s8) {
    std::function<int8_t(int8_t)> cast_func_s8 = [](int8_t x) { return x; };
    int8_t* weight_8bit = new int8_t[N * K];
    int8_t* src1_s8 = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    jd::pack<int8_t, int8_t>(weight_8bit, src1_s8, N, K, cast_func_s8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  } else if (src1_dtype == jd::data_type::f8_e4m3) {
    std::function<jd::float8_e4m3_t(jd::float8_e4m3_t)> cast_func_fp8 = [](jd::float8_e4m3_t x) { return x; };
    jd::float8_e4m3_t* src1_fp8 = reinterpret_cast<jd::float8_e4m3_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    jd::float8_e4m3_t* weight_8bit = new jd::float8_e4m3_t[N * K];
    jd::pack<jd::float8_e4m3_t, jd::float8_e4m3_t>(weight_8bit, src1_fp8, N, K, cast_func_fp8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  } else if (src1_dtype == jd::data_type::f8_e5m2) {
    std::function<jd::float8_e5m2_t(jd::float8_e5m2_t)> cast_func_fp8 = [](jd::float8_e5m2_t x) { return x; };
    jd::float8_e5m2_t* src1_fp8 = reinterpret_cast<jd::float8_e5m2_t*>(const_cast<void*>(rt_data1[io::SRC1]));
    jd::float8_e5m2_t* weight_8bit = new jd::float8_e5m2_t[N * K];
    jd::pack<jd::float8_e5m2_t, jd::float8_e5m2_t>(weight_8bit, src1_fp8, N, K, cast_func_fp8);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  }

  jd::operator_desc op_desc1(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference,
                             jd::engine_kind::cpu, ts_descs, attrs1, postop_attr);
  jd::operator_desc op_desc2(jd::kernel_kind::transpose_matmul, jd::kernel_prop::forward_inference,
                             jd::engine_kind::cpu, ts_descs, attrs2, postop_attr);

  op_args_t op_args = {op_desc1, rt_data1, contexts[0], nthr};
  op_args_t op_args_copy = {op_desc2, rt_data2, contexts[1], nthr};
  return {op_args, op_args_copy};
}

static auto case_func = []() {
  google::InitGoogleLogging("MMAVX512FP8KernelTest");
  std::vector<int> nthr_cases = {4};
  std::vector<test_params_t> cases;
  jd::postop_attr fp32_swish_attr{jd::data_type::fp32, jd::postop_type::eltwise, jd::postop_alg::swish, 2};
  for (int nthr : nthr_cases) {
    n_thread_t with_n_thread(nthr);
    cases.push_back({gen_case(4, 2, 16, jd::data_type::f8_e4m3, nthr, {{"alpha", "1.f"}, {"beta", "0.f"}})});
    cases.push_back({gen_case(4, 2, 16, jd::data_type::f8_e5m2, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 2, 16, jd::data_type::s8, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});

    cases.push_back(
        {gen_case(4, 2, 16, jd::data_type::f8_e4m3, nthr, {{"append_sum", "1"}, {"alpha", "1.f"}, {"beta", "0.f"}})});
    cases.push_back(
        {gen_case(4, 2, 16, jd::data_type::f8_e5m2, nthr, {{"append_sum", "1"}, {"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back(
        {gen_case(4, 2, 16, jd::data_type::s8, nthr, {{"append_sum", "1"}, {"alpha", "1.f"}, {"beta", "1.f"}})});

    cases.push_back({gen_case(4, 4096, 4096, jd::data_type::f8_e5m2, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 4096, jd::data_type::s8, nthr, {{"alpha", "1.f"}, {"beta", "1.f"}})});
    cases.push_back({gen_case(4, 4096, 4096, jd::data_type::bf16, nthr,
                              {{"alpha", "1.f"}, {"beta", "1.f"}, {"weight_type", "f8_e5m2"}})});
    cases.push_back({gen_case(4, 4096, 4096, jd::data_type::bf16, nthr,
                              {{"alpha", "1.f"},
                               {"beta", "1.f"},
                               {"thread_nums", std::to_string(nthr)},
                               {"weight_type", "f8_e5m2"},
                               {"postop_list", "fp32_swish"}},
                              {fp32_swish_attr})});
  }
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  auto attrs = tpi.param.args.first.op_desc.attrs();
  const auto shapes = tpi.param.args.first.op_desc.tensor_shapes();
  const auto dtypes = tpi.param.args.first.op_desc.tensor_dtypes();

  const dim_t M = shapes[io::SRC0][0];
  const dim_t K = shapes[io::SRC0][1];
  const dim_t N = shapes[io::SRC1][0];
  std::vector<std::string> params;
  params.push_back("c" + std::to_string(static_cast<int>(tpi.param.args.first.nthr)));
  params.push_back(std::to_string(M));
  params.push_back(std::to_string(K));
  params.push_back(std::to_string(N));
  params.push_back(jd::data_type_name.at(dtypes[io::SRC1]));
  if (attrs["alpha"] != "" && str_to_num<float>(attrs["alpha"]) != 1.f)
    params.push_back(std::string("alpha") + num2id(attrs["alpha"]));
  if (attrs["weight_type"] != "") {
    params.push_back(attrs["weight_type"]);
  }
  if (attrs["postop_list"] != "") {
    params.push_back(attrs["postop_list"]);
  }
  if (attrs["append_sum"] != "") {
    params.push_back("append_sum");
  }
  return join_str(params, "_");
}

INSTANTIATE_TEST_SUITE_P(SparseLib, MMAVX512FP8KernelTest, case_func(), test_suffix);
}  // namespace test
