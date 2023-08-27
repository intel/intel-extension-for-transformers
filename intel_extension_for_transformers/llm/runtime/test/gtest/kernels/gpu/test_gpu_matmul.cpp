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
#include "../unit_test_utils.hpp"
#include "gtest/gtest.h"
#include "kernel_desc.hpp"
#include "kernel.hpp"
#include "engine.hpp"
#include "stream.hpp"
#include "engine_factory.hpp"
#include "memory_storage.hpp"

namespace test {
struct op_args_t {
  jd::operator_desc op_desc;
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
    jd::engine_factory factory;
    const jd::engine_t* ocl_gpu_engine = factory.create(jd::engine_kind::gpu, jd::runtime_kind::opencl);
    const jd::engine_t* cpu_engine = factory.create(jd::engine_kind::cpu, jd::runtime_kind::undef);
    std::shared_ptr<jd::kernel_t> matmul_kernel;
    jd::stream_t* stream = nullptr;
    ocl_gpu_engine->create_stream(reinterpret_cast<jd::stream_t**>(&stream));
    ocl_gpu_engine->create_kernel(p.op_desc, matmul_kernel, stream);
    std::vector<jd::memory_storage_t*> mems;
    int M = p.op_desc.tensor_descs()[2].shape()[0];
    int N = p.op_desc.tensor_descs()[2].shape()[1];
    int K = p.op_desc.tensor_descs()[0].shape()[1];
    for (auto& num : {M, N, K}) {
      mems.emplace_back();
      cpu_engine->create_memory_storage(reinterpret_cast<jd::memory_storage_t**>(&mems.back()));
      mems.back()->copy(const_cast<void*>(reinterpret_cast<const void*>(&num)), sizeof(int*),
                        jd::copy_direction_t::host_to_host, nullptr);
    }
    size_t ele_num = p.op_desc.tensor_descs().size();
    for (size_t i = 0; i < ele_num; i++) {
      mems.emplace_back();
      ocl_gpu_engine->create_memory_storage(reinterpret_cast<jd::memory_storage_t**>(&mems.back()));
      mems.back()->copy(const_cast<void*>(p.rt_data[i]), p.op_desc.tensor_descs()[i].size() * sizeof(float*),
                        jd::copy_direction_t::host_to_device, stream);
    }

    jd::exec_context_t context(stream);
    size_t i = 0;
    for (; i < 5; i++) {
      context.add_input(mems[i]);
    }
    for (; i < 6; i++) {
      context.add_output(mems[i]);
    }

    matmul_kernel->init(context);
    matmul_kernel->execute();
    mems.back()->copy(const_cast<void*>(p.rt_data[2]), p.op_desc.tensor_descs()[2].size() * sizeof(float*),
                      jd::copy_direction_t::device_to_host, stream);

    const float* A = reinterpret_cast<const float*>(q.rt_data[0]);
    const float* B = reinterpret_cast<const float*>(q.rt_data[1]);
    float* C = const_cast<float*>(reinterpret_cast<const float*>(q.rt_data[2]));
    M = q.op_desc.tensor_descs()[2].shape()[0];
    N = q.op_desc.tensor_descs()[2].shape()[1];
    K = q.op_desc.tensor_descs()[0].shape()[1];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
      }
    }
  } catch (const std::exception& e) {
    if (t.expect_to_fail) {
      return true;
    } else {
      return false;
    }
  }
  auto buf1 = p.rt_data[2];
  auto size1 = p.op_desc.tensor_descs()[2].size();
  auto buf2 = q.rt_data[2];
  auto size2 = q.op_desc.tensor_descs()[2].size();

  return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
}

class GPUMatmulKernelTest : public testing::TestWithParam<test_params_t> {
 protected:
  GPUMatmulKernelTest() {}
  virtual ~GPUMatmulKernelTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(GPUMatmulKernelTest, ) {
  test_params_t t = testing::TestWithParam<test_params_t>::GetParam();
  EXPECT_TRUE(check_result(t));

  const auto& op_desc = t.args.first.op_desc;
  auto op_attrs = op_desc.attrs();
}

std::pair<op_args_t, op_args_t> gen_case(dim_t M, dim_t K, dim_t N) {
  jd::tensor_desc src_desc = {{M, K}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc wei_desc = {{K, N}, jd::data_type::fp32, jd::format_type::ab};
  jd::tensor_desc dst_desc = {{M, N}, jd::data_type::fp32, jd::format_type::ab};
  std::vector<jd::tensor_desc> ts_descs = {src_desc, wei_desc, dst_desc};
  jd::operator_desc op_desc(jd::kernel_kind::matmul, jd::kernel_prop::forward_inference, jd::engine_kind::gpu,
                            jd::runtime_kind::opencl, ts_descs, {});

  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == 2);
    float data_sparsity = 0;
    auto ranges = std::vector<float>{-10, 10};
    auto data_pair = make_data_obj(tsd.shape(), tsd.dtype(), is_clear, ranges, data_sparsity);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  op_args_t op_args_p = {op_desc, rt_data1, 1};
  op_args_t op_args_q = {op_desc, rt_data2, 1};

  return {op_args_p, op_args_q};
}
static auto case_func = []() {
  google::InitGoogleLogging("GPUMatmulKernelTest");
  std::vector<test_params_t> cases;
  cases.push_back({gen_case(16, 16, 16)});
  return ::testing::ValuesIn(cases);
};

std::string test_suffix(testing::TestParamInfo<test_params_t> tpi) {
  (void)tpi;
  return "example_test";
}

INSTANTIATE_TEST_SUITE_P(SparseLib, GPUMatmulKernelTest, case_func(), test_suffix);

}  // namespace test
