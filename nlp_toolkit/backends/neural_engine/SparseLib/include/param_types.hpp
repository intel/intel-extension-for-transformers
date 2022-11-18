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

#ifndef ENGINE_SPARSELIB_INCLUDE_PARAM_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_PARAM_TYPES_HPP_
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <map>
namespace jd {
// The main kinds of kernel.
enum class kernel_kind : uint8_t {
  undef,
  sparse_matmul,
  eltwiseop,
  layernorm_ba,
  transpose_matmul,
  softmax,
  gather,
  attention
};

enum class postop_alg : uint8_t { undef, exp, tanh, gelu, relu, quantize, dequantize, linear, eltop_int_lut };

enum class binaryop_alg : uint8_t { undef, add, sub, mul, per_channel_quant, per_channel_dequant };

enum class postop_type : uint8_t { eltwise };

static std::map<postop_alg, const char*> postop_alg_name = {
    {postop_alg::exp, "exp"},
    {postop_alg::gelu, "gelu"},
};

enum class reg_type : uint8_t { mask, zmm, reg64 };

// The propagation kind of kernel, temporarily defined as a specific function or
// scenario. Further, the specific function can be implemented by different
// algorithms, e.g.: gemm, brgemm, ref.
enum class kernel_prop : uint8_t {
  undef,
  forward_inference,
};

// Data type.
enum class data_type : uint8_t {
  undef,
  u8,
  s8,
  u16,
  s16,
  fp16,
  bf16,
  fp32,
  s32,
};

// Format type.
enum class format_type : uint8_t {
  undef,
  a,
  ab,  // shape permutation = {0, 1}
  ba,  // shape permutation = {1, 0}
  abc,

  // encoding format of sparse matrix
  uncoded,
  csr,
  csc,
  bsr,
  bsc,
  csrp,
};

// Engine kind.
enum class engine_kind : uint8_t {
  undef,
  cpu,
};

// postop attribute for op-fusion
class postop_attr {
 public:
  data_type dt;
  postop_type op_type;
  postop_alg op_alg;
  float alpha = 0;
  float beta = 0;
  float scale = 0;

  postop_attr() {}

  postop_attr(const data_type& dt, const postop_type& op_type, const postop_alg& op_alg, float alpha = 0.0,
              float beta = 0.0, float scale = 0.0)
      : dt(dt), op_type(op_type), op_alg(op_alg), alpha(alpha), beta(beta), scale(scale) {}
};

class binaryop_attr {
 public:
  void* static_addr;
  float* scale;
  float* zp;
  binaryop_alg op_alg;
  data_type op_dt;

  binaryop_attr(binaryop_alg alg, data_type dt) : op_alg(alg), op_dt(dt) {
    static_addr = nullptr;
    scale = nullptr;
    zp = nullptr;
  }
  binaryop_attr(void* ptr, binaryop_alg alg, data_type dt) : static_addr(ptr), op_alg(alg), op_dt(dt) {}
  void set_scale(float* scale) { this->scale = scale; }
  void set_zp(float* zp) { this->zp = zp; }
};

static std::unordered_map<data_type, const int> type_size = {
    {data_type::fp32, 4}, {data_type::s32, 4}, {data_type::fp16, 2},
    {data_type::bf16, 2}, {data_type::u8, 1},  {data_type::s8, 1},
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_PARAM_TYPES_HPP_
