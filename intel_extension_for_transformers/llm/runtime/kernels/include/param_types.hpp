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
#include <cassert>
#include <cstdint>
#include <unordered_map>

namespace jd {
// The main kinds of kernel.
enum class kernel_kind : uint8_t {
  undef,
  sparse_matmul,
  matmul,
  eltwiseop,
  groupnorm,
  layernorm_ba,
  layernormalized_spmm,
  transpose_matmul,
  dynamic_quant_matmul,
  softmax,
  gather,
  attention,
  transpose_mha,
  mha_dense,
  slice,
  dynamic_quant
};

enum class postop_alg : uint8_t {
  undef,
  exp,
  tanh,
  gelu,
  relu,
  low_precision_exp,
  swish,
  quantize,
  dequantize,
  linear,
  eltop_int_lut
};

enum class binaryop_alg : uint8_t { undef, add, sub, mul, per_channel_quant, per_channel_dequant };

enum class postop_type : uint8_t { eltwise };

static std::unordered_map<postop_alg, const char*> postop_alg_name = {
    {postop_alg::exp, "exp"},           {postop_alg::tanh, "tanh"},
    {postop_alg::gelu, "gelu"},         {postop_alg::relu, "relu"},
    {postop_alg::quantize, "quantize"}, {postop_alg::dequantize, "dequantize"},
    {postop_alg::linear, "linear"},     {postop_alg::eltop_int_lut, "eltop_int_lut"},
    {postop_alg::swish, "swish"}};

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
  s4,
  f8_e4m3,
  f8_e5m2,
  u8,
  s8,
  u16,
  s16,
  fp16,
  bf16,
  fp32,
  s32,
};
const std::unordered_map<data_type, const char*> data_type_name{
    {data_type::u8, "u8"},           {data_type::s8, "s8"},     {data_type::f8_e4m3, "f8_e4m3"},
    {data_type::f8_e5m2, "f8_e5m2"}, {data_type::u16, "u16"},   {data_type::s16, "s16"},
    {data_type::fp16, "fp16"},       {data_type::bf16, "bf16"}, {data_type::fp32, "fp32"},
    {data_type::s32, "s32"},
};

// Format type.
enum class format_type : uint8_t {
  undef,
  a,
  ab,  // shape permutation = {0, 1}
  ba,  // shape permutation = {1, 0}
  abc,
  abcd,
  acbd,

  // encoding format of sparse matrix
  uncoded,
  csr,
  csc,
  bsr,
  bsc,
  csrp,
};
constexpr format_type plain_format(const int n) {
  return n == 1   ? format_type::a
         : n == 2 ? format_type::ab
         : n == 3 ? format_type::abc
         : n == 4 ? format_type::abcd
                  : (assert(false), format_type::undef);
}

static const std::unordered_map<format_type, const char*> format_type_name = {
    {format_type::a, "a"},       {format_type::ab, "ab"},     {format_type::ba, "ba"},     {format_type::abc, "abc"},
    {format_type::abcd, "abcd"}, {format_type::acbd, "acbd"}, {format_type::csr, "csr"},   {format_type::csc, "csc"},
    {format_type::bsr, "bsr"},   {format_type::bsc, "bsc"},   {format_type::csrp, "csrp"},
};

// Engine kind.
enum class engine_kind : uint8_t {
  undef,
  cpu,
  gpu,
};

// Runtime kind.
enum class runtime_kind : uint8_t { undef, opencl, sycl, thread_pool };

// postop attribute for op-fusion
class postop_attr {
 public:
  data_type dt = data_type::undef;
  postop_type op_type = postop_type::eltwise;
  postop_alg op_alg = postop_alg::undef;
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
  binaryop_alg op_alg = binaryop_alg::undef;
  data_type op_dt = data_type::undef;

  binaryop_attr(binaryop_alg alg, data_type dt) : op_alg(alg), op_dt(dt) {
    static_addr = nullptr;
    scale = nullptr;
    zp = nullptr;
  }
  binaryop_attr(void* ptr, binaryop_alg alg, data_type dt) : static_addr(ptr), op_alg(alg), op_dt(dt) {
    scale = nullptr;
    zp = nullptr;
  }
  void set_scale(float* scale) { this->scale = scale; }
  void set_zp(float* zp) { this->zp = zp; }
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_PARAM_TYPES_HPP_
