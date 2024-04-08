//  Copyright (c) 2023 Intel Corporation
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
#include "dispatcher/include/dispatcher_utils.hpp"
#include "dispatcher/include/bestla_gemm_dispatcher.hpp"
#include "dispatcher/include/bestla_weightonly_dispatcher.hpp"
#include "dispatcher/include/bestla_packq_impl.hpp"
#include "include/dropout.hpp"
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <cassert>
#include <map>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

static std::map<torch::ScalarType, dispatcher_utils::QBITS_DT> qbits_dt_map{
    {torch::kFloat32, dispatcher_utils::QBITS_FP32}, {torch::kBFloat16, dispatcher_utils::QBITS_BF16}};

static dispatcher_utils::QBITS_DT get_qbits_dt(torch::Tensor* tensor) {
  TORCH_CHECK(qbits_dt_map.count(tensor->scalar_type()) != 0, "unsupported qbits data type.");
  return qbits_dt_map[tensor->scalar_type()];
}

template <woq::WOQ_TASK TASK>
static void inline init_woq_config_param(woq::woq_config_param* p, woq::woq_runtime_ctx* ctx,
                                         const std::string& compute_type, const std::string& weight_type,
                                         const std::string& scale_type, bool asym) {
  p->compute_type = compute_type;
  p->weight_type = weight_type;
  p->scale_type = scale_type;
  p->asym = asym;
  switch (TASK) {
    case woq::WOQ_QUANTIZE:
    case woq::WOQ_DEQUANTIZE:
      p->src_dt = dispatcher_utils::QBITS_FP32;
      p->dst_dt = dispatcher_utils::QBITS_FP32;  // bestla doesn't care about dst_dt in quantize/dequant task,so set
                                                 // fp32 as default.
      break;
    case woq::WOQ_LINEAR:
      p->src_dt = get_qbits_dt(ctx->activation);
      p->dst_dt = get_qbits_dt(ctx->output);
      break;
  }
}

static torch::Tensor repack_quantized_weight(const torch::Tensor& qweight, const torch::Tensor& scale,
                                             const torch::Tensor& zp, const torch::Tensor& g_idx,
                                             const std::string& weight_type, const std::string& scale_type,
                                             const std::string& compute_type, bool asym, int64_t blocksize) {
  torch::Tensor output;
  woq::repack_quantized_weight_param p{compute_type,      weight_type, scale_type, asym, static_cast<int>(blocksize),
                                       g_idx.numel() != 0};
  woq::repack_quantized_weight_ctx ctx{const_cast<torch::Tensor*>(&qweight),
                                       const_cast<torch::Tensor*>(&scale),
                                       const_cast<torch::Tensor*>(&zp),
                                       const_cast<torch::Tensor*>(&g_idx),
                                       &output,
                                       static_cast<int>(qweight.sizes()[1]),
                                       static_cast<int>(qweight.sizes()[0])};
  woq::bestla_packq(&p, &ctx, woq::WOQ_REPACK);
  return output;
}

static size_t get_packed_weight_size(int k, int n, const std::string& weight_type, const std::string& scale_type,
                                     const std::string& compute_type, bool asym, int64_t blocksize, bool act_shuf) {
  woq::repack_quantized_weight_param p{compute_type, weight_type, scale_type, asym, static_cast<int>(blocksize),
                                       act_shuf};
  woq::repack_quantized_weight_ctx ctx;
  ctx.n = n;
  ctx.k = k;
  woq::bestla_packq(&p, &ctx, woq::WOQ_GET_PACKW_SIZE);
  return ctx.packw_size;
}

static torch::Tensor quantize_to_packed_weight(const torch::Tensor& fp32_weight, bool transpose, int64_t blocksize,
                                               const std::string& compute_type, const std::string& weight_type,
                                               const std::string& scale_type, bool asym) {
  torch::Tensor output;
  woq::woq_config_param p;
  woq::woq_runtime_ctx ctx{nullptr, const_cast<torch::Tensor*>(&fp32_weight), nullptr, &output, transpose};
  init_woq_config_param<woq::WOQ_QUANTIZE>(&p, &ctx, compute_type, weight_type, scale_type, asym);
  p.blocksize = static_cast<int>(blocksize);
  woq::dispatch_woq_task(&p, &ctx, woq::WOQ_QUANTIZE);
  return output;
}

static void dequantize_packed_weight(const torch::Tensor& compressed_weight, torch::Tensor& dequantize_weight,
                                     bool transpose, const std::string& compute_type, const std::string& weight_type,
                                     const std::string& scale_type) {
  woq::woq_config_param p;
  woq::woq_runtime_ctx ctx{nullptr, const_cast<torch::Tensor*>(&compressed_weight), nullptr, &dequantize_weight,
                           transpose};
  init_woq_config_param<woq::WOQ_DEQUANTIZE>(&p, &ctx, compute_type, weight_type, scale_type,
                                             false);  // zp is packed to compressed-weight, it's ok to set false here.
  woq::dispatch_woq_task(&p, &ctx, woq::WOQ_DEQUANTIZE);
}

static void woq_linear(const torch::Tensor& activation, const torch::Tensor& weight, const torch::Tensor& bias,
                       torch::Tensor& output, const std::string& compute_type, const std::string& weight_type,
                       const std::string& scale_type, bool asym) {
  woq::woq_config_param p;
  torch::Tensor* rt_bias = bias.numel() == 0 ? &output : const_cast<torch::Tensor*>(&bias);
  woq::woq_runtime_ctx ctx{
      const_cast<torch::Tensor*>(&activation),
      const_cast<torch::Tensor*>(&weight),
      rt_bias,
      &output,
  };
  ctx.lda = static_cast<int>(activation.sizes()[1]);
  ctx.ldo = static_cast<int>(output.sizes()[1]);
  ctx.m = static_cast<int>(activation.sizes()[0]);
  ctx.k = static_cast<int>(activation.sizes()[1]);
  ctx.n = ctx.ldo;
  ctx.alpha = 1.f;
  ctx.beta = bias.numel() != 0 ? 1.f : 0.f;
  init_woq_config_param<woq::WOQ_LINEAR>(&p, &ctx, compute_type, weight_type, scale_type, asym);
  woq::dispatch_woq_task(&p, &ctx, woq::WOQ_LINEAR);
}

static void set_woq_workspace(const torch::Tensor& workspace) {
  woq::set_woq_workspace(const_cast<torch::Tensor*>(&workspace));
}

static void bestlaop_gemm(const torch::Tensor& matA, const torch::Tensor& matB, const torch::Tensor& matC,
                          bool matB_trans) {
  TORCH_CHECK(matA.dim() == 2 && matB.dim() == 2 && matC.dim() == 2,
              "Qbits: only support 2-dim input-tensor in bestla gemm op.");
  bestla_gemm::bestla_gemm_runtime_ctx ctx;
  ctx.matA = const_cast<torch::Tensor*>(&matA);
  ctx.matB = const_cast<torch::Tensor*>(&matB);
  ctx.matC = const_cast<torch::Tensor*>(&matC);
  ctx.matB_trans = matB_trans;
  ctx.m = static_cast<int>(matA.sizes()[0]);
  ctx.n = static_cast<int>(matC.sizes()[1]);
  ctx.k = static_cast<int>(matA.sizes()[1]);
  TORCH_CHECK(matB_trans ? ctx.k == matB.sizes()[1] : ctx.k == matB.sizes()[0],
              "QBits: input shape mismatch in bestla gemm op.");
  return bestla_gemm::dispatch_bestla_gemm(&ctx);
}

static torch::Tensor acquire_packed_weight_info(torch::Tensor& packw, int64_t acquire_type) {
  return woq::get_packw_info(packw, static_cast<woq::PACKW_ACQUIRE_TYPE>(acquire_type));
}

static torch::Tensor qbits_dropout_fwd(torch::Tensor& output, double p) { return dropout_fwd(output, p); }

static void qbits_dropout_bwd(torch::Tensor& grad, torch::Tensor& scale) { dropout_bwd(grad, scale); }

static bool check_isa_supported(std::string isa) {
  if (isa == "AMX") return dispatcher_utils::check_amx();
  if (isa == "AVX512_VNNI") return dispatcher_utils::check_avx512_vnni();
  if (isa == "AVX_VNNI") return dispatcher_utils::check_avx_vnni();
  if (isa == "AVX512F") return dispatcher_utils::check_avx512f();
  if (isa == "AVX2") return dispatcher_utils::check_avx2();
  return false;
}

PYBIND11_MODULE(qbits, m) {
  m.def("quantize_to_packed_weight", &quantize_to_packed_weight);
  m.def("woq_linear", &woq_linear);
  m.def("dequantize_packed_weight", &dequantize_packed_weight);
  m.def("repack_quantized_weight", &repack_quantized_weight);
  m.def("get_packed_weight_size", &get_packed_weight_size);
  m.def("set_woq_workspace", &set_woq_workspace);
  m.def("matmul", &bestlaop_gemm);
  m.def("acquire_packed_weight_info", &acquire_packed_weight_info);
  m.def("dropout_fwd", &qbits_dropout_fwd);
  m.def("dropout_bwd", &qbits_dropout_bwd);
  m.def("check_isa_supported", &check_isa_supported);
}
