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
#pragma once

#include <torch/serialize/input-archive.h>
namespace bestla_gemm {
struct bestla_gemm_runtime_ctx {
  torch::Tensor *matA, *matB, *matC;
  bool matB_trans;
  int m, n, k;
};

void dispatch_bestla_gemm(bestla_gemm_runtime_ctx* ctx);
}  // namespace bestla_gemm
