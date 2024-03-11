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
#include "bestla_weightonly_dispatcher.hpp"
namespace woq {

enum PACKW_ACQUIRE_TYPE {
  SIZE = 0,
  BLOCKSIZE,
  K,
  N,
  ACT_SHUFFLE,
  G_IDX,
  WEI_TYPE,
  CMPT_TYPE,
  SCALE_TYPE,
};

void bestla_packq(woq_packq_param* p, woq_packq_ctx* ctx);
torch::Tensor get_packw_info(torch::Tensor& packw, PACKW_ACQUIRE_TYPE ACQ_T);
}  // namespace woq
