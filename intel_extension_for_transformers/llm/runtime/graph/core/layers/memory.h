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

#include "ele_wise.h"
#include "core/ne.h"

#ifdef __cplusplus
extern "C" {
#endif

void ne_attention_padding_mask_f32_forward(const int bs, const int nr_qk, const int qlen, const int ith, const int nth,
                                           const void* padding, const float p_value, struct ne_tensor* dst);

#ifdef __cplusplus
}
#endif
