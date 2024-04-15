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
#include "../include/dispatcher_utils.hpp"
#include <c10/core/ScalarType.h>
#include <torch/types.h>

namespace dispatcher_utils {
string get_torch_dt_name(torch::Tensor* tensor) {
  string ret = "unrecognized";
  if (tensor->scalar_type() == torch::kFloat32) ret = "fp32";
  if (tensor->scalar_type() == torch::kBFloat16) ret = "bf16";
  return ret;
}
}  // namespace dispatcher_utils
