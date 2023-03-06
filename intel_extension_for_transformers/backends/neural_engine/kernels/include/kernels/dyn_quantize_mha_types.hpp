//  Copyright (c) 2022 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_DYN_QUANTIZE_MHA_TYPES_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_DYN_QUANTIZE_MHA_TYPES_HPP_

#include <vector>

#include "amx_utils.hpp"
#include "param_types.hpp"

namespace jd {
namespace ssd {
namespace dyn_quantize_mha_io {
enum io {
  Q,
  K,
  MASK,
  V,
  DST,
  TMP,  // size of K + size of V + ~1M

  Q_SCALE,
  Q_ZP,
  K_SCALE,
  K_ZP,
  V_SCALE,
  V_ZP,
  DST_SCALE,
  DST_ZP,

  BATCH_SIZE,
  HEAD_NUM,
  HEAD_SIZE,
  M,  // "seq_len" for Q & DST
  N,  // "seq_len" for K & V
  dyn_quantize_mha_io_MAX = N,
};
}  // namespace dyn_quantize_mha_io

}  // namespace ssd
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_DYN_QUANTIZE_MHA_TYPES_HPP_
