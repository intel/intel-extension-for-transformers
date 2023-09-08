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

#include "kernels/amx_utils.hpp"
#include "src/cpu/jit_domain/jit_amx_configure.hpp"
#include "src/utils.hpp"
namespace jd {

static const jit_amx_config_t tilecfg;
static const jit_amx_release_t tilerls;

void amx_tile_config_t::amx_tile_configure(int thread_x, tile_param_t param) {
  if (param != param_[thread_x]) {
    param_[thread_x] = param;
    configure_tiles(param, config_);
    tilecfg.tile_configure(reinterpret_cast<void*>(config_));
  }
}

void amx_tile_config_t::amx_tile_release(int thread_x) {
  tilerls.tile_release();
  param_[thread_x] = tile_param_t();
}

#ifdef WITH_GCC_FLAGS
#pragma GCC push_options
#pragma GCC optimize("O0")
#endif
void configure_tiles(tile_param_t param, tileconfig_t* sparselib_tc) {
  // Filling tile configure structure. Could be done offline.
  sparselib_tc->palette_id = 1;
  int sizeof_src_dtype = 1;
  if (param.is_bf16) {
    sizeof_src_dtype = 2;
  }
  int sizeof_dst_dtype = 4;
  // zeros reserved
  for (int t = 0; t < 15; ++t) {
    sparselib_tc->reserved[t] = 0;
  }
  int inc = 0;
  // Configure C tiles
  for (; inc < param.C_tile_num; ++inc) {
    sparselib_tc->rows[inc] = param.M_tile;
    sparselib_tc->colb[inc] = param.N_tile * sizeof_dst_dtype;
  }
  // Configure A tiles
  for (; inc < param.A_tile_num + param.C_tile_num; ++inc) {
    sparselib_tc->rows[inc] = param.M_tile;
    sparselib_tc->colb[inc] = param.K_tile * sizeof_src_dtype;
  }
  // Configure B tile. B effectively has 64 rows and 16 columns.
  for (; inc < 8; ++inc) {
    sparselib_tc->rows[inc] = param.K_tile / param.K_pack;
    sparselib_tc->colb[inc] = param.N_tile * param.K_pack * sizeof_src_dtype;
  }
  // Zeroing other cols & rows
  for (int t = 8; t < 16; ++t) {
    sparselib_tc->rows[t] = 0;
    sparselib_tc->colb[t] = 0;
  }
}
#ifdef WITH_GCC_FLAGS
#pragma GCC pop_options
#endif
}  // namespace jd
