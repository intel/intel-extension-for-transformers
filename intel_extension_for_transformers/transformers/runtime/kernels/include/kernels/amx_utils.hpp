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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_AMX_UTILS_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_AMX_UTILS_HPP_
#include <omp.h>
#include <cstdint>
#include <vector>

#include "src/cpu/jit_domain/jit_amx_configure.hpp"
namespace jd {

class tile_param_t {
 public:
  int M_tile;
  int N_tile;
  int K_tile;
  bool is_bf16;
  int K_pack;
  int C_tile_num = 4;
  int A_tile_num = 2;
  int B_tile_num = 2;

  tile_param_t()
      : M_tile(0), N_tile(0), K_tile(0), is_bf16(false), K_pack(0), C_tile_num(4), A_tile_num(2), B_tile_num(2) {}
  tile_param_t(int m_tile, int n_tile, int k_tile, bool bf16, int k_pack, int c_tile_num = 4, int a_tile_num = 2,
               int b_tile_num = 2)
      : M_tile(m_tile),
        N_tile(n_tile),
        K_tile(k_tile),
        is_bf16(bf16),
        K_pack(k_pack),
        C_tile_num(c_tile_num),
        A_tile_num(a_tile_num),
        B_tile_num(b_tile_num) {
    SPARSE_LOG_IF(FATAL, (c_tile_num + a_tile_num + b_tile_num) != 8) << "sum of a,b,c tile must be 8.";
  }

 public:
  bool operator!=(const tile_param_t& rhs) const {
    return (M_tile != rhs.M_tile) || (K_tile != rhs.K_tile) || (N_tile != rhs.N_tile) || (is_bf16 != rhs.is_bf16) ||
           (K_pack != rhs.K_pack) || (C_tile_num != rhs.C_tile_num) || (A_tile_num != rhs.A_tile_num) ||
           (B_tile_num != rhs.B_tile_num);
  }
};

struct tileconfig_t;

void configure_tiles(tile_param_t param, tileconfig_t* sparselib_tc);

// Tile configure structure
struct tileconfig_t {
  uint8_t palette_id = 0;
  uint8_t reserved[15] = {0};
  uint16_t colb[16] = {64};
  uint8_t rows[16] = {16};
  tileconfig_t() = default;
  explicit tileconfig_t(const tile_param_t& param) : tileconfig_t() { configure_tiles(param, this); }
};

/**
 * The amx_tile_config_t is in amx_tile_config_t mode to ensure all primitive share the
 * same configure. defines the `GetInstance` method that serves as an
 * alternative to constructor and lets clients access the same instance of this
 * class over and over.
 */
class amx_tile_config_t {
 public:
  amx_tile_config_t() {
    tilecfg.create_kernel();
    tilerls.create_kernel();
    int nthr = omp_get_max_threads();
    param_.resize(nthr);
  }
  ~amx_tile_config_t() {
    if (config_ != nullptr) {
      delete config_;
      config_ = nullptr;
    }
  }

 protected:
  std::vector<tile_param_t> param_;
  tileconfig_t* config_ = new tileconfig_t();

 public:
  /**
   * amx_tile_config_ts should not be cloneable.
   */
  amx_tile_config_t(amx_tile_config_t& other) = delete;  // NOLINT
  /**
   * amx_tile_config_ts should not be assignable.
   */
  void operator=(const amx_tile_config_t&) = delete;
  /**
   * Finally, any singleton should define some business logic, which can be
   * executed on its instance.
   */
  void amx_tile_configure(int thread_x, tile_param_t param);
  void amx_tile_release(int thread_x);
  jit_amx_config_t tilecfg;
  jit_amx_release_t tilerls;
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_AMX_UTILS_HPP_
