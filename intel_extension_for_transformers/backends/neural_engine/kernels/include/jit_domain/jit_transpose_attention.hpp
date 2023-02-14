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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_TRANSPOSE_ATTENTION_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_TRANSPOSE_ATTENTION_HPP_

#include "jit_generator.hpp"
#include "amx_utils.hpp"
#include "kernels/transpose_attention_types.hpp"
#include "cpu_isa.hpp"

namespace jd {
typedef Xbyak::util::StackFrame StackFrame;

void configure_tiles(tileconfig_t* tc, int TILE_K);
#define TILE_M 16  // Number of rows in an A or C tile
#define TILE_N 16  // Number of columns in a B or C tile
#define KPACK 4    // Vertical K packing into dword

class MHA_kernel : public jit_generator {
 public:
  int MTile, NTile, KTile;  // only for gemm related kernel
  int NPacked;              // only for norm-reorder kernel
  tileconfig_t tc;
  MHA_kernel(size_t size, int m, int n, int k, int npacked)
      : jit_generator(size, 0), MTile(m), NTile(n), KTile(k), NPacked(npacked) {}
  virtual ~MHA_kernel() {}
  virtual void generate() {}
};

class MHA_stage1_kernel : public MHA_kernel {
 public:
  MHA_stage1_kernel(size_t size, int m, int n, int k, int npacked) : MHA_kernel(size, m, n, k, npacked) {}
  virtual ~MHA_stage1_kernel() {}
  virtual void generate() {}

 protected:
  void packedf32_bf16(int idx0, int idx1);
  // x and y can be same register
  // len(tmp)==2
  void exp_f32_lowprecision(Zmm log2e, Zmm ln2, Zmm c[3], Zmm x, Zmm y, Zmm tmp[]);
};

class MHA_stage2_kernel : public MHA_kernel {
 public:
  MHA_stage2_kernel(size_t size, int m, int n, int k, int npacked) : MHA_kernel(size, m, n, k, npacked) {}
  virtual ~MHA_stage2_kernel() {}
  virtual void generate() {}

 protected:
  void loadbf16_norm_rows(int idx, RegExp load_addr, Zmm scale);
  // n=exp/max(sumexp,epsilon)  0~1
  // o=n*255  0~255
  void normalize(int idx, Zmm scale);
};

class TransposeCopy8x8_1B_kernel : public MHA_kernel {
 public:
  TransposeCopy8x8_1B_kernel(size_t size = 16 * 1024) : MHA_kernel(size, 0, 0, 0, 0) {}

 protected:
  void generate();
  std::vector<Ymm> transpose8x4B(Ymm* rows, Ymm* tmp);
};

class MHA_s8s8s8_row_amx_32x32_batchk_binary_exp : public MHA_stage1_kernel {
 public:
  int const BatchK = 1;

 public:
  MHA_s8s8s8_row_amx_32x32_batchk_binary_exp(int k, size_t size = 32 * 1024) : MHA_stage1_kernel(size, 32, 32, k, 0) {}

 protected:
  void generate();
};

class MHA_s8s8s8_row_vnni_8x32_batchk_binary_exp : public MHA_stage1_kernel {
 public:
  MHA_s8s8s8_row_vnni_8x32_batchk_binary_exp(size_t size = 32 * 1024) : MHA_stage1_kernel(size, 8, 32, 8, 0) {}

 protected:
  void generate();
};

class MHA_norm_quantize_reorder_prescale_packed : public MHA_stage2_kernel {
 public:
  MHA_norm_quantize_reorder_prescale_packed(int npacked, int tile, size_t size = 16 * 1024)
      : MHA_stage2_kernel(size, 0, tile, 0, npacked) {}

 protected:
  void generate();
  void vnni_interleave_load_6regs(int startIdx);
};

class MHA_norm_quantize_reorder_vnniw_prescale_packed : public MHA_stage2_kernel {
 public:
  MHA_norm_quantize_reorder_vnniw_prescale_packed(int npacked, int tile, size_t size = 16 * 1024)
      : MHA_stage2_kernel(size, 0, tile, 0, npacked) {}

 protected:
  void generate();
  void vnni_word_interleave_load_3regs(int startIdx);
};

class MHA_Matmul_s8u8u8_amx_32x32 : public MHA_kernel {
 public:
  MHA_Matmul_s8u8u8_amx_32x32(int k, size_t size = 32 * 1024) : MHA_kernel(size, 32, 32, k, 0) {}

 protected:
  void generate();
};

class MHA_Matmul_s8u8u8_vnni_word_8x32 : public MHA_kernel {
 public:
  MHA_Matmul_s8u8u8_vnni_word_8x32(size_t size = 32 * 1024) : MHA_kernel(size, 8, 32, 8, 0) {}

 protected:
  void generate();
};

class SeqCopy_1B_avx512_Nx4_Temp : public MHA_kernel {
 public:
  SeqCopy_1B_avx512_Nx4_Temp(int N = 32, size_t size = 16 * 1024) : MHA_kernel(size, 0, N, 0, 0) {}

 protected:
  void generate();
  void vnni_interleave_load_6regs(int startIdx);
};

class SeqCopy_1B_avx512_Nx2_Temp : public MHA_kernel {
 public:
  SeqCopy_1B_avx512_Nx2_Temp(int N = 32, size_t size = 16 * 1024) : MHA_kernel(size, 0, N, 0, 0) {}

 protected:
  void generate();
  void vnni_word_interleave_load_3regs(int startIdx);
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_DOMAIN_JIT_TRANSPOSE_ATTENTION_HPP_
