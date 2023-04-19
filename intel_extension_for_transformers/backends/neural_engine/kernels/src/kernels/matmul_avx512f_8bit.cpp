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

#include "kernels/matmul_avx512f_8bit.hpp"
#include "fp8.hpp"
#include "kernels/matmul_types.hpp"
#include "singleton.hpp"
#include "cpu_parallel.hpp"
#include "utils.hpp"

namespace jd {

using io = ssd::matmul_io::io;
inline std::vector<std::vector<dim_t>> get_tensor_shapes(const std::vector<tensor_desc>& descs) {
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  return shapes;
}

bool matmul_avx512f_8bit_kd_t::init() {
  if (!isa_available(avx512_core_bf16)) return false;
  params_init();
  return true;
}

bool matmul_avx512f_8bit_kd_t::params_init() {
  auto& descs = op_desc_.tensor_descs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  auto attrs = op_desc_.attrs();

  dim_t M = shapes[io::SRC0][0];
  dim_t K = shapes[io::SRC0][1];
  dim_t N = shapes[io::SRC1][0];
  jit_param_.M = M;
  jit_param_.K = K;
  jit_param_.N = N;

  bool has_bias = !shapes[io::SRC2].empty();
  if (attrs["alpha"] != "") jit_param_.alpha = str_to_num<float>(attrs["alpha"]);
  SPARSE_LOG_IF(WARNING, jit_param_.alpha == 0.f)
      << "Alpha for matmul is set to 0 meaning that the base result will be discarded";

  if (has_bias) {
    if (attrs["beta"] != "") jit_param_.beta = str_to_num<float>(attrs["beta"]);
    SPARSE_LOG_IF(WARNING, has_bias && jit_param_.beta == 0.f)
        << "Beta for matmul is set to 0 meaning the binary-add does nothing";
  } else {
    jit_param_.beta = 0.f;  // set beta to 0 to avoid generate unnecessary asm ascode
  }

  auto iter = attrs.find("append_op");
  jit_param_.has_gelu = (iter != attrs.end() && iter->second == "gelu_tanh") ? true : false;

  if (descs[io::SRC1].dtype() == data_type::bf16) {
    jit_param_.weight_bf16 = reinterpret_cast<bfloat16_t*>(str_to_num<intptr_t>(attrs["weight_bf16"]));
    jit_param_.weight_type = data_type::f8_e4m3;
    for (auto& it : data_type_name) {
      if (it.second == attrs["weight_type"]) {
        jit_param_.weight_type = it.first;
        break;
      }
    }
  } else if (descs[io::SRC1].dtype() == data_type::s8 || descs[io::SRC1].dtype() == data_type::f8_e4m3 ||
             descs[io::SRC1].dtype() == data_type::f8_e5m2) {
    jit_param_.weight_type = descs[io::SRC1].dtype();
  }
  jit_param_.weight_fp8 = reinterpret_cast<float8_t*>(str_to_num<intptr_t>(attrs["weight_8bit"]));

  if (attrs["thread_nums"] != "") {
    jit_param_.thread_num = str_to_num<intptr_t>(attrs["thread_nums"]);
  }

  jit_param_.postop_attrs = op_desc_.apply_postops_list();
  packBF16();
  return true;
}

void matmul_avx512f_8bit_kd_t::reference(bfloat16_t* srcptr, float8_t* dstptr, int row, int col, int rowpad, int colpad,
                                         int srcstride, int dststride) {
  int srcld = srcstride / 2;
  auto sptr = reinterpret_cast<bfloat16_t*>(srcptr);
  auto dptr = reinterpret_cast<uint8_t*>(dstptr);
  int NTile = 16;
  for (int irow = 0; irow < rowpad; irow += NTile) {
    for (int icol = 0; icol < colpad; icol += 1) {
      for (int iin = 0; iin < NTile; iin++) {
        if (irow + iin < row) {
          if (icol < col) {
            *(dptr + irow * dststride + icol * NTile + iin) = float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(
                bf16_to_fp32(*(sptr + (irow + iin) * srcld + icol)));
          } else {
            *(dptr + irow * dststride + icol * NTile + iin) =
                float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(static_cast<float>(0));
          }
        } else {
          *(dptr + irow * dststride + icol * NTile + iin) =
              float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(static_cast<float>(0));
        }
      }
    }
  }
}
void matmul_avx512f_8bit_kd_t::reference(bfloat16_t* srcptr, int8_t* dstptr, int row, int col, int rowpad, int colpad,
                                         int srcstride, int dststride) {
  int srcld = srcstride / 2;
  auto sptr = reinterpret_cast<bfloat16_t*>(srcptr);
  auto dptr = (dstptr);
  int NTile = 16;
  for (int irow = 0; irow < rowpad; irow += NTile) {
    for (int icol = 0; icol < colpad; icol += 1) {
      for (int iin = 0; iin < NTile; iin++) {
        if (irow + iin < row) {
          if (icol < col) {
            *(dptr + irow * dststride + icol * NTile + iin) =
                fp32_to_int8(bf16_to_fp32(*(sptr + (irow + iin) * srcld + icol)));
          } else {
            *(dptr + irow * dststride + icol * NTile + iin) = fp32_to_int8(static_cast<float>(0));
          }
        } else {
          *(dptr + irow * dststride + icol * NTile + iin) = fp32_to_int8(static_cast<float>(0));
        }
      }
    }
  }
}

void matmul_avx512f_8bit_kd_t::packBF16() {
  CpuDevice* cpudevice = Singleton<CpuDevice>::GetInstance();
  bfloat16_t* matB = jit_param_.weight_bf16;
  int n = jit_param_.N;
  int k = jit_param_.K;
  int ldb = k;
  int npad = pad_to(n, 16);
  int kpad = pad_to(k, 1);
  auto ncores = cpudevice->getThreads();
  Parallel2DRowMajor _para;
  _para.update(npad, kpad, 16, 1, ncores);
#pragma omp parallel
  {
    int tidx = omp_get_thread_num();
    int colidx, rowidx, rowsize, colsize;
    _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      int rowremain = remainsize(rowidx, n, rowsize);
      int colremain = remainsize(colidx, k, colsize);
      if (op_desc_.tensor_descs()[io::SRC1].dtype() == data_type::bf16) {
        if (jit_param_.weight_type == data_type::s8) {
          reference(matB + rowidx * ldb + colidx, jit_param_.weight_int8 + rowidx * kpad + colidx * 16, rowremain,
                    colremain, rowsize, colsize, k * sizeof(bfloat16_t), kpad);
        } else if (jit_param_.weight_type == data_type::f8_e4m3 || jit_param_.weight_type == data_type::f8_e5m2) {
          reference(matB + rowidx * ldb + colidx, jit_param_.weight_fp8 + rowidx * kpad + colidx * 16, rowremain,
                    colremain, rowsize, colsize, k * sizeof(bfloat16_t), kpad);
        } else {
          SPARSE_LOG(ERROR) << "Not Support Weight type";
        }
      }
    }
  }
}

matmul_avx512f_8bit_k_t::matmul_avx512f_8bit_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      t_shapes_(get_tensor_shapes(kd->get_operator_desc().tensor_descs())),
      M_(t_shapes_[io::SRC0][0]),
      K_(t_shapes_[io::SRC0][1]),
      N_(t_shapes_[io::SRC1][0]) {}

bool matmul_avx512f_8bit_k_t::init() {
  CpuDevice* cpudevice = Singleton<CpuDevice>::GetInstance();
  auto& ker_param = derived_kd()->jit_param();
  if (ker_param.thread_num > 0) {
    cpudevice->setThreads(ker_param.thread_num);
  }
  auto ker = new jit_gemm_avx512f_8bit_t(ker_param);
  if (ker == nullptr) return false;
  if (!ker->create_kernel()) return false;
  jit_ker_ = ker;
  mCacheAdapter.update(M_, N_, K_, cpudevice->L2Cache, 16, 240);
  mParallel.update(M_, N_, jit_ker_->MTile, jit_ker_->NTile, cpudevice->getThreads(), mCacheAdapter);
  mCacheAdapter.set_N(mParallel.mNStep, mParallel.mThdRow <= jit_ker_->MTile);
  mCacheAdapter.mKBatch = K_;

  lda = K_;
  ldb = K_;
  ldc = N_;
  ldd = 0;

  return true;
}

bool matmul_avx512f_8bit_k_t::execute(const std::vector<const void*>& rt_data) const {
  bfloat16_t* matA = const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(rt_data[io::SRC0]));
  bfloat16_t* matC = const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(rt_data[io::DST0]));
  bfloat16_t* matD = const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(rt_data[io::SRC2]));
#pragma omp parallel
  {
    int tidx = omp_get_thread_num();
    int colidx, rowidx, rowsize, colsize;
    mParallel.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      int rowremain = remainsize(rowidx, M_, rowsize);
      int colremain = remainsize(colidx, N_, colsize);
      int kbatch = pad_to_le(mCacheAdapter.mKBatch, jit_ker_->KTile);
      auto cptr = matC + colidx + rowidx * ldc;
      auto dptr = matD + colidx + rowidx * ldd;
      for (int iterk = 0; iterk < K_; iterk += kbatch) {
        int kbatch_remain = iterk + kbatch <= K_ ? kbatch : K_ - iterk;
        auto aptr = matA + rowidx * lda + iterk;
        auto bptr = derived_kd()->jit_param().weight_fp8 + colidx * K_ + iterk * 16;
        for (int j = 0; j < colremain; j += mParallel.mNStep) {
          for (int i = 0; i < rowremain; i += jit_ker_->MTile) {
            for (int in = 0; in < mParallel.mNStep; in += jit_ker_->NTile) {
              int tmpcol = j + in;
              if (j + in < colremain) {
                int nsize = remainsize(j + in, colremain, jit_ker_->NTile);
                ssd::matmul_fp8_data_t parm = ssd::matmul_fp8_data_t{aptr + i * lda,
                                                                     bptr + tmpcol * K_,
                                                                     cptr + i * ldc + tmpcol,
                                                                     dptr + i * ldd + tmpcol,
                                                                     kbatch_remain,
                                                                     nsize,
                                                                     lda * 2,
                                                                     static_cast<int>(K_),
                                                                     ldc * 2,
                                                                     ldd,
                                                                     iterk,
                                                                     derived_kd()->jit_param().alpha,
                                                                     derived_kd()->jit_param().beta};
                (*jit_ker_)(&parm);
              }
            }
          }
        }
      }
    }
  }
  return true;
}
}  // namespace jd
