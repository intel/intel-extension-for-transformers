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

#include "matmul_avx512f_8bit.hpp"

#include "kernels/data_pack.hpp"
#include "kernels/matmul_types.hpp"
#include "src/singleton.hpp"
#include "src/utils.hpp"

namespace jd {

using io = ssd::matmul_io::io;
using input = ssd::matmul_input::input;
using output = ssd::matmul_output::output;

bool matmul_avx512f_8bit_kd_t::init() {
  if (!isa_available(avx512_core_bf16)) return false;
  params_init();
  return true;
}

bool matmul_avx512f_8bit_kd_t::params_init() {
  const auto shapes = op_desc_.tensor_shapes();
  const auto dtypes = op_desc_.tensor_dtypes();
  auto attrs = op_desc_.attrs();

  dim_t M = shapes[io::SRC0][0];
  dim_t K = shapes[io::SRC0][1];
  dim_t N = shapes[io::SRC1][0];
  jit_param_.M = M;
  jit_param_.K = K;
  jit_param_.N = N;

  jit_param_.has_scale0 = shapes.size() > io::SCALE0 && !shapes[io::SCALE0].empty();
  bool has_bias = shapes.size() > io::SRC2 && !shapes[io::SRC2].empty();
  if (attrs["alpha"] != "") jit_param_.alpha = str_to_num<float>(attrs["alpha"]);
  SPARSE_LOG_IF(WARNING, jit_param_.alpha == 0.f)
      << "Alpha for matmul is set to 0 meaning that the base result will be discarded";

  bool has_append_sum = shapes.size() > io::APPEND_SUM && !shapes[io::APPEND_SUM].empty();
  jit_param_.has_append_sum = has_append_sum;
  if (has_bias) {
    if (attrs["beta"] != "") jit_param_.beta = str_to_num<float>(attrs["beta"]);
    SPARSE_LOG_IF(WARNING, has_bias && jit_param_.beta == 0.f)
        << "Beta for matmul is set to 0 meaning the binary-add does nothing";
  } else {
    jit_param_.beta = 0.f;  // set beta to 0 to avoid generate unnecessary asm ascode
  }

  if (attrs["thread_nums"] != "") {
    jit_param_.thread_num = str_to_num<intptr_t>(attrs["thread_nums"]);
  }

  jit_param_.postop_attrs = op_desc_.apply_postops_list();
  jit_param_.weight_8bit = reinterpret_cast<uint8_t*>(str_to_num<intptr_t>(attrs["weight_8bit"]));

  if (dtypes[io::SRC1] == data_type::bf16) {
    jit_param_.weight_bf16 = reinterpret_cast<bfloat16_t*>(str_to_num<intptr_t>(attrs["weight_bf16"]));
    jit_param_.weight_type = data_type::f8_e5m2;
    for (auto& it : data_type_name) {
      if (it.second == attrs["weight_type"]) {
        jit_param_.weight_type = it.first;
        break;
      }
    }
    if (jit_param_.weight_type == data_type::f8_e4m3) {
      std::function<float8_e4m3_t(bfloat16_t)> cast_func = [&](bfloat16_t bf16) -> float8_e4m3_t {
        float fp32 = static_cast<float>(bfloat16_t(bf16));
        return float8_e4m3_t(fp32);
      };
      pack<float8_e4m3_t, bfloat16_t>(jit_param_.weight_f8_e4m3, jit_param_.weight_bf16, N, K, cast_func);
    } else if (jit_param_.weight_type == data_type::f8_e5m2) {
      std::function<float8_e5m2_t(bfloat16_t)> cast_func = [&](bfloat16_t bf16) -> float8_e5m2_t {
        float fp32 = static_cast<float>(bfloat16_t(bf16));
        return float8_e5m2_t(fp32);
      };
      pack<float8_e5m2_t, bfloat16_t>(jit_param_.weight_f8_e5m2, jit_param_.weight_bf16, N, K, cast_func);
    }
  } else if (dtypes[io::SRC1] == data_type::s8 || dtypes[io::SRC1] == data_type::f8_e4m3 ||
             dtypes[io::SRC1] == data_type::f8_e5m2) {
    jit_param_.weight_type = dtypes[io::SRC1];
  }
  return true;
}

matmul_avx512f_8bit_k_t::matmul_avx512f_8bit_k_t(const std::shared_ptr<const kd_t>& kd)
    : kernel_t(kd),
      t_shapes_(kd->get_operator_desc().tensor_shapes()),
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
  float* scale = const_cast<float*>(reinterpret_cast<const float*>(rt_data[io::SCALE0]));
  bfloat16_t* matE = derived_kd()->jit_param().has_append_sum
                         ? const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(rt_data[io::APPEND_SUM]))
                         : nullptr;
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
      auto eptr = derived_kd()->jit_param().has_append_sum ? matE + colidx + rowidx * ldc : nullptr;
      auto scaleptr = scale + colidx;
      for (int iterk = 0; iterk < K_; iterk += kbatch) {
        int kbatch_remain = iterk + kbatch <= K_ ? kbatch : K_ - iterk;
        auto aptr = matA + rowidx * lda + iterk;
        auto bptr = derived_kd()->jit_param().weight_8bit + colidx * K_ + iterk * 16;
        for (int j = 0; j < colremain; j += mParallel.mNStep) {
          for (int i = 0; i < rowremain; i += jit_ker_->MTile) {
            for (int in = 0; in < mParallel.mNStep; in += jit_ker_->NTile) {
              int tmpcol = j + in;
              if (j + in < colremain) {
                int nsize = remainsize(j + in, colremain, jit_ker_->NTile);
                ssd::matmul_fp8_data_t parm =
                    ssd::matmul_fp8_data_t{aptr + i * lda,
                                           bptr + tmpcol * K_,
                                           cptr + i * ldc + tmpcol,
                                           dptr + i * ldd + tmpcol,
                                           derived_kd()->jit_param().has_append_sum ? eptr + i * ldc + tmpcol : nullptr,
                                           scaleptr + tmpcol,
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

bool matmul_avx512f_8bit_k_t::execute(const exec_context_t& context) const {
  bfloat16_t* matA = nullptr;
  bfloat16_t* matC = nullptr;
  bfloat16_t* matD = nullptr;
  bfloat16_t* matE = nullptr;
  float* scale = nullptr;

  context.input(input::SRC0)->get_handle(reinterpret_cast<void**>(&matA));
  context.input(input::SRC2)->get_handle(reinterpret_cast<void**>(&matD));
  context.input(input::SCALE0)->get_handle(reinterpret_cast<void**>(&scale));

  context.output(output::DST0)->get_handle(reinterpret_cast<void**>(&matC));
  dim_t M = context.get_dynamic_shape().empty() ? M_ : context.get_dynamic_shape().front();
  if (M_ != M) {
    M_ = M;
    CpuDevice* cpudevice = Singleton<CpuDevice>::GetInstance();
    mCacheAdapter.update(M_, N_, K_, cpudevice->L2Cache, 16, 240);
    mParallel.update(M_, N_, jit_ker_->MTile, jit_ker_->NTile, cpudevice->getThreads(), mCacheAdapter);
  }

  if (derived_kd()->jit_param().has_append_sum) {
    context.input(input::APPEND_SUM)->get_handle(reinterpret_cast<void**>(&matE));
  }
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
      auto eptr = derived_kd()->jit_param().has_append_sum ? matE + colidx + rowidx * ldc : nullptr;
      auto scaleptr = scale + colidx;
      for (int iterk = 0; iterk < K_; iterk += kbatch) {
        int kbatch_remain = iterk + kbatch <= K_ ? kbatch : K_ - iterk;
        auto aptr = matA + rowidx * lda + iterk;
        auto bptr = derived_kd()->jit_param().weight_8bit + colidx * K_ + iterk * 16;
        for (int j = 0; j < colremain; j += mParallel.mNStep) {
          for (int i = 0; i < rowremain; i += jit_ker_->MTile) {
            for (int in = 0; in < mParallel.mNStep; in += jit_ker_->NTile) {
              int tmpcol = j + in;
              if (j + in < colremain) {
                int nsize = remainsize(j + in, colremain, jit_ker_->NTile);
                ssd::matmul_fp8_data_t parm =
                    ssd::matmul_fp8_data_t{aptr + i * lda,
                                           bptr + tmpcol * K_,
                                           cptr + i * ldc + tmpcol,
                                           dptr + i * ldd + tmpcol,
                                           derived_kd()->jit_param().has_append_sum ? eptr + i * ldc + tmpcol : nullptr,
                                           scaleptr + tmpcol,
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
