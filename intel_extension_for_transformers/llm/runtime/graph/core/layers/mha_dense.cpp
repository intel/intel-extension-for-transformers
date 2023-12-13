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
#include "layers/mha_dense.h"

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#ifdef NE_TESTS
#include <memory>
#include <tuple>

#include "layers/ne_test_layers_utils.hpp"
#endif

#include "core/data_types.h"
#include "layers/jblas_common.hpp"
#include "jblas/jit_blas.h"
#include "jblas/jit_blas_epilogue.h"
#include "jblas/jit_blas_gemm.h"
#include "jblas/jit_blas_parallel.h"
#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_prologue_a.h"
#include "jblas/jit_blas_storage.h"
#include "jblas/jit_blas_wrapper.h"

#define MHA_2ND_EXP 1
constexpr bool MHA_PREFER_AVX512FP16 = true;

#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512vl")
#if CompileBF16()
#pragma GCC target("avx512bf16")
#endif
#if CompileFP16()
#pragma GCC target("avx512fp16")
#endif
#endif

namespace {
using jblas::utils::bf16;
using jblas::utils::fp16;
using jblas::utils::padto;
using jblas::utils::padto_le;
using jblas::utils::remainsize;
using jblas::utils::updiv;
namespace utils = jblas::utils;

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
struct attn_fwd_args_t {
  Q_T* Q;
  K_T* K;
  V_T* V;
  DST_T* dst;
  float Q_sc, K_sc, V_sc, dst_sc;
  char* tmp;
  float QK_scale;
  ne_attn_flags_t attn_flags;
  int batch_size, head_num, heads_kv, head_size, sl_q, sl_kv;
  ATTN_FWD_LAYOUT Q_layout, K_layout, V_layout, dst_layout;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl, step_k_head_size;
  int step_v_bs, step_v_head_num, step_v_sl, step_v_head_size;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
};

struct MHAProblem {
  int batch_size, head_num, heads_kv, head_size, sl_q, sl_kv;
};

inline __m512 poly_scale_2nd_ps(const __m512 z, const __m512 f, const __m512 c0, const __m512 c1, const __m512 c2) {
  const auto y = _mm512_fmadd_ps(_mm512_fmadd_ps(f, c0, c1), f, c2);  // auto y = (f * c0 + c1) * f + c2;
  const auto exp = _mm512_scalef_ps(y, z);
  return exp;
}

inline __m512 exp_ps_0_1(const __m512 x) {
  static const auto c0 = _mm512_set1_ps(0.240226507f);
  static const auto c1 = _mm512_set1_ps(0.452920674f);
  static const auto c2 = _mm512_set1_ps(0.713483036f);
  static const float v_log2e = std::log2(std::exp(1.f));
  static const auto log2e = _mm512_set1_ps(v_log2e);
  static const auto half = _mm512_set1_ps(.5f);

  const auto x1 = _mm512_fmadd_ps(x, log2e, half);  // auto x1 = x * log2e + _mm512_set1_ps(.5f);
  const auto z = _mm512_floor_ps(x1);
  const auto f = _mm512_sub_ps(x1, z);  // auto f = x1 - z;

  return poly_scale_2nd_ps(z, f, c0, c1, c2);
}

inline float mha_exp_ref(float x) {
#if MHA_2ND_EXP
  static const float log2e = std::log2(std::exp(1.f));
  static const float ln2 = std::log(2.f);
  const float x1 = x * log2e + .5f;
  const float z = std::floor(x1);
  const float f = x1 - z;
  constexpr std::array<float, 3> coeff{0.240226507f, 0.452920674f, 0.713483036f};
  return ldexpf(coeff[0] * f * f + coeff[1] * f + coeff[2], z);  // same as a * std::pow(2, z) but more precise
#else
  return expf(x)
#endif
}

#ifdef NOT_CURRENTLY_USED
inline __m512 exp_2nd_ph(const __m512 z, const __m512 f, const __m512 c0, const __m512 c1, const __m512 c2) {
  const auto y = _mm512_fmadd_ph(_mm512_fmadd_ph(f, c0, c1), f, c2);  // auto y = (f * c0 + c1) * f + c2;
  const auto exp = _mm512_scalef_ph(y, z);
  return exp;
}

inline __m512 exp_ph_0_1(const __m512 x) {
  static const auto c0 = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(0.240226507f).x));
  static const auto c1 = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(0.452920674f).x));
  static const auto c2 = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(0.713483036f).x));
  static const float v_log2e = std::log2(std::exp(1.f));
  static const auto log2e = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(v_log2e).x));
  static const auto half = _mm512_castsi512_ph(_mm512_set1_epi16(fp16(.5f).x));

  const auto x1 = _mm512_fmadd_ph(x, log2e, half);  // auto x1 = x * log2e + _mm512_set1_ph(.5f);
  const auto z = _mm512_floor_ph(x1);
  const auto f = _mm512_sub_ph(x1, z);  // auto f = x1 - z;

  return exp_2nd_ph(z, f, c0, c1, c2);
}
#endif

/**
 * @brief An Epilogue that optionally apply a casual mask and scale the fp32 result, performing exp, accumulating sum of
 * each line of exp, and storing exp as bf16 results
 */
template <JBLAS_ISA ISA_T, typename T_DST>
class ScaleExpAccSumFp32 {
 public:
  struct Param {
    T_DST* dst;
    float* dst_sum;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };

  JBLAS_CODE forward(const float* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p, void* /* tmpcache */, size_t /* cachesize */) const {
    assert(("alibi not supported!", p.alibi_slope == 0.f));
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_sum = p.dst_sum + M_offset;
#if MHA_2ND_EXP && CompileBF16()
    static_assert(std::is_same<T_DST, bf16>::value, "bf16 support only");
    const auto v_scale = _mm512_set1_ps(p.scale);
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_sum = _mm512_setzero_ps();
      for (; j < N_unmasked - 15; j += 16) {
        const auto v_exp = exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_loadu_ps(src + i * src_step + j)));
        v_sum = _mm512_add_ps(v_sum, v_exp);
        _mm256_storeu_epi16(dst + i * p.ld_dst + j, (__m256i)_mm512_cvtneps_pbh(v_exp));
      }
      if (j < N_unmasked) {
        const auto v_exp = exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_maskz_loadu_ps(v_mask, src + i * src_step + j)));
        v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
        _mm256_storeu_epi16(dst + i * p.ld_dst + j, (__m256i)_mm512_maskz_cvtneps_pbh(v_mask, v_exp));
        j += 16;
      }
      dst_sum[i] += _mm512_reduce_add_ps(v_sum);

      if (j < jblas::utils::padto(N, 64))
        memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (jblas::utils::padto(N, 64) - j));
    }
#else
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);
      for (int j = 0; j < N_unmasked; ++j) {
        const auto exp_ = expf(src[i * src_step + j] * p.scale);
        dst[i * p.ld_dst + j] = static_cast<T_DST>(exp_);
        dst_sum[i] += exp_;
      }
      if (N_unmasked < jblas::utils::padto(N, 64))
        memset(dst + i * p.ld_dst + N_unmasked, 0, sizeof(*dst) * (jblas::utils::padto(N, 64) - N_unmasked));
    }
#endif

    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using ScaleExpAccSumFp32Bf16 = ScaleExpAccSumFp32<ISA_T, bf16>;

/**
 * @brief An Epilogue that scale the fp32 result, convert to bf16 and write back to memory
 */
template <JBLAS_ISA ISA_T, typename T_SRC, typename T_DST>
class ScaleWriteBack {
 public:
  using SType = T_SRC;
  using DType = T_DST;
  struct Param {
    const float* scale;
    DType* dst;
    int ld_dst;
  };

  JBLAS_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p, void* /* tmpcache */, size_t /* cachesize */) {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto scale = p.scale + M_offset;
    // TODO(Yi): high performance implementation
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)  //
        dst[i * p.ld_dst + j] = static_cast<DType>(scale[i] * src[i * src_step + j]);

    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using ScaleWriteBackFp32Bf16 = ScaleWriteBack<ISA_T, float, bf16>;
template <JBLAS_ISA ISA_T>
using ScaleWriteBackFp32Fp32 = ScaleWriteBack<ISA_T, float, float>;
template <JBLAS_ISA ISA_T>
using ScaleWriteBackS32S8 = ScaleWriteBack<ISA_T, int32_t, int8_t>;

/**
 * @brief PackedWeight(Default) with batch
 */
class StoragePackedWeightBatch : public jblas::storage::gemm::IWeightBase {
  using Base = jblas::storage::gemm::IWeightBase;

 public:
  int mBatch;
  jblas::storage::ObjectAlignedBuffer<NE_ALIGNMENT> mWBuf;
  // size_t mWSize;

  explicit StoragePackedWeightBatch(uint64_t _core_id) : Base(_core_id), mBatch(0) {}
  size_t resize(int NPad, int KPad, int N, int K, int num_batch, JBLAS_DTYPE dtype) {
    IWeightBase::resize(NPad, KPad, N, K, dtype);
    mBatch = num_batch;
    auto bsize = static_cast<size_t>(mBatch) * NPad * KPad * jblas::utils::jblas_dtype_size(dtype);
    mWBuf.resize(bsize);
    mSize = utils::padto(IWeightBase::getSerializedSize() + mWBuf.getSerializedSize(), NE_ALIGNMENT);
    return mSize;
  }

  template <typename T>
  inline constexpr T* WPtr() const {
    return mWBuf.get<T>();
  }

  void assign(int8_t* buf) override {
    deserializeBuffer(buf, true);
    mWBuf.deserializeBuffer(buf, true);
  }

  void serialize(int8_t* wptr) override {
    serializeToBuffer(wptr);
    mWBuf.serializeToBuffer(wptr);
  }

  void deserialize(int8_t* rptr) override {
    deserializeBuffer(rptr, false);
    mWBuf.deserializeBuffer(rptr, false);
  }

 protected:
  size_t getSerializedSize() override { return Base::getSerializedSize() + sizeof(mBatch); }

  void serializeToBuffer(int8_t*& wptr) override {
    Base::serializeToBuffer(wptr);
    utils::serialize(wptr, mBatch);
  }
  void deserializeBuffer(int8_t*& rptr, bool map_buf) override {
    Base::deserializeBuffer(rptr, map_buf);
    if (!map_buf) {
      mBatch = utils::deserialize<int>(rptr);
    } else {
      utils::serialize<int>(rptr, mBatch);
    }
  }
};

/**
 * @brief An weight Prologue that Packs transposed Bf16 weight; optimized for runtime packing. It is the base type of
 * that for transposed / non-transposed source
 */
template <class GemmCore_T, JBLAS_ISA ISA_T, bool IsTrans, typename T_SRC = typename GemmCore_T::BType>
class WeightPackBatchBf16Base {
 public:
  using WType = typename GemmCore_T::BType;      // weight type
  using SType = T_SRC;                           // source type (before packed)
  using StorageType = StoragePackedWeightBatch;  // packed weight type

  struct Param {
    const SType* B;
    const int ldb;
    const StorageType* packedW;
  };

  JBLAS_CODE getWeight(...) = delete;

  JBLAS_CODE getWeight(WType** dstptr, int* dststep, int /* b_size */, int /* k_size */, int /* n_size */, int b_offset,
                       int k_offset, int n_offset, const Param& param, void* /* tmpcache */, size_t /* cachesize */) {
    const auto wptr = param.packedW;
    if (!wptr) return JblasInvalidParam;
    assert(k_offset % GemmCore_T::KTILE == 0);
    assert(n_offset % GemmCore_T::NTILE == 0);
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    *dstptr = wptr->template WPtr<WType>() + n_offset * KPad + k_offset * GemmCore_T::NTILE;
    *dststep = KPad;
    return JblasSuccess;
  }

  JBLAS_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                       const Param& param, void* tmpcache, size_t cachesize) {
    return getWeight(dstptr, dststep, 1, k_size, n_size, 0, k_offset, n_offset, param, tmpcache, cachesize);
  }

  JBLAS_CODE packWeight(...) = delete;
};

template <class GemmCore_T, JBLAS_ISA ISA_T, typename T_SRC = typename GemmCore_T::BType>
class WeightPackBatchBf16Trans : public WeightPackBatchBf16Base<GemmCore_T, ISA_T, true, T_SRC> {
  using Base = WeightPackBatchBf16Base<GemmCore_T, ISA_T, true, T_SRC>;

 public:
  using typename Base::Param;
  using typename Base::StorageType;
  using typename Base::SType;
  using typename Base::WType;

  /// Reorder job of a thread
  void run(const Param& p, const jblas::parallel::ThreadProblem2D& thdp, std::function<int(int)> step_batch) {
    if (!thdp.valid) return;
    const auto pw = dynamic_cast<const StorageType*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after transpose & padding
    const int NPad = pw->mNPad;  // N size after transpose & padding
    assert(pw->mK <= KPad);
    assert(pw->mN <= NPad);

    // y for batch; x for major-dim of the source data (N-dim of the packed weight)
    const auto [y, x] = thdp.loc;
    const auto [ny, nx] = thdp.size;
    const auto nx_pad = jblas::utils::padto(nx, GemmCore_T::NTILE);

    assert(padto(pw->mK, GemmCore_T::KTILE) == KPad);

    using KernInterleave = typename jblas::kernel::wrapper::PaddingTransInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::template forward<ISA_T, T_SRC, WType>(  //
          p.B + step_batch(ibat) + x * p.ldb,                                           //
          pw->template WPtr<WType>() + ibat * KPad * NPad + x * KPad,                   //
          nx, pw->mK,                                                                   // size
          nx_pad, KPad,                                                                 // padded size
          p.ldb, KPad);                                                                 // step
      assert(forward_stat == JblasSuccess);
    }
  }
};

template <class GemmCore_T, JBLAS_ISA ISA_T, typename T_SRC = typename GemmCore_T::BType>
class WeightPackBatchBf16NonTr : public WeightPackBatchBf16Base<GemmCore_T, ISA_T, false, T_SRC> {
  using Base = WeightPackBatchBf16Base<GemmCore_T, ISA_T, false, T_SRC>;

 public:
  using typename Base::Param;
  using typename Base::StorageType;
  using typename Base::SType;
  using typename Base::WType;

  /// Reorder job of a thread
  void run(const Param& p, const jblas::parallel::ThreadProblem2D& thdp, std::function<int(int)> step_batch) {
    if (!thdp.valid) return;
    const auto pw = dynamic_cast<const StorageType*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after padding
    const int NPad = pw->mNPad;  // N size after padding
    assert(pw->mK <= KPad);
    assert(pw->mN <= NPad);
    assert(padto(pw->mN, GemmCore_T::NTILE) == NPad);

    auto [y, x] = thdp.loc;
    auto [ny, nx] = thdp.size;
    const auto nx_pad = jblas::utils::padto(nx, GemmCore_T::KTILE);

    using KernInterleave = typename jblas::kernel::wrapper::PaddingInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::template forward<ISA_T, T_SRC, WType>(  //
          p.B + step_batch(ibat) + x * p.ldb,                                           //
          pw->template WPtr<WType>() + ibat * KPad * NPad + x * GemmCore_T::NTILE,      //
          nx, pw->mN,                                                                   // size
          nx_pad, NPad,                                                                 // padded size
          p.ldb, KPad);                                                                 // stride
      assert(forward_stat == JblasSuccess);
    }
  }
};

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class ActivationIdentity {
 public:
  using AType = typename _GemmCore_T::AType;
  struct Param {
    const AType* A;
    int lda;
  };
  ActivationIdentity() {}

  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset, void* /* tmpcache */, size_t /* cachesize */) {
    auto aptr = const_cast<AType*>(_param.A);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return JblasSuccess;
  }
};

/**
 * @brief LauncherBase with addition input as packed weight offset
 */
template <JBLAS_ISA RT_ISA_, class _GemmCore_T, template <class, JBLAS_ISA> class _PrologueA_T,
          template <class, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class LauncherBaseOff                             //
    : public jblas::wrapper::gemm::LauncherBase<  //
          RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T> {
  using Base = jblas::wrapper::gemm::LauncherBase<  //
      RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T>;

 public:
  using typename Base::GemmCore;
  using Param = typename Base::Param;
  using AType = typename Base::AType;
  using BType = typename Base::BType;
  using CType = typename Base::CType;
  static constexpr auto RT_ISA = RT_ISA_;

  void run(const Param& _param, const jblas::parallel::gemm::ThreadProblemBase& _config,
           int w_offset /* weight offset for batching */) {
    // TO(Yi) temperarily configure to max tiling size
    this->mGemmCore.configure(16, 16, 16);  // Need 'this->' here; See：https://stackoverflow.com/questions/11405
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = tmpC + _config.block[0] * _config.block[1];
    tmpCache = utils::cpu_pointer_align(tmpCache);

    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, w_offset, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

 protected:
  void run_block(const Param& _param, const jblas::parallel::gemm::ThreadProblemBase& _config,
                 int w_offset /* weight offset for batching */, int blk_m, int blk_n, int blk_msize, int blk_nsize,
                 AType* tmpA, BType* /*tmpB*/, CType* tmpC, void* tmpcache) {
    int n_padded = padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = utils::remainsize(iterk, _param.problem.dims[3], _config.block[2]);
      int k_padded = padto(k_remain, GemmCore::KTILE);
      int k_paddedle = padto_le(k_remain, GemmCore::KTILE);
      BType* bptr_cache = nullptr;
      int bcache_step = 0;
      this->mProB.getWeight(&bptr_cache, &bcache_step,      // See：https://stackoverflow.com/questions/11405
                            k_padded, n_padded,             //
                            iterk, _config.loc[1] + blk_n,  //
                            _param.paramB, tmpcache, _config.tmpcachesize);
      bptr_cache += w_offset;
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);

        int acache_step = 0;
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                    blk_m + i + _config.loc[0], iterk, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                                  acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                                  _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                    blk_m + i + _config.loc[0], iterk + k_paddedle, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                                  GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                                  iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    this->mEpilogue.forward(tmpC, _config.block[1], _config.loc[0] + blk_m, _config.loc[1] + blk_n, blk_msize,
                            blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

/**
 * @brief LauncherBase with addition input as packed weight offset
 */
template <JBLAS_ISA RT_ISA_, class _GemmCore_T, template <class, JBLAS_ISA> class _PrologueA_T,
          template <class, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class LauncherBaseWeight                          //
    : public jblas::wrapper::gemm::LauncherBase<  //
          RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T> {
  using Base = jblas::wrapper::gemm::LauncherBase<  //
      RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T>;

 public:
  using typename Base::GemmCore;
  using Param = typename Base::Param;
  using AType = typename Base::AType;
  using BType = typename Base::BType;
  using CType = typename Base::CType;
  static constexpr auto RT_ISA = RT_ISA_;

  void run(const Param& _param, const jblas::parallel::gemm::ThreadProblemBase& _config) {
    this->mGemmCore.configure(16, 16, 16);  // Need 'this->' here; See：https://stackoverflow.com/questions/11405
    auto StackTmp = alloca(_config.stacksize);
    auto tmpB = reinterpret_cast<BType*>(StackTmp);
    tmpB = utils::cpu_pointer_align(tmpB);
    auto tmpA = reinterpret_cast<AType*>(tmpB + static_cast<size_t>(_config.block[1]) * _config.block[2]);
    tmpA = utils::cpu_pointer_align(tmpA);
    auto tmpC = reinterpret_cast<CType*>(tmpA + static_cast<size_t>(GemmCore::MTILE) * _config.block[2]);
    tmpC = utils::cpu_pointer_align(tmpC);
    auto tmpCache = tmpC + _config.block[0] * _config.block[1];
    tmpCache = utils::cpu_pointer_align(tmpCache);

    for (int itern = 0; itern < _config.size[1]; itern += _config.block[1]) {
      int n_remain = utils::remainsize(itern, _config.size[1], _config.block[1]);
      for (int iterm = 0; iterm < _config.size[0]; iterm += _config.block[0]) {
        int m_remain = utils::remainsize(iterm, _config.size[0], _config.block[0]);
        run_block(_param, _config, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC, tmpCache);
      }
    }
  }

 protected:
  void run_block(const Param& _param, const jblas::parallel::gemm::ThreadProblemBase& _config, int blk_m, int blk_n,
                 int blk_msize, int blk_nsize, AType* tmpA, BType* tmpB, CType* tmpC, void* tmpcache) {
    int n_padded = padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.problem.dims[3]; iterk += _config.block[2]) {
      int k_remain = remainsize(iterk, _param.problem.dims[3], _config.block[2]);
      int k_padded = padto(k_remain, GemmCore::KTILE);
      int k_paddedle = padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_step = 0;

      this->mProB.getWeight(&bptr_cache, &bcache_step, _param.paramB, k_padded, blk_nsize, iterk,
                            _config.loc[1] + blk_n, tmpcache, _config.tmpcachesize);
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.block[1];
        int ccache_stride = _config.block[1] * sizeof(CType);

        int acache_step = 0;
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                    (blk_m + i + _config.loc[0]), iterk, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                                  acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk, tmpcache,
                                  _config.tmpcachesize);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                    (blk_m + i + _config.loc[0]), iterk + k_paddedle, tmpcache, _config.tmpcachesize);
          this->mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                                  GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                                  iterk + k_paddedle, tmpcache, _config.tmpcachesize);
        }
      }
    }
    this->mEpilogue.forward(tmpC, _config.block[1], (_config.loc[0] + blk_m), _config.loc[1] + blk_n, blk_msize,
                            blk_nsize, _param.paramC, tmpcache, _config.tmpcachesize);
  }
};

/**
 * @brief MHA interface
 *
 * @tparam L_ExpSum Launcher type of the QK exp sum matmul
 * @tparam L_Scale Launcher type of the PV scale matmul (S for that in the flash-attn paper)
 */
template </* class Parallel_T, */ class L_ExpSum, class L_Scale>
class MHAInterface {
 public:
  using PrologueQ = typename L_ExpSum::PrologueA;
  using PrologueK = typename L_ExpSum::PrologueB;
  using QKProQArgs = typename PrologueQ::Param;
  using QKProKArgs = typename PrologueK::Param;
  using QKArgs = typename L_ExpSum::Param;
  using QKEpiArgs = typename L_ExpSum::EpiParam;

  using PrologueS = typename L_Scale::PrologueA;
  using PrologueV = typename L_Scale::PrologueB;
  using PVProPArgs = typename PrologueS::Param;
  using PVProVArgs = typename PrologueV::Param;
  using PVArgs = typename L_Scale::Param;
  using PVEpiArgs = typename L_Scale::EpiParam;

  using GemmQK = typename L_ExpSum::GemmCore;
  using GemmPV = typename L_Scale::GemmCore;
  using Q_T = typename std::remove_const<typename std::remove_pointer<decltype(QKProQArgs::A)>::type>::type;
  using K_T = typename PrologueK::SType;
  using V_T = typename PrologueV::SType;
  using DST_T = typename std::remove_const<typename std::remove_pointer<decltype(PVEpiArgs::dst)>::type>::type;

  static_assert(GemmQK::MTILE == GemmPV::MTILE, "2 GEMM should have the same M_TILE.");

  JBLAS_CODE compute(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p, const jblas::parallel::IThreading& th) {
    static constexpr auto M_TILE = GemmQK::MTILE;
    assert(p.Q_sc == 1 && p.K_sc == 1 && p.V_sc == 1 && p.dst_sc == 1);
    assert(p.Q_layout == ATTN_FWD_LAYOUT_PLAIN && p.K_layout == ATTN_FWD_LAYOUT_PLAIN &&
           p.V_layout == ATTN_FWD_LAYOUT_PLAIN && p.dst_layout == ATTN_FWD_LAYOUT_PLAIN);
    assert(p.step_v_head_size == 1);
    assert(p.step_k_head_size == 1 || p.step_k_sl == 1);
    const auto num_heads = p.batch_size * p.head_num;  // Total number of heads
    jblas::device::CpuBase cb;                         // Note: DO NOT use cb.mNumThreads; use th.num_threads() instead

    const bool is_causal = (p.attn_flags & NE_ATTN_FLAG_IS_CAUSAL) != 0;
    const bool is_alibi = (p.attn_flags & NE_ATTN_FLAG_IS_ALIBI8) != 0;
    assert(!is_causal || p.sl_q <= p.sl_kv);
    assert(("alibi not supported!", !is_alibi));
    assert(("GQA not supported!", p.head_num == p.heads_kv));
    const auto sl_diff = p.sl_kv - p.sl_q;

    // prepare memory for packed weight
    // TODO(Yi): init packed weight with p.tmp
    StoragePackedWeightBatch /*<typename GemmQK::BType>*/ K_pack(GemmQK::ID);  // packed K
    K_pack.resize(padto(p.sl_kv, GemmQK::NTILE), padto(p.head_size, GemmQK::KTILE), p.sl_kv, p.head_size, num_heads,
                  jblas::utils::jblas_dtype<typename GemmQK::BType>);
    auto bufferK = jblas::utils::amalloc<int8_t>(K_pack.mSize);
    K_pack.assign(bufferK);
    StoragePackedWeightBatch /*<typename GemmPV::BType>*/ V_pack(GemmPV::ID);  // packed V
    V_pack.resize(padto(p.head_size, GemmPV::NTILE), padto(p.sl_kv, GemmPV::KTILE), p.head_size, p.sl_kv, num_heads,
                  jblas::utils::jblas_dtype<typename GemmPV::BType>);
    auto bufferV = jblas::utils::amalloc<int8_t>(V_pack.mSize);
    V_pack.assign(bufferV);
    const auto K_pack_batch_off = K_pack.mKPad * K_pack.mNPad;
    const auto V_pack_batch_off = V_pack.mKPad * V_pack.mNPad;

    const auto step_batch_k = [step_bs = p.step_k_bs, step_hn = p.step_k_head_num, hn = p.heads_kv](int ibat) {
      return (ibat / hn) * step_bs + (ibat % hn) * step_hn;
    };
    const auto step_batch_v = [step_bs = p.step_v_bs, step_hn = p.step_v_head_num, hn = p.heads_kv](int ibat) {
      return (ibat / hn) * step_bs + (ibat % hn) * step_hn;
    };

    // prepare parallel scheduler for packed weight
    using Scheduler2D = typename jblas::parallel::Scheduler2D;
    using ThreadProblem2D = typename jblas::parallel::ThreadProblem2D;
    const auto schK = p.step_k_head_size == 1
                          ? Scheduler2D({th.num_threads(), {num_heads, p.sl_kv}, {1, GemmQK::NTILE}})
                          : Scheduler2D({th.num_threads(), {num_heads, p.head_size}, {1, GemmQK::KTILE}});
    const auto schV = Scheduler2D({th.num_threads(), {num_heads, p.sl_kv}, {1, GemmPV::KTILE}});

    const MHAProblem problem = {p.batch_size, p.head_num, p.heads_kv, p.head_size, p.sl_q, p.sl_kv};
    const auto m_tiles = updiv(p.sl_q, M_TILE);
    const auto num_tasks = num_heads * m_tiles;
    const Scheduler2D parl({th.num_threads(), {num_tasks, 1}, {1, 1}});

    th.parallel_for([&](int tid) {
      {  // reorder K & V
        ThreadProblem2D thdpK{tid};
        schK.getIndex(thdpK);
        l_expsum.mProB.run(  // pack K
            QKProKArgs{
                /* .B = */ p.K,
                /* .ldb = */ p.step_k_sl * p.step_k_head_size,  //  use the non-one step
                /* .StorageType = */ &K_pack,
            },
            thdpK, step_batch_k);

        ThreadProblem2D thdpV{tid};
        schV.getIndex(thdpV);
        l_scale.mProB.run(  // pack V
            PVProVArgs{
                /* .B = */ p.V,
                /* .ldb = */ p.step_v_sl,
                /* .StorageType = */ &V_pack,
            },
            thdpV, step_batch_v);
      }

      th.sync();

      // calculate mm + softmax + mm
      {
        const int tmp_exp_size = M_TILE * padto(p.sl_kv, GemmQK::NTILE) * sizeof(ne_bf16_t);  // TODO(Yi): alignment?
        const auto tmp = p.tmp + tid * tmp_exp_size;
        ThreadProblem2D thdp{tid};
        parl.getIndex(thdp);
        const auto [task_start, _assert0] = thdp.loc;
        auto [task_size, _assert_max1] = thdp.size;
        assert(task_size == 0 || _assert0 == 0);
        assert(task_size == 0 || _assert_max1 == 1 || _assert_max1 == 0);
        if (_assert_max1 == 0 || !thdp.valid) task_size = 0;

        for (int task_id = task_start; task_id < task_start + task_size; ++task_id) {
          const int ibat = task_id / m_tiles;
          const int i_m = task_id % m_tiles * M_TILE;
          const int ibs = ibat / p.head_num;
          const int ihn = ibat % p.head_num;
          const int m_size = std::min(M_TILE, p.sl_q - i_m);
          // TODO(Yi): heads_kv

          float exp_sum[M_TILE]{};
          memset(exp_sum, 0, sizeof(exp_sum));

          // ptr to Q / dst matrix of the current head
          const auto head_q = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num;
          const auto head_dst = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num;
          const auto unmasked_size = is_causal ? std::min(p.sl_kv, p.sl_kv - p.sl_q + i_m + M_TILE - 1 + 1) : p.sl_kv;

          const auto unmasked_size_pad_qk = std::min(p.sl_kv, padto(unmasked_size, GemmQK::NTILE));
          const auto unmasked_size_pad_pv = std::min(p.sl_kv, padto(unmasked_size, GemmPV::KTILE));
          const auto ld_tmp_exp = padto(padto(unmasked_size_pad_pv, GemmQK::NTILE), GemmPV::KTILE);

          typename jblas::parallel::gemm::ThreadProblemBase tpQK{
              /* ThreadProblem2D */ {tid, {}, {i_m, 0}, {m_size, unmasked_size_pad_qk}, true},
              /* .block = */ {M_TILE, GemmQK::NTILE, p.head_size},
              /* .stacksize = */ cb.mL2Cache,
              /* .tmpcachesize = */ cb.mL2Cache,
          };
          const auto bf16_tmp = reinterpret_cast<bf16*>(tmp);
          l_expsum.run(  // QxK => S ==exp==> P
              QKArgs{
                  jblas::utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ p.sl_q,
                      /* .N = */ unmasked_size_pad_qk,
                      /* .K = */ p.head_size,
                  },
                  /* .paramA = */ QKProQArgs{head_q, p.step_q_sl},
                  /* .paramB = */ QKProKArgs{nullptr, 0, &K_pack},
                  /* .paramC = */
                  QKEpiArgs{
                      /* .dst = */ bf16_tmp - i_m * ld_tmp_exp,  // pretend that there is a whole exp mat
                      /* .dst_sum = */ exp_sum - i_m,            // pretend that there is a whole exp sum
                      /* .ld_dst = */ ld_tmp_exp,
                      /* .scale = */ p.QK_scale,
                      /* .causal_offset = */ is_causal ? sl_diff : -1,
                      /* .alibi_slope = */ 0.f,
                  },
                  // /* .workspace = */ nullptr,
              },
              tpQK, /* w_offset */ ibat * K_pack_batch_off);
          for (int ii = 0; ii < M_TILE; ++ii) exp_sum[ii] = 1.f / exp_sum[ii];

          typename jblas::parallel::gemm::ThreadProblemBase tpPV{
              /* ThreadProblem2D */ {tid, {}, {0, 0}, {m_size, p.head_size}, true},
              /* .block = */ {M_TILE, GemmPV::NTILE, unmasked_size_pad_qk},
              /* .stacksize = */ cb.mL2Cache,
              /* .tmpcachesize = */ cb.mL2Cache,
          };
          l_scale.run(  // PxV => O
              PVArgs{
                  jblas::utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ std::min(p.sl_q - i_m, M_TILE),
                      /* .N = */ p.head_size,
                      /* .K = */ unmasked_size_pad_qk,
                  },
                  /* .paramA = */ PVProPArgs{(jblas::utils::bf16*)tmp, ld_tmp_exp},
                  /* .paramB = */ PVProVArgs{nullptr, 0, &V_pack},
                  /* .paramC = */
                  PVEpiArgs{
                      /* .scale = */ exp_sum,
                      /* .dst = */ head_dst + i_m * p.step_dst_sl,
                      /* .ld_dst = */ p.step_dst_sl,
                  },
                  // /* .workspace = */ nullptr,
              },
              tpPV, /* w_offset */ ibat * V_pack_batch_off);
        }
      }
    });
    jblas::utils::afree(bufferK);
    jblas::utils::afree(bufferV);
    return JblasSuccess;
  }

 protected:
  L_ExpSum l_expsum;
  L_Scale l_scale;
};

/**
 * @brief An Epilogue that optionally apply a casual mask (but may not filling zero) and scale the fp32 result, update
 * the maximun of each line of the reuslt, and storing exp as bf16 results
 */
template <JBLAS_ISA ISA_T, typename T_SRC, typename T_DST>
class ScaleTrackMax {
 public:
  using DType = T_DST;
  using SType = T_SRC;
  struct Param;

  JBLAS_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p) const {
    assert(false);
    return JblasNotSupport;
  }
};
template <JBLAS_ISA ISA_T>
class ScaleTrackMax<ISA_T, fp16, float> {
 public:
  using DType = float;
  using SType = fp16;
  struct Param {
    DType* dst;
    DType* dst_max;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };

  JBLAS_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p, void* /* tmpcache */, size_t /* cachesize */) const {
    assert(("alibi not supported!", p.alibi_slope == 0.f));
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
#if CompileFP16()
#if MHA_2ND_EXP
    const auto v_scale = _mm512_set1_ps(p.scale);

    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_max = _mm512_set1_ps(-INFINITY);
      for (; j < N_unmasked - 15; j += 16) {
        const auto xs = _mm512_mul_ps(v_scale, _mm512_cvtxph_ps(_mm256_loadu_ph(src + i * src_step + j)));
        v_max = _mm512_max_ps(v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
      }
      if (j < N_unmasked) {
        const auto xs = _mm512_mul_ps(
            v_scale, _mm512_cvtxph_ps(_mm256_castsi256_ph(_mm256_maskz_loadu_epi16(v_mask, src + i * src_step + j))));
        v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        j += 16;
      }
      dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));

      // if (j < jblas::utils::padto(N, 64))
      //   memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (jblas::utils::padto(N, 64) - j));
    }
#else
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);
      for (int j = 0; j < N_unmasked; ++j) {
        const auto val_ = src[i * src_step + j] * p.scale;
        dst[i * p.ld_dst + j] = static_cast<T_DST>(val_);
        dst_max[i] = std::max(dst_max[i], val_);
      }
      if (N_unmasked < jblas::utils::padto(N, 64))
        memset(dst + i * p.ld_dst + N_unmasked, 0, sizeof(*dst) * (jblas::utils::padto(N, 64) - N_unmasked));
    }
#endif

    return JblasSuccess;
#else
    return JblasNotSupport;
#endif
  }
};
template <JBLAS_ISA ISA_T>
using ScaleTrackMaxFp16Fp32 = ScaleTrackMax<ISA_T, fp16, float>;

template <JBLAS_ISA ISA_T>
class ScaleTrackMax<ISA_T, float, float> {
 public:
  using DType = float;
  using SType = float;
  struct Param {
    DType* dst;
    DType* dst_max;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };
  static constexpr float seq15[16]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  JBLAS_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p, void* /* tmpcache */, size_t /* cachesize */) const {
    return p.alibi_slope == 0 ? forward_<false>(src, src_step, M_offset, N_offset, M, N, p)
                              : forward_<true>(src, src_step, M_offset, N_offset, M, N, p);
  }

  template <bool HAS_ALIBI>
  JBLAS_CODE forward_(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                      const int N, const Param& p) const {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
#if CompileAVX512F()
#if MHA_2ND_EXP
    const auto v_scale = _mm512_set1_ps(p.scale);
    const auto v_seq15 = _mm512_loadu_ps(seq15);
    const auto alibi_slope = _mm512_set1_ps(p.alibi_slope);
    const auto alibi_base = _mm512_mul_ps(alibi_slope, _mm512_add_ps(v_seq15, _mm512_set1_ps(N_offset)));
    const auto alibi_step = _mm512_set1_ps(p.alibi_slope * 16);

    for (int i = 0; i < M; ++i) {
      auto alibi_curr = alibi_base;
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_max = _mm512_set1_ps(-INFINITY);
      for (; j < N_unmasked - 15; j += 16) {
        const auto xs = _mm512_fmadd_ps(v_scale, _mm512_loadu_ps(src + i * src_step + j), alibi_curr);
        v_max = _mm512_max_ps(v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        if constexpr (HAS_ALIBI) alibi_curr = _mm512_add_ps(alibi_curr, alibi_step);
      }
      if (j < N_unmasked) {
        const auto xs = _mm512_fmadd_ps(v_scale, _mm512_maskz_loadu_ps(v_mask, src + i * src_step + j), alibi_curr);
        v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        if constexpr (HAS_ALIBI) alibi_curr = _mm512_add_ps(alibi_curr, alibi_step);
        j += 16;
      }
      dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));

      // if (j < jblas::utils::padto(N, 64))
      //   memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (jblas::utils::padto(N, 64) - j));
    }
#else
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);
      for (int j = 0; j < N_unmasked; ++j) {
        const auto val_ = src[i * src_step + j] * p.scale;
        dst[i * p.ld_dst + j] = static_cast<T_DST>();
        dst_max[i] = std::max(dst_max[i], val_);
      }
      if (N_unmasked < jblas::utils::padto(N, 64))
        memset(dst + i * p.ld_dst + N_unmasked, 0, sizeof(*dst) * (jblas::utils::padto(N, 64) - N_unmasked));
    }
#endif

    return JblasSuccess;
#else
    return JblasNotSupport;
#endif
  }
};
template <JBLAS_ISA ISA_T>
using ScaleTrackMaxFp32Fp32 = ScaleTrackMax<ISA_T, float, float>;

template <JBLAS_ISA ISA_T>
class ScaleTrackMax<ISA_T, int32_t, float> {
 public:
  using DType = float;
  using SType = int32_t;
  struct Param {
    DType* dst;
    DType* dst_max;
    int ld_dst;  // #elements
    float scale;
    int causal_offset;  // offset for causal mask; negative value for disabling causal mask
    float alibi_slope;  // m-factor in the alibi paper for current head: https://arxiv.org/abs/2108.12409
  };

  JBLAS_CODE forward(const SType* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p, void* /* tmpcache */, size_t /* cachesize */) const {
    assert(("alibi not supported!", p.alibi_slope == 0.f));
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_max = p.dst_max + M_offset;
#if CompileAVX512F()
    const auto v_scale = _mm512_set1_ps(p.scale);

    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_max = _mm512_set1_ps(-INFINITY);
      for (; j < N_unmasked - 15; j += 16) {
        const auto xs = _mm512_mul_ps(v_scale, _mm512_cvtepi32_ps(_mm512_loadu_si512(src + i * src_step + j)));
        v_max = _mm512_max_ps(v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
      }
      if (j < N_unmasked) {
        const auto xs =
            _mm512_mul_ps(v_scale, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(v_mask, src + i * src_step + j)));
        v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
        _mm512_storeu_ps(dst + i * p.ld_dst + j, xs);
        j += 16;
      }
      dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));
      // if (j < jblas::utils::padto(N, 64))
      //   memset(dst + i * p.ld_dst + j, 0, sizeof(*dst) * (jblas::utils::padto(N, 64) - j));
    }
    return JblasSuccess;
#else
    return JblasNotSupport;
#endif
  }
};
template <JBLAS_ISA ISA_T>
using ScaleTrackMaxS32Fp32 = ScaleTrackMax<ISA_T, int32_t, float>;

template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightBase {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = BType;
  struct Param {
    const SType* B;
    int ldb;
    bool is_padded;
  };
  WeightBase() {}
  JBLAS_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size, int k_offset,
                       int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    if ((n_size % _GemmCore_T::NTILE == 0) && std::is_same<SType, BType>::value &&
        0) {  // TODO(Yi) : use a gemm core accept step for K or reorder at runtime
      *dst_ptr = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      *dst_step = p.ldb;
      return JblasSuccess;
    } else if (*dst_ptr != nullptr && std::is_same<SType, BType>::value) {
      const auto src = const_cast<SType*>(p.B) + k_offset * p.ldb + n_offset;
      const auto npad = padto(n_size, _GemmCore_T::NTILE);
      *dst_step = npad;
      for (int k = 0; k < k_size; ++k) {
        memcpy(*dst_ptr + k * npad, src + k * p.ldb, sizeof(BType) * n_size);
        memset(*dst_ptr + k * npad + n_size, 0, sizeof(BType) * (npad - n_size));
      }
      return JblasSuccess;
    } else {
      assert(false);
      return JblasNotSupport;
    }
  }
};
template <class _GemmCore_T, JBLAS_ISA ISA_T>
class WeightForwardNTile48 {
 public:
  using BType = typename _GemmCore_T::BType;
  using SType = BType;
  struct Param {
    const SType* B;
    int ldb;
    bool is_padded;
  };
  WeightForwardNTile48() {}
  JBLAS_CODE getWeight(BType** dst_ptr, int* dst_step, const Param& p, int k_size, int n_size, int k_offset,
                       int n_offset, void* /* tmpcache */, size_t /* cachesize */) {
    assert(p.is_padded);
    *dst_ptr = const_cast<SType*>(p.B) + k_offset * 48 + n_offset * p.ldb;
    *dst_step = p.ldb;
    return JblasSuccess;
  }
};
template <class SRC_T, class DST_T>
struct InplacePrecomputeMaxSoftmax {
  // nsize is the staring n-size when causal mask enabled
  // src and dst cam be on the same address if sizeof(SRC_T) >= sizeof(DST_T) and ld is correctly set
  // s_max and expsum cam be on the same address
  static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, SRC_T* src, DST_T* dst,
                      const SRC_T* s_max, float* expsum, int ld_src, int ld_dst) {
    assert(false);
  }
};
#if CompileFP16()
template <>
struct InplacePrecomputeMaxSoftmax<float, fp16> {
  static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, float* src, fp16* dst, const float* s_max,
                      float* expsum, int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const uint16_t v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
      {  // subtract max
        const auto row_max = _mm512_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
          _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
        }
      }
      auto v_sum = _mm512_setzero_ps();
      {  // exp & sum
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_exp = exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
          v_sum = _mm512_add_ps(v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp = exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
          v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }
        expsum[ii] = _mm512_reduce_add_ps(v_sum);
        v_sum = _mm512_set1_ps(expsum[ii]);
      }
      {  // scale & fp16
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_softmax = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          _mm256_storeu_ph(i_dst + jj, _mm512_cvtxps_ph(v_softmax));
        }
        if (jj < curr_n_size) {
          const auto v_softmax = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          _mm256_storeu_ph(i_dst + jj, _mm512_maskz_cvtxps_ph(v_mask, v_softmax));
          jj += 16;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(fp16) * (n_pad_size - jj));
      }
    }
  }
};
#endif

#if CompileBF16()
template <>
struct InplacePrecomputeMaxSoftmax<float, bf16> {
  static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, float* src, bf16* dst, const float* s_max,
                      float* expsum, int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const auto v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
      const auto v_mask32 = _cvtu32_mask32((1U << (curr_n_size % 32)) - 1);
      {  // subtract max
        const auto row_max = _mm512_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
          _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
        }
      }
      auto v_sum = _mm512_setzero_ps();
      {  // exp & sum
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_exp = exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
          v_sum = _mm512_add_ps(v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp = exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
          v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }
        expsum[ii] = _mm512_reduce_add_ps(v_sum);
        v_sum = _mm512_set1_ps(expsum[ii]);
      }
      {  // scale & bf16
        int jj = 0;
        for (; jj < curr_n_size / 32 * 32; jj += 32) {
          const auto v_softmax0 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          const auto v_softmax1 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj + 16), v_sum);
          _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_cvtne2ps_pbh(v_softmax1, v_softmax0));
        }
        if (jj < curr_n_size) {
          const auto v_softmax0 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
          const auto v_softmax1 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj + 16), v_sum);
#if defined(__GNUC__) && (__GNUC__ == 13) && (__GNUC_MINOR__ <= 2)
          // There is a bug on gcc 13.1/13.2 what reverse the parameter order;
          // A GUN team member said that it will befixed in GCC 13.3
          _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_maskz_cvtne2ps_pbh(v_mask32, v_softmax0, v_softmax1));
#else
          _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_maskz_cvtne2ps_pbh(v_mask32, v_softmax1, v_softmax0));
#endif
          jj += 32;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(bf16) * (n_pad_size - jj));
      }
    }
  }
};
#endif

template <>
struct InplacePrecomputeMaxSoftmax<float, uint8_t> {
  static void forward(int m_size, int n_size, int n_pad_size, bool is_causal, float* src, uint8_t* dst, float* s_max,
                      float* expsum, int ld_src, int ld_dst) {
    for (int ii = 0; ii < m_size; ++ii) {
      const auto i_src = src + ii * ld_src;
      const auto i_dst = dst + ii * ld_dst;
      const auto curr_n_size = n_size + (is_causal ? ii : 0);
      const uint16_t v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
      {  // subtract max
        const auto row_max = _mm512_set1_ps(s_max[ii]);
        for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
          _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
        }
      }
      {  // exp & sum
        auto v_sum = _mm512_setzero_ps();
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_exp = exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
          v_sum = _mm512_add_ps(v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);
        }
        if (jj < curr_n_size) {
          const auto v_exp = exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
          v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
          _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
        }
        expsum[ii] = _mm512_reduce_add_ps(v_sum);
      }
      {  // scale & int8
        const auto v_scale = _mm512_set1_ps(UINT8_MAX);
        int jj = 0;
        for (; jj < curr_n_size / 16 * 16; jj += 16) {
          const auto v_softmax = _mm512_mul_ps(_mm512_loadu_ps(i_src + jj), v_scale);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(i_dst + jj),
                           _mm512_cvtusepi32_epi8(_mm512_cvtps_epu32(v_softmax)));
        }
        if (jj < curr_n_size) {
          const auto v_softmax = _mm512_mul_ps(_mm512_loadu_ps(i_src + jj), v_scale);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(i_dst + jj),
                           _mm512_maskz_cvtusepi32_epi8(v_mask, _mm512_cvtps_epu32(v_softmax)));
          jj += 16;
        }
        if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(uint8_t) * (n_pad_size - jj));
      }
    }
  }
};

/**
 * @brief MHA interface with N-dim parallelism & stable softmax
 *
 * @tparam L_Max Launcher type of the QK matmul; tracking the dst max value of each row
 * @tparam L_Scale Launcher type of the PV scale matmul (S for that in the flash-attn paper)
 */
template </* class Parallel_T, */ class L_Max, class L_Scale>
class MHAStableInterface {
  template <class EpiArgs, bool HAS_SCALE, class T>
  static inline typename std::enable_if<!HAS_SCALE, EpiArgs>::type composeEpiArgs(float*, T* dst, int ld_dst) {
    return {dst, ld_dst};
  }
  template <class EpiArgs, bool HAS_SCALE, class T>
  static inline typename std::enable_if<HAS_SCALE, EpiArgs>::type composeEpiArgs(float* scale, T* dst, int ld_dst) {
    return {scale, dst, ld_dst};
  }

 public:
  using PrologueQ = typename L_Max::PrologueA;
  using PrologueK = typename L_Max::PrologueB;
  using QKProQArgs = typename PrologueQ::Param;
  using QKProKArgs = typename PrologueK::Param;
  using QKArgs = typename L_Max::Param;
  using QKEpiArgs = typename L_Max::EpiParam;

  using PrologueS = typename L_Scale::PrologueA;
  using PrologueV = typename L_Scale::PrologueB;
  using PVProPArgs = typename PrologueS::Param;
  using PVProVArgs = typename PrologueV::Param;
  using PVArgs = typename L_Scale::Param;
  using PVEpiArgs = typename L_Scale::EpiParam;

  using GemmQK = typename L_Max::GemmCore;
  using GemmPV = typename L_Scale::GemmCore;
  using Q_T = typename std::remove_const<typename std::remove_pointer<decltype(QKProQArgs::A)>::type>::type;
  using K_T = typename PrologueK::SType;
  using V_T = typename PrologueV::SType;
  using DST_T = typename L_Scale::Epilogue::DType;

  static_assert(GemmQK::MTILE == GemmPV::MTILE, "2 GEMM should have the same M_TILE.");
  static constexpr auto M_TILE = GemmQK::MTILE;

  JBLAS_CODE compute(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p, const jblas::parallel::IThreading& th) {
    assert((std::is_same<Q_T, int8_t>::value || p.Q_sc == 1));
    assert((std::is_same<K_T, int8_t>::value || p.K_sc == 1));
    assert((std::is_same<V_T, int8_t>::value || p.V_sc == 1));
    assert((std::is_same<DST_T, int8_t>::value || p.dst_sc == 1));

    assert((p.Q_layout == ATTN_FWD_LAYOUT_PLAIN && p.dst_layout == ATTN_FWD_LAYOUT_PLAIN));
    assert((p.K_layout == ATTN_FWD_LAYOUT_PLAIN ||
            (std::is_same<K_T, int8_t>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
            (std::is_same<K_T, bf16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2)));
    assert((p.V_layout == ATTN_FWD_LAYOUT_PLAIN ||
            (std::is_same<V_T, int8_t>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
            (std::is_same<V_T, bf16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2)));

    assert((!std::is_same<PrologueK, ::WeightForwardNTile48<typename L_Max::GemmCore, L_Max::RT_ISA>>::value) ||
           p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ||
           p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);  // WeightForward can only be used with preprocessed layout
    assert((!std::is_same<PrologueV, ::WeightForwardNTile48<typename L_Scale::GemmCore, L_Scale::RT_ISA>>::value) ||
           p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ||
           p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);  // WeightForward can only be used with preprocessed layout

    assert((p.K_layout != ATTN_FWD_LAYOUT_PLAIN || p.step_v_head_size == 1));
    assert((p.V_layout != ATTN_FWD_LAYOUT_PLAIN || p.step_k_sl == 1));
    const auto num_heads = p.batch_size * p.head_num;  // Total number of heads
    jblas::device::CpuBase cb;                         // Note: DO NOT use cb.mNumThreads; use th.num_threads() instead
    const bool is_causal = (p.attn_flags & NE_ATTN_FLAG_IS_CAUSAL) != 0;
    const bool is_alibi = (p.attn_flags & NE_ATTN_FLAG_IS_ALIBI8) != 0;
    assert(!is_causal || p.sl_q <= p.sl_kv);
    assert(("head_num must be a multiple of heads_kv!", p.head_num % p.heads_kv == 0));
    const auto group_heads = p.head_num / p.heads_kv;
    const auto sl_diff = p.sl_kv - p.sl_q;

    // alibi slope
    const int n_heads_log2_floor = 1 << static_cast<int>(floor(log2(p.head_num)));
    const float m0 = powf(2.0f, -(8.f) / n_heads_log2_floor);
    const float m1 = powf(2.0f, -(8.f / 2.0f) / n_heads_log2_floor);

    const auto m_tiles = updiv(p.sl_q, M_TILE);
    const auto num_tasks = num_heads * m_tiles;

    using Scheduler2D = typename jblas::parallel::Scheduler2D;
    const Scheduler2D parl({th.num_threads(), {num_tasks, 1}, {1, 1}});  // main parallel scheduler

    th.parallel_for([&](int tid) {
      const int tmp_s_size = M_TILE * padto(padto(p.sl_kv, GemmQK::NTILE), GemmPV::KTILE);
      const int tmp_p_size = tmp_s_size;
      const int tmp_bytes = tmp_s_size * sizeof(float);  // S & exp
      const auto tmp_s = reinterpret_cast<float*>(p.tmp + tid * tmp_bytes);
      using PType = typename GemmPV::AType;
      const auto tmp_p = reinterpret_cast<PType*>(tmp_s);  // overwrite tmp_s row-wisely

      // calculate mm + softmax + mm
      {
        typename jblas::parallel::ThreadProblem2D thdp{tid};
        parl.getIndex(thdp);
        const auto [task_start, _assert0] = thdp.loc;
        auto [task_size, _assert_max1] = thdp.size;
        assert(task_size == 0 || _assert0 == 0);
        assert(task_size == 0 || _assert_max1 == 1 || _assert_max1 == 0);
        if (_assert_max1 == 0 || !thdp.valid) task_size = 0;

        for (int task_id = task_start; task_id < task_start + task_size; ++task_id) {
          const int ibat = task_id / m_tiles;
          const int i_m = task_id % m_tiles * M_TILE;
          const int ibs = ibat / p.head_num;
          const int ihn = ibat % p.head_num;
          const int ihkv = ihn / group_heads;
          const int m_size = std::min(M_TILE, p.sl_q - i_m);

          const auto alibi_ihn_m = !is_alibi                    ? 0.f
                                   : (ihn < n_heads_log2_floor) ? powf(m0, ihn + 1)
                                                                : powf(m1, 2 * (ihn - n_heads_log2_floor) + 1);

          float s_max[M_TILE]{};  // maximum for each row of the S matrix
          std::fill_n(s_max, M_TILE, -INFINITY);

          // ptr to Q / dst matrix of the current head
          const auto head_q = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num;
          const auto head_k = p.K + ibs * p.step_k_bs + ihkv * p.step_k_head_num;
          const auto head_v = p.V + ibs * p.step_v_bs + ihkv * p.step_v_head_num;
          const auto head_dst = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num;
          const auto unmasked_size = is_causal ? std::min(p.sl_kv, sl_diff + i_m + M_TILE - 1 + 1) : p.sl_kv;

          const auto unmasked_size_pad_qk = std::min(p.sl_kv, padto(unmasked_size, GemmQK::NTILE));
          const auto unmasked_size_pad_pv = std::min(p.sl_kv, padto(unmasked_size, GemmPV::KTILE));
          const int ld_tmp_s = padto(padto(unmasked_size_pad_pv, GemmQK::NTILE), GemmPV::KTILE);
          static_assert(sizeof(float) >= sizeof(PType), "PType exceeded float size!");
          const int ld_tmp_p = ld_tmp_s * sizeof(float) / sizeof(PType);
          const auto qk_prok_ldb = p.step_k_sl == 1                                 ? p.step_k_head_size
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ? p.step_k_sl
                                   : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? p.step_k_sl
                                                                                    : (assert(0), 0);

          typename jblas::parallel::gemm::ThreadProblemBase tpQK{
              /* ThreadProblem2D */ {tid, {}, {i_m, 0}, {m_size, unmasked_size_pad_qk}, true},
              /* .block = */ {M_TILE, GemmQK::NTILE, p.head_size},
              /* .stacksize = */ cb.mL2Cache,
              /* .tmpcachesize = */ cb.mL2Cache,
          };
          l_qk.run(  // QxK => S ==exp==> P
              QKArgs{
                  jblas::utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ p.sl_q,
                      /* .N = */ unmasked_size_pad_qk,
                      /* .K = */ p.head_size,
                  },
                  /* .paramA = */
                  QKProQArgs{
                      head_q,
                      p.step_q_sl,
                  },
                  /* .paramB = */
                  QKProKArgs{
                      /* .B = */ head_k,
                      /* .ldb = */ qk_prok_ldb,
                      /* .is_padded = */ true,
                  },  // K should be pre-transposed
                  /* .paramC = */
                  QKEpiArgs{
                      /* .dst = */ tmp_s - i_m * ld_tmp_s,  // pretend that there is a whole S mat
                      /* .dst_sum = */ s_max - i_m,         // pretend that there is a whole S mat
                      /* .ld_dst = */ ld_tmp_s,
                      /* .scale = */ p.QK_scale * p.Q_sc * p.K_sc,
                      /* .causal_offset = */ is_causal ? sl_diff : -1,
                      /* .alibi_slope = */ alibi_ihn_m,
                  },
                  // /* .workspace = */ nullptr,
              },
              tpQK);

          // softmax (with pre-computed row_max)
          const auto unmasked_size_start = is_causal ? std::min(sl_diff + i_m + 1, p.sl_kv) : p.sl_kv;
          float expsum[M_TILE]{};  // maximum for each row of the S matrix
          const auto softmax_npad_size = padto(unmasked_size_pad_pv, GemmPV::KTILE);
          InplacePrecomputeMaxSoftmax<float, PType>::forward(               //
              m_size, unmasked_size_start, softmax_npad_size,               // m / n
              is_causal, tmp_s, tmp_p, s_max, expsum, ld_tmp_s, ld_tmp_p);  //

          const auto pv_scale = expsum;
          for (int i = 0; i < M_TILE; ++i) pv_scale[i] = p.V_sc / UINT8_MAX / expsum[i] / p.dst_sc;

          const auto pv_prov_ldb = p.step_v_head_size == 1                          ? p.step_v_sl
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 ? p.step_v_head_size
                                   : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? p.step_v_head_size
                                                                                    : (assert(0), 0);

          typename jblas::parallel::gemm::ThreadProblemBase tpPV{
              /* ThreadProblem2D */ {tid, {}, {0, 0}, {m_size, p.head_size}, true},
              /* .block = */ {M_TILE, GemmPV::NTILE, unmasked_size_pad_pv},
              /* .stacksize = */ cb.mL2Cache,
              /* .tmpcachesize = */ cb.mL2Cache,
          };
          l_pv.run(  // PxV => O
              PVArgs{
                  jblas::utils::GemmProblem{
                      /* .batch */ 1,
                      /* .M = */ std::min(p.sl_q - i_m, M_TILE),
                      /* .N = */ p.head_size,
                      /* .K = */ unmasked_size_pad_pv,
                  },
                  /* .paramA = */ PVProPArgs{tmp_p, ld_tmp_p},
                  /* .paramB = */
                  PVProVArgs{
                      /* .B = */ head_v,
                      /* .ldb = */ pv_prov_ldb,
                      /* .is_padded = */ true,
                  },
                  /* .paramC = */
                  composeEpiArgs<PVEpiArgs, std::is_same<V_T, int8_t>::value>(  //
                      pv_scale, head_dst + i_m * p.step_dst_sl, p.step_dst_sl),
                  // /* .workspace = */ nullptr,
              },
              tpPV);
        }
      }
    });
    return JblasSuccess;
  }

 protected:
  L_Max l_qk;
  L_Scale l_pv;
};

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
void jblas_fusion_attn_forward(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& params) = delete;

template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchBf16Bf16NonTr = WeightPackBatchBf16NonTr<GEMM_T, ISA_T, bf16>;
template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchBf16Bf16Trans = WeightPackBatchBf16Trans<GEMM_T, ISA_T, bf16>;
template <>
void jblas_fusion_attn_forward<bf16, bf16, bf16, bf16>(const attn_fwd_args_t<bf16, bf16, bf16, bf16>& p) {
  using GemmKernelBF16ExpSum = ::LauncherBaseOff<  //
      JblasAMX_BF16,                               //
      jblas::gemm::HCoreRowNAmxbf16<64, 16>,       //
      jblas::prologue_a::gemm::ActivationBase,     //
      WeightPackBatchBf16Bf16Trans,                //
      ::ScaleExpAccSumFp32Bf16>;                   //
  using GemmKernelBF16 = ::LauncherBaseOff<        //
      JblasAMX_BF16,                               //
      jblas::gemm::HCoreRowNAmxbf16<64, 16>,       //
      jblas::prologue_a::gemm::ActivationBase,     //
      WeightPackBatchBf16Bf16NonTr,                //
      ::ScaleWriteBackFp32Bf16>;
  static MHAInterface<GemmKernelBF16ExpSum, GemmKernelBF16> kernel;
  const auto pth = ne_jblas::ne_threading::get();
  [[maybe_unused]] const auto ret = kernel.compute(p, *pth);
  assert(ret == JblasSuccess);
}

template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchFp16Bf16NonTr = WeightPackBatchBf16NonTr<GEMM_T, ISA_T, fp16>;
template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchFp16Bf16Trans = WeightPackBatchBf16Trans<GEMM_T, ISA_T, fp16>;
template <>
void jblas_fusion_attn_forward<float, fp16, fp16, float>(const attn_fwd_args_t<float, fp16, fp16, float>& params) {
  GetCPUDevice();
  const auto pth = ne_jblas::ne_threading::get();
  if (MHA_PREFER_AVX512FP16 && _cd->AVX512_FP16() && params.step_k_sl == 1) {
    using GemmKernelFP16TrackMax = ::LauncherBaseWeight<   //
        JblasAVX512_FP16,                                  //
        jblas::gemm::HCoreRowNAvx512fp16<64, 8>,           //
        jblas::prologue_a::gemm::ActivationConverterFp32,  //
        ::WeightBase,                                      //
        ::ScaleTrackMaxFp16Fp32>;                          //
    using GemmKernelFP16 = ::LauncherBaseWeight<           //
        JblasAVX512_FP16,                                  //
        jblas::gemm::HCoreRowNAvx512fp16<64, 8>,           //
        jblas::prologue_a::gemm::ActivationBase,           //
        ::WeightBase,                                      //
        jblas::epilogue::gemm::AccumulatorWriteBackFp16Fp32>;
    static MHAStableInterface<GemmKernelFP16TrackMax, GemmKernelFP16> kernel;
    [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
    assert(ret == JblasSuccess);
  } else if (_cd->AMX_BF16()) {
    if (params.step_k_head_size == 1) {
      using GemmKernelFP32FP16BF16ExpSum = ::LauncherBaseOff<  //
          JblasAMX_BF16,                                       //
          jblas::gemm::HCoreRowNAmxbf16<64, 16>,               //
          jblas::prologue_a::gemm::ActivationConverterFp32,    //
          WeightPackBatchFp16Bf16Trans,                        //
          ::ScaleExpAccSumFp32Bf16>;                           //
      using GemmKernelBF16FP16FP32 = ::LauncherBaseOff<        //
          JblasAMX_BF16,                                       //
          jblas::gemm::HCoreRowNAmxbf16<64, 16>,               //
          jblas::prologue_a::gemm::ActivationBase,             //
          WeightPackBatchFp16Bf16NonTr,                        //
          ::ScaleWriteBackFp32Fp32>;
      static MHAInterface<GemmKernelFP32FP16BF16ExpSum, GemmKernelBF16FP16FP32> kernel;
      [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
      assert(ret == JblasSuccess);
    } else if (params.step_k_sl == 1) {
      using GemmKernelFP32FP16BF16ExpSum = ::LauncherBaseOff<  //
          JblasAMX_BF16,                                       //
          jblas::gemm::HCoreRowNAmxbf16<64, 16>,               //
          jblas::prologue_a::gemm::ActivationConverterFp32,    //
          WeightPackBatchFp16Bf16NonTr,                        //
          ::ScaleExpAccSumFp32Bf16>;                           //
      using GemmKernelBF16FP16FP32 = ::LauncherBaseOff<        //
          JblasAMX_BF16,                                       //
          jblas::gemm::HCoreRowNAmxbf16<64, 16>,               //
          jblas::prologue_a::gemm::ActivationBase,             //
          WeightPackBatchFp16Bf16NonTr,                        //
          ::ScaleWriteBackFp32Fp32>;
      static MHAInterface<GemmKernelFP32FP16BF16ExpSum, GemmKernelBF16FP16FP32> kernel;
      [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
      assert(ret == JblasSuccess);
    }
  } else {
    assert(false);  // no suitbale launcher
  }
}

template <>
void jblas_fusion_attn_forward<fp16, fp16, fp16, fp16>(const attn_fwd_args_t<fp16, fp16, fp16, fp16>& params) {
  GetCPUDevice();
  const auto pth = ne_jblas::ne_threading::get();
  if (_cd->AMX_BF16()) {
    using GemmKernelFP16TrackMax = ::LauncherBaseWeight<  //
        JblasAVX512_FP16,                                 //
        jblas::gemm::HCoreRowNAvx512fp16<64, 8>,          //
        jblas::prologue_a::gemm::ActivationBase,          //
        ::WeightBase,                                     //
        ::ScaleTrackMaxFp16Fp32>;                         //
    using GemmKernelFP16 = ::LauncherBaseWeight<          //
        JblasAVX512_FP16,                                 //
        jblas::gemm::HCoreRowNAvx512fp16<64, 8>,          //
        jblas::prologue_a::gemm::ActivationBase,          //
        ::WeightBase,                                     //
        jblas::epilogue::gemm::AccumulatorWriteBackFp16>;
    static MHAStableInterface<GemmKernelFP16TrackMax, GemmKernelFP16> kernel;
    [[maybe_unused]] const auto ret = kernel.compute(params, *pth);
    assert(ret == JblasSuccess);
  } else {
    assert(0);
  }
}

template <>
void jblas_fusion_attn_forward<int8_t, int8_t, int8_t, int8_t>(
    const attn_fwd_args_t<int8_t, int8_t, int8_t, int8_t>& params) {
  GetCPUDevice();
  const auto pth = ne_jblas::ne_threading::get();
  if (/* params.sl_q > 4 &&  */ _cd->AMX_INT8()) {         // TODO(Yi): add vnni impl
    using GemmKernelInt32TrackMax = ::LauncherBaseWeight<  //
        JblasAMX_INT8,                                     //
        jblas::gemm::ICoreRowNAmxint8SS<48, 16>,           //
        jblas::prologue_a::gemm::ActivationBase,           //
        ::WeightForwardNTile48,                            //
        ::ScaleTrackMaxS32Fp32>;                           //
    using GemmKernelInt32 = ::LauncherBaseWeight<          //
        JblasAMX_INT8,                                     //
        jblas::gemm::ICoreRowNAmxint8<48, 16>,             //
        jblas::prologue_a::gemm::ActivationBase,           //
        ::WeightForwardNTile48,                            //
        ::ScaleWriteBackS32S8>;                            //
    static MHAStableInterface<GemmKernelInt32TrackMax, GemmKernelInt32> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, *pth);
    assert(ret == JblasSuccess);
  } else if (_cd->AVX512_VNNI()) {
    // using GemmKernelInt32TrackMax = ::LauncherBaseWeight<  //
    //     JblasAMX_INT8,                                         // TODO(Yi): s8s8 vnni kernel?
    //     jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8,           //
    //     jblas::prologue::gemm::ActivationBase,                 //
    //     ::WeightForwardNTile48,                                          //
    //     ::ScaleTrackMaxS32Fp32>;                               //
    // using GemmKernelInt32 = ::LauncherBaseWeight<          //
    //     JblasAVX512_VNNI,                                      //
    //     jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,         //
    //     jblas::prologue::gemm::ActivationBase,                 //
    //     ::WeightForwardNTile48,                                          //
    //     ::ScaleWriteBackS32S8>;                                //
    // static MHAStableInterface<GemmKernelInt32TrackMax, GemmKernelInt32> mha;
    // [[maybe_unused]] const auto ret = mha.compute(params);
    // assert(ret == JblasSuccess);
    assert(0);
  } else {
    assert(0);
  }
}

template <>
void jblas_fusion_attn_forward<float, bf16, bf16, float>(const attn_fwd_args_t<float, bf16, bf16, float>& params) {
  GetCPUDevice();
  const auto pth = ne_jblas::ne_threading::get();
  if (/* params.sl_q > 4 &&  */ _cd->AMX_BF16()) {         // TODO(Yi): add vdpbf16ps impl
    using GemmKernelBF16TrackMax = ::LauncherBaseWeight<   //
        JblasAMX_BF16,                                     //
        jblas::gemm::HCoreRowNAmxbf16<48, 16>,             //
        jblas::prologue_a::gemm::ActivationConverterFp32,  //
        ::WeightForwardNTile48,                            //
        ::ScaleTrackMaxFp32Fp32>;                          //
    using GemmKernelBF16 = ::LauncherBaseWeight<           //
        JblasAMX_BF16,                                     //
        jblas::gemm::HCoreRowNAmxbf16<48, 16>,             //
        ::ActivationIdentity,                              // pretty sure we have enough paddings for P-matrix
        ::WeightForwardNTile48,                            //
        jblas::epilogue::gemm::AccumulatorWriteBackFp32>;  //
    static MHAStableInterface<GemmKernelBF16TrackMax, GemmKernelBF16> mha;
    [[maybe_unused]] const auto ret = mha.compute(params, *pth);
    assert(ret == JblasSuccess);
  } else {
    assert(0);
  }
}

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
void jblas_fusion_attn_forward_ref(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p) {
  const bool is_causal = (p.attn_flags & NE_ATTN_FLAG_IS_CAUSAL) != 0;
  const bool is_alibi = (p.attn_flags & NE_ATTN_FLAG_IS_ALIBI8) != 0;
  assert(!is_causal || p.sl_q <= p.sl_kv);
  assert(("head_num must be a multiple of heads_kv!", p.head_num % p.heads_kv == 0));
  const auto group_heads = p.head_num / p.heads_kv;
  attn_shape_t attn_shape{
      p.batch_size, p.head_num, p.heads_kv, p.head_size, p.sl_q, p.sl_kv,
  };
  const auto workspace_size = jblas_fusion_attn_workspace_size(&attn_shape);
  static std::mt19937 rng;
  static std::uniform_int_distribution<> dist;
#ifdef NE_TESTS
  init_vector(p.tmp, workspace_size, INT8_MIN - 1, INT8_MAX + 1, dist(rng));
#else
  std::fill_n(p.tmp, workspace_size, 'f');
#endif
  const bool IS_BF16_GEMM =
      (std::is_same<Q_T, float>::value && std::is_same<K_T, fp16>::value && std::is_same<V_T, fp16>::value &&
       std::is_same<DST_T, float>::value && (!MHA_PREFER_AVX512FP16 || (p.step_k_head_size == 1))) ||
      (std::is_same<Q_T, float>::value && std::is_same<K_T, bf16>::value && std::is_same<V_T, bf16>::value &&
       std::is_same<DST_T, float>::value);
  assert(p.Q_layout == ATTN_FWD_LAYOUT_PLAIN);
  assert(p.dst_layout == ATTN_FWD_LAYOUT_PLAIN);
  assert((p.K_layout == ATTN_FWD_LAYOUT_PLAIN ||
          (std::is_same<K_T, int8_t>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
          (std::is_same<K_T, bf16>::value && p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2)));
  assert((p.V_layout == ATTN_FWD_LAYOUT_PLAIN ||
          (std::is_same<V_T, int8_t>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4) ||
          (std::is_same<V_T, bf16>::value && p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2)));

  const auto NTILE = p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                     : p.K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                                                                      : 0;
  const auto ROWPACK = p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                       : p.V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                                                                        : 0;
  const int n_heads_log2_floor = 1 << static_cast<int>(floor(log2(p.head_num)));
  const float m0 = powf(2.0f, -(8.f) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(8.f / 2.0f) / n_heads_log2_floor);

#pragma omp parallel for collapse(3)
  for (int ibs = 0; ibs < p.batch_size; ++ibs)
    for (int ihn = 0; ihn < p.head_num; ++ihn)
      for (int i = 0; i < p.sl_q; ++i) {
        const auto ihkv = ihn / group_heads;
        const auto q_curr = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num + i * p.step_q_sl;
        const auto dst_curr = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num + i * p.step_dst_sl;

        const auto k_curr = p.K + ibs * p.step_k_bs + ihkv * p.step_k_head_num;
        const auto v_curr = p.V + ibs * p.step_v_bs + ihkv * p.step_v_head_num;

        const auto sl_diff = p.sl_kv - p.sl_q;
        const auto unmasked = is_causal ? sl_diff + i + 1 : p.sl_kv;
        const auto curr_row = std::unique_ptr<float[]>(new float[unmasked]);

        const auto alibi_ihn_m = !is_alibi                    ? 0.f
                                 : (ihn < n_heads_log2_floor) ? powf(m0, ihn + 1)
                                                              : powf(m1, 2 * (ihn - n_heads_log2_floor) + 1);

        // Q x K
        float row_max = -INFINITY;
        for (int j = 0; j < unmasked; ++j) {
          curr_row[j] = 0.f;
          for (int k = 0; k < p.head_size; ++k) {
            if (p.K_layout != ATTN_FWD_LAYOUT_PLAIN) {
              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto k_remain = k % ROWPACK;
              const auto k_block = k - k_remain;
              const auto k_value =
                  static_cast<float>(k_curr[j_block * p.step_k_sl + k_block * NTILE + j_remain * ROWPACK + k_remain]);
              const auto q_value =
                  IS_BF16_GEMM ? static_cast<float>(static_cast<bf16>(q_curr[k])) : static_cast<float>(q_curr[k]);
              curr_row[j] += k_value * q_value;
            } else if (IS_BF16_GEMM) {
              curr_row[j] += static_cast<float>(static_cast<bf16>(q_curr[k])) *  // TODO(Yi) fp16 acc
                             static_cast<float>(static_cast<bf16>(k_curr[j * p.step_k_sl + k * p.step_k_head_size]));
            } else {
              curr_row[j] += static_cast<float>(q_curr[k]) *  // TODO(Yi) fp16 acc
                             static_cast<float>(k_curr[j * p.step_k_sl + k * p.step_k_head_size]);
            }
          }
          curr_row[j] = curr_row[j] * p.QK_scale * p.Q_sc * p.K_sc + j * alibi_ihn_m;
          row_max = std::max(row_max, curr_row[j]);
        }

        // exp
        float exp_sum = 0.f;
        for (int j = 0; j < unmasked; ++j) {
          curr_row[j] = mha_exp_ref(curr_row[j] - row_max);
          exp_sum += curr_row[j];
        }

        // softmax
        if (std::is_same<V_T, int8_t>::value) {
          for (int j = 0; j < unmasked; ++j) curr_row[j] = roundf(curr_row[j] * UINT8_MAX) / UINT8_MAX / exp_sum;
        } else {
          for (int j = 0; j < unmasked; ++j) curr_row[j] /= exp_sum;
        }
        if (IS_BF16_GEMM) {
          for (int j = 0; j < unmasked; ++j) curr_row[j] = static_cast<float>(static_cast<bf16>(curr_row[j]));
        }

        // P x V
        for (int j = 0; j < p.head_size; ++j) {
          float dst_f32_val = 0.f;
          for (int k = 0; k < unmasked; ++k) {
            if (p.V_layout != ATTN_FWD_LAYOUT_PLAIN) {
              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto k_remain = k % ROWPACK;
              const auto k_block = k - k_remain;
              const auto v_value = static_cast<float>(
                  v_curr[j_block * p.step_v_head_size + k_block * NTILE + j_remain * ROWPACK + k_remain]);
              dst_f32_val += curr_row[k] * v_value;
            } else if (IS_BF16_GEMM) {
              dst_f32_val += curr_row[k] * static_cast<float>(static_cast<bf16>(v_curr[k * p.step_v_sl + j]));
            } else {
              dst_f32_val += curr_row[k] * static_cast<float>(v_curr[k * p.step_v_sl + j]);
            }
          }
          dst_curr[j] = static_cast<DST_T>(dst_f32_val * p.V_sc / p.dst_sc);
        }
      }
}
}  // namespace

void jblas_fusion_attn_bf16_forward(const attn_bf16_fwd_args_t* params) {
  return jblas_fusion_attn_forward(*reinterpret_cast<const attn_fwd_args_t<bf16, bf16, bf16, bf16>*>(params));
}

bool jblas_fusion_attn_fp32_fp16_fp16_fp32_support(const attn_shape_t* params) {
#if CompileBF16()
  GetCPUDevice();
  // TODO(Yi): check K V's layout
  return _cd->AMX_BF16();
#endif
  return false;
}
void jblas_fusion_attn_fp32_fp16_fp16_fp32_forward(const attn_fp32_fp16_fp16_fp32_fwd_args_t* params) {
  return jblas_fusion_attn_forward(*reinterpret_cast<const attn_fwd_args_t<float, fp16, fp16, float>*>(params));
  // return jblas_fusion_attn_forward_ref(*reinterpret_cast<const attn_fwd_args_t<float, fp16, fp16, float>*>(params));
}

bool jblas_fusion_attn_fp16_support(const attn_shape_t* params) {
#if CompileFP16()
  GetCPUDevice();
  // TODO(Yi): check K V's layout
  return _cd->AMX_BF16();
#endif
  return false;
}
void jblas_fusion_attn_fp16_forward(const attn_fp16_fwd_args_t* params) {
  return jblas_fusion_attn_forward<fp16, fp16, fp16, fp16>(
      *reinterpret_cast<const attn_fwd_args_t<fp16, fp16, fp16, fp16>*>(params));
}
void jblas_fusion_attn_int8_forward(const attn_int8_fwd_args_t* params) {
  return jblas_fusion_attn_forward<int8_t, int8_t, int8_t, int8_t>(
      *reinterpret_cast<const attn_fwd_args_t<int8_t, int8_t, int8_t, int8_t>*>(params));
}
size_t jblas_fusion_attn_workspace_size(const attn_shape_t* params) {
  const auto& p = *params;  // TODO(Yi): Better way to get tmp size?
  return size_t(omp_get_max_threads() * sizeof(float) * 16) * padto(padto(p.sl_kv, 48), 64);
}

bool jblas_reordered_attn_fp32_support(const attn_shape_t* params) {
#if CompileBF16()
  GetCPUDevice();
  // TODO(Yi): check K V's layout
  return _cd->AMX_BF16();
#endif
  return false;
}
// kv cache sizes in bytes per layer per batch per beam for;
void jblas_reordered_attn_fp32_batch_kv_info(const kv_shape_t* params, kv_cache_info_t* out) {
  // use bf16 for kv-cache
  const auto p = *params;
  out->k_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
  out->v_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;

  out->stride_k_head_size = sizeof(bf16) * 48;
  out->stride_k_sl = sizeof(bf16) * padto(static_cast<int>(p.head_size), 32);
  out->stride_k_head_num = out->stride_k_sl * padto(static_cast<int>(p.sl_kv_max), 48);
  out->k_bytes = out->stride_k_head_num * p.heads_kv;

  out->stride_v_sl = sizeof(bf16) * 48;
  out->stride_v_head_size = sizeof(bf16) * padto(static_cast<int>(p.sl_kv_max), 32);
  out->stride_v_head_num = out->stride_v_head_size * padto(static_cast<int>(p.head_size), 48);
  out->v_bytes = out->stride_v_head_num * p.heads_kv;
}

void jblas_reordered_attn_fp32_forward(const jblas_reordered_attn_fp32_fp32_fwd_args_t* params) {
  assert(params->K_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);
  assert(params->V_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);

  const attn_fwd_args_t<float, bf16, bf16, float> jblas_params = {
      /* .Q = */ params->Q,
      /* .K = */ reinterpret_cast<bf16*>(params->K),
      /* .V = */ reinterpret_cast<bf16*>(params->V),
      /* .dst = */ params->dst,
      /* .Q_sc = */ params->Q_sc,
      /* .K_sc = */ params->K_sc,
      /* .V_sc = */ params->V_sc,
      /* .dst_sc = */ params->dst_sc,
      /* .tmp = */ params->tmp,
      /* .QK_scale = */ params->QK_scale,
      /* .attn_flags = */ params->attn_flags,
      /* .batch_size = */ params->batch_size,
      /* .head_num = */ params->head_num,
      /* .heads_kv = */ params->heads_kv,
      /* .head_size = */ params->head_size,
      /* .sl_q = */ params->sl_q,
      /* .sl_kv = */ params->sl_kv,
      /* .Q_layout = */ params->Q_layout,
      /* .K_layout = */ params->K_layout,
      /* .V_layout = */ params->V_layout,
      /* .dst_layout = */ params->dst_layout,
      /* .step_q_bs = */ params->step_q_bs,
      /* .step_q_head_num = */ params->step_q_head_num,
      /* .step_q_sl = */ params->step_q_sl,
      /* .step_k_bs = */ static_cast<int>(params->stride_k_bs / sizeof(bf16)),
      /* .step_k_head_num = */ static_cast<int>(params->stride_k_head_num / sizeof(bf16)),
      /* .step_k_sl = */ static_cast<int>(params->stride_k_sl / sizeof(bf16)),
      /* .step_k_head_size = */ 48,
      /* .step_v_bs = */ static_cast<int>(params->stride_v_bs / sizeof(bf16)),
      /* .step_v_head_num = */ static_cast<int>(params->stride_v_head_num / sizeof(bf16)),
      /* .step_v_sl = */ 48,
      /* .step_v_head_size = */ static_cast<int>(params->stride_v_head_size / sizeof(bf16)),
      /* .step_dst_bs = */ params->step_dst_bs,
      /* .step_dst_head_num = */ params->step_dst_head_num,
      /* .step_dst_sl = */ params->step_dst_sl,
  };
  return jblas_fusion_attn_forward<float, bf16, bf16, float>(jblas_params);
}

template <bool zero_padding>
void jblas_reordered_attn_fp32_update_k_(const jblas_fusion_attn_fp32_update_kv_args_t* params) {
  const auto p = *params;
  NE_ASSERT(p.step_head_size == 1);
  const auto pad_headsize = padto(p.head_size, 32);
  const auto pad_seq_max = padto(p.seq_max, 48);
  const auto cache_step_head_num = pad_headsize * pad_seq_max;
  const auto cache_step_bs = p.heads_kv * cache_step_head_num;
  GetCPUDevice();
  const bool use_jit = _cd->AVX512_BF16() && (p.seq_off == 0) && zero_padding;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < p.batch_size; ++ibs) {
    for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
      const auto dst = reinterpret_cast<bf16*>(p.cache) + ibs * cache_step_bs + ihn * cache_step_head_num;
      const auto src = p.src + ibs * p.step_bs + ihn * p.step_head_num;

      if (use_jit) {
        jblas::kernel::jit::PaddingTransInterleaveCvt::forward<48>(  //
            src, dst, p.seq_size, p.head_size, padto(p.seq_size, 48), padto(p.head_size, 32), p.step_seq, pad_headsize);
      } else {
        for (int i = 0; i < p.seq_size; ++i) {      // QK_GEMM should not require 0-padding on seq_kv (i.e. N-dim)
          for (int j = 0; j < pad_headsize; ++j) {  // K-dim padding for QK_GEMM
            const auto i_dst = p.seq_off + i;
            const auto ii = i_dst % 48;
            const auto i_blk = i_dst - ii;
            const auto jj = j % 2;
            const auto j_blk = j - jj;
            if constexpr (zero_padding) {
              dst[i_blk * pad_headsize + ii * 2 + j_blk * 48 + jj] =
                  j < p.head_size ? static_cast<bf16>(src[i * p.step_seq + j]) : bf16(0);
            } else {
              if (j < p.head_size)
                dst[i_blk * pad_headsize + ii * 2 + j_blk * 48 + jj] = static_cast<bf16>(src[i * p.step_seq + j]);
            }
          }
        }
      }
    }
  }
}
void jblas_reordered_attn_fp32_update_k(const jblas_fusion_attn_fp32_update_kv_args_t* params) {
  return params->no_zeroing ? jblas_reordered_attn_fp32_update_k_<false>(params)
                            : jblas_reordered_attn_fp32_update_k_<true>(params);
}

template <bool zero_padding>
void jblas_reordered_attn_fp32_update_v_(const jblas_fusion_attn_fp32_update_kv_args_t* params) {
  const auto p = *params;
  NE_ASSERT(p.step_head_size == 1);
  const auto pad_headsize = padto(p.head_size, 48);
  const auto pad_seq_max = padto(p.seq_max, 32);
  const auto step_cache_head_num = pad_headsize * pad_seq_max;
  const auto step_cache_bs = p.heads_kv * step_cache_head_num;
  GetCPUDevice();
  const bool use_jit = _cd->AVX512_BF16() && (p.seq_off == 0) && zero_padding;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < p.batch_size; ++ibs) {
    for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
      const auto dst = reinterpret_cast<bf16*>(p.cache) + ibs * step_cache_bs + ihn * step_cache_head_num;
      const auto src = p.src + ibs * p.step_bs + ihn * p.step_head_num;
      if (use_jit) {
        jblas::kernel::jit::PaddingInterleaveCvt::forward<48>(  //
            src, dst, p.seq_size, p.head_size, padto(p.seq_size, 32), padto(p.head_size, 48), p.step_seq, pad_seq_max);
      } else {
        for (int i = 0; i < padto(p.seq_off + p.seq_size, 32) - p.seq_off; ++i) {  // K-dim padding for PV_GEMM
          for (int j = 0; j < p.head_size; ++j) {  // PV_GEMM should't require 0-padding on head_size (i.e. N-dim)
            const auto i_dst = p.seq_off + i;
            const auto ii = i_dst % 2;
            const auto i_blk = i_dst - ii;
            const auto jj = j % 48;
            const auto j_blk = j - jj;
            if constexpr (zero_padding) {
              dst[i_blk * 48 + ii + j_blk * pad_seq_max + jj * 2] =
                  i < p.seq_size ? static_cast<bf16>(src[i * p.step_seq + j]) : bf16(0);
            } else {
              if (i < p.seq_size)
                dst[i_blk * 48 + ii + j_blk * pad_seq_max + jj * 2] = static_cast<bf16>(src[i * p.step_seq + j]);
            }
          }
        }
      }
    }
  }
}
void jblas_reordered_attn_fp32_update_v(const jblas_fusion_attn_fp32_update_kv_args_t* params) {
  return params->no_zeroing ? jblas_reordered_attn_fp32_update_v_<false>(params)
                            : jblas_reordered_attn_fp32_update_v_<true>(params);
}

void jblas_reordered_attn_fp32_shift_rope_k(char* cache, const ne_fp16_t* cossin, int batch_size, int heads_kv,
                                            int head_size, int seq_max, int seq_keep) {
  const auto pad_headsize = padto(head_size, 32);
  const auto pad_seq_max = padto(seq_max, 48);
  const auto cache_step_head_num = pad_headsize * pad_seq_max;
  const auto cache_step_bs = heads_kv * cache_step_head_num;

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < batch_size; ++ibs)
    for (int ihn = 0; ihn < heads_kv; ++ihn) {
      const auto src = reinterpret_cast<bf16*>(cache) + ibs * cache_step_bs + ihn * cache_step_head_num;
      jblas::kernel::jit::CScaleInterleavedBF16FP16::forward<48>(  // NOLINT [build/include_what_you_use]
          src, reinterpret_cast<const fp16*>(cossin), head_size, pad_seq_max, pad_headsize, seq_keep);
    }
}

template <bool zero_padding>
void jblas_fusion_attn_fp32_batch_cpy_k_(const jblas_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  static constexpr auto N_TILE = 48;
  static constexpr auto K_TILE = 32;
  static constexpr auto K_PACK = 2;
  const auto p = *params;
  const auto pad_headsize = padto(p.head_size, K_TILE);
  const auto pad_seq_max = padto(p.seq_max, N_TILE);
  const auto step_head_num = pad_headsize * pad_seq_max;

  const auto seq_unaligned = std::min(padto(p.seq_off, N_TILE) - p.seq_off, p.seq_size);
  const auto size_aligned_cpy = pad_headsize * (padto(p.seq_off + p.seq_size, N_TILE) - padto(p.seq_off, N_TILE));
#pragma omp parallel for
  for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
    const auto dst = reinterpret_cast<bf16*>(p.dst) + ihn * step_head_num;
    const auto src = reinterpret_cast<bf16*>(p.src) + ihn * step_head_num;

    if (seq_unaligned) {
      const auto ii = p.seq_off % N_TILE;
      const auto i_blk = p.seq_off - ii;
      const auto off = i_blk * pad_headsize + ii * K_PACK;
      for (int j = 0; j < pad_headsize; j += K_PACK) {  // K-dim padding for QK_GEMM
        memcpy(dst + off + j * N_TILE, src + off + j * N_TILE, sizeof(bf16) * K_PACK * seq_unaligned);
      }
    }
    if constexpr (zero_padding) {
      if (size_aligned_cpy) {
        const auto off = padto(p.seq_off, N_TILE) * pad_headsize;
        memcpy(dst + off, src + off, sizeof(bf16) * size_aligned_cpy);
      }
    } else {
      assert(("Unimplemented!", false));
    }
  }
}
void jblas_fusion_attn_fp32_batch_cpy_k(const jblas_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  return params->no_zeroing ? jblas_fusion_attn_fp32_batch_cpy_k_<false>(params)
                            : jblas_fusion_attn_fp32_batch_cpy_k_<true>(params);
}

template <bool zero_padding>
void jblas_fusion_attn_fp32_batch_cpy_v_(const jblas_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  static constexpr auto N_TILE = 48;
  static constexpr auto K_TILE = 32;
  static constexpr auto K_PACK = 2;
  const auto p = *params;
  const auto pad_headsize = padto(p.head_size, N_TILE);
  const auto pad_seq_max = padto(p.seq_max, K_TILE);
  const auto step_head_num = pad_headsize * pad_seq_max;

  const auto seq_off_aligned = padto(p.seq_off, K_PACK);
  const auto seq_end_aligned = padto(p.seq_off + p.seq_size, K_TILE);
  const auto seq_size_aligned = seq_end_aligned - seq_off_aligned;
#pragma omp parallel for collapse(2)
  for (int ihn = 0; ihn < p.heads_kv; ++ihn) {
    for (int j = 0; j < p.head_size; j += N_TILE) {
      const auto dst = reinterpret_cast<bf16*>(p.dst) + ihn * step_head_num + pad_seq_max * j;
      const auto src = reinterpret_cast<bf16*>(p.src) + ihn * step_head_num + pad_seq_max * j;
      if (p.seq_off != seq_off_aligned) {  // seq_size_unaligen must be 0 or 1 as K_PACK = 2
        const auto off = (seq_off_aligned - K_PACK) * N_TILE + 1;
        for (int jj = 0; jj < N_TILE; ++jj) dst[off + jj * K_PACK] = src[off + jj * K_PACK];
      }
      if constexpr (zero_padding) {
        if (seq_off_aligned != seq_end_aligned) {
          const auto off = seq_off_aligned * N_TILE;
          memcpy(dst + off, src + off, sizeof(bf16) * N_TILE * seq_size_aligned);
        }
      } else {
        assert(("Unimplemented!", false));
      }
    }
  }
}
void jblas_fusion_attn_fp32_batch_cpy_v(const jblas_fusion_attn_fp32_batch_cpy_kv_args_t* params) {
  return params->no_zeroing ? jblas_fusion_attn_fp32_batch_cpy_v_<false>(params)
                            : jblas_fusion_attn_fp32_batch_cpy_v_<true>(params);
}

#ifdef __GNUC__
#pragma GCC pop_options
#endif

#ifdef NE_TESTS
#define CheckISA(ISA)                         \
  {                                           \
    GetCPUDevice();                           \
    if (!_cd->ISA()) {                        \
      printf("Wrong Device ISA: " #ISA "\n"); \
      return;                                 \
    }                                         \
  }

namespace {
bool ret_ok = true;

class TestMhaDese {
 public:
  TestMhaDese() {
    printf("Test suit: %s\n", __FUNCTION__);
    CheckISA(AMX_BF16);
    GetCPUDevice();
    ne_jblas::ne_threading::get()->set_threads(std::min(_cd->getThreads(), omp_get_max_threads()));
    jblas::utils::request_perm_xtile_data();

#if CompileFP16()
    ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE);
    ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE);
    ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE);
    ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 63, 63}, NE_ATTN_FLAG_NONE);
    ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE);
    ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL);

    ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<fp16, fp16, fp16, fp16>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<fp16, fp16, fp16, fp16>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<fp16, fp16, fp16, fp16>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<fp16, fp16, fp16, fp16>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, true);

    ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<float, fp16, fp16, float>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<float, fp16, fp16, float>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, true);
    ret_ok &= test_case<float, fp16, fp16, float>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, true);
#endif

    const auto s8layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK4;
    ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, false, s8layout);
    ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, false, s8layout);
    ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, false, s8layout);
    ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, false, s8layout);
    ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, false, s8layout);
    ret_ok &= test_case<int8_t, int8_t, int8_t, int8_t>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, false, s8layout);

    const auto BA48b2a = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
    ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 32, 128, 64}, NE_ATTN_FLAG_NONE, false, BA48b2a, 1e-3f);
    ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 32, 64, 128}, NE_ATTN_FLAG_NONE, false, BA48b2a, 1e-3f);
    ret_ok &= test_case<float, bf16, bf16, float>({2, 5, 5, 80, 128, 77}, NE_ATTN_FLAG_NONE, false, BA48b2a, 1e-3f);
    ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 256, 63, 63}, NE_ATTN_FLAG_NONE, false, BA48b2a, 1e-3f);
    ret_ok &= test_case<float, bf16, bf16, float>({3, 4, 4, 256, 1, 384}, NE_ATTN_FLAG_NONE, false, BA48b2a, 1e-3f);
    ret_ok &= test_case<float, bf16, bf16, float>({1, 1, 1, 64, 64, 64}, NE_ATTN_FLAG_IS_CAUSAL, false, BA48b2a, 1e-3f);

    ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 32, 128, 64}, 64, NE_ATTN_FLAG_NONE);
    ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 5, 32, 64, 128}, 256, NE_ATTN_FLAG_NONE);
    ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 5, 80, 128, 77}, 256, NE_ATTN_FLAG_NONE);
    ret_ok &= test_reorder_pipe<float, float, float, float>({2, 5, 1, 80, 128, 77}, 256, NE_ATTN_FLAG_NONE);
    ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 256, 63, 63}, 256, NE_ATTN_FLAG_NONE);
    ret_ok &= test_reorder_pipe<float, float, float, float>({3, 4, 4, 256, 1, 384}, 384, NE_ATTN_FLAG_NONE);
    ret_ok &= test_reorder_pipe<float, float, float, float>({3, 4, 2, 256, 1, 384}, 384, NE_ATTN_FLAG_NONE);
    ret_ok &= test_reorder_pipe<float, float, float, float>({1, 1, 1, 64, 64, 64}, 128, NE_ATTN_FLAG_IS_CAUSAL);
    ret_ok &= test_reorder_pipe<float, float, float, float>({1, 8, 8, 64, 64, 64}, 128,
                                                            NE_ATTN_FLAG_IS_CAUSAL | NE_ATTN_FLAG_IS_ALIBI8);
    printf("Test suit done: %s\n", __FUNCTION__);
  }

  template <class T>
  static constexpr float init_min_val = std::is_same<T, int8_t>::value    ? -127.f
                                        : std::is_same<T, uint8_t>::value ? 0.f
                                                                          : -1.f;
  template <class T>
  static constexpr float init_max_val = std::is_same<T, int8_t>::value    ? 127.f
                                        : std::is_same<T, uint8_t>::value ? 255.f
                                                                          : 1.f;
  template <class T>
  static constexpr float init_scale_val = 1.f / init_max_val<T>;

#ifdef _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

  template <class Q_T, class K_T, class V_T, class DST_T>
  bool test_case(const attn_shape_t& s, ne_attn_flags_t flags, bool k_trans = false,
                 ATTN_FWD_LAYOUT kv_layout = ATTN_FWD_LAYOUT_PLAIN, float eps = 1e-2f) {
    assert(kv_layout == ATTN_FWD_LAYOUT_PLAIN || !k_trans);
    const auto batch_size = s.batch_size;
    const auto head_num = s.head_num;
    const auto heads_kv = s.heads_kv;
    const auto head_size = s.head_size;
    const auto sl_q = s.sl_q;
    const auto sl_kv = s.sl_kv;
    assert(("GQA not supported!", s.head_num == s.heads_kv));

    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("bs_%d hn_%d hkv_%d hs_%d sl_q_%d sk_kv_%d %s %s\n", batch_size, head_num, heads_kv, head_size, sl_q, sl_kv,
           flags & NE_ATTN_FLAG_IS_CAUSAL ? "maksed" : "unmask", flags & NE_ATTN_FLAG_IS_ALIBI8 ? "alibi8" : "");

    const auto NTILE = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                                                                       : 0;
    const auto ROWPACK = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                                                                         : 0;
    const auto ROWPAD = ROWPACK * 16;
    const auto k_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, ROWPAD) : head_size;
    const auto k_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, NTILE) : sl_kv;
    const auto v_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, ROWPAD) : sl_kv;
    const auto v_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, NTILE) : head_size;

    std::vector<Q_T> src_q(batch_size * head_num * sl_q * head_size);
    std::vector<K_T> src_k(batch_size * heads_kv * k_rows_pad * k_cols_pad);
    std::vector<V_T> src_v(batch_size * heads_kv * v_rows_pad * v_cols_pad);
    std::vector<DST_T> dst(batch_size * head_num * sl_q * head_size);
    std::vector<DST_T> ref(batch_size * head_num * sl_q * head_size);  // reference result
    std::vector<char> tmp(jblas_fusion_attn_workspace_size(&s));

    // init vector
    static std::mt19937 rng(1);
    std::uniform_int_distribution<> dist;
    init_vector(&src_q, init_min_val<Q_T>, init_max_val<Q_T>, dist(rng));
    init_vector(&src_k, init_min_val<K_T>, init_max_val<K_T>, dist(rng));
    init_vector(&src_v, init_min_val<V_T>, init_max_val<V_T>, dist(rng));

    // pad0 for padded layouts
    if (kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4 || kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2) {
#pragma omp parallel for collapse(2)
      for (int ibs = 0; ibs < batch_size; ++ibs) {
        for (int ihn = 0; ihn < heads_kv; ++ihn) {
          // K
          const auto k_off = (ibs * heads_kv + ihn) * k_rows_pad * k_cols_pad;
          for (int i = 0; i < k_rows_pad; ++i) {
            for (int j = 0; j < k_cols_pad; ++j) {
              if (i < head_size && j < sl_kv) continue;

              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto i_remain = i % ROWPACK;
              const auto i_block = i - i_remain;
              src_k[k_off + j_block * k_rows_pad + i_block * NTILE + j_remain * ROWPACK + i_remain] = K_T(0);
            }
          }
          // V
          const auto v_off = (ibs * heads_kv + ihn) * v_rows_pad * v_cols_pad;
          for (int i = 0; i < v_rows_pad; ++i) {
            for (int j = 0; j < v_cols_pad; ++j) {
              if (i < sl_kv && j < head_size) continue;

              const auto j_remain = j % NTILE;
              const auto j_block = j - j_remain;
              const auto i_remain = i % ROWPACK;
              const auto i_block = i - i_remain;
              src_v[v_off + j_block * v_rows_pad + i_block * NTILE + j_remain * ROWPACK + i_remain] = V_T(0);
            }
          }
        }
      }
    }

    attn_fwd_args_t<Q_T, K_T, V_T, DST_T> args{
        /* .Q = */ src_q.data(),
        /* .K = */ src_k.data(),
        /* .V = */ src_v.data(),
        /* .dst = */ ref.data(),
        /* .Q_sc = */ init_scale_val<Q_T>,
        /* .K_sc = */ init_scale_val<K_T>,
        /* .V_sc = */ init_scale_val<V_T>,
        /* .dst_sc = */ init_scale_val<V_T>,
        /* .tmp = */ tmp.data(),
        /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
        /* .attn_flags = */ flags,
        /* .batch_size = */ batch_size,
        /* .head_num = */ head_num,
        /* .heads_kv = */ heads_kv,
        /* .head_size = */ head_size,
        /* .sl_q = */ sl_q,
        /* .sl_kv = */ sl_kv,
        /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .K_layout = */ kv_layout,
        /* .V_layout = */ kv_layout,
        /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .step_q_bs = */ sl_q * head_num * head_size,
        /* .step_q_head_num = */ head_size,
        /* .step_q_sl = */ head_num * head_size,
        /* .step_k_bs = */ sl_kv * heads_kv * head_size,
        /* .step_k_head_num = */ k_trans ? head_size * sl_kv : head_size,
        /* .step_k_sl = */ k_trans ? 1 : heads_kv * head_size,
        /* .step_k_head_size = */ k_trans ? sl_kv : 1,
        /* .step_v_bs = */ sl_kv * heads_kv * head_size,
        /* .step_v_head_num = */ head_size,
        /* .step_v_sl = */ heads_kv * head_size,
        /* .step_v_head_size = */ 1,
        /* .step_dst_bs = */ sl_q * head_num * head_size,
        /* .step_dst_head_num = */ head_size,
        /* .step_dst_sl = */ head_num * head_size,
    };
    if (kv_layout != ATTN_FWD_LAYOUT_PLAIN) {
      args.step_k_bs = heads_kv * k_rows_pad * k_cols_pad;
      args.step_k_head_num = k_rows_pad * k_cols_pad;
      args.step_k_sl = k_rows_pad;
      args.step_k_head_size = NTILE;
      args.step_v_bs = heads_kv * v_rows_pad * v_cols_pad;
      args.step_v_head_num = v_rows_pad * v_cols_pad;
      args.step_v_sl = NTILE;
      args.step_v_head_size = v_rows_pad;
    }

    jblas_fusion_attn_forward_ref(args);

    args.dst = dst.data();
    jblas_fusion_attn_forward(args);

    // Check result
    return compare_data(dst.data(), ref.data(), dst.size(), eps);
  }

  template <class Q_T, class K_T, class V_T, class DST_T>
  bool test_reorder_pipe(const attn_shape_t& s, int sl_kv_max, ne_attn_flags_t flags) {
    const auto batch_size = s.batch_size;
    const auto head_num = s.head_num;
    const auto heads_kv = s.heads_kv;
    const auto head_size = s.head_size;
    const auto sl_q = s.sl_q;
    const auto sl_kv = s.sl_kv;
    assert(("head_num must be a multiple of heads_kv!", head_num % heads_kv == 0));

    printf("\ntest_case: %s\t", __PRETTY_FUNCTION__);
    printf("bs_%d hn_%d hkv_%d hs_%d sl_q_%d sk_kv_%d %s %s\n", batch_size, head_num, heads_kv, head_size, sl_q, sl_kv,
           flags & NE_ATTN_FLAG_IS_CAUSAL ? "maksed" : "unmask", flags & NE_ATTN_FLAG_IS_ALIBI8 ? "alibi8" : "");

    assert(sl_kv_max >= sl_kv);

    kv_shape_t kv_shape = {
        /* .heads_kv */ static_cast<uint32_t>(heads_kv),
        /* .head_size */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max */ static_cast<uint32_t>(sl_kv_max),
    };
    kv_cache_info_t kv_cache_info;
    jblas_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
    assert(kv_cache_info.k_layout >= kv_cache_info.v_layout);
    const auto kv_layout = kv_cache_info.k_layout;
    const auto NTILE = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 48
                       : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 48
                                                                       : 0;
    const auto ROWPACK = kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK4   ? 4
                         : kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2 ? 2
                                                                         : 0;
    const auto ROWPAD = ROWPACK * 16;
    const auto k_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, ROWPAD) : head_size;
    const auto k_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, NTILE) : sl_kv;
    const auto v_rows_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(sl_kv, ROWPAD) : sl_kv;
    const auto v_cols_pad = kv_layout != ATTN_FWD_LAYOUT_PLAIN ? padto(head_size, NTILE) : head_size;

    std::vector<Q_T> src_q(batch_size * head_num * sl_q * head_size);
    std::vector<K_T> src_k(batch_size * heads_kv * sl_kv * head_size);
    std::vector<V_T> src_v(batch_size * heads_kv * sl_kv * head_size);
    std::vector<char> k_cache(batch_size * kv_cache_info.k_bytes);
    std::vector<char> v_cache(batch_size * kv_cache_info.v_bytes);
    std::vector<DST_T> dst(batch_size * head_num * sl_q * head_size);
    std::vector<DST_T> ref(batch_size * head_num * sl_q * head_size);  // reference result
    std::vector<char> tmp(jblas_fusion_attn_workspace_size(&s));

    // init vector
    static std::mt19937 rng(1);
    std::uniform_int_distribution<> dist;
    init_vector(&src_q, init_min_val<Q_T>, init_max_val<Q_T>, dist(rng));
    init_vector(&src_k, init_min_val<K_T>, init_max_val<K_T>, dist(rng));
    init_vector(&src_v, init_min_val<V_T>, init_max_val<V_T>, dist(rng));

    // undefined values
    init_vector(&k_cache, INT8_MIN, INT8_MAX, dist(rng));
    init_vector(&v_cache, INT8_MIN, INT8_MAX, dist(rng));

    int step_src_k_bs = sl_kv * heads_kv * head_size;
    int step_src_k_head_num = head_size;
    int step_src_k_sl = heads_kv * head_size;
    int step_src_k_head_size = 1;
    int step_src_v_bs = sl_kv * heads_kv * head_size;
    int step_src_v_head_num = head_size;
    int step_src_v_sl = heads_kv * head_size;
    int step_src_v_head_size = 1;
    attn_fwd_args_t<Q_T, K_T, V_T, DST_T> ref_args{
        /* .Q = */ src_q.data(),
        /* .K = */ src_k.data(),
        /* .V = */ src_v.data(),
        /* .dst = */ ref.data(),
        /* .Q_sc = */ init_scale_val<Q_T>,
        /* .K_sc = */ init_scale_val<K_T>,
        /* .V_sc = */ init_scale_val<V_T>,
        /* .dst_sc = */ init_scale_val<V_T>,
        /* .tmp = */ tmp.data(),
        /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
        /* .attn_flags = */ flags,
        /* .batch_size = */ batch_size,
        /* .head_num = */ head_num,
        /* .heads_kv = */ heads_kv,
        /* .head_size = */ head_size,
        /* .sl_q = */ sl_q,
        /* .sl_kv = */ sl_kv,
        /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .K_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .V_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
        /* .step_q_bs = */ sl_q * head_num * head_size,
        /* .step_q_head_num = */ head_size,
        /* .step_q_sl = */ head_num * head_size,

        /* .step_k_bs = */ step_src_k_bs,
        /* .step_k_head_num = */ step_src_k_head_num,
        /* .step_k_sl = */ step_src_k_sl,
        /* .step_k_head_size = */ step_src_k_head_size,
        /* .step_v_bs = */ step_src_v_bs,
        /* .step_v_head_num = */ step_src_v_head_num,
        /* .step_v_sl = */ step_src_v_sl,
        /* .step_v_head_size = */ step_src_v_head_size,

        /* .step_dst_bs = */ sl_q * head_num * head_size,
        /* .step_dst_head_num = */ head_size,
        /* .step_dst_sl = */ head_num * head_size,
    };
    jblas_fusion_attn_forward_ref(ref_args);

    if (std::is_same<std::tuple<Q_T, K_T, V_T, DST_T>, std::tuple<float, float, float, float>>::value) {
      assert(kv_layout == ATTN_FWD_LAYOUT_NTILE48_ROWPACK2);
      // for testing, first reorder sl_kv - 1 and than concat the last 1 line
      const auto seq_size_first = sl_kv - 1;
      const auto seq_size_next = 1;
      jblas_fusion_attn_fp32_update_kv_args_t update_k_args = {
          /* .src = */ src_k.data(),
          /* .cache = */ k_cache.data(),
          /* .batch_size = */ batch_size,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .seq_off = */ 0,
          /* .seq_size = */ seq_size_first,
          /* .seq_max = */ sl_kv_max,
          /* .step_bs = */ step_src_k_bs,
          /* .step_head_num = */ step_src_k_head_num,
          /* .step_seq = */ step_src_k_sl,
          /* .step_head_size = */ step_src_k_head_size,
      };
      jblas_reordered_attn_fp32_update_k(&update_k_args);

      jblas_fusion_attn_fp32_update_kv_args_t update_v_args = {
          /* .src = */ src_v.data(),
          /* .cache = */ v_cache.data(),
          /* .batch_size = */ batch_size,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .seq_off = */ 0,
          /* .seq_size = */ seq_size_first,
          /* .seq_max = */ sl_kv_max,
          /* .step_bs = */ step_src_v_bs,
          /* .step_head_num = */ step_src_v_head_num,
          /* .step_seq = */ step_src_v_sl,
          /* .step_head_size = */ step_src_v_head_size,
      };
      jblas_reordered_attn_fp32_update_v(&update_v_args);

      update_k_args.seq_off = seq_size_first;
      update_k_args.seq_size = seq_size_next;
      update_k_args.src = src_k.data() + seq_size_first * step_src_k_sl;
      jblas_reordered_attn_fp32_update_k(&update_k_args);

      update_v_args.seq_off = seq_size_first;
      update_v_args.seq_size = seq_size_next;
      update_v_args.src = src_v.data() + seq_size_first * step_src_v_sl;
      jblas_reordered_attn_fp32_update_v(&update_v_args);

      jblas_reordered_attn_fp32_fp32_fwd_args_t kern_args{
          /* .Q = */ reinterpret_cast<float*>(src_q.data()),
          /* .K = */ k_cache.data(),
          /* .V = */ v_cache.data(),
          /* .dst = */ reinterpret_cast<float*>(dst.data()),
          /* .Q_sc = */ init_scale_val<Q_T>,
          /* .K_sc = */ init_scale_val<K_T>,
          /* .V_sc = */ init_scale_val<V_T>,
          /* .dst_sc = */ init_scale_val<V_T>,
          /* .tmp = */ tmp.data(),
          /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(head_size)),
          /* .attn_flags = */ flags,
          /* .batch_size = */ batch_size,
          /* .head_num = */ head_num,
          /* .heads_kv = */ heads_kv,
          /* .head_size = */ head_size,
          /* .sl_q = */ sl_q,
          /* .sl_kv = */ sl_kv,
          /* .Q_layout = */ ATTN_FWD_LAYOUT_PLAIN,
          /* .K_layout = */ ATTN_FWD_LAYOUT_NTILE48_ROWPACK2,
          /* .V_layout = */ ATTN_FWD_LAYOUT_NTILE48_ROWPACK2,
          /* .dst_layout = */ ATTN_FWD_LAYOUT_PLAIN,
          /* .step_q_bs = */ sl_q * head_num * head_size,
          /* .step_q_head_num = */ head_size,
          /* .step_q_sl = */ head_num * head_size,

          /* .stride_k_bs = */ static_cast<int>(kv_cache_info.k_bytes),
          /* .stride_k_head_num = */ kv_cache_info.stride_k_head_num,
          /* .stride_k_sl = */ kv_cache_info.stride_k_sl,
          /* .stride_k_head_size = */ kv_cache_info.stride_k_head_size,
          /* .stride_v_bs = */ static_cast<int>(kv_cache_info.v_bytes),
          /* .stride_v_head_num = */ kv_cache_info.stride_v_head_num,
          /* .stride_v_sl = */ kv_cache_info.stride_v_sl,
          /* .stride_v_head_size = */ kv_cache_info.stride_v_head_size,

          /* .step_dst_bs = */ sl_q * head_num * head_size,
          /* .step_dst_head_num = */ head_size,
          /* .step_dst_sl = */ head_num * head_size,
      };
      jblas_reordered_attn_fp32_forward(&kern_args);
    }

    // Check result
    return compare_data(dst.data(), ref.data(), dst.size(), 1e-2f);
  }
};
static const TestMhaDese inst_;

}  // namespace

int main() {
  printf("NE_TESTS: mha_dense ");
  printf(ret_ok ? "OK\n" : "FAILED\n");
  return ret_ok ? 0 : -1;
}
#endif
