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

#include <cmath>
#ifdef NE_TESTS
#include <memory>
#include <random>

#include "layers/ne_test_layers_utils.hpp"
#endif

#include <immintrin.h>

#include "core/data_types.h"
#include "jblas/jit_blas_gemm.h"
#include "jblas/jit_blas_prologue.h"
#include "jblas/jit_blas_utils.h"
#include "jblas/jit_blas_wrapper.h"

#define MHA_2ND_EXP 1
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
using namespace jblas::utils;

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
struct attn_fwd_args_t {
  Q_T* Q;
  K_T* K;
  V_T* V;
  DST_T* dst;
  char* tmp;
  float QK_scale;
  bool is_causal;
  int batch_size, head_num, head_size, sl_q, sl_kv;
  int step_q_bs, step_q_head_num, step_q_sl;
  int step_k_bs, step_k_head_num, step_k_sl;
  int step_v_bs, step_v_head_num, step_v_sl;
  int step_dst_bs, step_dst_head_num, step_dst_sl;
};
using jblas_attn_fp32_fp16_fp16_fp32_fwd_args_t = attn_fwd_args_t<float, fp16, fp16, float>;

/**
 * @brief An Epilogue that scale the fp32 result, performing exp, accumulating sum of each line of exp, and storing exp
 * as bf16 results
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
  };

  static inline __m512 snd_order_poly_exp(__m512 z, __m512 f, const float c[]) {
    const auto c0 = _mm512_set1_ps(c[0]);
    const auto c1 = _mm512_set1_ps(c[1]);
    const auto c2 = _mm512_set1_ps(c[2]);

    auto y = _mm512_fmadd_ps(_mm512_fmadd_ps(f, c0, c1), f, c2);  // auto y = (f * c0 + c1) * f + c2;
    auto exp = _mm512_scalef_ps(y, z);

    return exp;
  }

  static inline __m512 exp_ps_0_1(__m512 x) {
    static const float v_log2e = std::log2(std::exp(1.f));
    const auto log2e = _mm512_set1_ps(v_log2e);
    const float _c[] = {0.240226507f, 0.452920674f, 0.713483036f};

    auto x1 = _mm512_fmadd_ps(x, log2e, _mm512_set1_ps(.5f));  // auto x1 = x * log2e + _mm512_set1_ps(.5f);
    auto z = _mm512_floor_ps(x1);
    auto f = _mm512_sub_ps(x1, z);  // auto f = x1 - z;

    return snd_order_poly_exp(z, f, _c);
  }

  JBLAS_CODE forward(const float* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p) const {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto dst_sum = p.dst_sum + M_offset;
#if MHA_2ND_EXP && CompileBF16()
    const auto v_scale = _mm512_set1_ps(p.scale);
    for (int i = 0; i < M; ++i) {
      const auto N_unmasked =
          std::min(N, p.causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + p.causal_offset + 1);

      const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
      int j = 0;
      auto v_sum = _mm512_setzero_ps();
      for (; j < N_unmasked - 15; j += 16) {
        const auto v_exp = exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_loadu_ps(src + i * src_step + j)));
        static_assert(std::is_same<T_DST, bf16>::value, "bf16 support only");
        v_sum = _mm512_add_ps(v_sum, v_exp);
        _mm256_storeu_epi16(dst + i * p.ld_dst + j, (__m256i)_mm512_cvtneps_pbh(v_exp));
      }
      if (j < N_unmasked) {
        const auto v_exp = exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_maskz_loadu_ps(v_mask, src + i * src_step + j)));
        static_assert(std::is_same<T_DST, bf16>::value, "bf16 support only");
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
template <JBLAS_ISA ISA_T, typename T_DST>
class ScaleWriteBackFp32 {
 public:
  struct Param {
    const float* scale;
    T_DST* dst;
    int ld_dst;
  };

  JBLAS_CODE forward(const float* src, const int src_step, const int M_offset, const int N_offset, const int M,
                     const int N, const Param& p) {
    const auto dst = p.dst + M_offset * p.ld_dst + N_offset;
    const auto scale = p.scale + M_offset;
    // TODO(Yi): high performance implementation
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)  //
        dst[i * p.ld_dst + j] = static_cast<T_DST>(src[i * src_step + j] * scale[i]);

    return JblasSuccess;
  }
};
template <JBLAS_ISA ISA_T>
using ScaleWriteBackFp32Bf16 = ScaleWriteBackFp32<ISA_T, bf16>;
template <JBLAS_ISA ISA_T>
using ScaleWriteBackFp32Fp32 = ScaleWriteBackFp32<ISA_T, float>;

/**
 * @brief PackedWeight(Default) with batch
 *
 * @tparam T element type of the weight
 */
template <typename T>
class PackedWeightBatch : public jblas::prologue::gemm::StorageWeight {
 public:
  explicit PackedWeightBatch(jblas::gemm::GemmCoreType _type)
      : jblas::prologue::gemm::StorageWeight(_type), mBatch(0) {}

  void resize(int NPad, int KPad) = delete;
  void resize(int NPad, int KPad, int num_batch) {
    mNPad = NPad;
    mKPad = KPad;
    mBatch = num_batch;

    mBuffer.resize((size_t)mBatch * NPad * KPad * jblas::gemm::getWeightSize(mCoreType));
    mRawPtr = mBuffer.data();
    mWPtr = getPtr<T>();
    mWSize = getSize<T>();
  }

  int mBatch;
  T* mWPtr;
  size_t mWSize;

 protected:
  virtual size_t getDataSerializedSize() override {
    return jblas::prologue::gemm::StorageWeight::getSerializedSize() + sizeof(mBatch);
  }
};

/**
 * @brief An weight Prologue that Packs transposed Bf16 weight; optimized for runtime packing. It is the base type of
 * that for transposed / non-transposed source
 */
template <class GemmCore_T, JBLAS_ISA ISA_T, bool IsTrans, typename T_SRC = typename GemmCore_T::BType>
class WeightPackBatchBf16Base {
 public:
  using WType = typename GemmCore_T::BType;  // weight type
  using PW_T = PackedWeightBatch<WType>;     // packed weight type
  using Parallel = parallel::Parallel2DRowMajor;

  struct Param {
    const jblas::prologue::PackedWeight* packedW;
  };

  // additional parameter to pack weight at runtime
  struct PackParam {
    T_SRC* src;
    int ld;
    std::function<int(int)> step_batch;
    int K;
    int N;
  };

  JBLAS_CODE getWeight(...) = delete;

  JBLAS_CODE getWeight(WType** dstptr, int* dststep, int /* b_size */, int /* k_size */, int /* n_size */, int b_offset,
                       int k_offset, int n_offset, const jblas::prologue::PackedWeight* ptr) {
    const auto wptr = dynamic_cast<const PW_T*>(ptr);
    if (!wptr) return JblasInvalidParam;
    assert(k_offset % GemmCore_T::KTILE == 0);
    assert(n_offset % GemmCore_T::NTILE == 0);
    auto KPad = wptr->mKPad;
    auto NPad = wptr->mNPad;
    *dstptr = wptr->mWPtr + n_offset * KPad + k_offset * GemmCore_T::NTILE;
    *dststep = KPad;
    return JblasSuccess;
  }

  JBLAS_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                       const jblas::prologue::PackedWeight* ptr) {
    return getWeight(dstptr, dststep, 1, k_size, n_size, 0, k_offset, n_offset, ptr);
  }

  JBLAS_CODE packWeight(...) = delete;

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  jblas::prologue::PackedWeight* packWeight(const int N, const int K, const WType* src, const int ld) {
    static_assert(std::is_same<WType, ne_bf16_t>::value, "Only support BF16 weight pack.");
    const auto KPad = padto(K, GemmCore_T::KTILE);
    const auto NPad = padto(N, GemmCore_T::NTILE);
    const auto pw = new PackedWeightBatch<WType>(GemmCore_T::TYPE);
    pw->resize(NPad, KPad);
    assert(false);  // TODO(Yi): call reorderT
    return pw;
  }

  /**
   * @brief Create a Parallel object (batch major)
   *
   * @param b batch dim size
   * @param k K dim size (or N dim for transposed pack)
   * @param bblock batch dim block
   * @param kblock
   * @return Parallel
   */
  Parallel createParallel(int b, int k, int bblock, int kblock) {
    Parallel _paral;
    auto cb = CpuBase();
    _paral.update(b, k, bblock, kblock, cb.mNumThreads);
    return _paral;
  }

  /// Reorder job of a thread
  void reorderT(const Param& p, const int tid, const PackParam& pp, const Parallel& paral) {
    assert(false);  // Use the overload function
  };
};

template <class GemmCore_T, JBLAS_ISA ISA_T, typename T_SRC = typename GemmCore_T::BType>
class WeightPackBatchBf16Trans : public WeightPackBatchBf16Base<GemmCore_T, ISA_T, true, T_SRC> {
  using Base = WeightPackBatchBf16Base<GemmCore_T, ISA_T, true, T_SRC>;

 public:
  using typename Base::PackParam;
  using typename Base::Parallel;
  using typename Base::Param;
  using typename Base::PW_T;
  using typename Base::WType;

  /// Reorder job of a thread
  void reorderT(const Param& p, const int tid, const PackParam& pp, const Parallel& paral) {
    const auto pw = dynamic_cast<const PW_T*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after transpose & padding
    const int NPad = pw->mNPad;  // N size after transpose & padding
    assert(pp.K <= KPad);
    assert(pp.N <= NPad);

    int y, x, ny, nx_pad;  // y for batch; x for major-dim of the source data (N-dim of the packed weight)
    paral.getIndex(tid, &y, &x, &ny, &nx_pad);
    if (ny <= 0 || nx_pad <= 0) return;
    ny = remainsize(y, paral.mRows, ny);
    int nx = remainsize(x, paral.mCols, nx_pad);
    assert(paral.mRows == pw->mBatch);
    assert(paral.mPadRow == 1);  // batch should not have paddings (pad tp 1)

    assert(padto(pp.N, GemmCore_T::NTILE) == NPad);

    using KernInterleave = typename jblas::kernel::wrapper::PaddingTransInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::template forward<ISA_T, T_SRC, WType>(  //
          pp.src + pp.step_batch(ibat) + x * pp.ld,                                     //
          pw->mWPtr + ibat * KPad * NPad + x * KPad,                                    //
          nx, pp.K,                                                                     // size
          nx_pad, KPad,                                                                 // padded size
          pp.ld, KPad);                                                                 // step
      assert(forward_stat == JblasSuccess);
    }
  };
};

template <class GemmCore_T, JBLAS_ISA ISA_T, typename T_SRC = typename GemmCore_T::BType>
class WeightPackBatchBf16NonTr : public WeightPackBatchBf16Base<GemmCore_T, ISA_T, false, T_SRC> {
  using Base = WeightPackBatchBf16Base<GemmCore_T, ISA_T, false, T_SRC>;

 public:
  using typename Base::PackParam;
  using typename Base::Parallel;
  using typename Base::Param;
  using typename Base::PW_T;
  using typename Base::WType;

  /// Reorder job of a thread
  void reorderT(const Param& p, const int tid, const PackParam& pp, const Parallel& paral) {
    const auto pw = dynamic_cast<const PW_T*>(p.packedW);
    assert(pw != nullptr);
    const int KPad = pw->mKPad;  // K size after padding
    const int NPad = pw->mNPad;  // N size after padding
    assert(pp.K <= KPad);
    assert(pp.N <= NPad);

    int y, x, ny, nx_pad;  // y for batch; x for major-dim of the source data (K-dim)
    paral.getIndex(tid, &y, &x, &ny, &nx_pad);
    if (ny <= 0 || nx_pad <= 0) return;
    ny = remainsize(y, paral.mRows, ny);
    int nx = remainsize(x, paral.mCols, nx_pad);
    assert(paral.mRows == pw->mBatch);
    assert(paral.mPadRow == 1);  // batch should not have paddings (pad tp 1)

    assert(padto(pp.N, GemmCore_T::NTILE) == NPad);

    using KernInterleave = typename jblas::kernel::wrapper::PaddingInterleaveMN<  //
        GemmCore_T::NTILE, GemmCore_T::PACK_ROW>;

    for (int ibat = y; ibat < y + ny; ++ibat) {
      const auto forward_stat = KernInterleave::template forward<ISA_T, T_SRC, WType>(  //
          pp.src + pp.step_batch(ibat) + x * pp.ld,                                     //
          pw->mWPtr + ibat * KPad * NPad + x * GemmCore_T::NTILE,                       //
          nx, pp.N,                                                                     // size
          nx_pad, NPad,                                                                 // padded size
          pp.ld, KPad);                                                                 // stride
      assert(forward_stat == JblasSuccess);
    }
  };
};

/**
 * @brief GemmLauncherPackWeight with addition input as packed weight offset
 */
template <JBLAS_ISA RT_ISA_, class _GemmCore_T, template <class, JBLAS_ISA> class _PrologueA_T,
          template <class, JBLAS_ISA> class _PrologueB_T, template <JBLAS_ISA> class _Epilogue_T>
class GemmLauncherPackWeightOff                                         //
    : public jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
          RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T> {
  using Base = jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
      RT_ISA_, _GemmCore_T, _PrologueA_T, _PrologueB_T, _Epilogue_T>;

 public:
  using typename Base::GemmCore;
  using Param = typename Base::Param;
  using AType = typename Base::AType;
  using BType = typename Base::BType;
  using CType = typename Base::CType;
  static constexpr auto RT_ISA = RT_ISA_;

  struct ParallelConfig {
    const int rowidx, colidx;
    const int rowsize, colsize;
    const int MStep, NStep, KStep;
    const int w_offset;  // weight offset for batching
    const size_t StackSize;
  };

  void launch(const ParallelConfig& _config, const Param& _param) {
    int rowremain = remainsize(_config.rowidx, _param.M, _config.rowsize);
    int colremain = remainsize(_config.colidx, _param.N, _config.colsize);
    const auto StackSize = _config.StackSize
                               ? _config.StackSize
                               : sizeof(BType) * _config.NStep * _config.KStep +      //
                                     sizeof(AType) * _config.MStep * _config.KStep +  //
                                     sizeof(CType) * padto(rowremain, _config.MStep) * padto(colremain, _config.NStep);
    auto StackTmp = alloca(StackSize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + _config.NStep * _config.KStep);
    auto tmpC = (CType*)(tmpA + _config.MStep * _config.KStep);
    for (int itern = 0; itern < colremain; itern += _config.NStep) {
      int n_remain = remainsize(itern, colremain, _config.NStep);
      for (int iterm = 0; iterm < rowremain; iterm += _config.MStep) {
        int m_remain = remainsize(iterm, rowremain, _config.MStep);
        run_block(_config, _param, iterm, itern, m_remain, n_remain, tmpA, tmpB, tmpC);
      }
    }
  }

 protected:
  void run_block(const ParallelConfig& _config, const Param& _param, int blk_m, int blk_n, int blk_msize, int blk_nsize,
                 AType* tmpA, BType* /*tmpB*/, CType* tmpC) {
    int n_padded = padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.KStep) {
      int k_remain = remainsize(iterk, _param.K, _config.KStep);
      int k_padded = padto(k_remain, GemmCore::KTILE);
      int k_paddedle = padto_le(k_remain, GemmCore::KTILE);
      BType* bptr_cache = nullptr;
      int bcache_step = 0;
      this->mProB.getWeight(&bptr_cache, &bcache_step,      // See：https://stackoverflow.com/questions/11405
                            k_padded, n_padded,             //
                            iterk, _config.colidx + blk_n,  //
                            _param.paramB.packedW);
      bptr_cache += _config.w_offset;
      int bcache_stride = bcache_step * sizeof(BType);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.NStep;
        int ccache_stride = _config.NStep * sizeof(CType);

        int acache_step = 0;
        if (k_paddedle) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
                                    (blk_m + i + _config.rowidx), iterk);
          this->mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain, n_padded, k_paddedle,
                                  acache_step * sizeof(AType), bcache_stride, ccache_stride, iterk);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          AType* aptr_cache = tmpA;
          this->mProA.getActivation(&aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
                                    blk_m + i + _config.rowidx, iterk + k_paddedle);
          this->mGemmCore.forward(aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache, m_remain, n_padded,
                                  GemmCore::KTILE, acache_step * sizeof(AType), bcache_stride, ccache_stride,
                                  iterk + k_paddedle);
        }
      }
    }
    this->mEpilogue.forward(tmpC, _config.NStep, _config.rowidx + blk_m, _config.colidx + blk_n, blk_msize, blk_nsize,
                            _param.paramC);
  }
};

/**
 * @brief MHA interface
 *
 * @tparam L_ExpSum Launcher type of the QK exp sum matmul
 * @tparam L_Scale Launcher type of the PV scale matmul (S for that in the flash-attn paper)
 */
template <class L_ExpSum, class L_Scale>
class MHAInterface {
 public:
  using PC_QK = typename L_ExpSum::ParallelConfig;
  using PC_PV = typename L_Scale::ParallelConfig;

  using PrologueQ = typename L_ExpSum::PrologueA;
  using PrologueK = typename L_ExpSum::PrologueB;
  using QKProQArgs = typename PrologueQ::Param;
  using QKProKArgs = typename PrologueK::Param;
  using QKProKPackArgs = typename PrologueK::PackParam;
  using QKArgs = typename L_ExpSum::Param;
  using QKEpiArgs = typename L_ExpSum::EpiParam;

  using PrologueS = typename L_Scale::PrologueA;
  using PrologueV = typename L_Scale::PrologueB;
  using PVProPArgs = typename PrologueS::Param;
  using PVProVArgs = typename PrologueV::Param;
  using PVProVPackArgs = typename PrologueV::PackParam;
  using PVArgs = typename L_Scale::Param;
  using PVEpiArgs = typename L_Scale::EpiParam;

  using GemmQK = typename L_ExpSum::GemmCore;
  using GemmPV = typename L_Scale::GemmCore;
  using Parallel2DRowMajor = parallel::Parallel2DRowMajor;

  static_assert(GemmQK::MTILE == GemmPV::MTILE, "2 GEMM should have the same M_TILE.");

  template <typename Q_T, typename K_T, typename V_T, typename DST_T>
  JBLAS_CODE compute(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>& p) {
    static constexpr auto M_TILE = GemmQK::MTILE;
    const auto num_heads = p.batch_size * p.head_num;  // Total number of heads
    const auto cb = CpuBase();
    omp_set_num_threads(cb.mNumThreads);
    assert(!p.is_causal || p.sl_q <= p.sl_kv);
    const auto sl_diff = p.sl_kv - p.sl_q;

    // prepare memory for packed weight
    // TODO(Yi): init packed weight with p.tmp
    PackedWeightBatch<typename GemmQK::BType> K_pack(jblas::gemm::GemmCoreType::AMX_BF16_16x64);  // packed K
    K_pack.resize(padto(p.sl_kv, GemmQK::NTILE), padto(p.head_size, GemmQK::KTILE), num_heads);
    PackedWeightBatch<typename GemmPV::BType> V_pack(jblas::gemm::GemmCoreType::AMX_BF16_16x64);  // packed V
    V_pack.resize(padto(p.head_size, GemmPV::NTILE), padto(p.sl_kv, GemmPV::KTILE), num_heads);
    const auto K_pack_batch_off = K_pack.mKPad * K_pack.mNPad;
    const auto V_pack_batch_off = V_pack.mKPad * V_pack.mNPad;

    // prepare parallel scheme for packed weight
    const auto paralK = l_expsum.mProB.createParallel(num_heads, p.sl_kv, 1, GemmQK::NTILE);
    const auto paralV = l_scale.mProB.createParallel(num_heads, p.sl_kv, 1, GemmPV::KTILE);

    const auto step_batch_k = [step_bs = p.step_k_bs, step_hn = p.step_k_head_num, hn = p.head_num](int ibat) {
      return (ibat / hn) * step_bs + (ibat % hn) * step_hn;
    };
    const auto step_batch_v = [step_bs = p.step_v_bs, step_hn = p.step_v_head_num, hn = p.head_num](int ibat) {
      return (ibat / hn) * step_bs + (ibat % hn) * step_hn;
    };

    Parallel2DRowMajor parl;  // w1&w3 from Seq* Fin=>FMid
    const auto m_tiles = updiv(p.sl_q, M_TILE);
    const auto num_tasks = num_heads * m_tiles;
    parl.update(num_tasks, 1, 1, 1, cb.mNumThreads);

#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      const int tmp_exp_size = M_TILE * padto(p.sl_kv, GemmQK::NTILE) * sizeof(ne_bf16_t);  // TODO
      const auto tmp = p.tmp + tid * tmp_exp_size;

      // pack K/V
      {
        l_expsum.mProB.reorderT(  // pack K
            QKProKArgs{&K_pack}, tid,
            QKProKPackArgs{
                /* .src = */ p.K,
                /* .ld = */ p.step_k_sl,
                /* .step_batch = */ step_batch_k,
                /* .K = */ p.head_size,
                /* .N = */ p.sl_kv,
            },
            paralK);
        l_scale.mProB.reorderT(  // pack V
            PVProVArgs{&V_pack}, tid,
            PVProVPackArgs{
                /* .src = */ p.V,
                /* .ld = */ p.step_v_sl,
                /* .step_batch = */ step_batch_v,
                /* .K = */ p.sl_kv,
                /* .N = */ p.head_size,
            },
            paralV);
      }

#pragma omp barrier

      // calculate mm + softmax + mm
      {
        int task_start, _assert0, task_size, _assert_max1;
        parl.getIndex(tid, &task_start, &_assert0, &task_size, &_assert_max1);
        assert(task_size == 0 || _assert0 == 0);
        assert(task_size == 0 || _assert_max1 == 1 || _assert_max1 == 0);
        if (_assert_max1 == 0) task_size = 0;

        for (int task_id = task_start; task_id < task_start + task_size; ++task_id) {
          const int ibat = task_id / m_tiles;
          const int i_m = task_id % m_tiles * M_TILE;
          const int ibs = ibat / p.head_num;
          const int ihn = ibat % p.head_num;
          float exp_sum[M_TILE]{};
          memset(exp_sum, 0, sizeof(exp_sum));

          // ptr to Q / dst matrix of the current head
          const auto head_q = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num;
          const auto head_dst = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num;
          const auto max_unmasked = p.is_causal ? std::min(p.sl_kv, p.sl_kv - p.sl_q + i_m + M_TILE - 1 + 1) : p.sl_kv;

          const auto max_unmasked_pad_qk = std::min(p.sl_kv, padto(max_unmasked, GemmQK::NTILE));
          const auto max_unmasked_pad_pv = std::min(p.sl_kv, padto(max_unmasked, GemmPV::KTILE));
          const auto ld_tmp_exp = padto(padto(max_unmasked_pad_pv, GemmQK::NTILE), GemmPV::KTILE);
          l_expsum.launch(  // QxK => S ==exp==> P
              PC_QK{
                  /* rowidx = */ i_m,
                  /* colidx = */ 0,
                  /* rowsize = */ M_TILE,
                  /* colsize = */ max_unmasked_pad_qk,
                  /* MStep = */ M_TILE,
                  /* NStep = */ GemmQK::NTILE,
                  /* KStep = */ p.head_size,
                  /* w_offset = */ ibat * K_pack_batch_off,
                  /* StackSize = */ 0,
              },
              QKArgs{
                  /* .M = */ p.sl_q,
                  /* .N = */ max_unmasked_pad_qk,
                  /* .K = */ p.head_size,
                  /* .paramA = */ QKProQArgs{head_q, p.step_q_sl},
                  /* .paramB = */ QKProKArgs{&K_pack},
                  /* .paramC = */
                  QKEpiArgs{
                      /* .dst = */ (bf16*)tmp - i_m * ld_tmp_exp,  // pretend that we have a whole exp mat
                      /* .dst_sum = */ exp_sum - i_m,              // pretend that we have a whole exp sum
                      /* .ld_dst = */ ld_tmp_exp,
                      /* .scale = */ p.QK_scale,
                      /* .causal_offset = */ p.is_causal ? sl_diff : -1,
                  },
                  /* .workspace = */ nullptr,
              });
          for (int ii = 0; ii < M_TILE; ++ii) exp_sum[ii] = 1.f / exp_sum[ii];
          l_scale.launch(  // PxV => O
              PC_PV{
                  /* rowidx = */ 0,
                  /* colidx = */ 0,
                  /* rowsize = */ M_TILE,
                  /* colsize = */ p.head_size,
                  /* MStep = */ M_TILE,
                  /* NStep = */ GemmPV::NTILE,
                  /* KStep = */ max_unmasked_pad_qk,  // TODO(Yi): pad?
                  /* w_offset = */ ibat * V_pack_batch_off,
                  /* StackSize = */ 0,
              },
              PVArgs{
                  /* .M = */ std::min(p.sl_q - i_m, M_TILE),
                  /* .N = */ p.head_size,
                  /* .K = */ max_unmasked_pad_qk,
                  /* .paramA = */ PVProPArgs{(jblas::utils::bf16*)tmp, ld_tmp_exp},
                  /* .paramB = */ PVProVArgs{&V_pack},
                  /* .paramC = */
                  PVEpiArgs{
                      /* .scale = */ exp_sum,
                      /* .dst = */ head_dst + i_m * p.step_dst_sl,
                      /* .ld_dst = */ p.step_dst_sl,
                  },
                  /* .workspace = */ nullptr,
              });
        }
      }
    }
    return JblasSuccess;
  }

 protected:
  L_ExpSum l_expsum;
  L_Scale l_scale;
};

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
void jblas_attn_forward(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>* params) = delete;

template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchBf16Bf16NonTr = WeightPackBatchBf16NonTr<GEMM_T, ISA_T, bf16>;
template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchBf16Bf16Trans = WeightPackBatchBf16Trans<GEMM_T, ISA_T, bf16>;
template <>
void jblas_attn_forward<bf16, bf16, bf16, bf16>(const attn_fwd_args_t<bf16, bf16, bf16, bf16>* params) {
  using GemmKernelBF16ExpSum = ::GemmLauncherPackWeightOff<  //
      JblasAMX_BF16,                                         //
      jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,           //
      jblas::prologue::gemm::ActivationBase,                 //
      WeightPackBatchBf16Bf16Trans,                          //
      ::ScaleExpAccSumFp32Bf16>;                             //
  using GemmKernelBF16 = ::GemmLauncherPackWeightOff<        //
      JblasAMX_BF16,                                         //
      jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,           //
      jblas::prologue::gemm::ActivationBase,                 //
      WeightPackBatchBf16Bf16NonTr,                          //
      ::ScaleWriteBackFp32Bf16>;
  static MHAInterface<GemmKernelBF16ExpSum, GemmKernelBF16> kernel;
  kernel.compute(*params);
}

template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchFp16Bf16NonTr = WeightPackBatchBf16NonTr<GEMM_T, ISA_T, fp16>;
template <class GEMM_T, JBLAS_ISA ISA_T>
using WeightPackBatchFp16Bf16Trans = WeightPackBatchBf16Trans<GEMM_T, ISA_T, fp16>;
template <>
void jblas_attn_forward<float, fp16, fp16, float>(const attn_fwd_args_t<float, fp16, fp16, float>* params) {
  GetCPUDevice();
  if (_cd->AMX_BF16()) {
    using GemmKernelFP32FP16BF16ExpSum = ::GemmLauncherPackWeightOff<  //
        JblasAMX_BF16,                                                 //
        jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,                   //
        jblas::prologue::gemm::ActivationConverterFp32,                //
        WeightPackBatchFp16Bf16Trans,                                  //
        ::ScaleExpAccSumFp32Bf16>;                                     //
    using GemmKernelBF16FP16FP32 = ::GemmLauncherPackWeightOff<        //
        JblasAMX_BF16,                                                 //
        jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,                   //
        jblas::prologue::gemm::ActivationBase,                         //
        WeightPackBatchFp16Bf16NonTr,                                  //
        ::ScaleWriteBackFp32Fp32>;
    static MHAInterface<GemmKernelFP32FP16BF16ExpSum, GemmKernelBF16FP16FP32> kernel;
    kernel.compute(*params);
  } else {
    assert(0);
  }
}

template <typename Q_T, typename K_T, typename V_T, typename DST_T>
void jblas_attn_forward_ref(const attn_fwd_args_t<Q_T, K_T, V_T, DST_T>* params) {
  const auto p = *params;
  assert(!p.is_causal || p.sl_q <= p.sl_kv);
  attn_shape_t attn_shape{
      p.batch_size, p.head_num, p.head_size, p.sl_q, p.sl_kv,
  };
  const auto workspace_size = jblas_fusion_attn_bf16_workspace_size(&attn_shape);
  std::fill_n(params->tmp, workspace_size, 'f');

#pragma omp parallel for collapse(3)
  for (int ibs = 0; ibs < p.batch_size; ++ibs)
    for (int ihn = 0; ihn < p.head_num; ++ihn)
      for (int i = 0; i < p.sl_q; ++i) {
        const auto q_curr = p.Q + ibs * p.step_q_bs + ihn * p.step_q_head_num + i * p.step_q_sl;
        const auto dst_curr = p.dst + ibs * p.step_dst_bs + ihn * p.step_dst_head_num + i * p.step_dst_sl;

        const auto k_curr = p.K + ibs * p.step_k_bs + ihn * p.step_k_head_num;
        const auto v_curr = p.V + ibs * p.step_v_bs + ihn * p.step_v_head_num;

        const auto sl_diff = p.sl_kv - p.sl_q;
        const auto unmasked = p.is_causal ? sl_diff + i + 1 : p.sl_kv;
        const auto curr_row = std::unique_ptr<float[]>(new float[unmasked]);

        // Q x K
        float row_max = -INFINITY;
        for (int j = 0; j < unmasked; ++j) {
          curr_row[j] = 0.f;
          for (int k = 0; k < p.head_size; ++k) {
            curr_row[j] += static_cast<float>(static_cast<bf16>(q_curr[k])) *  // TODO(Yi) fp16 acc
                           static_cast<float>(static_cast<bf16>(k_curr[j * p.step_k_sl + k]));
          }
          curr_row[j] *= p.QK_scale;
          row_max = std::max(row_max, curr_row[j]);
        }

        // exp
        float exp_sum = 0.f;
        for (int j = 0; j < unmasked; ++j) {
          curr_row[j] = expf(curr_row[j] /* - row_max */);
          exp_sum += curr_row[j];
        }

        // softmax
        for (int j = 0; j < unmasked; ++j) curr_row[j] /= exp_sum;

        // P x V
        for (int j = 0; j < p.head_size; ++j) {
          float dst_f32_v = 0.f;
          for (int k = 0; k < unmasked; ++k) {
            dst_f32_v += curr_row[k] * static_cast<float>(static_cast<bf16>(v_curr[k * p.step_v_sl + j]));
          }
          dst_curr[j] = static_cast<DST_T>(dst_f32_v);
        }
      }
}
}  // namespace

void jblas_attn_bf16_forward(const attn_bf16_fwd_args_t* params) {
  return jblas_attn_forward(reinterpret_cast<const attn_fwd_args_t<bf16, bf16, bf16, bf16>*>(params));
}
bool jblas_fusion_attn_fp32_fp16_fp16_fp32_support(const attn_shape_t* params) {
  GetCPUDevice();
  // TODO check K V's layout
  return _cd->AMX_BF16();
}

void jblas_fusion_attn_fp32_fp16_fp16_fp32_forward(const attn_fp32_fp16_fp16_fp32_fwd_args_t* params) {
  return jblas_attn_forward(reinterpret_cast<const attn_fwd_args_t<float, fp16, fp16, float>*>(params));
  // return jblas_attn_forward_ref(reinterpret_cast<const attn_fwd_args_t<float, fp16, fp16, float>*>(params));
}

size_t jblas_fusion_attn_bf16_workspace_size(const attn_shape_t* params) {
  const auto& p = *params;  // TODO(Yi): Better way to get tmp size?
  return size_t(omp_get_max_threads() * sizeof(float) * 16) * padto(p.sl_kv, 64);
}

#ifdef __GNUC__
#pragma GCC pop_options
#endif

#ifdef NE_TESTS
namespace {
bool return_success = true;

class TestMhaDese {
 public:
  TestMhaDese() {
    printf("Test suit: %s\n", __FUNCTION__);
    CheckISA(AMX_BF16);
    jblas::utils::request_perm_xtile_data();
    return_success &= test_case({1, 1, 32, 128, 64});
    return_success &= test_case({2, 5, 32, 64, 128});
    return_success &= test_case({2, 5, 80, 128, 77});
    return_success &= test_case({1, 1, 256, 63, 63});
    return_success &= test_case({3, 4, 256, 1, 384});
    return_success &= test_case({1, 1, 64, 64, 64}, true);
    printf("Test suit done: %s\n", __FUNCTION__);
  }
  bool test_case(const attn_shape_t& s, bool is_causal = false) {
    using namespace jblas::utils;
    const auto batch_size = s.batch_size;
    const auto head_num = s.head_num;
    const auto head_size = s.head_size;
    const auto sl_q = s.sl_q;
    const auto sl_kv = s.sl_kv;

    printf("Test case : bs_%d hn_%d hs_%d sl_q_%d sk_kv_%d %s\n", batch_size, head_num, head_size, sl_q, sl_kv,
           is_causal ? "maksed" : "unmask");
    std::vector<float> src_q(batch_size * head_num * sl_q * head_size);
    std::vector<fp16> src_k(batch_size * head_num * sl_kv * head_size);
    std::vector<fp16> src_v(batch_size * head_num * sl_kv * head_size);
    std::vector<float> dst(batch_size * head_num * sl_q * head_size);
    std::vector<float> ref(batch_size * head_num * sl_q * head_size);  // reference result
    std::vector<char> tmp(jblas_fusion_attn_bf16_workspace_size(&s));

    // init vector
    static std::mt19937 rng(1);
    std::uniform_int_distribution<> dist;
    init_vector(&src_q, -1.f, 1.f, dist(rng));
    init_vector(&src_k, -1.f, 1.f, dist(rng));
    init_vector(&src_v, -1.f, 1.f, dist(rng));

    jblas_attn_fp32_fp16_fp16_fp32_fwd_args_t args{
        /* .Q = */ src_q.data(),
        /* .K = */ src_k.data(),
        /* .V = */ src_v.data(),
        /* .dst = */ ref.data(),
        /* .tmp = */ tmp.data(),
        /* .QK_scale = */ 1.f / sqrtf(static_cast<float>(batch_size)),
        /* .is_causal = */ is_causal,
        /* .batch_size = */ batch_size,
        /* .head_num = */ head_num,
        /* .head_size = */ head_size,
        /* .sl_q = */ sl_q,
        /* .sl_kv = */ sl_kv,
        /* .step_q_bs = */ sl_q * head_num * head_size,
        /* .step_q_head_num = */ head_size,
        /* .step_q_sl = */ head_num * head_size,
        /* .step_k_bs = */ sl_kv * head_num * head_size,
        /* .step_k_head_num = */ head_size,
        /* .step_k_sl = */ head_num * head_size,
        /* .step_v_bs = */ sl_kv * head_num * head_size,
        /* .step_v_head_num = */ head_size,
        /* .step_v_sl = */ head_num * head_size,
        /* .step_dst_bs = */ sl_q * head_num * head_size,
        /* .step_dst_head_num = */ head_size,
        /* .step_dst_sl = */ head_num * head_size,
    };

    jblas_attn_forward_ref(&args);

    args.dst = dst.data();
    jblas_attn_forward(&args);

    // Check result
    return compare_data(dst.data(), ref.data(), dst.size(), 1e-2f);
  }
};
static const TestMhaDese inst_;

}  // namespace

int main() {
  printf("NE_TESTS: mha_dense ");
  printf(return_success ? "OK\n" : "FAILED\n");
  return return_success ? 0 : -1;
}
#endif
