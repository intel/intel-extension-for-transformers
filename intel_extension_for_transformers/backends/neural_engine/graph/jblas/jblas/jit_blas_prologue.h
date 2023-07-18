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
#include <immintrin.h>

#include "jit_base.hpp"
#include "jit_blas.h"
#include "jit_blas_gemm.h"
#include "jit_blas_utils.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue {
class PackedWeight {
 public:
  PackedWeight(jblas::gemm::GemmCoreType type) {
    mNPad = 0;
    mKPad = 0;
    mSize = 0;
    mCoreType = type;
  }

  virtual ~PackedWeight() {}

  virtual size_t getSerializedSize() {
    size_t totalsize = 0;
    totalsize += sizeof(mSize);
    totalsize += sizeof(mCoreType);
    totalsize += sizeof(mType);
    totalsize += sizeof(mNPad);
    totalsize += sizeof(mKPad);
    totalsize += getDataSerializedSize();
    return totalsize;
  }

  virtual void serializeToBuffer(void* buf) {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    mSize = getSerializedSize();
    utils::serialize(wptr, mSize);
    utils::serialize(wptr, mCoreType);
    utils::serialize(wptr, mType);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
    serializeDataToBuffer(wptr);
  }

  virtual void deserializeBuffer(void* buf, int memalloc) {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mSize = utils::deserialize<size_t>(rptr);
    mCoreType = utils::deserialize<jblas::gemm::GemmCoreType>(rptr);
    mType = utils::deserialize<int>(rptr);
    mNPad = utils::deserialize<int>(rptr);
    mKPad = utils::deserialize<int>(rptr);
    deserializeDataBuffer(rptr, memalloc);
  }
  size_t mSize;
  jblas::gemm::GemmCoreType mCoreType = jblas::gemm::GemmCoreType::Undef;
  int mType = -1;
  int mNPad = 0, mKPad = 0;
  static int constexpr TypeOffset = sizeof(mSize) + sizeof(mCoreType);

 protected:
  virtual size_t getDataSerializedSize() = 0;
  virtual void serializeDataToBuffer(void* buf) = 0;
  virtual void deserializeDataBuffer(void* buf, int memalloc) = 0;
};

namespace gemm {

template <class _GemmCore_T>
class ActivationBase {
 public:
  using AType = typename _GemmCore_T::AType;
  struct Param {
    const AType* A;
    int lda;
  };
  ActivationBase() {}
  template <JBLAS_ISA ISA_T>
  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const Param& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<AType*>(_param.A);
    if (k_size % _GemmCore_T::KTILE == 0) {
      *dstptr = aptr + m_offset * _param.lda + k_offset;
      *dststep = _param.lda;
      return JblasSuccess;
    } else {
      auto k_pad = utils::padto(k_size, _GemmCore_T::KTILE);
      *dststep = k_pad;
      return kernel::wrapper::Memcpy2D::forward<ISA_T>(aptr + m_offset * _param.lda + k_offset, *dstptr, m_size,
                                                       k_size * sizeof(AType), _param.lda * sizeof(AType),
                                                       k_pad * sizeof(AType));
    }
    return JblasSuccess;
  }
};

template <class _GemmCore_T>
class ActivationF32U8KBlockQuantize {
 public:
  using AType = typename _GemmCore_T::AType;
  using AQType = uint8_t;
  using SType = float;
  struct Param {
    const float* A;
    int lda;
  };
  struct QuanParam {
    AQType* A;
    AQType* zp;
    int lda;
    SType* scales;
    int lds;
    int kblock;

    void resize(int m, int kpad, int _kblock) {
      kblock = _kblock;
      lda = kpad;
      mA.resize(m * lda);
      A = mA.data();
      lds = utils::updiv(kpad, _kblock);
      mScales.resize(m * lds);
      mZp.resize(m * lds);
      scales = mScales.data();
      zp = mZp.data();
    }
    utils::aligned_vector<AQType> mA;
    utils::aligned_vector<AQType> mZp;
    utils::aligned_vector<SType> mScales;
  };
  using Parallel = utils::parallel::Parallel2DRowMajorColBlock;

  Parallel createParallel(int m, int k, int kblock) {
    Parallel _paral;
    auto cb = utils::CpuBase();
    _paral.update(m, k, 1, 16, kblock, cb.mNumThreads);
    return _paral;
  }

  QuanParam createObj(int m, int k, int kblock) {
    QuanParam quan;
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    quan.resize(m, kpad, kblock);
    return quan;
  }

  template <JBLAS_ISA ISA_T>
  void quantizeT(const Param& _param, int tidx, QuanParam& quan, Parallel& para) {
    int colidx, rowidx, rowsize, colsize;
    int blkidx, idxinblk;
    para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize, &blkidx, &idxinblk);

    if (rowsize > 0 && colsize > 0) {
      // min max
      auto srcptr = _param.A + rowidx * _param.lda + colidx;
      int rowremain = utils::remainsize(rowidx, para.mRows, rowsize);
      int colremain = utils::remainsize(colidx, para.mCols, colsize);
      auto thdqptr = quan.A + rowidx * quan.lda + colidx;
      auto thdsptr = quan.scales + rowidx * quan.lds + blkidx;
      auto thdzptr = quan.zp + rowidx * quan.lds + blkidx;
      kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T>(
          rowremain, colremain, srcptr, _param.lda, thdqptr, quan.lda, thdsptr, quan.lds, thdzptr, para.mColBlock);
    }
  }

  template <JBLAS_ISA ISA_T>
  QuanParam quantize(const Param& _param, int m, int k, int kblock) {
    utils::parallel::Parallel2DRowMajorColBlock paral;
    utils::CpuBase cb;
    if (m == 1) {
      cb.mNumThreads = 1;
    }
    paral.update(m, k, 1, 16, kblock, cb.mNumThreads);
    int kpad = utils::padto(k, _GemmCore_T::KTILE);
    QuanParam quan;
    quan.resize(m, kpad, kblock);
    if (m == 1) {
      auto srcptr = _param.A;
      int rowremain = m;
      int colremain = k;
      auto thdqptr = quan.A;
      auto thdsptr = quan.scales;
      auto thdzptr = quan.zp;
      kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T>(rowremain, colremain, srcptr, _param.lda, thdqptr,
                                                                   quan.lda, thdsptr, quan.lds, thdzptr, kblock);
      return quan;
    }

    omp_set_num_threads(cb.mNumThreads);
    if (paral.mThdsPerBlock == 1) {  // no barrier
#pragma omp parallel
      {
        int tidx = omp_get_thread_num();
        int colidx, rowidx, rowsize, colsize;
        int blkidx, idxinblk;
        paral.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize, &blkidx, &idxinblk);
        if (rowsize > 0 && colsize > 0) {
          // min max
          auto srcptr = _param.A + rowidx * _param.lda + colidx;
          int rowremain = utils::remainsize(rowidx, m, rowsize);
          int colremain = utils::remainsize(colidx, k, colsize);
          auto thdqptr = quan.A + rowidx * quan.lda + colidx;
          auto thdsptr = quan.scales + rowidx * quan.lds + blkidx;
          auto thdzptr = quan.zp + rowidx * quan.lds + blkidx;
          kernel::wrapper::QuantizeU8ColBlock::template forward<ISA_T>(
              rowremain, colremain, srcptr, _param.lda, thdqptr, quan.lda, thdsptr, quan.lds, thdzptr, kblock);
        }
      }
    } else {
      assert(0);
    }
    return quan;
  }

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE getActivation(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size, int m_offset,
                           int k_offset) {
    auto aptr = const_cast<AType*>(_param.A);
    *dstptr = aptr + m_offset * _param.lda + k_offset;
    *dststep = _param.lda;
    return JblasSuccess;
  }

  template <JBLAS_ISA ISA_T>
  JBLAS_CODE getScale(SType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size, int m_offset,
                      int k_offset) {
    auto ptr = const_cast<SType*>(_param.scales);
    *dstptr = ptr + m_offset * _param.lds + k_offset / _param.kblock;
    *dststep = _param.lds;
    return JblasSuccess;
  }

  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE getZp(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size,
                                 int m_offset, int k_offset) {
    *dstptr = &_param.zp[(m_offset)*_param.lds + k_offset / _param.kblock];
    *dststep = _param.lds;
    return JblasSuccess;
  }

  template <JBLAS_ISA ISA_T>
  static inline JBLAS_CODE getZpBroadcast(AType** dstptr, int* dststep, const QuanParam& _param, int m_size, int k_size,
                                          int m_offset, int k_offset) {
    for (size_t i = 0; i < m_size; i++) {
      auto zpval = _param.zp[(m_offset + i) * _param.lds + k_offset / _param.kblock];
      kernel::wrapper::Broadcast::template forward<ISA_T>(_param.kblock, zpval, *dstptr + i * _param.kblock);
    }
    return JblasSuccess;
  }
};

enum class PackType : int {
  Default = 0,
};

template <typename WType>
class PackedWeightDefault : public prologue::PackedWeight {
 public:
  PackedWeightDefault(jblas::gemm::GemmCoreType _type) : PackedWeight(_type) {
    mWPtr = NULL;
    mWSize = 0;
    mType = 0;
  }

  void resize(int NPad, int KPad) {
    mNPad = NPad;
    mKPad = KPad;
    mWeights.resize((size_t)NPad * KPad);
    mWPtr = mWeights.data();
    mWSize = mWeights.size();
  }
  WType* mWPtr;
  size_t mWSize;

 protected:
  virtual size_t getDataSerializedSize() override {
    size_t totalsize = 0;
    totalsize += sizeof(mWSize);
    totalsize += mWSize * sizeof(mWPtr[0]);
    return totalsize;
  }
  virtual void serializeDataToBuffer(void* buf) override {
    auto wptr = reinterpret_cast<int8_t*>(buf);
    utils::serialize(wptr, mWSize);
    for (size_t i = 0; i < mWSize; i++) {
      utils::serialize(wptr, mWPtr[i]);
    }
  }
  virtual void deserializeDataBuffer(void* buf, int memalloc) override {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    size_t rsize = utils::deserialize<size_t>(rptr);
    if (memalloc) {
      mWeights.resize(rsize);
      std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
    } else {
      mWPtr = (WType*)rptr;
      mWSize = rsize;
    }
    rptr += rsize * sizeof(mWeights[0]);
  }
  utils::aligned_vector<WType> mWeights;
};

template <typename WType>
class PackedWeightBase {
 public:
  static PackedWeight* deserialBuffer(void* serialized_buf, int memalloc = 0) {
    auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
    size_t tsize = utils::deserialize<size_t>(rptr);
    int mType = utils::deserialize<int>(rptr);
    rptr = reinterpret_cast<int8_t*>(serialized_buf);
    auto type = static_cast<PackType>(mType);
    if (type == PackType::Default) {
      auto ptr = new PackedWeightDefault<WType>(jblas::gemm::GemmCoreType::Undef);
      ptr->deserializeBuffer(rptr, memalloc);
      return ptr;
    }
    return NULL;
  }
};

template <class _GemmCore_T>
class WeightPack {
 public:
  struct Param {
    const prologue::PackedWeight* packedW;
  };
  using WType = typename _GemmCore_T::BType;

  template <JBLAS_ISA ISA_T>
  PackedWeight* packTranspose(const int N, const int K, const WType* B, const int ldb,
                              PackType type = PackType::Default) {
    utils::aligned_vector<float> B_NT(N * K);
    transposeWeight<ISA_T>(N, K, B, ldb, B_NT.data(), N);
    return packWeight<ISA_T>(N, K, B_NT.data(), N, type);
  }

  template <JBLAS_ISA ISA_T>
  PackedWeight* pack(const int N, const int K, const WType* B, const int ldb, PackType type = PackType::Default) {
    return packWeight<ISA_T>(N, K, B, N, type);
  }

  template <JBLAS_ISA ISA_T, typename _T>
  inline JBLAS_CODE getWeight(_T* dstptr, int k_size, int n_size, int k_offset, int n_offset, const PackedWeight* ptr) {
    return JblasNotSupport;
  }

  template <JBLAS_ISA ISA_T>
  inline JBLAS_CODE getWeight(WType** dstptr, int* dststep, int k_size, int n_size, int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    {
      auto wptr = dynamic_cast<const PackedWeightDefault<WType>*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad + k_offset * _GemmCore_T::NTILE;
        *dstptr = bptr;
        *dststep = KPad;
        return JblasSuccess;
      }
    }
    return JblasInvalidParam;
  }

 protected:
  template <JBLAS_ISA ISA_T>
  void transposeWeight(const int N, const int K, const WType* src, const int ld_src, WType* dst, const int ld_dst) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(N, K, 16, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, N,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, K, colsize);
        kernel::wrapper::Transpose2D<WType>::template forward<ISA_T>(
            src + rowidx * ld_src + colidx, dst + rowidx + colidx * ld_dst, rowremain, colremain, ld_src, ld_dst);
      }
    }
  }

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  template <JBLAS_ISA ISA_T>
  PackedWeight* packWeight(const int N, const int K, const WType* B, const int ldb, PackType type) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    PackedWeight* ptr = NULL;
    WType* wptr = NULL;
    if (type == PackType::Default) {
      auto tmp = new PackedWeightDefault<WType>(_GemmCore_T::TYPE);
      tmp->resize(NPad, KPad);
      wptr = tmp->mWPtr;
      ptr = tmp;
    }
    if (ptr == NULL) {
      return ptr;
    }
    reorder<ISA_T>(N, K, B, ldb, wptr);
    return ptr;
  }

  template <JBLAS_ISA ISA_T>
  void reorder(const int N, const int K, const WType* B, const int ldb, WType* dstptr) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, _GemmCore_T::KTILE, _GemmCore_T::NTILE, cb.mNumThreads);
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(rowidx, K,
                                          rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        auto ret = kernel::wrapper::PaddingInterleaveMN<_GemmCore_T::NTILE, sizeof(B[0]), _GemmCore_T::PACK_ROW>::
            template forward<ISA_T>((void*)(B + rowidx * ldb + colidx),
                                    dstptr + rowidx * _GemmCore_T::NTILE + colidx * KPad, rowremain, colremain, rowsize,
                                    colsize, ldb * sizeof(B[0]), KPad * sizeof(dstptr[0]));
        assert(ret == JblasSuccess);
      }
    }
  }
};

}  // namespace gemm
}  // namespace prologue
}  // namespace jblas