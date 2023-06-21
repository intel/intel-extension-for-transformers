#pragma once
#include "jit_blas_wrapper.h"
#include "kernel_wrapper.h"

namespace jblas {
namespace prologue {
namespace weight_comp {
class PackedWeight {
 public:
  PackedWeight() {
    mNPad = 0;
    mKPad = 0;
  }

  virtual ~PackedWeight() {}

  virtual size_t getSerializedSize() {
    size_t totalsize = 0;
    totalsize += sizeof(size_t);
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
    utils::serialize(wptr, mType);
    utils::serialize(wptr, mNPad);
    utils::serialize(wptr, mKPad);
    serializeDataToBuffer(wptr);
  }

  virtual void deserializeBuffer(void* buf, int memalloc) {
    auto rptr = reinterpret_cast<int8_t*>(buf);
    mSize = utils::deserialize<size_t>(rptr);
    mType = utils::deserialize<int>(rptr);
    mNPad = utils::deserialize<int>(rptr);
    mKPad = utils::deserialize<int>(rptr);
    deserializeDataBuffer(rptr, memalloc);
  }
  size_t mSize;
  int mType = -1;
  int mNPad = 0, mKPad = 0;

 protected:
  virtual size_t getDataSerializedSize() = 0;
  virtual void serializeDataToBuffer(void* buf) = 0;
  virtual void deserializeDataBuffer(void* buf, int memalloc) = 0;
};

namespace gemm {

template <class _GemmCore_T>
class WeightS4_KBlock {
 public:
  struct Param {
    const PackedWeight* packedW;
  };
  enum class S4Type : int {
    S4_F32 = 0,
    S4_Bf16,
  };

  class PackedWeightS4F32 : public weight_comp::PackedWeight {
   public:
    PackedWeightS4F32() {
      mWPtr = NULL;
      mWSize = 0;
      mSPtr = NULL;
      mSSize = 0;
      mBlockSize = 0;
      mType = static_cast<int>(S4Type::S4_F32);
    }

    void resize(int NPad, int KPad, int Block) {
      mNPad = NPad;
      mKPad = KPad;
      mWeights.resize((size_t)NPad * KPad / 2);
      mBlockSize = Block;
      int nk_scale = utils::updiv(KPad, Block);
      mScales.resize(nk_scale * NPad);
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
      mSPtr = mScales.data();
      mSSize = mScales.size();
    }

    utils::int4x2* mWPtr;
    size_t mWSize;
    float* mSPtr;
    size_t mSSize;
    int mBlockSize;

   protected:
    virtual size_t getDataSerializedSize() override {
      size_t totalsize = 0;
      totalsize += sizeof(mBlockSize);
      totalsize += sizeof(mWSize);
      totalsize += mWSize * sizeof(mWPtr[0]);
      totalsize += sizeof(mSSize);
      totalsize += mSSize * sizeof(mSPtr[0]);
      return totalsize;
    }
    virtual void serializeDataToBuffer(void* buf) override {
      auto wptr = reinterpret_cast<int8_t*>(buf);
      utils::serialize(wptr, mBlockSize);
      utils::serialize(wptr, mWSize);
      for (size_t i = 0; i < mWSize; i++) {
        utils::serialize(wptr, mWPtr[i]);
      }
      utils::serialize(wptr, mSSize);
      for (size_t i = 0; i < mSSize; i++) {
        utils::serialize(wptr, mSPtr[i]);
      }
    }
    virtual void deserializeDataBuffer(void* buf, int memalloc) override {
      auto rptr = reinterpret_cast<int8_t*>(buf);
      mBlockSize = utils::deserialize<int>(rptr);
      size_t rsize = utils::deserialize<size_t>(rptr);
      if (memalloc) {
        mWeights.resize(rsize);
        std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
        mWPtr = mWeights.data();
        mWSize = mWeights.size();
      } else {
        mWPtr = (utils::int4x2*)rptr;
        mWSize = rsize;
      }
      rptr += rsize * sizeof(mWeights[0]);
      rsize = utils::deserialize<size_t>(rptr);
      if (memalloc) {
        mScales.resize(rsize);
        std::memcpy(mScales.data(), rptr, rsize * sizeof(mScales[0]));
        mSPtr = mScales.data();
        mSSize = mScales.size();
      } else {
        mSPtr = (float*)rptr;
        mSSize = rsize;
      }
      rptr += rsize * sizeof(mScales[0]);
    }
    utils::aligned_vector<utils::int4x2> mWeights;
    utils::aligned_vector<float> mScales;
  };

  class PackedWeightS4Bf16 : public weight_comp::PackedWeight {
   public:
    PackedWeightS4Bf16() {
      mWPtr = NULL;
      mWSize = 0;
      mSPtr = NULL;
      mSSize = 0;
      mBlockSize = 0;
      mType = static_cast<int>(S4Type::S4_Bf16);
    }

    void resize(int NPad, int KPad, int Block) {
      mNPad = NPad;
      mKPad = KPad;
      mWeights.resize((size_t)NPad * KPad / 2);
      mBlockSize = Block;
      int nk_scale = utils::updiv(KPad, Block);
      mScales.resize(nk_scale * NPad);
      mWPtr = mWeights.data();
      mWSize = mWeights.size();
      mSPtr = mScales.data();
      mSSize = mScales.size();
    }

    utils::int4x2* mWPtr;
    size_t mWSize;
    utils::bf16* mSPtr;
    size_t mSSize;
    int mBlockSize;

   protected:
    virtual size_t getDataSerializedSize() override {
      size_t totalsize = 0;
      totalsize += sizeof(mBlockSize);
      totalsize += sizeof(mWSize);
      totalsize += mWSize * sizeof(mWPtr[0]);
      totalsize += sizeof(mSSize);
      totalsize += mSSize * sizeof(mSPtr[0]);
      return totalsize;
    }
    virtual void serializeDataToBuffer(void* buf) override {
      auto wptr = reinterpret_cast<int8_t*>(buf);
      utils::serialize(wptr, mBlockSize);
      utils::serialize(wptr, mWSize);
      for (size_t i = 0; i < mWSize; i++) {
        utils::serialize(wptr, mWPtr[i]);
      }
      utils::serialize(wptr, mSSize);
      for (size_t i = 0; i < mSSize; i++) {
        utils::serialize(wptr, mSPtr[i]);
      }
    }
    virtual void deserializeDataBuffer(void* buf, int memalloc) override {
      auto rptr = reinterpret_cast<int8_t*>(buf);
      mBlockSize = utils::deserialize<int>(rptr);
      size_t rsize = utils::deserialize<size_t>(rptr);
      if (memalloc) {
        mWeights.resize(rsize);
        std::memcpy(mWeights.data(), rptr, rsize * sizeof(mWeights[0]));
        mWPtr = mWeights.data();
        mWSize = mWeights.size();
      } else {
        mWPtr = (utils::int4x2*)rptr;
        mWSize = rsize;
      }
      rptr += rsize * sizeof(mWeights[0]);
      rsize = utils::deserialize<size_t>(rptr);
      if (memalloc) {
        mScales.resize(rsize);
        std::memcpy(mScales.data(), rptr, rsize * sizeof(mScales[0]));
        mSPtr = mScales.data();
        mSSize = mScales.size();
      } else {
        mSPtr = (utils::bf16*)rptr;
        mSSize = rsize;
      }
      rptr += rsize * sizeof(mScales[0]);
    }
    utils::aligned_vector<utils::int4x2> mWeights;
    utils::aligned_vector<utils::bf16> mScales;
  };

  class PackedWeightBase {
   public:
    static PackedWeight* deserialBuffer(void* serialized_buf,
                                        int memalloc = 0) {
      auto rptr = reinterpret_cast<int8_t*>(serialized_buf);
      size_t tsize = utils::deserialize<size_t>(rptr);
      int mType = utils::deserialize<int>(rptr);
      rptr = reinterpret_cast<int8_t*>(serialized_buf);
      auto type = static_cast<S4Type>(mType);
      if (type == S4Type::S4_F32) {
        auto ptr = new PackedWeightS4F32();
        ptr->deserializeBuffer(rptr, memalloc);
        return ptr;
      }
      if (type == S4Type::S4_Bf16) {
        auto ptr = new PackedWeightS4Bf16();
        ptr->deserializeBuffer(rptr, memalloc);
        return ptr;
      }
      return NULL;
    }
  };

  template <JBLAS_ISA ISA_T>
  void quantizeWeight(const int N, const int K, const float* B, const int ldb,
                      int blocksize, int8_t* qB, float* scales) {
    utils::parallel::Parallel2DRowMajor _para;
    utils::CpuBase cb;
    _para.update(K, N, blocksize, 16, cb.mNumThreads);
    omp_set_num_threads(cb.mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        int rowremain = utils::remainsize(
            rowidx, K,
            rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        kernel::wrapper::QuantizeS8KBlock::forward<ISA_T>(
            B + rowidx * ldb + colidx, qB + rowidx * N + colidx, rowremain,
            colremain, ldb, N, scales + rowidx / blocksize * N + colidx,
            blocksize);
      }
    }
  }

  template <JBLAS_ISA ISA_T>
  void transposeWeight(const int N, const int K, const float* src,
                       const int ld_src, float* dst, const int ld_dst) {
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
        int rowremain = utils::remainsize(
            rowidx, N,
            rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, K, colsize);
        kernel::wrapper::Transpose2D<float>::forward<ISA_T>(
            src + rowidx * ld_src + colidx, dst + rowidx + colidx * ld_dst,
            rowremain, colremain, ld_src, ld_dst);
      }
    }
  }

  // from KxN int8 symmetric weight to packed N//NtilexKPadxNTile int4 weight
  template <JBLAS_ISA ISA_T>
  PackedWeight* compressWeight(const int N, const int K, const int8_t* B,
                               const int ldb, const float* scales,
                               int blocksize, S4Type type) {
    int KPad = utils::padto(K, _GemmCore_T::KTILE);
    int NPad = utils::padto(N, _GemmCore_T::NTILE);
    int nk_scale = utils::updiv(KPad, blocksize);
    PackedWeight* ptr = NULL;
    utils::int4x2* wptr = NULL;
    if (type == S4Type::S4_F32) {
      auto tmp = new PackedWeightS4F32;
      tmp->resize(NPad, KPad, blocksize);
      wptr = tmp->mWPtr;
      ptr = tmp;
#pragma omp parallel for
      for (int i = 0; i < nk_scale; i++) {
        std::memcpy(tmp->mSPtr + i * NPad, scales + i * N,
                    N * sizeof(scales[0]));
      }
    } else if (type == S4Type::S4_Bf16) {
      auto tmp = new PackedWeightS4Bf16;
      tmp->resize(NPad, KPad, blocksize);
      wptr = tmp->mWPtr;
      ptr = tmp;
#pragma omp parallel for
      for (int i = 0; i < nk_scale; i++) {
        for (int j = 0; j < N; j++) {
          *(tmp->mSPtr + i * NPad + j) =
              utils::cast<float, utils::bf16>(*(scales + i * N + j));
        }
      }
    }
    if (ptr == NULL) {
      return ptr;
    }
    reorderCompress<ISA_T>(N, K, B, ldb, scales, wptr, blocksize);
    return ptr;
  }

  template <JBLAS_ISA ISA_T>
  void reorderCompress(const int N, const int K, const int8_t* B,
                                const int ldb, const float* scales,
                                utils::int4x2* dstptr, int blocksize) {
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
        int rowremain = utils::remainsize(
            rowidx, K,
            rowsize);  // rowremain: src valid size. rowsize: padded size
        int colremain = utils::remainsize(colidx, N, colsize);
        utils::aligned_vector<int8_t> tmp;
        tmp.resize(colsize * rowsize);
        auto ret = kernel::wrapper::PaddingInterleaveMN<
            _GemmCore_T::NTILE, sizeof(B[0]), _GemmCore_T::PACK_ROW>::
            template forward<ISA_T>((void*)(B + rowidx * ldb + colidx),
                                    tmp.data(), rowremain, colremain, rowsize,
                                    colsize, ldb * sizeof(B[0]),
                                    rowsize * sizeof(tmp[0]));
        assert(ret == JblasSuccess);
        ret =
            kernel::wrapper::CompressS8S4<_GemmCore_T::NTILE>::template forward<
                ISA_T>(
                tmp.data(),
                dstptr + rowidx * _GemmCore_T::NTILE / 2 + colidx * KPad / 2,
                rowsize, colsize, rowsize * sizeof(tmp[0]),
                KPad * sizeof(dstptr[0]) / 2);
        assert(ret == JblasSuccess);
      }
    }
  }

  template <JBLAS_ISA ISA_T>
  PackedWeight* compressWeightTranspose(const int N, const int K,
                                        const float* B, const int ldb,
                                        int blocksize, S4Type type) {
    utils::aligned_vector<float> B_NT(N * K);
    transposeWeight<ISA_T>(N, K, B, ldb, B_NT.data(), N);
    return compressWeight<ISA_T>(N, K, B_NT.data(), N, blocksize, type);
  }

  template <JBLAS_ISA ISA_T>
  PackedWeight* compressWeight(const int N, const int K, const float* B,
                               const int ldb, int blocksize, S4Type type) {
    int nk_scale = utils::updiv(K, blocksize);
    utils::aligned_vector<int8_t> quanW(N * K);
    utils::aligned_vector<float> scales(nk_scale * N);
    quantizeWeight<ISA_T>(N, K, B, ldb, blocksize, quanW.data(), scales.data());
    return compressWeight<ISA_T>(N, K, quanW.data(), N, scales.data(),
                                 blocksize, type);
  }

  template <JBLAS_ISA ISA_T, typename _T>
  inline JBLAS_CODE getWeight(_T* dstptr, int k_size, int n_size, int k_offset,
                              int n_offset, const PackedWeight* ptr) {
    return JblasNotSupport;
  }

  template <JBLAS_ISA ISA_T>
  inline JBLAS_CODE getWeight(float* dstptr, int k_size, int n_size,
                              int k_offset, int n_offset,
                              const PackedWeight* ptr) {
    static_assert(_GemmCore_T::PACK_ROW == 1);  // float PackRow==1
    {
      auto wptr = dynamic_cast<const PackedWeightS4F32*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad / 2 +
                    k_offset * _GemmCore_T::NTILE / 2;
        for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
          kernel::wrapper::DecompressKBlockS4F32::forward<ISA_T, float>(
              bptr + i * KPad / 2, dstptr + i * k_size,
              k_size / _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW / 2,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
              wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad);
        }
        return JblasSuccess;
      }
    }
    {
      auto wptr = dynamic_cast<const PackedWeightS4Bf16*>(ptr);
      if (wptr) {
        auto NPad = wptr->mNPad;
        auto KPad = wptr->mKPad;
        auto bptr = wptr->mWPtr + n_offset * KPad / 2 +
                    k_offset * _GemmCore_T::NTILE / 2;
        for (int i = 0; i < n_size; i += _GemmCore_T::NTILE) {
          kernel::wrapper::DecompressKBlockS4F32::forward<ISA_T, utils::bf16>(
              bptr + i * KPad / 2, dstptr + i * k_size,
              k_size / _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW / 2,
              _GemmCore_T::NTILE * _GemmCore_T::PACK_ROW,
              wptr->mSPtr + n_offset + i, k_offset, wptr->mBlockSize, NPad);
        }
        return JblasSuccess;
      }
    }
    return JblasInvalidParam;
  }
};

}  // namespace gemm
}  // namespace weight_comp
}  // namespace prologue
namespace wrapper {
namespace gemm_weight_comp {

template <JBLAS_ISA _RT_ISA_T, class _GemmCore_T,
          template <class _T> class _PrologueA_T,
          template <class _T> class _PrologueB_T, class _Epilogue_T>
class GemmLauncherPackWeight {
 public:
  using GemmCore = _GemmCore_T;
  using PrologueA = _PrologueA_T<GemmCore>;
  using PrologueB = _PrologueB_T<GemmCore>;
  using AType = typename GemmCore::AType;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::BType;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::CType;
  using EpiParam = typename _Epilogue_T::Param;
  static_assert(GemmCore::ISA >= _RT_ISA_T,
                "RunTime ISA should cover GEMM's ISA");
  struct Param {
    const int M, N, K;
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
    void* workspace;
  };
  struct ParallelConfig {
    const int rowidx, colidx;
    const int rowsize, colsize;
    const int MStep, NStep, KStep;
    const size_t StackSize;
  };
  _GemmCore_T mGemmCore;
  PrologueA mProA;
  PrologueB mProB;
  _Epilogue_T mEpilogue;
  GemmLauncherPackWeight() {}

  void launch(const ParallelConfig& _config, const Param& _param) {
    int rowremain =
        utils::remainsize(_config.rowidx, _param.M, _config.rowsize);
    int colremain =
        utils::remainsize(_config.colidx, _param.N, _config.colsize);
    auto StackTmp = alloca(_config.StackSize);
    auto tmpB = (BType*)(StackTmp);
    auto tmpA = (AType*)(tmpB + _config.NStep * _config.KStep);
    auto tmpC = (CType*)(tmpA + GemmCore::MTILE * GemmCore::KTILE);
    for (int itern = 0; itern < colremain; itern += _config.NStep) {
      int n_remain = utils::remainsize(itern, colremain, _config.NStep);
      for (int iterm = 0; iterm < rowremain; iterm += _config.MStep) {
        int m_remain = utils::remainsize(iterm, rowremain, _config.MStep);
        run_block(_config, _param, iterm, itern, m_remain, n_remain, tmpA, tmpB,
                  tmpC);
      }
    }
  }

 protected:
  void run_block(const ParallelConfig& _config, const Param& _param, int blk_m,
                 int blk_n, int blk_msize, int blk_nsize, AType* tmpA,
                 BType* tmpB, CType* tmpC) {
    int n_padded = utils::padto(blk_nsize, GemmCore::NTILE);
    for (int iterk = 0; iterk < _param.K; iterk += _config.KStep) {
      int k_remain = utils::remainsize(iterk, _param.K, _config.KStep);
      int k_padded = utils::padto(k_remain, GemmCore::KTILE);
      int k_paddedle = utils::padto_le(k_remain, GemmCore::KTILE);
      auto bptr_cache = tmpB;
      int bcache_stride = k_padded * sizeof(BType);
      mProB.template getWeight<_RT_ISA_T>(bptr_cache, k_padded, n_padded, iterk,
                                          _config.colidx + blk_n,
                                          _param.paramB.packedW);
      for (int i = 0; i < blk_msize; i += GemmCore::MTILE) {
        int m_remain = utils::remainsize(i, blk_msize, GemmCore::MTILE);
        auto cptr_cache = tmpC + i * _config.NStep;
        int ccache_stride = _config.NStep * sizeof(CType);

        AType* aptr_cache = nullptr;
        int acache_step = 0;
        if (k_paddedle) {
          mProA.template getActivation<_RT_ISA_T>(
              &aptr_cache, &acache_step, _param.paramA, m_remain, k_paddedle,
              (blk_m + i + _config.rowidx), iterk);
          mGemmCore.forward(aptr_cache, bptr_cache, cptr_cache, m_remain,
                            n_padded, k_paddedle, acache_step * sizeof(AType),
                            bcache_stride, ccache_stride, iterk);
        }
        int k_tail = k_remain - k_paddedle;
        if (k_tail) {
          aptr_cache = tmpA;
          mProA.template getActivation<_RT_ISA_T>(
              &aptr_cache, &acache_step, _param.paramA, m_remain, k_tail,
              (blk_m + i + _config.rowidx), iterk + k_paddedle);
          mGemmCore.forward(
              aptr_cache, bptr_cache + k_paddedle * GemmCore::NTILE, cptr_cache,
              m_remain, n_padded, k_padded, acache_step * sizeof(AType),
              bcache_stride, ccache_stride, iterk + k_paddedle);
        }
      }
    }
    mEpilogue.template forward<_RT_ISA_T>(
        tmpC, _config.NStep, (_config.rowidx + blk_m), _config.colidx + blk_n,
        blk_msize, blk_nsize, _param.paramC);
  }
};

template <class _Launcher_T, class _Parallel_T>
class GemmInterfacePackWeight : protected utils::CpuBase {
 public:
  using Arguments = typename _Launcher_T::Param;
  using Config = typename _Launcher_T::ParallelConfig;
  using WeightType = typename _Launcher_T::PrologueB;
  using GemmCore = typename _Launcher_T::GemmCore;
  GemmInterfacePackWeight(int M = 0, int N = 0, int K = 0)
      : mParallel(sizeof(typename _Launcher_T::BType),
                  sizeof(typename _Launcher_T::CType)) {
    update_problem(M, N, K);
  }
  WeightType* getWeightPtr() { return &mLauncher.mProB; }
  // forward=packB+compute
  JBLAS_CODE compute(const Arguments& _param) {
    mNumThreads = utils::parallel::CpuDevice::getInstance()->getThreads();
    update_problem(_param.M, _param.N, _param.K);
    omp_set_num_threads(mNumThreads);
#pragma omp parallel
    {
      int tidx = omp_get_thread_num();
      int colidx, rowidx, rowsize, colsize;
      mParallel.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
      if (rowsize > 0 && colsize > 0) {
        Config _config{rowidx,
                       colidx,
                       rowsize,
                       colsize,
                       mParallel.getMStep(),
                       mParallel.getNStep(),
                       mParallel.getKStep(),
                       mL2Cache};
        mLauncher.launch(_config, _param);
      }
    }
    return JblasSuccess;
  }

 protected:
  void update_problem(int M, int N, int K) {
    if (!mParallel.sameProblem(M, N, K)) {
      mParallel.update(M, N, K, mNumThreads, mL2Cache, GemmCore::MTILE,
                       GemmCore::NTILE, GemmCore::KTILE, GemmCore::PREFERED_N);
      static bool dbgprint = false;
      if (dbgprint) {
        mParallel.print();
        dbgprint = false;
      }
    }
  }

 protected:
  _Launcher_T mLauncher;
  _Parallel_T mParallel;
};

}  // namespace gemm_weight_comp
namespace gemm_default {
namespace weight_comp {

namespace avx512f {
JBLAS_ISA constexpr DefaultISA = JblasAVX512F;
using GemmKernelS4KBlock =
    jblas::wrapper::gemm_weight_comp::GemmInterfacePackWeight<
        jblas::wrapper::gemm_weight_comp::GemmLauncherPackWeight<
            DefaultISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
            jblas::prologue::gemm::ActivationBase,
            jblas::prologue::weight_comp::gemm::WeightS4_KBlock,
            jblas::epilogue::gemm::AlphaBetaProcessFp32>,
        DefaultParallel>;
}  // namespace avx512f
}  // namespace weight_comp
}  // namespace gemm_default
}  // namespace wrapper
}  // namespace jblas
