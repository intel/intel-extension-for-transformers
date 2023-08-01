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
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "jit_blas.h"
#include "xbyak/xbyak_util.h"

// As long as the compiler supports the ISA, we will enable it.
// Only the ISA you use in your project will be compiled.
#ifdef __GNUC__
#define CompileAVX512F() (defined(__GNUC__) && (__GNUC__ >= 6))
#define CompileAVX2() (defined(__GNUC__) && (__GNUC__ >= 5))
#define CompileAMX() (defined(__GNUC__) && (__GNUC__ >= 11))
#define CompileAMXBF16() (CompileAMX()))
#define CompileAMXINT8() (CompileAMX()))
#else
#define CompileAVX512F() _MSC_VER && (_MSC_VER >= 1911)
#define CompileAVX2() _MSC_VER && (_MSC_VER >= 1900)
#define CompileAMX() 0
#define CompileAMXBF16() 0
#define CompileAMXINT8() 0
#endif

namespace jblas {
namespace utils {

struct bf16 {
  uint16_t x;
  union bf16f32 {
    float f32;
    uint16_t bf16[2];
  };
  float tofloat() {
    bf16f32 tmp = {0.f};
    tmp.bf16[1] = x;
    return tmp.f32;
  }
  void fromfloat(float _v) {
    bf16f32 tmp = {0.f};
    tmp.f32 = _v;
    x = tmp.bf16[1];
  }
};

struct int4x2 {
  int8_t x : 4;
  int8_t y : 4;
  static int8_t convert(int8_t src) {
    int16_t dst = src;
    dst += 7;
    dst >>= 4;
    return dst > 7 ? 7 : dst;
  }
};

#ifndef _WIN32
#include <err.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/signal.h>
#include <sys/syscall.h>
#include <unistd.h>

#define fatal_error(msg, ...) err(1, "[FAIL]\t" msg, ##__VA_ARGS__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

static void request_perm_xtile_data() {
  unsigned long bitmask;
  long rc;

  rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc) fatal_error("XTILE_DATA request failed: %ld", rc);

  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (rc) fatal_error("prctl(ARCH_GET_XCOMP_PERM) error: %ld", rc);

  if (bitmask & XFEATURE_MASK_XTILE) printf("ARCH_REQ_XCOMP_PERM XTILE_DATA successful.\n");
}
#else
static void request_perm_xtile_data() {}
#endif

template <JBLAS_ISA ISA_T>
class isa_base {
 public:
  static bool constexpr avx = ISA_T >= JblasAVX;
  static bool constexpr avx2 = ISA_T >= JblasAVX2;
  static bool constexpr avx512f = ISA_T >= JblasAVX512F;
  static bool constexpr avx512_vnni = ISA_T >= JblasAVX512_VNNI;
  static bool constexpr amx_bf16 = ISA_T >= JblasAMX_BF16;
  static bool constexpr amx_int8 = ISA_T >= JblasAMX_INT8;
};

static inline int padto_le(int src, int padding) { return src / padding * padding; }

static inline int updiv(int a, int b) { return (a + b - 1) / b; }

static inline int downdiv(int a, int b) { return a / b; }

static inline int remainsize(int pos, int size, int N) { return pos + N <= size ? N : size - pos; }

template <typename _SRCT, typename _DSTT>
static inline _DSTT cast(_SRCT _src) {
  return static_cast<_DSTT>(_src);
}

template <>
int8_t cast(float _src) {
  _src = _src >= 0.f ? _src + 0.5f : _src - 0.5f;
  _src = std::min(_src, 127.f);
  _src = std::max(_src, -128.f);
  return static_cast<int8_t>(_src);
}

template <>
uint8_t cast(float _src) {
  _src += 0.5f;
  _src = std::min(_src, 255.f);
  _src = std::max(_src, 0.f);
  return static_cast<uint8_t>(_src);
}

template <>
float cast(bf16 _src) {
  return _src.tofloat();
}
template <>
bf16 cast(float _src) {
  bf16 tmp;
  tmp.fromfloat(_src);
  return tmp;
}

template <typename _T>
void serialize(int8_t*& buf, _T _val) {
  *(_T*)buf = _val;
  buf += sizeof(_T);
}

template <typename _T>
_T deserialize(int8_t*& buf) {
  auto val = *(_T*)buf;
  buf += sizeof(_T);
  return val;
}

static inline int padto(int a, int b) { return updiv(a, b) * b; }

template <typename _T, int _Alignment = 64>
class aligned_vector {
 public:
  aligned_vector() : mRawsize(0), mPtr(nullptr), mAlignedsize(0) {}
  aligned_vector(size_t _size, _T _val = {0}) {
    resize(_size);
    std::fill_n(mVec.begin(), mVec.size(), _val);
  }
  size_t size() { return mRawsize; }
  void resize(size_t size) {
    mRawsize = size;
    mAlignedsize = (mRawsize + _Alignment - 1) / _Alignment * _Alignment + _Alignment;
    mVec.resize(mAlignedsize);
    auto uptr = reinterpret_cast<uint64_t>(mVec.data());
    mPtr = reinterpret_cast<_T*>((uptr + _Alignment - 1) / _Alignment * _Alignment);
  }
  _T* data() const { return mPtr; }
  _T& operator[](size_t _n) noexcept { return mPtr[_n]; }

 protected:
  size_t mAlignedsize, mRawsize;
  std::vector<_T> mVec;
  _T* mPtr;
};

using milliseconds = std::chrono::milliseconds;
using nanoseconds = std::chrono::nanoseconds;
using microseconds = std::chrono::microseconds;
template <typename _DUR = std::chrono::milliseconds>
class timer {
 public:
  using sclock_t = std::chrono::steady_clock;
  using stime_point_t = std::chrono::time_point<sclock_t>;

  timer() { clear(); }

  void start() { startT = sclock_t::now(); }

  void clear() { startT = stime_point_t::min(); }

  bool null_state() { return startT == stime_point_t::min(); }

  float stop() { return static_cast<float>(std::chrono::duration_cast<_DUR>(sclock_t::now() - startT).count()); }

  stime_point_t startT;
};

template <typename T>
class minmax_statistics {
 public:
  minmax_statistics() { clear(); }

  void clear() {
    min_val = std::numeric_limits<T>::max();
    max_val = std::numeric_limits<T>::min();
    avg_val = 0;
    count = 0;
  }

  void add(T _val) {
    min_val = min_val > _val ? _val : min_val;
    max_val = max_val < _val ? _val : max_val;
    count += 1;
    avg_val = (avg_val * (count - 1) + _val) / count;
  }

  T min_val, max_val, avg_val;
  size_t count;
};

template <int _PRINT_CYCLE_MS = 100, typename _PRECISION = microseconds, typename _LOG_PRECISION = milliseconds>
class timer_statistics_logger {
 public:
  typedef timer<milliseconds> log_timer_t;
  timer_statistics_logger() {
    clear();
    log_ratio = (float)std::chrono::duration_cast<_PRECISION>(_LOG_PRECISION(1)).count();
  }

  void clear() {
    statis.clear();
    logtm.clear();
  }

  void start() {
    if (logtm.null_state()) {
      logtm.start();
    }
    tm.start();
  }

  bool stop() {
    auto elapsed = tm.stop();
    statis.add(elapsed);
    if (logtm.stop() >= _PRINT_CYCLE_MS) {
      record();
      clear();
      logtm.start();
      return true;
    }
    return false;
  }

  bool add(float time) {
    statis.add(time);
    if (logtm.stop() >= _PRINT_CYCLE_MS) {
      record();
      clear();
      logtm.start();
      return true;
    }
    return false;
  }

  const char* get_log_str() {
    sprintf(str, "Min:%.4f, Max:%.4f, Average:%.4f", min_val, max_val, avg_val);
    return str;
  }
  float min_val, max_val, avg_val;

 private:
  void record() {
    min_val = statis.min_val / log_ratio;
    max_val = statis.max_val / log_ratio;
    avg_val = statis.avg_val / log_ratio;
  }
  float log_ratio;
  char str[256];
  timer<_PRECISION> tm;
  minmax_statistics<float> statis;
  timer<milliseconds> logtm;
};

namespace parallel {

class CpuDevice {
 public:
  inline void setThreads(int _nth) {
    if (_nth <= 0) {
      numthreads = std::min(numcores, ompthreads);
    } else {
      numthreads = std::min(numcores, _nth);
      numthreads = std::min(ompthreads, _nth);
    }
#ifdef _OPENMP
    omp_set_num_threads(numthreads);
#endif
  }
  inline int getThreads() { return numthreads; }
  inline uint32_t getL2CacheSize() { return L2Cache; }
  inline uint32_t getL1CacheSize() { return L1Cache; }
  inline bool AVX() { return mHasAVX; }
  inline bool AVX2() { return mHasAVX2; }
  inline bool AVX_VNNI() { return mHasAVX_VNNI; }
  inline bool AVX512F() { return mHasAVX512F; }
  inline bool AVX512_VNNI() { return mHasAVX512_VNNI; }
  inline bool AMX_INT8() { return mHasAMX_INT8; }
  inline bool AMX_BF16() { return mHasAMX_BF16; }
#define ADD_FLAG(isa) mHas##isa = _cpu.has(_cpu.t##isa)
  CpuDevice() {
    static Xbyak::util::Cpu _cpu;
    L1Cache = _cpu.getDataCacheSize(0);
    L2Cache = _cpu.getDataCacheSize(1);
    ADD_FLAG(AVX);
    ADD_FLAG(AVX2);
    ADD_FLAG(AVX512F);
    ADD_FLAG(AVX512_VNNI);
    ADD_FLAG(AVX_VNNI);
    ADD_FLAG(AMX_BF16);
    ADD_FLAG(AMX_INT8);
    numcores = _cpu.getNumCores(Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    ompthreads = omp_get_max_threads();
    numthreads = std::min(numcores, ompthreads);
#ifdef FORCE_NUM_THREADS
    numthreads = FORCE_NUM_THREADS;
#endif
    omp_set_num_threads(numthreads);
  }

  static CpuDevice* getInstance() {
    static CpuDevice instance;
    return &instance;
  }
#undef ADD_FLAG

 protected:
  uint32_t L2Cache, L1Cache;
  bool mHasAVX2, mHasAVX_VNNI, mHasAVX, mHasAVX512_VNNI, mHasAMX_INT8, mHasAMX_BF16, mHasAVX512F;
  int numcores;
  int ompthreads;
  int numthreads;
};

#define GetCPUDevice() auto _cd = jblas::utils::parallel::CpuDevice::getInstance();

#define CheckISA(ISA)                       \
  {                                         \
    GetCPUDevice() if (!_cd->ISA()) {       \
      printf("Wrong Device ISA" #ISA "\n"); \
      return;                               \
    }                                       \
  }

struct Parallel2D {
  virtual void getIndex(int threadIdx, int* row, int* col, int* rowsize, int* colsize) {
    if (threadIdx >= mValidThreads) {
      *rowsize = 0;
      *colsize = 0;
      return;
    }
    int tx = threadIdx % mColThreads;
    int ty = threadIdx / mColThreads;
    *col = tx * mThdCol;
    *row = ty * mThdRow;
    *colsize = padto(remainsize(*col, mCols, mThdCol), mPadCol);
    *rowsize = padto(remainsize(*row, mRows, mThdRow), mPadRow);
  }

  void calc_valid_threads() { mValidThreads = mColThreads * int(std::ceil(float(mRows) / mThdRow)); }

  void print() {
    printf("Thread Block:(%d,%d)\n", mThdRow, mThdCol);
    printf("Thread in use:%d of %d, Nx%d\n", mValidThreads, mThreadsCount, mColThreads);
  }
  int mThdRow = 0, mThdCol = 0;
  int mColThreads = 0;
  int mRows = 0, mCols = 0;
  int mPadRow = 0, mPadCol = 0;
  int mValidThreads = 0, mThreadsCount = 0;
};

struct Parallel2DRowMajor : Parallel2D {
  void update(int row, int col, int minrow, int mincol, int ncores) {
    mCols = col;
    mRows = row;
    mPadCol = mincol;
    mPadRow = minrow;
    int colnum = updiv(col, mincol);
    int rownum = updiv(row, minrow);
    float ratio = colnum * rownum / float(ncores);
    if (ratio <= 1) {
      mThdRow = minrow;
      mColThreads = colnum;
      mThdCol = mincol;
      calc_valid_threads();
      return;
    }
    float colratio = ratio > colnum ? colnum : ceil(ratio);
    mThdCol = colratio * mincol;
    mColThreads = ceil(float(colnum) / colratio);
    mThdRow = ceil(rownum / (float(ncores) / mColThreads)) * minrow;
    calc_valid_threads();
  }
};
template <class _GemmCore_T>
struct Parallel2DGemm : Parallel2D {
 public:
  Parallel2DGemm() { mL2Size = CpuDevice::getInstance()->getL2CacheSize() * 0.8f; }
  static int constexpr BSize = sizeof(typename _GemmCore_T::BType);
  static int constexpr CSize = sizeof(typename _GemmCore_T::CType);
  bool update(int M, int N, int K, int threads) {
    mM = M;
    mN = N;
    mK = K;
    if (M == 0 || N == 0 || K == 0) {
      return false;
    }
    if (sameProblem(M, N, K, threads)) {
      return false;
    }
    mMPadded = padto(M, _GemmCore_T::MTILE);
    mNPadded = padto(N, _GemmCore_T::NTILE);
    mKPadded = padto(K, _GemmCore_T::KTILE);
    mPadCol = _GemmCore_T::NTILE;
    mPadRow = _GemmCore_T::MTILE;
    mRows = M;
    mCols = N;
    mThreadsCount = threads;
    int rownum = updiv(mRows, _GemmCore_T::MTILE);
    int colnum = updiv(mCols, _GemmCore_T::NTILE);
    mDensity = float(mRows) * mCols / (mRows + mCols);
    int maxN = 0;
    float maxScore = std::numeric_limits<float>::min();
    int core_enum = sqrt(mThreadsCount);
    for (int i = 1; i <= core_enum; i += 1) {
      generate_by_cores(i, mThreadsCount / i, rownum, colnum);
      auto thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = i;
      }
      generate_by_cores(mThreadsCount / i, i, rownum, colnum);
      thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = mThreadsCount / i;
      }
    }
    generate_by_cores(maxN, mThreadsCount / maxN, rownum, colnum);
    update_cache_blocking();

    float BA_ratio = float(N) / M;
    if (BA_ratio >= 10) {
      // B matrix is too big, need split K to reduce latency
      int const NStage = 10;
      int const K_Split = padto(updiv(K, 10), _GemmCore_T::KTILE);
      if (mKStep > K_Split) {
        mKStep = K_Split;
      }
    }
    return true;
  }
  inline int getN() { return mN; }
  inline int getM() { return mM; }
  inline int getK() { return mK; }
  inline int getPaddedN() { return mNPadded; }
  inline int getPaddedM() { return mMPadded; }
  inline int getPaddedK() { return mKPadded; }
  inline int getNStep() { return mNStep; }
  inline int getMStep() { return mMStep; }
  inline int getKStep() { return mKStep; }
  inline bool sameProblem(int m, int n, int k, int numthd) {
    return m == mM && n == mN && k == mK && numthd == mThreadsCount;
  }
  void print() {
    Parallel2D::print();
    printf("GEMM MStep:%d NStep:%d KStep:%d\n", getMStep(), getNStep(), getKStep());
    printf("Cache Size:%llu\n", mL2Size);
  }

 protected:
  float calculate_score() {
    int tmpnstep = mThdCol < _GemmCore_T::PREFERED_N ? mThdCol : _GemmCore_T::PREFERED_N;
    float threadratio = float(mValidThreads) / mThreadsCount;
    float density = float(tmpnstep) * mThdRow / (tmpnstep + mThdRow);
    const float Thres = 64;
    if (mDensity < Thres) {
      return (threadratio * 1.f + density * 0.0016f) * density / mDensity;
    }
    return (threadratio * 1.f + density * 0.0016f);
  }

  void generate_by_cores(int ny, int nx, int rownum, int colnum) {
    mThdRow = updiv(rownum, ny) * _GemmCore_T::MTILE;
    mThdCol = updiv(colnum, nx) * _GemmCore_T::NTILE;
    mColThreads = updiv(mCols, mThdCol);
    mValidThreads = updiv(mRows, mThdRow) * mColThreads;
  }

  // cache = mMStep * mNStep * CSize + mNStep * mKStep * BSize
  //       = mNStep * (mMStep*CSize + mKStep*BSize)
  // C Access = K/mKStep
  // B Access = M/mMStep
  // A Access = N/mNStep
  void update_cache_blocking() {
    int constexpr MRef = 256, KRef = 256;
    size_t csize_total = mL2Size - _GemmCore_T::PREFERED_N * KRef * BSize;
    int maxM = csize_total / _GemmCore_T::PREFERED_N / CSize;
    maxM = downdiv(maxM, _GemmCore_T::MTILE);
    int nthdm = mThdRow / _GemmCore_T::MTILE;
    if (maxM < nthdm) {
      int niter = updiv(nthdm, maxM);
      mMStep = updiv(nthdm, niter) * _GemmCore_T::MTILE;
    } else {
      mMStep = mThdRow;
    }
    int maxN = mL2Size / (mMStep * CSize + KRef * BSize);
    maxN = downdiv(maxN, _GemmCore_T::NTILE);
    int nthdn = mThdCol / _GemmCore_T::NTILE;
    if (maxN < nthdn) {
      int niter = updiv(nthdn, maxN);
      mNStep = updiv(nthdn, niter) * _GemmCore_T::NTILE;
    } else {
      mNStep = mThdCol;
    }
    update_kstep();
  }
  void update_kstep() {
    auto rawk = (mL2Size / mNStep - mMStep * CSize) / BSize;
    mKStep = padto_le(rawk, _GemmCore_T::KTILE);
  }

  size_t mL2Size = 0;
  int mNStep = 0;
  int mMStep = 0;
  int mKStep = 0;
  float mDensity = 0.f;
  int mM = 0, mN = 0, mK = 0;
  int mMPadded = 0, mNPadded = 0, mKPadded = 0;
};

struct Parallel2DRowMajorColBlock : Parallel2D {
  int mThdsPerBlock;
  int mBlockPerThread;
  int mBlocksPerCol;
  int mColBlock;
  int mTmpStride;
  int mTmpSize;
  void update(int row, int col, int minrow, int mincol, int colblock, int ncores) {
    mCols = col;
    mRows = row;
    mPadCol = mincol;
    mPadRow = minrow;
    mColBlock = colblock;
    mThreadsCount = ncores;
    int colnum = updiv(col, mColBlock);
    int blockcount = colnum * row;
    float blockperthd = float(blockcount) / mThreadsCount;
    float colratio = blockperthd > colnum ? colnum : ceil(blockperthd);
    if (blockperthd <= 1) {
      int tilecount = updiv(col, mincol);
      float tileperthd = float(tilecount) / mThreadsCount;
      colratio = tileperthd > tilecount ? tilecount : ceil(tileperthd);
      mThdCol = colratio * mincol;
      goto __COL_EPI;
    }
    mThdCol = padto(colratio * mColBlock, mPadCol);
  __COL_EPI:
    mBlocksPerCol = utils::updiv(mCols, mColBlock);
    if (mThdCol > mColBlock) {
      mThdsPerBlock = 1;
      mBlockPerThread = downdiv(mThdCol, mColBlock);
      mColThreads = updiv(mBlocksPerCol, mBlockPerThread);
    } else {
      mThdCol = mColBlock;
      mThdsPerBlock = 1;
      mBlockPerThread = 1;
      mColThreads = mThdsPerBlock * mBlocksPerCol;
    }

    mThdRow = ceil(mRows / (float(mThreadsCount) / mColThreads)) * mPadRow;
    calc_valid_threads();
  }

  virtual void getIndex(int threadIdx, int* row, int* col, int* rowsize, int* colsize, int* block, int* idxinblk) {
    if (threadIdx >= mValidThreads) {
      *rowsize = 0;
      *colsize = 0;
      return;
    }
    int tx = threadIdx % mColThreads;
    int tb = tx / mThdsPerBlock;
    int ty = threadIdx / mColThreads;
    if (mThdsPerBlock > 1) {
      *block = tb;
      *idxinblk = tx % mThdsPerBlock;
      *col = tb * mColBlock + *idxinblk * mThdCol;
      *colsize = padto(remainsize(*col, *col + mColBlock, mThdCol), mPadCol);
    } else {
      *idxinblk = 0;
      *block = tb * mBlockPerThread;
      *col = tx * mThdCol;
      *colsize = padto(remainsize(*col, mCols, mThdCol), mPadCol);
    }
    *row = ty * mThdRow;
    *rowsize = padto(remainsize(*row, mRows, mThdRow), mPadRow);
  }

  size_t getTmpSize(size_t elesize, int sizepadding = 64) {
    mTmpSize = padto(elesize, sizepadding);
    mTmpStride = mBlocksPerCol * mTmpSize * mThdsPerBlock;
    return mRows * mTmpStride;
  }
  void print() {
    Parallel2D::print();
    printf("Blocks per col:%d\n", mBlocksPerCol);
    printf("Blocks per Thread:%d\n", mBlockPerThread);
    printf("Threads per Block:%d\n", mThdsPerBlock);
  }
};

template <class _GemmCore_T>
struct Parallel2DGemmKBlock : Parallel2D {
 public:
  Parallel2DGemmKBlock() { mL2Size = CpuDevice::getInstance()->getL2CacheSize() * 0.8f; }
  static int constexpr BSize = sizeof(typename _GemmCore_T::BType);
  static int constexpr CSize = sizeof(typename _GemmCore_T::CType);
  bool update(int M, int N, int K, int KBlock, int threads) {
    mM = M;
    mN = N;
    mK = K;
    if (M == 0 || N == 0 || K == 0) {
      return false;
    }
    if (sameProblem(M, N, K, threads)) {
      return false;
    }
    if (_GemmCore_T::KTILE > KBlock || KBlock % _GemmCore_T::KTILE != 0) {
      assert(0);  // invalid parameter
      return false;
    }
    mMPadded = padto(M, _GemmCore_T::MTILE);
    mNPadded = padto(N, _GemmCore_T::NTILE);
    mKPadded = padto(K, _GemmCore_T::KTILE);
    mPadCol = _GemmCore_T::NTILE;
    mPadRow = _GemmCore_T::MTILE;
    mRows = M;
    mCols = N;
    mThreadsCount = threads;
    int rownum = updiv(mRows, _GemmCore_T::MTILE);
    int colnum = updiv(mCols, _GemmCore_T::NTILE);
    mDensity = float(mRows) * mCols / (mRows + mCols);
    int maxN = 0;
    float maxScore = std::numeric_limits<float>::min();
    int core_enum = sqrt(mThreadsCount);
    for (int i = 1; i <= core_enum; i += 1) {
      generate_by_cores(i, mThreadsCount / i, rownum, colnum);
      auto thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = i;
      }
      generate_by_cores(mThreadsCount / i, i, rownum, colnum);
      thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = mThreadsCount / i;
      }
    }
    generate_by_cores(maxN, mThreadsCount / maxN, rownum, colnum);
    update_cache_blocking(KBlock);
    return true;
  }
  inline int getN() { return mN; }
  inline int getM() { return mM; }
  inline int getK() { return mK; }
  inline int getPaddedN() { return mNPadded; }
  inline int getPaddedM() { return mMPadded; }
  inline int getPaddedK() { return mKPadded; }
  inline int getNStep() { return mNStep; }
  inline int getMStep() { return mMStep; }
  inline int getKStep() { return mKStep; }
  inline bool sameProblem(int m, int n, int k, int numthd) {
    return m == mM && n == mN && k == mK && numthd == mThreadsCount;
  }
  void print() {
    Parallel2D::print();
    printf("GEMM MStep:%d NStep:%d KStep:%d\n", getMStep(), getNStep(), getKStep());
    printf("Cache Size:%llu\n", mL2Size);
  }

 protected:
  float calculate_score() {
    int tmpnstep = mThdCol < _GemmCore_T::PREFERED_N ? mThdCol : _GemmCore_T::PREFERED_N;
    float threadratio = float(mValidThreads) / mThreadsCount;
    float density = float(tmpnstep) * mThdRow / (tmpnstep + mThdRow);
    const float Thres = 64;
    if (mDensity < Thres) {
      return (threadratio * 1.f + density * 0.0016f) * density / mDensity;
    }
    return (threadratio * 1.f + density * 0.0016f);
  }

  void generate_by_cores(int ny, int nx, int rownum, int colnum) {
    mThdRow = updiv(rownum, ny) * _GemmCore_T::MTILE;
    mThdCol = updiv(colnum, nx) * _GemmCore_T::NTILE;
    mColThreads = updiv(mCols, mThdCol);
    mValidThreads = updiv(mRows, mThdRow) * mColThreads;
  }

  void update_cache_blocking(int kblock) {
    int constexpr MRef = 256;
    int kRef = kblock > 256 ? 256 : kblock;
    size_t csize_total = mL2Size - _GemmCore_T::PREFERED_N * kRef * BSize;
    int maxM = csize_total / _GemmCore_T::PREFERED_N / CSize;
    maxM = downdiv(maxM, _GemmCore_T::MTILE);
    int nthdm = mThdRow / _GemmCore_T::MTILE;
    if (maxM < nthdm) {
      int niter = updiv(nthdm, maxM);
      mMStep = updiv(nthdm, niter) * _GemmCore_T::MTILE;
    } else {
      mMStep = mThdRow;
    }
    int maxN = mL2Size / (mMStep * CSize + kRef * BSize);
    maxN = downdiv(maxN, _GemmCore_T::NTILE);
    int nthdn = mThdCol / _GemmCore_T::NTILE;
    if (maxN < nthdn) {
      int niter = updiv(nthdn, maxN);
      mNStep = updiv(nthdn, niter) * _GemmCore_T::NTILE;
    } else {
      mNStep = mThdCol;
    }
    mKStep = kRef;
  }

  void update_kstep() {
    auto rawk = (mL2Size / mNStep - mMStep * CSize) / BSize;
    mKStep = padto_le(rawk, _GemmCore_T::KTILE);
  }

  size_t mL2Size = 0;
  int mNStep = 0;
  int mMStep = 0;
  int mKStep = 0;
  float mDensity = 0.f;
  int mM = 0, mN = 0, mK = 0;
  int mMPadded = 0, mNPadded = 0, mKPadded = 0;
};

template <class _GemmCore_T>
struct Parallel2DGemmKBlockFixed : Parallel2D {
 public:
  Parallel2DGemmKBlockFixed() { mL2Size = CpuDevice::getInstance()->getL2CacheSize() * 0.8f; }
  static int constexpr BSize = sizeof(typename _GemmCore_T::BType);
  static int constexpr CSize = sizeof(typename _GemmCore_T::CType);
  bool update(int M, int N, int K, int KBlock, int threads) {
    mM = M;
    mN = N;
    mK = K;
    if (M == 0 || N == 0 || K == 0) {
      return false;
    }
    if (sameProblem(M, N, K, threads)) {
      return false;
    }
    if (_GemmCore_T::KTILE > KBlock || KBlock % _GemmCore_T::KTILE != 0) {
      assert(0);  // invalid parameter
      return false;
    }
    mMPadded = padto(M, _GemmCore_T::MTILE);
    mNPadded = padto(N, _GemmCore_T::NTILE);
    mKPadded = padto(K, _GemmCore_T::KTILE);
    mPadCol = _GemmCore_T::NTILE;
    mPadRow = _GemmCore_T::MTILE;
    mRows = M;
    mCols = N;
    mThreadsCount = threads;
    int rownum = updiv(mRows, _GemmCore_T::MTILE);
    int colnum = updiv(mCols, _GemmCore_T::NTILE);
    mDensity = float(mRows) * mCols / (mRows + mCols);
    int maxN = 0;
    float maxScore = std::numeric_limits<float>::min();
    int core_enum = sqrt(mThreadsCount);
    for (int i = 1; i <= core_enum; i += 1) {
      generate_by_cores(i, mThreadsCount / i, rownum, colnum);
      auto thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = i;
      }
      generate_by_cores(mThreadsCount / i, i, rownum, colnum);
      thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = mThreadsCount / i;
      }
    }
    generate_by_cores(maxN, mThreadsCount / maxN, rownum, colnum);
    update_cache_blocking(KBlock);
    return true;
  }
  inline int getN() { return mN; }
  inline int getM() { return mM; }
  inline int getK() { return mK; }
  inline int getPaddedN() { return mNPadded; }
  inline int getPaddedM() { return mMPadded; }
  inline int getPaddedK() { return mKPadded; }
  inline int getNStep() { return mNStep; }
  inline int getMStep() { return mMStep; }
  inline int getKStep() { return mKStep; }
  inline bool sameProblem(int m, int n, int k, int numthd) {
    return m == mM && n == mN && k == mK && numthd == mThreadsCount;
  }
  void print() {
    Parallel2D::print();
    printf("GEMM MStep:%d NStep:%d KStep:%d\n", getMStep(), getNStep(), getKStep());
    printf("Cache Size:%llu\n", mL2Size);
  }

 protected:
  float calculate_score() {
    int tmpnstep = mThdCol < _GemmCore_T::PREFERED_N ? mThdCol : _GemmCore_T::PREFERED_N;
    float threadratio = float(mValidThreads) / mThreadsCount;
    float density = float(tmpnstep) * mThdRow / (tmpnstep + mThdRow);
    const float Thres = 64;
    if (mDensity < Thres) {
      return (threadratio * 1.f + density * 0.0016f) * density / mDensity;
    }
    return (threadratio * 1.f + density * 0.0016f);
  }

  void generate_by_cores(int ny, int nx, int rownum, int colnum) {
    mThdRow = updiv(rownum, ny) * _GemmCore_T::MTILE;
    mThdCol = updiv(colnum, nx) * _GemmCore_T::NTILE;
    mColThreads = updiv(mCols, mThdCol);
    mValidThreads = updiv(mRows, mThdRow) * mColThreads;
  }

  void update_cache_blocking(int kblock) {
    int constexpr MRef = 256;
    int kRef = 256;
    if (kblock > 256) {
      kRef = kblock / 2;
    }
    if (kRef % kblock != 0) {
      kRef = padto(kRef, kblock);
    }
    size_t csize_total = mL2Size - _GemmCore_T::PREFERED_N * kRef * BSize;
    int maxM = csize_total / _GemmCore_T::PREFERED_N / CSize;
    maxM = downdiv(maxM, _GemmCore_T::MTILE);
    int nthdm = mThdRow / _GemmCore_T::MTILE;
    if (maxM < nthdm) {
      int niter = updiv(nthdm, maxM);
      mMStep = updiv(nthdm, niter) * _GemmCore_T::MTILE;
    } else {
      mMStep = mThdRow;
    }
    int maxN = mL2Size / (mMStep * CSize + kRef * BSize);
    maxN = downdiv(maxN, _GemmCore_T::NTILE);
    int nthdn = mThdCol / _GemmCore_T::NTILE;
    if (maxN < nthdn) {
      int niter = updiv(nthdn, maxN);
      mNStep = updiv(nthdn, niter) * _GemmCore_T::NTILE;
    } else {
      mNStep = mThdCol;
    }
    mKStep = kRef;
  }

  void update_kstep() {
    auto rawk = (mL2Size / mNStep - mMStep * CSize) / BSize;
    mKStep = padto_le(rawk, _GemmCore_T::KTILE);
  }

  size_t mL2Size = 0;
  int mNStep = 0;
  int mMStep = 0;
  int mKStep = 0;
  float mDensity = 0.f;
  int mM = 0, mN = 0, mK = 0;
  int mMPadded = 0, mNPadded = 0, mKPadded = 0;
};

}  // namespace parallel

class CpuBase {
 public:
  CpuBase() {
    GetCPUDevice();
    mL2Cache = _cd->getL2CacheSize();
    mNumThreads = _cd->getThreads();
  }
  size_t mL2Cache;
  int mNumThreads;
};

}  // namespace utils
}  // namespace jblas