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

  if (bitmask & XFEATURE_MASK_XTILE)
    printf("ARCH_REQ_XCOMP_PERM XTILE_DATA successful.\n");
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

static inline int padto_le(int src, int padding) {
  return src / padding * padding;
}

static inline int updiv(int a, int b) { return (a + b - 1) / b; }

static inline int downdiv(int a, int b) { return a / b; }

static inline int remainsize(int pos, int size, int N) {
  return pos + N <= size ? N : size - pos;
}

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
void serialize(int8_t *&buf, _T _val) {
  *(_T *)buf = _val;
  buf += sizeof(_T);
}

template <typename _T>
_T deserialize(int8_t *&buf) {
  auto val = *(_T *)buf;
  buf += sizeof(_T);
  return val;
}

static inline int padto(int a, int b) { return updiv(a, b) * b; }

template <typename _T, int _Alignment = 64>
class aligned_vector {
 public:
  aligned_vector() : mRawsize(0), mPtr(nullptr), mAlignedsize(0) {}
  aligned_vector(size_t _size, _T _val = _T(0)) {
    resize(_size);
    std::fill_n(mVec.begin(), mVec.size(), _val);
  }
  size_t size() { return mRawsize; }
  void resize(size_t size) {
    mRawsize = size;
    mAlignedsize =
        (mRawsize + _Alignment - 1) / _Alignment * _Alignment + _Alignment;
    mVec.resize(mAlignedsize);
    auto uptr = reinterpret_cast<uint64_t>(mVec.data());
    mPtr = reinterpret_cast<_T *>((uptr + _Alignment - 1) / _Alignment *
                                  _Alignment);
  }
  _T *data() const { return mPtr; }
  _T &operator[](size_t _n) noexcept { return mPtr[_n]; }

 protected:
  size_t mAlignedsize, mRawsize;
  std::vector<_T> mVec;
  _T *mPtr;
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

  float stop() {
    return static_cast<float>(
        std::chrono::duration_cast<_DUR>(sclock_t::now() - startT).count());
  }

  stime_point_t startT;
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

  static CpuDevice *getInstance() {
    static CpuDevice instance;
    return &instance;
  }
#undef ADD_FLAG

 protected:
  uint32_t L2Cache, L1Cache;
  bool mHasAVX2, mHasAVX_VNNI, mHasAVX, mHasAVX512_VNNI, mHasAMX_INT8,
      mHasAMX_BF16, mHasAVX512F;
  int numcores;
  int ompthreads;
  int numthreads;
};

#define GetCPUDevice() \
  auto _cd = jblas::utils::parallel::CpuDevice::getInstance();

#define CheckISA(ISA)                       \
  {                                         \
    GetCPUDevice() if (!_cd->ISA()) {       \
      printf("Wrong Device ISA" #ISA "\n"); \
      return;                               \
    }                                       \
  }

struct Parallel2D {
  virtual void getIndex(int threadIdx, int *row, int *col, int *rowsize,
                        int *colsize) {
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

  void calc_valid_threads() {
    mValidThreads = mColThreads * int(std::ceil(float(mRows) / mThdRow));
  }

  void print() {
    printf("Thread Block:(%d,%d)\n", mThdRow, mThdCol);
    printf("Thread in use:%d of %d, Nx%d\n", mValidThreads, mThreadsCount,
           mColThreads);
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

struct Parallel2DGemm : Parallel2D {
 public:
  Parallel2DGemm(int _bsize, int _csize) : BSize(_bsize), CSize(_csize) {}

  void update(int M, int N, int K, int threads, size_t L2size, int MTile,
              int NTile, int KTile, int PreferedN) {
    mM = M;
    mN = N;
    mK = K;
    if (M == 0 || N == 0 || K == 0) {
      return;
    }
    mMPadded = padto(M, MTile);
    mNPadded = padto(N, NTile);
    mKPadded = padto(K, KTile);
    mKTile = KTile;
    mMTile = MTile;
    mNTile = NTile;
    mPadCol = mNTile;
    mPadRow = mMTile;
    mNRef = PreferedN;
    mL2Size = L2size;
    mRows = M;
    mCols = N;
    mThreadsCount = threads;
    int rownum = updiv(mRows, mMTile);
    int colnum = updiv(mCols, mNTile);
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
      int const K_Split = padto(updiv(K, 10), mKTile);
      if (mKStep > K_Split) {
        mKStep = K_Split;
      }
    }
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
  inline bool sameProblem(int m, int n, int k) {
    return m == mM && n == mN && k == mK;
  }
  void print() {
    Parallel2D::print();
    printf("GEMM MStep:%d NStep:%d KStep:%d\n", getMStep(), getNStep(),
           getKStep());
  }

 protected:
  float calculate_score() {
    int tmpnstep = mThdCol < mNRef ? mThdCol : mNRef;
    float threadratio = float(mValidThreads) / mThreadsCount;
    float density = float(tmpnstep) * mThdRow / (tmpnstep + mThdRow);
    const float Thres = 64;
    if (mDensity < Thres) {
      return (threadratio * 1.f + density * 0.0016f) * density / mDensity;
    }
    return (threadratio * 1.f + density * 0.0016f);
  }

  void generate_by_cores(int ny, int nx, int rownum, int colnum) {
    mThdRow = updiv(rownum, ny) * mMTile;
    mThdCol = updiv(colnum, nx) * mNTile;
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
    size_t csize_total = mL2Size - mNRef * KRef * BSize;
    int maxM = csize_total / mNRef / CSize;
    maxM = downdiv(maxM, mMTile);
    int nthdm = mThdRow / mMTile;
    if (maxM < nthdm) {
      int niter = updiv(nthdm, maxM);
      mMStep = updiv(nthdm, niter) * mMTile;
    } else {
      mMStep = mThdRow;
    }
    int maxN = mL2Size / (mMStep * CSize + KRef * BSize);
    maxN = downdiv(maxN, mNTile);
    int nthdn = mThdCol / mNTile;
    if (maxN < nthdn) {
      int niter = updiv(nthdn, maxN);
      mNStep = updiv(nthdn, niter) * mNTile;
    } else {
      mNStep = mThdCol;
    }
    update_kstep();
  }
  void update_kstep() {
    auto rawk = (mL2Size / mNStep - mMStep * CSize) / BSize;
    mKStep = padto_le(rawk, mKTile);
  }

  const int BSize, CSize;
  size_t mL2Size = 0;
  int mNStep = 0;
  int mMStep = 0;
  int mKStep = 0;
  int mKTile = 0, mNTile = 0, mMTile = 0;
  int mNRef = 0;
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