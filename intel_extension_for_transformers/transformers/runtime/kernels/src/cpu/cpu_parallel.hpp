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

#ifndef ENGINE_SPARSELIB_SRC_CPU_CPU_PARALLEL_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_CPU_PARALLEL_HPP_

#include <algorithm>
#include <limits>
#include "src/utils.hpp"

namespace jd {
struct CpuDevice {
  CpuDevice();
  int getThreads() const { return numthreads; }
  void setThreads(int thread_nums) { numthreads = thread_nums; }
  int numthreads;
  unsigned int L2Cache, L1Cache;
  bool mHasVNNI512, mHasAMXINT8, mHasAMXBF16, mHas512F;
  int numcores;
  int ompthreads;
};

template <typename _T>
struct GemmCacheAdpter {
  void update(int m, int n, int k, int cachesize, int minN, int preferedN, float cacheratio = 0.8f) {
    (void)m;
    mKTotal = k;
    int constexpr EleSize = sizeof(_T);
    mElesize = cachesize * cacheratio / EleSize;
    int n_start = mElesize / k;
    if (n_start >= n) {
      mNMax = pad_to_le(n_start, minN);
      mKBatch = k;
      return;
    }
    int k_min = mElesize / preferedN;
    float c_acccess_ratio = static_cast<float>(k) / k_min;
    float c_threshold = 1.5;  // merge C access by using small N
    if (c_acccess_ratio <= c_threshold) {
      mNMax = pad_to_le(n_start, minN);
      mKBatch = k;
      return;
    }
    mNMax = preferedN;
    mKBatch = k_min;
  }
  void set_N(int _N, bool _minm) {
    mNMax = _N;
    mKBatch = _minm ? mKTotal : mElesize / mNMax;
  }

  int mKBatch;
  int mNMax;
  int mKTotal;
  int mElesize;
};

struct Parallel2D {
  void getIndex(int threadIdx, int* row, int* col, int* rowsize, int* colsize) const {
    if (threadIdx >= mValidThreads) {
      *rowsize = 0;
      *colsize = 0;
      return;
    }
    int tx = threadIdx % mColThreads;
    int ty = threadIdx / mColThreads;
    *col = tx * mThdCol;
    *row = ty * mThdRow;
    *colsize = remainsize(*col, mCols, mThdCol);
    *rowsize = remainsize(*row, mRows, mThdRow);
  }

  void calc_valid_threads() { mValidThreads = mColThreads * std::ceil(static_cast<float>(mRows) / mThdRow); }

  void print() {
    printf("Thread Block:(%d,%d)\n", mThdRow, mThdCol);
    printf("Thread in use:%d of %d, Nx%d\n", mValidThreads, mThreadsCount, mColThreads);
  }
  int mThdRow, mThdCol;
  int mColThreads;
  int mRows, mCols;
  int mValidThreads, mThreadsCount;
};

struct Parallel2DRowMajor : Parallel2D {
  void update(int row, int col, int minrow, int mincol, int ncores) {
    mCols = col;
    mRows = row;
    int colnum = ceil_div(col, mincol);
    int rownum = ceil_div(row, minrow);
    float ratio = colnum * rownum / static_cast<float>(ncores);
    if (ratio <= 1) {
      mThdRow = minrow;
      mColThreads = colnum;
      mThdCol = mincol;
      calc_valid_threads();
      return;
    }
    float colratio = ratio > colnum ? colnum : ceil(ratio);
    mThdCol = colratio * mincol;
    mColThreads = ceil(static_cast<float>(colnum) / colratio);
    mThdRow = ceil(rownum / (static_cast<float>(ncores) / mColThreads)) * minrow;
    calc_valid_threads();
  }
};

template <typename _T>
struct Parallel2DGemmV2 : Parallel2D {
  void update(int row, int col, int minrow, int mincol, int ncores,
              GemmCacheAdpter<_T>& _adapter) {  // NOLINT
    mRows = row;
    mCols = col;
    mMinRow = minrow;
    mMinCol = mincol;
    mThreadsCount = ncores;
    mNStep = _adapter.mNMax;
    int rownum = ceil_div(mRows, mMinRow);
    int colnum = ceil_div(mCols, mMinCol);
    int maxN = 1;
    float maxScore = std::numeric_limits<float>::min();
    int core_enum = std::sqrt(mThreadsCount);
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
        maxN = ncores / i;
      }
    }
    generate_by_cores(maxN, mThreadsCount / maxN, rownum, colnum);
    mNStep = std::min(mNStep, mThdCol);
  }

  float calculate_score() {
    int tmpnstep = mThdCol < mNStep ? mThdCol : mNStep;
    float threadratio = static_cast<float>(mValidThreads) / mThreadsCount;
    float tileratio = static_cast<float>(mThdCol) / pad_to(mThdCol, tmpnstep);
    float density = static_cast<float>(tmpnstep) * mThdRow / (tmpnstep + mThdRow);
    density /= tmpnstep;
    return threadratio * 4.f + density * tileratio * 0.2f;
  }

  void generate_by_cores(int ny, int nx, int rownum, int colnum) {
    mThdRow = ceil_div(rownum, ny) * mMinRow;
    mThdCol = ceil_div(colnum, nx) * mMinCol;
    mColThreads = ceil_div(mCols, mThdCol);
    mValidThreads = ceil_div(mRows, mThdRow) * mColThreads;
  }

  void print() {
    Parallel2D::print();
    printf("GEMM NStep:%d\n", mNStep);
  }
  int mNStep;
  int mMinCol, mMinRow;
};
}  // namespace jd

#endif  // ENGINE_SPARSELIB_SRC_CPU_CPU_PARALLEL_HPP_
