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
#include "kernels/data_pack.hpp"
#include "src/singleton.hpp"
#include "cpu_parallel.hpp"

namespace jd {
template <typename dst_t, typename src_t>
void reference(
    dst_t* output, src_t* input, int row, int col, int rowpad, int colpad, int srcstride, int dststride,
    std::function<dst_t(src_t)> cast_func = [](src_t x) -> dst_t { return static_cast<dst_t>(x); }) {
  int srcld = srcstride / 2;
  int NTile = 16;
  for (int irow = 0; irow < rowpad; irow += NTile) {
    for (int icol = 0; icol < colpad; icol += 1) {
      for (int iin = 0; iin < NTile; iin++) {
        if (irow + iin < row) {
          if (icol < col) {
            *(output + irow * dststride + icol * NTile + iin) = cast_func(*(input + (irow + iin) * srcld + icol));
          } else {
            *(output + irow * dststride + icol * NTile + iin) = cast_func(static_cast<src_t>(0));
          }
        } else {
          *(output + irow * dststride + icol * NTile + iin) = cast_func(static_cast<src_t>(0));
        }
      }
    }
  }
}

template <typename dst_t, typename src_t>
void pack(dst_t* output, src_t* input, dim_t N, dim_t K, std::function<dst_t(src_t)> cast_func) {
  CpuDevice* cpudevice = Singleton<CpuDevice>::GetInstance();
  int npad = pad_to(N, 16);
  int kpad = pad_to(K, 1);
  auto ncores = cpudevice->getThreads();
  Parallel2DRowMajor _para;
  _para.update(npad, kpad, 16, 1, ncores);
#pragma omp parallel
  {
    int tidx = omp_get_thread_num();
    int colidx, rowidx, rowsize, colsize;
    _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      int rowremain = remainsize(rowidx, N, rowsize);
      int colremain = remainsize(colidx, K, colsize);
      reference<dst_t, src_t>(output + rowidx * kpad + colidx * 16, input + rowidx * K + colidx, rowremain, colremain,
                              rowsize, colsize, K * sizeof(bfloat16_t), kpad, cast_func);
    }
  }
}
#define DECLARE_DATA_PACK(dst_type, src_type)                                            \
  template SPARSE_API_ void pack<dst_type, src_type>(dst_type*, src_type*, dim_t, dim_t, \
                                                     std::function<dst_type(src_type)>);

DECLARE_DATA_PACK(int8_t, int8_t)
DECLARE_DATA_PACK(float8_e4m3_t, float8_e4m3_t)
DECLARE_DATA_PACK(float8_e5m2_t, float8_e5m2_t)
DECLARE_DATA_PACK(float8_e4m3_t, bfloat16_t)
DECLARE_DATA_PACK(float8_e5m2_t, bfloat16_t)
#undef DECLARE_DATA_PACK

}  // namespace jd
