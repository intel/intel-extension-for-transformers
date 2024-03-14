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
#include <omp.h>
#include <xbyak/xbyak_util.h>

#include "cpu_parallel.hpp"

#include "src/singleton.hpp"
#include "src/utils.hpp"

namespace jd {
CpuDevice::CpuDevice() {
  Xbyak::util::Cpu* cpu = Singleton<Xbyak::util::Cpu>::GetInstance();
  numcores = cpu->getNumCores(Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
  ompthreads = omp_get_max_threads();
  numthreads = std::min(numcores, ompthreads);
  omp_set_num_threads(numthreads);
  L1Cache = cpu->getDataCacheSize(0);
  L2Cache = cpu->getDataCacheSize(1);
  mHas512F = cpu->has(cpu->tAVX512F);
  mHasVNNI512 = cpu->has(cpu->tAVX512_VNNI);
  mHasAMXBF16 = cpu->has(cpu->tAMX_BF16);
  mHasAMXINT8 = cpu->has(cpu->tAMX_INT8);
}
}  // namespace jd
