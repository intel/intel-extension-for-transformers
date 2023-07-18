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
#include "hw_topology.hpp"

namespace hw {
template <int nSocket, int nHT>
class HardwareTopology : __hwtopo {
 public:
  HardwareTopology() = default;

 private:
  static const size_t nCores_;
  static const size_t nThreads_;

  static constexpr size_t nScoket_ = nSocket;
  static constexpr size_t nHT_ = nHT;
};

template <int nSocket, int nHT>
const size_t HardwareTopology<nSocket, nHT>::nCores_
  = kmp::KMPLauncher::getMaxProc() / nHT;

template <int nSocket, int nHT>
const size_t HardwareTopology<nSocket, nHT>::nThreads_
  = kmp::KMPLauncher::getMaxProc();
}  // namespace hw
