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
#include <utility>
#include "kmp_launcher.hpp"

namespace kmp {
  void KMPLauncher::pinRoot(int place) {
    KMPAffinityMask mask;
    mask.addCore(place).bind();
  }

  int KMPLauncher::getMaxProc() {
    return KMPAffinityMask::getMaxProc();
  }

  void KMPLauncher::pinThreads() {
    if (!affinityChanged_ && places_.size() > 0)
      return;

    setNumOfThreads(places_.size());

#pragma omp parallel
    {
      KMPAffinityMask mask;
      mask.addCore(places_[omp_get_thread_num()]).bind();
    }

    affinityChanged_ = false;
  }

  KMPLauncher& KMPLauncher::setAffinityPlaces(std::vector<int> newPlaces) {
    // std::sort(newPlaces.begin(), newPlaces.end(), std::less<int>());

    if (places_ != newPlaces) {
      places_ = std::move(newPlaces);
      affinityChanged_ = true;
    }

    return *this;
  }

  std::vector<int> KMPLauncher::getAffinityPlaces() const {
    return places_;
  }

  size_t KMPLauncher::getNumOfThreads() {
    return omp_get_thread_num();
  }

  void KMPLauncher::setNumOfThreads(int nCores) {
    omp_set_num_threads(nCores);
  }

}  // namespace kmp
