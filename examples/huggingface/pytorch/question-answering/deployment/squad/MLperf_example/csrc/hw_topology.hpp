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
#pragma once

#include <cstddef>
#include "kmp_launcher.hpp"

namespace hw {
//
// Hardware topology Description/Interface
//
class __hwtopo {
 public:
  // (TODO): Detect sockets and hyper-threading
  static __hwtopo* CreateHarewareTopology(size_t nSocket, bool bHT = true);
  virtual size_t AllocProc(int slot, int threadPerInstance) = 0;
};

}  // namespace hw
