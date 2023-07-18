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
#include "util.h"

int32_t get_num_physical_cores() {
#ifdef __linux__
  // enumerate the set of thread siblings, num entries is num cores
  std::unordered_set<std::string> siblings;
  for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
    std::ifstream thread_siblings("/sys/devices/system/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
    if (!thread_siblings.is_open()) {
      break;  // no more cpus
    }
    std::string line;
    if (std::getline(thread_siblings, line)) {
      siblings.insert(line);
    }
  }
  if (siblings.size() > 0) {
    return static_cast<int32_t>(siblings.size());
  }
#elif defined(__APPLE__) && defined(__MACH__)
  int32_t num_physical_cores;
  size_t len = sizeof(num_physical_cores);
  int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
  if (result == 0) {
    return num_physical_cores;
  }
  result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
  if (result == 0) {
    return num_physical_cores;
  }
#elif defined(_WIN32)
  // TODO: Implement
#endif
  unsigned int n_threads = std::thread::hardware_concurrency();
  return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}
