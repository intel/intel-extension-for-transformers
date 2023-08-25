//  Copyright (c) 2022 Intel Corporation
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

#include "src/cpu/cpu_isa.hpp"
#ifdef __linux__
#include <sys/syscall.h>
#endif  // __linux__

namespace jd {
bool init_amx() {
#ifdef __linux__

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

  unsigned long bitmask = 0;                                             // NOLINT
  long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);  // NOLINT
  if (0 != status) return false;
  if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

  status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (0 != status)
    return false;  // XFEATURE_XTILEDATA setup is failed, TMUL usage is not
                   // allowed
  status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

  // XFEATURE_XTILEDATA setup is failed, can't use TMUL
  if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

  // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed

  return true;
#else
  return true;
#endif
}

set_once_before_first_get_setting_t<bool>& amx_setting() {
  static set_once_before_first_get_setting_t<bool> setting(init_amx());
  return setting;
}

}  // namespace jd
