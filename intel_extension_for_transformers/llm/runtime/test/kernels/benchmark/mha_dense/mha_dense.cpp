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

#include "mha_dense.hpp"

#include "mha_dense_dynamic.hpp"
#include "mha_dense_static.hpp"

namespace bench {
bench_res_t mha_dense_bench::set_config(int argc, char** argv) {
  if (!strcmp(argv[0], "static")) {
    smb = std::make_shared<mha_dense_static_bench>();
  } else if (!strcmp(argv[0], "dynamic")) {
    smb = std::make_shared<mha_dense_dynamic_bench>();
  } else {
    LOG(ERROR) << "unknown kernel specification";
    return {bench_status::wrong_input};
  }
  return smb->set_config(--argc, ++argv);
}
}  // namespace bench
