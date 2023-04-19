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

#include "transpose_matmul/transpose_matmul.hpp"
#include "transpose_matmul/matmul_avx512f_p2031_p2013.hpp"
#include "transpose_matmul/matmul_avx512f_8bit.hpp"
#include "transpose_matmul/matmul_vnni_noperm_p2031_p1302.hpp"
#include "transpose_matmul/matmul_vnni_p2031_p2013.hpp"
#include "kernels/matmul_ref.hpp"

namespace jd {

void transpose_matmul_bench::get_true_data() {
  std::shared_ptr<const kernel_desc_t> ker_ref_desc;
  kernel_desc_t::create<matmul_ref_kd_t>(ker_ref_desc, args.second.op_desc);
  std::shared_ptr<const kernel_t> attention_ref_kernel;
  kernel_t::create<matmul_ref_k_t, matmul_ref_kd_t>(attention_ref_kernel, ker_ref_desc);
  attention_ref_kernel->execute(args.second.rt_data);
}

bench_res_t transpose_matmul_bench::set_config(int argc, char** argv) {
  if (!strcmp(argv[0], "avx512f_p2031_p2013")) {
    smb = std::make_shared<matmul_avx512f_p2031_p2013_bench>();
  } else if (!strcmp(argv[0], "vnni_noperm_p2031_p1302")) {
    smb = std::make_shared<matmul_vnni_noperm_p2031_p1302_bench>();
  } else if (!strcmp(argv[0], "vnni_p2031_p2013")) {
    smb = std::make_shared<matmul_vnni_p2031_p2013_bench>();
  } else if (!strcmp(argv[0], "avx512f_fp8")) {
    smb = std::make_shared<matmul_avx512f_8bit_bench>();
  } else {
    LOG(ERROR) << "unknown kernel specification";
    return {bench_status::wrong_input};
  }
  return smb->set_config(--argc, ++argv);
}

}  // namespace jd
