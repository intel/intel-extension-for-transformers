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
namespace jd {

double transpose_matmul_bench::calc_flop() const {
  std::vector<std::vector<dim_t>> shapes(ts_descs.size());
  std::transform(ts_descs.begin(), ts_descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  const dim_t M = shapes[ssd::SRC0][3];  // aka src0_perm_shape[2]
  const dim_t K = shapes[ssd::SRC0][1];  // aka src0_perm_shape[3]
  const dim_t N = shapes[ssd::SRC1][3];  // aka src1_perm_shape[3]
  const dim_t bs0 = shapes[ssd::DST0][0];
  const dim_t bs1 = shapes[ssd::DST0][1];

  return static_cast<double>(M) * N * K * bs0 * bs1 * 2;
}

bench_res_t transpose_matmul_bench::set_config(int argc, char** argv) {
  if (!strcmp(argv[0], "avx512f_p2031_p2013")) {
    smb = std::make_shared<matmul_avx512f_p2031_p2013_bench>();
  } else {
    LOG(ERROR) << "unknown kernel specification";
    return {bench_status::wrong_input};
  }
  return smb->set_config(--argc, ++argv);
}

}  // namespace jd
