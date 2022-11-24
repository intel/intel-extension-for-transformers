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

#include "transpose_matmul/matmul_avx512f_p2031_p2013.hpp"

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "utils.hpp"

namespace jd {

using dt = jd::data_type;
using ft = jd::format_type;

void matmul_avx512f_p2031_p2013_bench::get_true_data() {
  auto& op_desc = args.second.op_desc;
  auto& rt_data = args.second.rt_data;
  // shape configure alias
  auto& descs = op_desc.tensor_descs();
  auto attrs = op_desc.attrs();
  std::vector<std::vector<dim_t>> shapes(descs.size());
  std::transform(descs.begin(), descs.end(), shapes.begin(), [&](tensor_desc d) { return d.shape(); });
  std::vector<jd::data_type> dtypes(descs.size());
  std::transform(descs.begin(), descs.end(), dtypes.begin(), [&](tensor_desc d) { return d.dtype(); });

  const dim_t M = shapes[ssd::SRC0][3];  // aka src0_perm_shape[2]
  const dim_t K = shapes[ssd::SRC0][1];  // aka src0_perm_shape[3]
  const dim_t N = shapes[ssd::SRC1][3];  // aka src1_perm_shape[3]
  const dim_t bs0 = shapes[ssd::DST0][0];
  const dim_t bs1 = shapes[ssd::DST0][1];
  bool has_binary_add = !shapes[ssd::SRC2].empty();

  float alpha = 1.f, beta = 1.f;
  if (attrs["alpha"] != "") alpha = str_to_num<float>(attrs["alpha"]);
  if (attrs["beta"] != "") beta = str_to_num<float>(attrs["beta"]);

  const auto& left_dt = dtypes[ssd::SRC0];
  const auto& right_dt = dtypes[ssd::SRC1];
  const auto& dst_dt = dtypes[ssd::DST0];

  std::vector<dim_t> left_stride = {K * bs0 * M, bs0 * M, M, 1};
  std::vector<dim_t> right_stride = {K * bs0 * N, bs0 * N, N, 1};
  std::vector<dim_t> dst_stride = {bs1 * M * N, M * N, N, 1};

  // runtime data alias
  const auto left_data = rt_data[ssd::SRC0];
  const auto right_data = rt_data[ssd::SRC1];
  const auto badd_data = rt_data[ssd::SRC2];
  auto dst_data = const_cast<void*>(rt_data[ssd::DST0]);

  // buffer data
  auto left_fp32 = static_cast<const float*>(left_data);  // ptr alias
  auto right_fp32 = static_cast<const float*>(right_data);
  auto dst_fp32 = static_cast<float*>(dst_data);
  auto badd_fp32 = static_cast<const float*>(badd_data);

  // Computing the kernel

#pragma omp parallel for collapse(4)
  for (dim_t ibs0 = 0; ibs0 < bs0; ++ibs0)
    for (dim_t ibs1 = 0; ibs1 < bs1; ++ibs1)
      for (dim_t i = 0; i < M; ++i)
        for (dim_t j = 0; j < N; ++j) {
          float value = 0;
          dim_t dst_idx = ibs0 * dst_stride[0] + ibs1 * dst_stride[1] + i * dst_stride[2] + j * dst_stride[3];
#pragma omp simd
          for (dim_t k = 0; k < K; ++k) {
            /**
             *   src0:     bs1 k   bs0 m
             *   src1:     bs1 k   bs0 n
             *   src2/dst: bs0 bs1 m   n
             */
            dim_t l_idx = ibs1 * left_stride[0] + k * left_stride[1] + ibs0 * left_stride[2] + i * left_stride[3];
            dim_t r_idx = ibs1 * right_stride[0] + k * right_stride[1] + ibs0 * right_stride[2] + j * right_stride[3];
            auto l_value = left_dt == dt::fp32 ? left_fp32[l_idx] : 0;
            auto r_value = right_dt == dt::fp32 ? right_fp32[r_idx] : 0;
            value += l_value * r_value;
          }
          float badd_value = 0;
          if (has_binary_add) badd_value = dtypes[ssd::SRC2] == dt::fp32 ? badd_fp32[dst_idx] : 0;

          // Quantize dst data
          if (dst_dt == dt::fp32) {
            dst_fp32[dst_idx] = static_cast<float>(alpha * value + beta * badd_value);
          } else {
            LOG(FATAL) << "unsupported dst type";
          }
        }
}

bool matmul_avx512f_p2031_p2013_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data();
  auto buf1 = p.rt_data[ssd::DST0];
  auto size1 = p.op_desc.tensor_descs()[ssd::DST0].size();
  auto buf2 = q.rt_data[ssd::DST0];
  auto size2 = q.op_desc.tensor_descs()[ssd::DST0].size();
  const auto& dst_type = p.op_desc.tensor_descs()[ssd::DST0].dtype();
  if (dst_type == dt::fp32) {
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::s32) {
    return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::u8) {
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 5e-3);
  }
  return false;
}

std::pair<const void*, const void*> make_data_obj_matmul_avx512f_p2031_p2013(  //
    const std::vector<int64_t>& a_shape, const dt& a_dt, bool is_clear = false,
    const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<dim_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = aligned_allocator_t<char>::aligned_alloc(bytes_size, true);
  } else {
    if (a_dt == dt::fp32) {
      data_ptr = aligned_allocator_t<float>::aligned_alloc(elem_num);
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s32) {
      data_ptr = aligned_allocator_t<int32_t>::aligned_alloc(elem_num);
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::u8) {
      data_ptr = aligned_allocator_t<uint8_t>::aligned_alloc(elem_num);
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s8) {
      data_ptr = aligned_allocator_t<int8_t>::aligned_alloc(elem_num);
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    }
  }

  void* data_ptr_copy = aligned_allocator_t<char>::aligned_alloc(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}

void matmul_avx512f_p2031_p2013_bench::gen_case() {
  // Step 1: Construct operator config
  tensor_desc src0_desc = {{bs1, K, bs0, M}, dt::fp32, ft::ab};
  tensor_desc src1_desc = {{bs1, K, bs0, N}, dt::fp32, ft::ab};
  tensor_desc dst_desc = {{bs0, bs1, M, N}, dt::fp32, ft::ab};
  tensor_desc src2_desc = {{bs0, bs1, M, N}, dt::fp32, ft::ab};
  if (!has_binary_add) src2_desc = {{}, dt::fp32, ft::ab};
  ts_descs = {src0_desc, src1_desc, dst_desc, src2_desc};

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    if (index == ssd::SRC2 && !has_binary_add) {
      // insert nullptr as placeholder
      rt_data1.emplace_back(nullptr);
      rt_data2.emplace_back(nullptr);
      continue;
    }
    auto& tsd = ts_descs[index];
    bool is_clear = (index == ssd::DST0);
    auto ranges = std::vector<float>{-10, 10};
    auto data_pair = make_data_obj_matmul_avx512f_p2031_p2013(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }

  operator_desc op_desc(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                        op_attrs);

  // Step 3: op_args_t testcase pair
  op_args_t op_args = {op_desc, rt_data1};
  op_args_t op_args_copy = {op_desc, rt_data2};

  args = {op_args, op_args_copy};
}

bench_res_t matmul_avx512f_p2031_p2013_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "matmul_avx512f_p2031_p2013\n";
  if (argc < MATMUL_AVX512F_P2031_P2013_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);                                      // M
  K = str_to_num<int64_t>(argv[1]);                                      // K
  N = str_to_num<int64_t>(argv[2]);                                      // N
  bs0 = str_to_num<int64_t>(argv[3]);                                    // bs0
  bs1 = str_to_num<int64_t>(argv[4]);                                    // bs1
  has_binary_add = strcmp(argv[5], "1") == 0;                            // has_binary_add
  op_attrs["alpha"] = argc > 6 ? argv[6] : "1";                          // alpha
  op_attrs["beta"] = argc > 7 ? argv[7] : (has_binary_add ? "1" : "0");  // beta
  op_attrs["tile_m"] = argc > 8 ? argv[8] : "";                          // tile_m
  op_attrs["tile_n"] = argc > 9 ? argv[9] : "";                          // tile_n

  return {bench_status::success};
}

}  // namespace jd
