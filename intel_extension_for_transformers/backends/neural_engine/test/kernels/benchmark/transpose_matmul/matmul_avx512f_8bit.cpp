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

#include <xbyak/xbyak_util.h>

#include "transpose_matmul/matmul_avx512f_8bit.hpp"

#include "benchmark_utils.hpp"
#include "common_utils.hpp"
#include "cpu_parallel.hpp"
#include "kernels/fp8.hpp"
#include "singleton.hpp"
#include "utils.hpp"

namespace jd {

using dt = jd::data_type;
using ft = jd::format_type;
namespace {

int8_t to_8bit(float fp32, data_type type) {
  if (type == dt::s8) {
    return fp32_to_int8(fp32);
  } else if (type == dt::f8_e4m3) {
    return float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(fp32);
  } else if (type == dt::f8_e5m2) {
    return float8_base<FloatEncoding::E5M2>::convert_float_to_fp8(fp32);
  } else {
    return 0;
  }
}
void reference(int8_t* srcptr, int8_t* dstptr, data_type type, int row, int col, int rowpad, int colpad, int srcstride,
               int dststride) {
  int srcld = srcstride / 2;
  auto sptr = reinterpret_cast<int8_t*>(srcptr);
  auto dptr = (dstptr);
  int NTile = 16;
  for (int irow = 0; irow < rowpad; irow += NTile) {
    for (int icol = 0; icol < colpad; icol += 1) {
      for (int iin = 0; iin < NTile; iin++) {
        if (irow + iin < row) {
          if (icol < col) {
            *(dptr + irow * dststride + icol * NTile + iin) = *(sptr + (irow + iin) * srcld + icol);
          } else {
            *(dptr + irow * dststride + icol * NTile + iin) = to_8bit(static_cast<float>(0), type);
          }
        } else {
          *(dptr + irow * dststride + icol * NTile + iin) = to_8bit(static_cast<float>(0), type);
        }
      }
    }
  }
}

int8_t* pack(int8_t* input, int n, int k, data_type type, int ncores) {
  int ldb = k;
  int npad = pad_to(n, 16);
  int kpad = pad_to(k, 1);
  int8_t* output = aligned_allocator_t<int8_t>::allocate(npad * kpad, true);
  Parallel2DRowMajor _para;
  _para.update(npad, kpad, 16, 1, ncores);
#pragma omp parallel
  {
    int tidx = omp_get_thread_num();
    int colidx, rowidx, rowsize, colsize;
    _para.getIndex(tidx, &rowidx, &colidx, &rowsize, &colsize);
    if (rowsize > 0 && colsize > 0) {
      int rowremain = remainsize(rowidx, n, rowsize);
      int colremain = remainsize(colidx, k, colsize);
      reference(input + rowidx * ldb + colidx, output + rowidx * kpad + colidx * 16, type, rowremain, colremain,
                rowsize, colsize, k * sizeof(float8_t), kpad);
    }
  }
  return output;
}
}  // namespace

bool matmul_avx512f_8bit_bench::check_result() {
  const auto& p = args.first;
  const auto& q = args.second;

  get_true_data();
  auto buf1 = p.rt_data[io::DST0];
  auto size1 = p.op_desc.tensor_descs()[io::DST0].size();
  auto buf2 = q.rt_data[io::DST0];
  auto size2 = q.op_desc.tensor_descs()[io::DST0].size();
  const auto& dst_type = p.op_desc.tensor_descs()[io::DST0].dtype();
  if (dst_type == dt::fp32) {
    return compare_data<float>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::s32) {
    return compare_data<int32_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::u8) {
    return compare_data<uint8_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::s8) {
    return compare_data<int8_t>(buf1, size1, buf2, size2, 5e-3);
  } else if (dst_type == dt::bf16) {
    return compare_data<bfloat16_t>(buf1, size1, buf2, size2, 0.12);
  }
  return false;
}

std::pair<const void*, const void*> make_data_obj_matmul_avx512f_8bit(  //
    const std::vector<int64_t>& a_shape, const dt& a_dt, bool is_clear = false,
    const std::vector<float>& ranges = {-10, 10}) {
  int elem_num = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<dim_t>());
  int bytes_size = elem_num * type_size[a_dt];
  void* data_ptr = nullptr;
  if (is_clear) {
    data_ptr = aligned_allocator_t<char>::allocate(bytes_size, true);
  } else {
    if (a_dt == dt::fp32) {
      data_ptr = aligned_allocator_t<float>::allocate(elem_num);
      init_vector(static_cast<float*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s32) {
      data_ptr = aligned_allocator_t<int32_t>::allocate(elem_num);
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::u8) {
      data_ptr = aligned_allocator_t<uint8_t>::allocate(elem_num);
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::s8) {
      data_ptr = aligned_allocator_t<int8_t>::allocate(elem_num);
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::bf16) {
      data_ptr = aligned_allocator_t<bfloat16_t>::allocate(elem_num);
      init_vector(static_cast<bfloat16_t*>(data_ptr), elem_num, ranges[0], ranges[1]);
    } else if (a_dt == dt::f8_e4m3) {
      data_ptr = aligned_allocator_t<uint8_t>::allocate(elem_num);
      float8_t* fp8 = reinterpret_cast<float8_t*>(data_ptr);
      float* fp32 = aligned_allocator_t<float>::allocate(elem_num);
      init_vector(fp32, elem_num, ranges[0], ranges[1]);
      for (int i = 0; i < elem_num; i++) {
        fp8[i] = float8_base<FloatEncoding::E4M3>::convert_float_to_fp8(fp32[i]);
      }
      aligned_allocator_t<float, 64>::deallocate(fp32);
    }
  }

  void* data_ptr_copy = aligned_allocator_t<char>::allocate(bytes_size);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<const void*, const void*>{data_ptr, data_ptr_copy};
}
void matmul_avx512f_8bit_bench::gen_case() {
  // Step 1: Construct operator config
  tensor_desc src0_desc = {{M, K}, dt::bf16, ft::ab};
  tensor_desc src1_desc = {{N, K}, src1_dtype, ft::ab};
  tensor_desc dst_desc = {{M, N}, dt::bf16, ft::ab};
  tensor_desc bias_desc = {{N}, dt::bf16, ft::ab};
  ts_descs = {src0_desc, src1_desc, dst_desc, bias_desc};

  // Step 2: Construct runtime data
  std::vector<const void*> rt_data1;
  std::vector<const void*> rt_data2;
  int tensor_num = ts_descs.size();
  for (int index = 0; index < tensor_num; ++index) {
    auto& tsd = ts_descs[index];
    bool is_clear = (index == io::DST0);
    auto ranges = std::vector<float>{-2, 10};
    auto data_pair = make_data_obj_matmul_avx512f_8bit(tsd.shape(), tsd.dtype(), is_clear, ranges);
    rt_data1.emplace_back(data_pair.first);
    rt_data2.emplace_back(data_pair.second);
  }
  if (src1_dtype == data_type::s8) {
    op_attrs["weight_type"] = "s8";
  } else {
    op_attrs["weight_type"] = "f8_e4m3";
  }
  std::unordered_map<std::string, std::string> attrs1 = op_attrs;
  std::unordered_map<std::string, std::string> attrs2 = op_attrs;

  if (src1_dtype == dt::bf16) {
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(aligned_allocator_t<int8_t>::allocate(N * K)));
    attrs1["weight_bf16"] = std::to_string(reinterpret_cast<intptr_t>(rt_data1[io::SRC1]));
  } else {
    Xbyak::util::Cpu* cpu = Singleton<Xbyak::util::Cpu>::GetInstance();
    int numcores = cpu->getNumCores(Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    int ompthreads = omp_get_max_threads();
    int ncores = std::min(numcores, ompthreads);
    data_type type = data_type::f8_e4m3;

    for (auto& it : data_type_name) {
      if (it.second == op_attrs["weight_type"]) {
        type = it.first;
        break;
      }
    }
    int8_t* weight_8bit = pack(reinterpret_cast<int8_t*>(const_cast<void*>(rt_data1[io::SRC1])), N, K, type, ncores);
    attrs1["weight_8bit"] = std::to_string(reinterpret_cast<intptr_t>(weight_8bit));
  }

  operator_desc op_desc1(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                         attrs1);
  operator_desc op_desc2(kernel_kind::transpose_matmul, kernel_prop::forward_inference, engine_kind::cpu, ts_descs,
                         attrs2);

  op_args_t op_args = {op_desc1, rt_data1};
  op_args_t op_args_copy = {op_desc2, rt_data2};

  args = {op_args, op_args_copy};
}

bench_res_t matmul_avx512f_8bit_bench::set_config(int argc, char** argv) {
  LOG(INFO) << "matmul_avx512f_8bit\n";
  if (argc < matmul_avx512f_8bit_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  M = str_to_num<int64_t>(argv[0]);  // M
  K = str_to_num<int64_t>(argv[1]);  // K
  N = str_to_num<int64_t>(argv[2]);  // N
  switch (str_to_num<int64_t>(argv[3])) {
    case 0:
      src1_dtype = data_type::bf16;
      break;
    case 1:
      src1_dtype = data_type::s8;
      break;
    case 2:
      src1_dtype = data_type::f8_e4m3;
      break;

    default:
      src1_dtype = data_type::bf16;
      break;
  }
  op_attrs["alpha"] = argc > 4 ? argv[4] : "1";  // alpha
  op_attrs["beta"] = argc > 5 ? argv[5] : "1";   // beta

  return {bench_status::success};
}

}  // namespace jd
