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
#include "mha_dense_static.hpp"

#include <algorithm>
#include <utility>

#include "include/engine.hpp"
#include "include/engine_factory.hpp"
#include "src/cpu/kernels/mha_dense_ref.hpp"

using jd::engine_factory;
using jd::engine_t;
using jd::exec_context_t;
using jd::kernel_desc_t;
using jd::kernel_t;
using jd::memory_storage_t;
using jd::mha_dense_ref_k_t;
using jd::mha_dense_ref_kd_t;
using jd::stream_t;

namespace bench {
namespace {
std::pair<void*, void*> make_tensor_obj(const jd::tensor_desc& ts_desc, float min_value, float max_value) {
  int64_t elem_num = ts_desc.size();
  if (elem_num == 0) return {nullptr, nullptr};
  int bytes_size = elem_num * jd::type_size[ts_desc.dtype()];
  void* data_ptr = nullptr;
  if (min_value == 0.f && max_value == 0.f) {
    data_ptr = aligned_allocator_t<uint8_t>::allocate(pad_to(bytes_size, 64), true);
    memset(data_ptr, 0, bytes_size);
  } else {
    if (ts_desc.dtype() == jd::data_type::fp32) {
      data_ptr = aligned_allocator_t<float>::allocate(pad_to(elem_num, 16));
      init_vector(static_cast<float*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == jd::data_type::bf16) {
      data_ptr = aligned_allocator_t<jd::bfloat16_t>::allocate(pad_to(elem_num, 32));
      init_vector(static_cast<jd::bfloat16_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == jd::data_type::s32) {
      data_ptr = aligned_allocator_t<int32_t>::allocate(pad_to(elem_num, 16));
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == jd::data_type::u8) {
      data_ptr = aligned_allocator_t<uint8_t>::allocate(pad_to(elem_num, 64));
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, min_value, max_value);
    } else if (ts_desc.dtype() == jd::data_type::s8) {
      data_ptr = aligned_allocator_t<int8_t>::allocate(pad_to(elem_num, 64));
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, min_value, max_value);
    } else {
      SPARSE_LOG(FATAL) << "Unexpected dtype!";
    }
  }
  void* data_ptr_copy = aligned_allocator_t<uint8_t>::allocate(pad_to(bytes_size, 64), true);
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return {data_ptr, data_ptr_copy};
}
std::pair<void*, void*> make_tensor_obj(const jd::tensor_desc& ts_desc, float value) {
  return make_tensor_obj(ts_desc, value, value);
}
std::pair<void*, void*> make_tensor_obj(const jd::tensor_desc& ts_desc) {
  return ts_desc.dtype() == jd::data_type::bf16 ? make_tensor_obj(ts_desc, -1.f, 1.f)
                                                : make_tensor_obj(ts_desc, -10.f, 10.f);
}
static engine_factory factory;
static const engine_t* cpu_engine = factory.create(jd::engine_kind::cpu, jd::runtime_kind::undef);
auto create_cpu_memory_storage(void* ptr) {
  memory_storage_t* mem;
  cpu_engine->create_memory_storage(&mem);
  if (ptr) mem->set_handle(ptr);
  return mem;
}
}  // namespace

double mha_dense_static_bench::calc_flop() const {
  double flops = 0;
  flops += 2. * sl_m * head_size * sl_n;  // Q x K
  flops += 6. * sl_m * sl_n;              // softmax: 1max + 3reduction + 2softmax  (copied from softmax benchmark)
  flops += 2. * sl_n * sl_m * head_size;  // A x V
  flops *= head_num * batch_size;
  return flops;
}
bench_res_t mha_dense_static_bench::set_config(int argc, char** argv) {
  if (argc < MIN_ARG_NUM) {
    LOG(ERROR) << "Not enough arguments passed";
    return {bench_status::wrong_input};
  }
  LOG(INFO) << "mha_dense_static\n";
  batch_size = str_to_num<int64_t>(argv[0]);
  sl_m = str_to_num<int64_t>(argv[1]);
  head_num = str_to_num<int64_t>(argv[2]);
  head_size = str_to_num<int64_t>(argv[3]);
  dt_dst = (argc <= 4)                    ? jd::data_type::u8
           : strcmp(argv[4], "fp32") == 0 ? jd::data_type::fp32
           : strcmp(argv[4], "s8") == 0   ? jd::data_type::s8
           : strcmp(argv[4], "u8") == 0   ? jd::data_type::u8
           : strcmp(argv[4], "bf16") == 0 ? jd::data_type::bf16
                                          : jd::data_type::undef;
  dt_src = (argc <= 5)                    ? jd::data_type::s8
           : strcmp(argv[5], "s8") == 0   ? jd::data_type::s8
           : strcmp(argv[5], "bf16") == 0 ? jd::data_type::bf16
                                          : jd::data_type::undef;
  if (argc > 6) mask = str_to_num<int32_t>(argv[6]);
  if (argc > 7) badd_dim = str_to_num<int32_t>(argv[7]);
  if (argc > 8) sl_n = str_to_num<int32_t>(argv[8]);
  ft_kv = (argc <= 9)                    ? jd::format_type::abcd  //
          : strcmp(argv[9], "abcd") == 0 ? jd::format_type::abcd
          : strcmp(argv[9], "acbd") == 0 ? jd::format_type::acbd
                                         : jd::format_type::undef;
  stable_softmax = (argc <= 10)                ? dt_src == jd::data_type::s8  // s8 static uses stable by default
                   : strcmp(argv[10], "true")  ? true
                   : strcmp(argv[10], "false") ? false
                                               : (SPARSE_LOG(ERROR) << "Unexpected arg: stable_softmax ", false);
  if (argc > 11) return {bench_status::wrong_input};
  if (sl_n <= 0) sl_n = sl_m;
  if (mask <= 0) mask = sl_n;
  if (dt_dst == jd::data_type::undef || dt_src == jd::data_type::undef) return {bench_status::wrong_input};
  if (mask > sl_n) return {bench_status::wrong_input};
  if (badd_dim > 4) return {bench_status::wrong_input};
  if (ft_kv == jd::format_type::undef) return {bench_status::wrong_input};
  return {bench_status::success};
}

void mha_dense_static_bench::get_true_data() {
  std::shared_ptr<const kernel_desc_t> mha_dense_ref_desc;
  kernel_desc_t::create<mha_dense_ref_kd_t>(mha_dense_ref_desc, bench_data.op_desc);
  std::shared_ptr<const kernel_t> mha_dense_ref_kernel;
  kernel_t::create<mha_dense_ref_k_t, mha_dense_ref_kd_t>(mha_dense_ref_kernel, mha_dense_ref_desc);
  const auto ref_workspace_size = mha_dense_ref_kernel->get_workspace_size();
  const auto tmp_ref =
      std::shared_ptr<char>(aligned_allocator_t<char>::allocate(std::max(static_cast<size_t>(64), ref_workspace_size)),
                            [](char* ptr) { aligned_allocator_t<char>::deallocate(ptr); });
  std::unique_ptr<memory_storage_t> tmp_ref_mem(create_cpu_memory_storage(tmp_ref.get()));
  bench_data.ctx_ref.set_workspace(tmp_ref_mem.get());
  mha_dense_ref_kernel->execute(bench_data.ctx_ref);
}

bool mha_dense_static_bench::check_result() {
  get_true_data();
  void *buf1, *buf2;
  bench_data.ctx_kern.output(io_dst::DST)->get_handle(&buf1);
  bench_data.ctx_ref.output(io_dst::DST)->get_handle(&buf2);
  auto dst_size = bench_data.op_desc.tensor_descs()[io::DST].size();
  if (buf1 == buf2) return false;  // Should compare buffer with different addresses
  switch (bench_data.op_desc.tensor_descs()[io::DST].dtype()) {
    case jd::data_type::fp32:
      return compare_data<float>(buf1, dst_size, buf2, dst_size, 5e-3);
    case jd::data_type::bf16:
      return compare_data<jd::bfloat16_t>(buf1, dst_size, buf2, dst_size, 5e-2);
    case jd::data_type::u8:
      return compare_data<uint8_t>(buf1, dst_size, buf2, dst_size, 8e-3);
    case jd::data_type::s8:
      return compare_data<int8_t>(buf1, dst_size, buf2, dst_size, 8e-3);
    default:
      SPARSE_LOG(ERROR) << "Unexpected dst type";
  }
  return false;
}
void mha_dense_static_bench::gen_case() {
  op_attrs.clear();
  op_attrs["approx_exp"] = "True";
  op_attrs["stable_softmax"] = stable_softmax ? "True" : "False";
  if (dt_src == jd::data_type::s8)
    op_attrs["softmax_rescale"] = std::to_string(float{UINT8_MAX});  // TODO(Yi): workaround for accuracy of int8 gptj
  // Step 1: Construct runtime data for equivalent merged spmm
  std::vector<dim_t> badd_full = {batch_size, head_num, sl_m, sl_n};
  ts_descs.assign(io::SIZE, jd::tensor_desc{{}, jd::data_type::undef, jd::format_type::undef});
  ts_descs[io::SRC_Q] = {{batch_size, sl_m, head_num, head_size}, dt_src, jd::format_type::abcd};
  ts_descs[io::SRC_K] = {{batch_size, sl_n, head_num, head_size}, dt_src, ft_kv};
  ts_descs[io::SRC_V] = {{batch_size, sl_n, head_num, head_size}, dt_src, ft_kv};
  if (dt_src != jd::data_type::bf16)
    ts_descs[io::MASK] = {
        {batch_size}, jd::data_type::s32, jd::format_type::a};  // TODO(Yi): change given dt_src is confusing
  ts_descs[io::DST] = {{batch_size, sl_m, head_num, head_size}, dt_dst, jd::format_type::abcd};
  if (badd_dim > 0) {
    SPARSE_LOG_IF(FATAL, badd_dim > 4) << "Unsupported binary add dimension";
    ts_descs[io::BINARY_ADD] = {std::vector<dim_t>(badd_full.cend() - badd_dim, badd_full.cend()), jd::data_type::fp32,
                                jd::plain_format(badd_dim)};
  }
  ts_descs[io::ATT_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  if (dt_src == jd::data_type::s8) {
    ts_descs[io::Q_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
    ts_descs[io::K_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
    ts_descs[io::V_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  }
  if (dt_src != jd::data_type::bf16) {  // TODO(Yi): support dst sc/zp for bf16 MHA
    ts_descs[io::SRC_DST_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
    ts_descs[io::SRC_DST_ZP] = {{1}, jd::data_type::s32, jd::format_type::a};
  }
  // Step 2: Construct Tensor ptr
  auto Qs = make_tensor_obj(ts_descs[io::SRC_Q]);
  auto Ks = make_tensor_obj(ts_descs[io::SRC_K]);
  auto Vs = make_tensor_obj(ts_descs[io::SRC_V]);
  auto masks = make_tensor_obj(ts_descs[io::MASK], mask, mask);
  auto dsts = make_tensor_obj(ts_descs[io::DST], 0, 0);
  auto badds = make_tensor_obj(ts_descs[io::BINARY_ADD], -1.f, 1.f);
  auto att_scales = make_tensor_obj(ts_descs[io::ATT_SCALE], 1.f / std::sqrt(sl_n));
  auto q_scales = make_tensor_obj(ts_descs[io::Q_SCALE], 1.1f);
  auto k_scales = make_tensor_obj(ts_descs[io::K_SCALE], 0.9f);
  auto v_scales = make_tensor_obj(ts_descs[io::V_SCALE], 1.2f);
  auto dst_scales = make_tensor_obj(ts_descs[io::SRC_DST_SCALE], 1.2f);
  auto dst_zps = make_tensor_obj(ts_descs[io::SRC_DST_ZP], 10.f);

  stream_t *stream = nullptr, *ref_stream = nullptr;
  cpu_engine->create_stream(&stream);
  cpu_engine->create_stream(&ref_stream);
  exec_context_t ctx_kern(stream), ctx_ref(ref_stream);

  std::vector<memory_storage_t*> mem_src(io_src::SIZE), mem_dst(io_dst::SIZE);
  std::vector<dim_t> dynamic_shapes(io_shape::SIZE);
  mem_src[io_src::SRC_Q] = create_cpu_memory_storage(Qs.first);
  mem_src[io_src::SRC_K] = create_cpu_memory_storage(Ks.first);
  mem_src[io_src::SRC_V] = create_cpu_memory_storage(Vs.first);
  mem_src[io_src::MASK] = create_cpu_memory_storage(masks.first);
  mem_src[io_src::BINARY_ADD] = create_cpu_memory_storage(badds.first);
  mem_src[io_src::ATT_SCALE] = create_cpu_memory_storage(att_scales.first);
  mem_src[io_src::Q_SCALE] = create_cpu_memory_storage(q_scales.first);
  mem_src[io_src::Q_ZP] = create_cpu_memory_storage(nullptr);
  mem_src[io_src::K_SCALE] = create_cpu_memory_storage(k_scales.first);
  mem_src[io_src::K_ZP] = create_cpu_memory_storage(nullptr);
  mem_src[io_src::V_SCALE] = create_cpu_memory_storage(v_scales.first);
  mem_src[io_src::V_ZP] = create_cpu_memory_storage(nullptr);
  mem_src[io_src::SRC_DST_SCALE] = create_cpu_memory_storage(dst_scales.first);
  mem_src[io_src::SRC_DST_ZP] = create_cpu_memory_storage(dst_zps.first);
  mem_dst[io_dst::DST] = create_cpu_memory_storage(dsts.first);
  mem_dst[io_dst::DST_SCALE] = create_cpu_memory_storage(nullptr);
  mem_dst[io_dst::DST_ZP] = create_cpu_memory_storage(nullptr);
  dynamic_shapes[io_shape::BATCH_SIZE] = 0;
  dynamic_shapes[io_shape::HEAD_NUM] = 0;
  dynamic_shapes[io_shape::HEAD_SIZE] = 0;
  dynamic_shapes[io_shape::M] = 0;
  dynamic_shapes[io_shape::N] = 0;
  ctx_kern.set_outputs(mem_dst);
  ctx_kern.set_inputs(mem_src);
  ctx_kern.set_dynamic_shape(dynamic_shapes);

  std::vector<memory_storage_t*> ref_mem_src(io_src::SIZE), ref_mem_dst(io_dst::SIZE);
  std::vector<dim_t> ref_dynamic_shapes(io_shape::SIZE);
  ref_mem_src[io_src::SRC_Q] = create_cpu_memory_storage(Qs.second);
  ref_mem_src[io_src::SRC_K] = create_cpu_memory_storage(Ks.second);
  ref_mem_src[io_src::SRC_V] = create_cpu_memory_storage(Vs.second);
  ref_mem_src[io_src::MASK] = create_cpu_memory_storage(masks.second);
  ref_mem_src[io_src::BINARY_ADD] = create_cpu_memory_storage(badds.second);
  ref_mem_src[io_src::ATT_SCALE] = create_cpu_memory_storage(att_scales.second);
  ref_mem_src[io_src::Q_SCALE] = create_cpu_memory_storage(q_scales.second);
  ref_mem_src[io_src::Q_ZP] = create_cpu_memory_storage(nullptr);
  ref_mem_src[io_src::K_SCALE] = create_cpu_memory_storage(k_scales.second);
  ref_mem_src[io_src::K_ZP] = create_cpu_memory_storage(nullptr);
  ref_mem_src[io_src::V_SCALE] = create_cpu_memory_storage(v_scales.second);
  ref_mem_src[io_src::V_ZP] = create_cpu_memory_storage(nullptr);
  ref_mem_src[io_src::SRC_DST_SCALE] = create_cpu_memory_storage(dst_scales.second);
  ref_mem_src[io_src::SRC_DST_ZP] = create_cpu_memory_storage(dst_zps.second);
  ref_mem_dst[io_dst::DST] = create_cpu_memory_storage(dsts.second);
  ref_mem_dst[io_dst::DST_SCALE] = create_cpu_memory_storage(nullptr);
  ref_mem_dst[io_dst::DST_ZP] = create_cpu_memory_storage(nullptr);
  ref_dynamic_shapes[io_shape::BATCH_SIZE] = 0;
  ref_dynamic_shapes[io_shape::HEAD_NUM] = 0;
  ref_dynamic_shapes[io_shape::HEAD_SIZE] = 0;
  ref_dynamic_shapes[io_shape::M] = 0;
  ref_dynamic_shapes[io_shape::N] = 0;
  ctx_ref.set_outputs(ref_mem_dst);
  ctx_ref.set_inputs(ref_mem_src);
  ctx_ref.set_dynamic_shape(ref_dynamic_shapes);

  jd::operator_desc op_desc(jd::kernel_kind::mha_dense, jd::kernel_prop::forward_inference, jd::engine_kind::cpu,
                            ts_descs, op_attrs);

  // Step 3: op_args_t testcase pair
  bench_data = {op_desc, ctx_kern, ctx_ref};
}
}  // namespace bench
