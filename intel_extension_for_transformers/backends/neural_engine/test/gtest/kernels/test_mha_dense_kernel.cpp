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

#include <algorithm>
#include <map>
#include <string>

#include "engine.hpp"
#include "engine_factory.hpp"
#include "gtest/gtest.h"
#include "interface.hpp"
#include "src/cpu/kernels/mha_dense_ref.hpp"
#include "unit_test_utils.hpp"

namespace test {
using io = jd::exposed_enum::mha_dense::io;
using io_src = jd::exposed_enum::mha_dense_src::src;
using io_dst = jd::exposed_enum::mha_dense_dst::dst;
using io_shape = jd::exposed_enum::mha_dense_shape::shape;
using jd::engine_factory;
using jd::engine_t;
using jd::exec_context_t;
using jd::kernel_t;
using jd::memory_storage_t;
using jd::stream_t;

struct mha_dims_t {
  dim_t bs;
  dim_t sl_m;
  dim_t sl_n;
  dim_t head_num;
  dim_t head_size;

  mha_dims_t upper_bound(const mha_dims_t& rhs) {
    return {
        std::max(bs, rhs.bs),
        std::max(sl_m, rhs.sl_m),
        std::max(sl_n, rhs.sl_n),
        std::max(head_num, rhs.head_num),
        std::max(head_size, rhs.head_size),
    };
  }
};

std::ostream& operator<<(std::ostream& os, const mha_dims_t& dims) {
  os << dims.bs << '_' << dims.sl_m << '_' << dims.sl_n << '_' << dims.head_num << '_' << dims.head_size;
  return os;
}

mha_dims_t max_dims(std::vector<mha_dims_t> dims) {
  mha_dims_t max_(dims[0]);
  for (size_t i = 1; i < dims.size(); ++i) max_ = max_.upper_bound(dims[i]);
  return max_;
}

template <typename T>
struct test_params_t {
  T dims;
  int badd_dim /* = 0*/;
  jd::data_type dt_dst /* = jd::data_type::u8*/;
  jd::format_type ft_kv /* = jd::format_type::u8*/;
  int nthr;
  bool expect_to_fail;
};

struct test_data_t {
  jd::operator_desc op_desc;
  mutable exec_context_t ctx_kern;  // mutable to set_workspace
  mutable exec_context_t ctx_ref;   // mutable to set_workspace
};

static std::mt19937 rand_gen(1);
static engine_factory factory;
static const engine_t* cpu_engine = factory.create(jd::engine_kind::cpu, jd::runtime_kind::undef);
static const stream_t* stream = []() {
  stream_t* stream = nullptr;
  cpu_engine->create_stream(&stream);
  return stream;
}();

auto create_cpu_memory_storage(void* ptr) {
  memory_storage_t* mem;
  cpu_engine->create_memory_storage(&mem);
  if (ptr) mem->set_handle(ptr);
  return mem;
}
inline static std::string to_string(const testing::TestParamInfo<test_params_t<mha_dims_t>>& tpi) {
  auto&& p = tpi.param;
  std::vector<std::string> params_str;
  params_str.push_back("c" + std::to_string(p.nthr));
  params_str.push_back(std::to_string(p.dims.bs));               // bs
  params_str.push_back(std::to_string(p.dims.sl_m));             // sl_m
  params_str.push_back(std::to_string(p.dims.sl_n));             // sl_n
  params_str.push_back(std::to_string(p.dims.head_num));         // head_num
  params_str.push_back(std::to_string(p.dims.head_size));        // head_size
  params_str.push_back("badddim" + std::to_string(p.badd_dim));  // badddim
  params_str.push_back(jd::data_type_name.at(p.dt_dst) + std::string{"dst"});
  params_str.push_back(jd::format_type_name.at(p.ft_kv));  // kv_ft
  return join_str(params_str, "_");
}

bool compare_data(jd::data_type dt, size_t size, const exec_context_t& ctx1, const exec_context_t& ctx2) {
  void *buf1, *buf2;
  ctx1.output(io_dst::DST)->get_handle(&buf1);
  ctx2.output(io_dst::DST)->get_handle(&buf2);
  // Should compare buffer with different addresses
  EXPECT_NE(buf1, buf2);
  switch (dt) {
    case jd::data_type::fp32:
      return compare_data<float>(buf1, size, buf2, size, 5e-3);
    case jd::data_type::s32:
      return compare_data<int32_t>(buf1, size, buf2, size, 5e-3);
    case jd::data_type::u8:
      return compare_data<uint8_t>(buf1, size, buf2, size, 4e-3);
    case jd::data_type::s8:
      return compare_data<int8_t>(buf1, size, buf2, size, 4e-3);
    case jd::data_type::bf16:
      return compare_data<jd::bfloat16_t>(buf1, size, buf2, size, 1e-2);
    default:
      SPARSE_LOG(ERROR) << "Unexpected dst type";
  }
  return false;
}

template <class T>
std::shared_ptr<memory_storage_t> prepare_workspace(exec_context_t* ctx, const T& kern) {
  const auto workspace_size = kern.get_workspace_size();
  const auto ws = aligned_allocator_t<char>::allocate(std::max(static_cast<size_t>(64), workspace_size));
  std::shared_ptr<memory_storage_t> workspace_mem(create_cpu_memory_storage(ws), [ws](memory_storage_t* mem) {
    aligned_allocator_t<char>::deallocate(ws);
    delete mem;
  });
  ctx->set_workspace(workspace_mem.get());
  return workspace_mem;
}

bool check_result(const int nthr, const bool expect_to_fail, const test_data_t& d) {
  try {
    std::shared_ptr<const jd::kernel_desc_t> mha_dense_ref_desc;
    jd::kernel_desc_t::create<jd::mha_dense_ref_kd_t>(mha_dense_ref_desc, d.op_desc);
    std::shared_ptr<const kernel_t> ref_kern;
    kernel_t::create<jd::mha_dense_ref_k_t, jd::mha_dense_ref_kd_t>(ref_kern, mha_dense_ref_desc);
    const auto ref_ws_mem = prepare_workspace(&d.ctx_ref, *ref_kern);
    ref_kern->execute(d.ctx_ref);

    n_thread_t with_n_thread(nthr);
    jd::mha_dense_desc mha_dense_desc(d.op_desc);
    jd::mha_dense mha_dense_kernel(mha_dense_desc);
    const auto kern_ws_mem = prepare_workspace(&d.ctx_kern, mha_dense_kernel);
    mha_dense_kernel.execute(d.ctx_kern);
  } catch (const std::exception& e) {
    SPARSE_LOG(ERROR) << e.what();
    return expect_to_fail;
  }

  if (!expect_to_fail) {
    const auto dst_dt = d.op_desc.tensor_dtypes()[io::DST];
    const auto dst_size = d.op_desc.tensor_descs()[io::DST].size();
    return compare_data(dst_dt, dst_size, d.ctx_kern, d.ctx_ref);
  }
  return false;
}

std::pair<void*, void*> make_tensor_obj(const jd::tensor_desc& ts_desc, float min_value, float max_value) {
  dim_t elem_num = std::accumulate(ts_desc.shape().begin(), ts_desc.shape().end(), 1LL, std::multiplies<dim_t>());
  int bytes_size = elem_num * jd::type_size[ts_desc.dtype()];
  void* data_ptr = nullptr;
  if (min_value == 0.f && max_value == 0.f) {
    data_ptr = new uint8_t[bytes_size];
    memset(data_ptr, 0, bytes_size);
  } else {
    const auto seed = std::uniform_int_distribution<>()(rand_gen);
    if (ts_desc.dtype() == jd::data_type::fp32) {
      data_ptr = new float[elem_num];
      init_vector(static_cast<float*>(data_ptr), elem_num, min_value, max_value, seed);
    } else if (ts_desc.dtype() == jd::data_type::s32) {
      data_ptr = new int32_t[elem_num];
      init_vector(static_cast<int32_t*>(data_ptr), elem_num, min_value, max_value, seed);
    } else if (ts_desc.dtype() == jd::data_type::u8) {
      data_ptr = new uint8_t[elem_num];
      init_vector(static_cast<uint8_t*>(data_ptr), elem_num, min_value, max_value, seed);
    } else if (ts_desc.dtype() == jd::data_type::s8) {
      data_ptr = new int8_t[elem_num];
      init_vector(static_cast<int8_t*>(data_ptr), elem_num, min_value, max_value, seed);
    }
  }

  void* data_ptr_copy = new uint8_t[bytes_size];
  memcpy(data_ptr_copy, data_ptr, bytes_size);
  return std::pair<void*, void*>{data_ptr, data_ptr_copy};
}
std::pair<void*, void*> make_tensor_obj(const jd::tensor_desc& ts_desc) { return make_tensor_obj(ts_desc, -10, 10); }
std::pair<void*, void*> make_tensor_obj(const jd::tensor_desc& ts_desc, float value) {
  return make_tensor_obj(ts_desc, value, value);
}

jd::operator_desc gen_opdesc(const dim_t bs, const dim_t sl_m, const dim_t sl_n, const dim_t head_num,
                             const dim_t head_size, int badd_dim = 0, const jd::data_type dt_dst = jd::data_type::u8,
                             const jd::format_type kv_ft = jd::format_type::abcd) {
  std::vector<dim_t> badd_fullshape = {bs, head_num, sl_m, sl_n};
  std::vector<jd::tensor_desc> ts_descs(io::SIZE, jd::tensor_desc{});
  ts_descs[io::SRC_Q] = {{bs, sl_m, head_num, head_size}, jd::data_type::s8, jd::format_type::abcd};
  ts_descs[io::SRC_K] = {{bs, sl_n, head_num, head_size}, jd::data_type::s8, kv_ft};
  ts_descs[io::SRC_V] = {{bs, sl_n, head_num, head_size}, jd::data_type::s8, kv_ft};
  ts_descs[io::MASK] = {{bs}, jd::data_type::s32, jd::format_type::a};
  ts_descs[io::DST] = {{bs, sl_m, head_num, head_size}, dt_dst, jd::format_type::abcd};
  if (badd_dim > 0) {
    SPARSE_LOG_IF(FATAL, badd_dim > 4) << "Unsupported binary add dimention";
    ts_descs[io::BINARY_ADD] = {std::vector<dim_t>(badd_fullshape.cend() - badd_dim, badd_fullshape.cend()),
                                jd::data_type::fp32, jd::plain_format(badd_dim)};
  }
  ts_descs[io::ATT_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::Q_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::K_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::V_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::SRC_DST_SCALE] = {{1}, jd::data_type::fp32, jd::format_type::a};
  ts_descs[io::SRC_DST_ZP] = {{1}, jd::data_type::s32, jd::format_type::a};

  // Step 1.1: Construct Operator config obj
  std::unordered_map<std::string, std::string> attr_map;
  attr_map["approx_exp"] = "True";
  attr_map["stable_softmax"] = "True";
  attr_map["softmax_rescale"] = std::to_string(float{UINT8_MAX});  // TODO(Yi): Use 255?
  return {jd::kernel_kind::mha_dense, jd::kernel_prop::forward_inference, jd::engine_kind::cpu, ts_descs, attr_map};
}

// generate data and setup context for kernel and reference
void set_ctx(const jd::operator_desc& op_desc, exec_context_t* ctx_kern, exec_context_t* ctx_ref,
             bool dynamic_shape = false) {
  const auto& ts_descs = op_desc.tensor_descs();
  const auto batch_size = ts_descs[io::SRC_Q].shape()[0];
  const auto head_num = ts_descs[io::SRC_Q].shape()[2];
  const auto head_size = ts_descs[io::SRC_Q].shape()[3];
  const auto sl_m = ts_descs[io::SRC_Q].shape()[1];
  const auto sl_n = ts_descs[io::SRC_K].shape()[1];

  // Step 2: Construct Tensor ptr
  auto Qs = make_tensor_obj(ts_descs[io::SRC_Q]);
  auto Ks = make_tensor_obj(ts_descs[io::SRC_K]);
  auto Vs = make_tensor_obj(ts_descs[io::SRC_V]);
  auto masks = make_tensor_obj(ts_descs[io::MASK], 1, sl_n);
  auto dsts = make_tensor_obj(ts_descs[io::DST], 0);

  auto badds = make_tensor_obj(ts_descs[io::BINARY_ADD], -1.f, 1.f);

  auto att_scales = make_tensor_obj(ts_descs[io::ATT_SCALE], 1.f);  // TODO(Yi): 1/sqrt
  auto q_scales = make_tensor_obj(ts_descs[io::Q_SCALE], 1.1f);
  auto k_scales = make_tensor_obj(ts_descs[io::K_SCALE], 0.9f);
  auto v_scales = make_tensor_obj(ts_descs[io::V_SCALE], 1.2f);
  auto dst_scales = make_tensor_obj(ts_descs[io::SRC_DST_SCALE], 1.2f);
  auto dst_zps = make_tensor_obj(ts_descs[io::SRC_DST_ZP], 110);

  std::vector<dim_t> dynamic_shapes(io_shape::SIZE, 0);
  if (dynamic_shape) {
    dynamic_shapes[io_shape::BATCH_SIZE] = batch_size;
    dynamic_shapes[io_shape::HEAD_NUM] = head_num;
    dynamic_shapes[io_shape::HEAD_SIZE] = head_size;
    dynamic_shapes[io_shape::M] = sl_m;
    dynamic_shapes[io_shape::N] = sl_n;
  }

  std::vector<memory_storage_t*> mem_src(io_src::SIZE), mem_dst(io_dst::SIZE);
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
  ctx_kern->set_outputs(mem_dst);
  ctx_kern->set_inputs(mem_src);
  ctx_kern->set_dynamic_shape(dynamic_shapes);

  std::vector<memory_storage_t*> ref_mem_src(io_src::SIZE), ref_mem_dst(io_dst::SIZE);
  std::vector<dim_t> ref_dynamic_shapes(io_shape::SIZE, 0);
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
  ctx_ref->set_outputs(ref_mem_dst);
  ctx_ref->set_inputs(ref_mem_src);
  ctx_ref->set_dynamic_shape(dynamic_shapes);
}

// Free memory allocated in `set_ctx`
void free_ctx(exec_context_t* ctx_kern, exec_context_t* ctx_ref) {
  for (auto& ctx : {ctx_kern, ctx_ref}) {
    for (auto& mems : {ctx->inputs(), ctx->outputs()})
      for (auto& mem : mems) {
        void* h;
        mem->get_handle(&h);
        delete[] reinterpret_cast<const char*>(h);
        delete mem;
      }
    ctx->set_inputs({});
    ctx->set_outputs({});
  }
}

static auto case_func = []() {
  std::vector<test_params_t<mha_dims_t>> cases;

  // gencase: bs seqlen head_num head_size

  cases.push_back({{1, 64, 64, 1, 64}, 0, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({{2, 64, 64, 1, 32}, 0, jd::data_type::u8, jd::format_type::abcd, 1, false});

  // acbd
  cases.push_back({{2, 64, 64, 1, 32}, 0, jd::data_type::u8, jd::format_type::acbd, 1, false});

  // headsize 256
  cases.push_back({{1, 64, 64, 1, 256}, 0, jd::data_type::u8, jd::format_type::abcd, 1, false});

  // binary add
  cases.push_back({{3, 64, 64, 2, 256}, 1, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({{3, 64, 64, 2, 256}, 2, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({{3, 64, 64, 2, 256}, 3, jd::data_type::u8, jd::format_type::abcd, 1, false});
  cases.push_back({{3, 64, 64, 2, 256}, 4, jd::data_type::u8, jd::format_type::abcd, 1, false});

  // dt_dst
  cases.push_back({{1, 64, 64, 1, 256}, 0, jd::data_type::bf16, jd::format_type::abcd, 1, false});

  // seqlen 2k
  cases.push_back({{1, 2048, 2048, 1, 32}, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});
  cases.push_back({{1, 2041, 2041, 1, 32}, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});
  cases.push_back({{1, 512, 512, 1, 256}, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});

  // head_size = 64 / 32
  cases.push_back({{4, 384, 384, 16, 64}, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});
  cases.push_back({{4, 384, 384, 16, 32}, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});

  // a variety of seqlen
  for (int seq_len : {32, 33, 47, 48, 49, 63, 64, 65, 128, 384})
    cases.push_back({{12, seq_len, seq_len, 4, 32}, 0, jd::data_type::u8, jd::format_type::abcd, 0, false});

  // kv-cache
  for (int sl_n : {32, 33, 37, 63, 64}) {
    cases.push_back({{4, 1, sl_n, 16, 256}, 2, jd::data_type::u8, jd::format_type::acbd, 0, false});
    cases.push_back({{4, 1, sl_n, 16, 256}, 2, jd::data_type::bf16, jd::format_type::acbd, 0, false});
  }

  return ::testing::ValuesIn(cases);
};

class MhaDenseKernTest : public testing::TestWithParam<test_params_t<mha_dims_t>> {};

TEST_P(MhaDenseKernTest, ) {
  exec_context_t ctx_kern(stream), ctx_ref(stream);

  const auto& t = testing::TestWithParam<test_params_t<mha_dims_t>>::GetParam();
  const auto od =
      gen_opdesc(t.dims.bs, t.dims.sl_m, t.dims.sl_n, t.dims.head_num, t.dims.head_size, t.badd_dim, t.dt_dst);
  const std::shared_ptr<void> with_ctx{(set_ctx(od, &ctx_kern, &ctx_ref), nullptr),
                                       [&](...) { free_ctx(&ctx_kern, &ctx_ref); }};
  EXPECT_TRUE(check_result(t.nthr, t.expect_to_fail, {od, ctx_kern, ctx_ref}));
}

INSTANTIATE_TEST_SUITE_P(Kernels, MhaDenseKernTest, case_func(), to_string);

class MhaDenseKernDynShapeTest : public testing::TestWithParam<test_params_t<std::vector<mha_dims_t>>> {};

TEST_P(MhaDenseKernDynShapeTest, ) {
  const auto& t = testing::TestWithParam<test_params_t<std::vector<mha_dims_t>>>::GetParam();
  const auto dims = max_dims(t.dims);  // dims' upper bound to initialize the kernels
  const auto od = gen_opdesc(dims.bs, dims.sl_m, dims.sl_n, dims.head_num, dims.head_size, t.badd_dim, t.dt_dst);

  exec_context_t ctx_kern(stream), ctx_ref(stream);
  std::shared_ptr<const jd::kernel_desc_t> ref_kern_desc;
  jd::kernel_desc_t::create<jd::mha_dense_ref_kd_t>(ref_kern_desc, od);
  std::shared_ptr<const kernel_t> ref_kern;
  kernel_t::create<jd::mha_dense_ref_k_t, jd::mha_dense_ref_kd_t>(ref_kern, ref_kern_desc);
  const auto ref_ws_mem = prepare_workspace(&ctx_ref, *ref_kern);

  n_thread_t with_n_thread(t.nthr);
  jd::mha_dense_desc mha_dense_desc(od);
  jd::mha_dense mha_dense_kernel(mha_dense_desc);
  const auto kern_ws_mem = prepare_workspace(&ctx_kern, mha_dense_kernel);

  for (auto& dims : t.dims) {
    SPARSE_LOG(INFO) << "Executing shape: " << dims;
    const auto it_od = gen_opdesc(dims.bs, dims.sl_m, dims.sl_n, dims.head_num, dims.head_size, t.badd_dim, t.dt_dst);
    const std::shared_ptr<void> with_ctx{(set_ctx(it_od, &ctx_kern, &ctx_ref, true), nullptr),
                                         [&](...) { free_ctx(&ctx_kern, &ctx_ref); }};
    {
      n_thread_t with_n_thread(0);  // speedup reference
      ref_kern->execute(ctx_ref);
    }
    mha_dense_kernel.execute(ctx_kern);

    const auto size_dst = it_od.tensor_descs()[io::DST].size();
    ASSERT_TRUE(compare_data(t.dt_dst, size_dst, ctx_kern, ctx_ref));
  }
}

INSTANTIATE_TEST_SUITE_P(  //
    Kernels, MhaDenseKernDynShapeTest,
    ::testing::ValuesIn(std::vector<test_params_t<std::vector<mha_dims_t>>>{
        {{{4, 1, 33, 16, 256}, {4, 1, 34, 16, 256}, {4, 1, 35, 16, 256}, {4, 1, 36, 16, 256}},
         2,
         jd::data_type::u8,
         jd::format_type::acbd,
         0,
         false},
        {{{1, 64, 64, 1, 64}, {1, 64, 128, 1, 64}, {1, 128, 64, 1, 64}},
         2,
         jd::data_type::u8,
         jd::format_type::acbd,
         0,
         false},
    }));

}  // namespace test
