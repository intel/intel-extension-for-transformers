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

#include "attention_ref.hpp"
#include "kernels/attention_types.hpp"

#include "kernel_desc.hpp"
#include "kernel.hpp"
#include "spmm_ref.hpp"
#include "softmax_ref.hpp"
#include "matmul_ref.hpp"

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "Attention kernel requires `" << #f << "`"; \
    return false;                                                    \
  }

/**
 * Entries of op_desc_.attrs:
 *   q_weight_ptr: address of q_weight_ptr in plain format
 *   k_weight_ptr: address of k_weight_ptr in plain format
 *   v_weight_ptr: address of v_weight_ptr in plain format
 *   q_bias_ptr: address of q_bias_ptr in plain format
 *   k_bias_ptr: address of k_bias_ptr in plain format
 *   v_bias_ptr: address of v_bias_ptr in plain format
 *   q_scales_ptr: address of q_scales_ptr in plain format
 *   k_scales_ptr: address of k_scales_ptr in plain format
 *   v_scales_ptr: address of v_scales_ptr in plain format
 *
 *   out_scale: scale for matmul value of the first matmul
 *   softmax_in_scale: as name; also used for the outout of first matmul
 *   softmax_in_zero_point: as name; also used for the outout of first matmul
 *   softmax_out_scale: as name
 *   softmax_out_zero_point: as name
 */
namespace {
enum SubKernel {
  QK_SPMM = 0,
  Q_K_GEMM = 1,
  SOFTMAX = 2,
  V_SPMM = 3,
  QK_V_MATMUL = 4,
  NUM = 5,
};
}  // namespace

namespace jd {
using matmul_io = ssd::matmul_io::io;

attention_ref_kd_t::~attention_ref_kd_t() {
  if (fused_bias_addr_ != nullptr) aligned_allocator_t<char>::deallocate(fused_bias_addr_);
  if (fused_scales_addr_ != nullptr) aligned_allocator_t<char>::deallocate(fused_scales_addr_);
  if (qk_sparse_ptr_ != nullptr) delete qk_sparse_ptr_;
  if (v_sparse_ptr_ != nullptr) delete v_sparse_ptr_;
  if (qk_weight_addr_ != nullptr) aligned_allocator_t<char>::deallocate(qk_weight_addr_);
}

template <typename T_kd>
inline bool attention_ref_kd_t::add_kernel_desc(const operator_desc& op_desc, const char* name) {
  std::shared_ptr<const kernel_desc_t> kd;
  if (!kernel_desc_t::create<T_kd>(kd, op_desc)) {
    SPARSE_LOG(WARNING) << "Attention failed to create sub-kernel: " << name;
    return false;
  }
  kernel_descs_.push_back(kd);
  return true;
}

bool attention_ref_kd_t::init() {
  if (!isa_available(avx512_core)) return false;  // deatiled check are left to sub-kernels
  // part 0
  const auto& src_desc = op_desc_.tensor_descs()[attention_io::MERGE_SRC];
  const auto& q_weight_desc = op_desc_.tensor_descs()[attention_io::Q_WEIGHT];
  const auto& k_weight_desc = op_desc_.tensor_descs()[attention_io::K_WEIGHT];
  const auto& v_weight_desc = op_desc_.tensor_descs()[attention_io::V_WEIGHT];
  const auto& q_bias_desc = op_desc_.tensor_descs()[attention_io::Q_BIAS];
  const auto& k_bias_desc = op_desc_.tensor_descs()[attention_io::K_BIAS];
  const auto& v_bias_desc = op_desc_.tensor_descs()[attention_io::V_BIAS];
  const auto& q_scales_desc = op_desc_.tensor_descs()[attention_io::Q_SCALES];
  const auto& k_scales_desc = op_desc_.tensor_descs()[attention_io::K_SCALES];
  const auto& v_scales_desc = op_desc_.tensor_descs()[attention_io::V_SCALES];
  const auto& q_k_src2_desc = op_desc_.tensor_descs()[attention_io::Q_K_SRC2];  // for binary add
  const auto& dst_desc = op_desc_.tensor_descs()[attention_io::MERGE_DST];
  SPARSE_LOG_IF(FATAL, q_bias_desc.shape().empty() || k_bias_desc.shape().empty() || v_bias_desc.shape().empty())
      << "Attension shoulf have valid bias" << std::endl;

  // check tensor dim
  KERNEL_INIT_CHECK(dst_desc.shape().size() == 4)
  const dim_t head_num = dst_desc.shape()[0];
  const dim_t head_size = dst_desc.shape()[1];
  const dim_t batch_size = dst_desc.shape()[2];
  const dim_t seq_len = dst_desc.shape()[3];
  const dim_t ip_chanel = head_num * head_size;  // channel width for any of the three linear layer

  KERNEL_INIT_CHECK(src_desc.shape() == std::vector<dim_t>({batch_size, ip_chanel, seq_len}))
  KERNEL_INIT_CHECK(src_desc.dtype() == data_type::u8)
  for (auto& wei_desc : {q_weight_desc, k_weight_desc, v_weight_desc}) {
    KERNEL_INIT_CHECK(wei_desc.shape() == std::vector<dim_t>({ip_chanel, ip_chanel}))
    KERNEL_INIT_CHECK(wei_desc.dtype() == data_type::s8)
    KERNEL_INIT_CHECK(wei_desc.ftype() == format_type::bsr)
  }
  for (auto& bias_desc : {q_bias_desc, k_bias_desc, v_bias_desc}) {
    KERNEL_INIT_CHECK(bias_desc.shape() == std::vector<dim_t>({ip_chanel, 1}))
    KERNEL_INIT_CHECK(bias_desc.dtype() == data_type::s32)
    KERNEL_INIT_CHECK(bias_desc.ftype() == format_type::ab)
  }
  for (auto& scale_desc : {q_scales_desc, k_scales_desc, v_scales_desc}) {
    KERNEL_INIT_CHECK(scale_desc.shape() == std::vector<dim_t>({ip_chanel, 1}))
    KERNEL_INIT_CHECK(scale_desc.dtype() == data_type::fp32)
    KERNEL_INIT_CHECK(scale_desc.ftype() == format_type::ab)
  }
  KERNEL_INIT_CHECK(q_k_src2_desc.shape() == std::vector<dim_t>({batch_size, head_num, seq_len, seq_len}))
  KERNEL_INIT_CHECK(q_k_src2_desc.dtype() == data_type::fp32)

  const tensor_desc qkv_weight_desc({ip_chanel * 3, ip_chanel}, q_weight_desc.dtype(), q_weight_desc.ftype());
  const tensor_desc qkv_bias_desc({ip_chanel * 3, 1}, q_bias_desc.dtype(), q_bias_desc.ftype());
  const tensor_desc qkv_scales_desc({ip_chanel * 3, 1}, q_scales_desc.dtype(), q_scales_desc.ftype());
  const tensor_desc qk_weight_desc({ip_chanel * 2, ip_chanel}, q_weight_desc.dtype(), q_weight_desc.ftype());
  const tensor_desc qk_bias_desc({ip_chanel * 2, 1}, q_bias_desc.dtype(), q_bias_desc.ftype());
  const tensor_desc qk_scales_desc({ip_chanel * 2, 1}, q_scales_desc.dtype(), q_scales_desc.ftype());
  const tensor_desc qk_dst_desc_s8({ip_chanel * 2, batch_size * seq_len}, data_type::s8, format_type::ab);
  const tensor_desc qk_dst_desc_f32({ip_chanel * 2, batch_size * seq_len}, data_type::fp32, format_type::ab);

  auto op_attrs = op_desc_.attrs();
  const auto q_weight_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["q_weight_ptr"]));
  const auto k_weight_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["k_weight_ptr"]));
  const auto v_weight_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["v_weight_ptr"]));
  const auto q_bias_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["q_bias_ptr"]));
  const auto k_bias_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["k_bias_ptr"]));
  const auto v_bias_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["v_bias_ptr"]));
  const auto q_scales_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["q_scales_ptr"]));
  const auto k_scales_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["k_scales_ptr"]));
  const auto v_scales_addr = reinterpret_cast<void*>(str_to_num<intptr_t>(op_attrs["v_scales_ptr"]));

  // QKV weight bias scales merge
  const size_t wei_bytes = ip_chanel * ip_chanel * type_size[q_weight_desc.dtype()];
  const size_t bias_bytes = ip_chanel * 1 * type_size[q_bias_desc.dtype()];
  const size_t scale_bytes = ip_chanel * 1 * type_size[q_scales_desc.dtype()];

  // weight merge
  // TODO(Yi): merge QKV together
  qk_weight_addr_ = reinterpret_cast<void*>(aligned_allocator_t<int8_t>::allocate(ip_chanel * ip_chanel * 2));
  memcpy(qk_weight_addr_, q_weight_addr, wei_bytes);
  void* qk_weight_addr_offset = reinterpret_cast<void*>(reinterpret_cast<intptr_t>(qk_weight_addr_) + wei_bytes);
  memcpy(qk_weight_addr_offset, k_weight_addr, wei_bytes);

  qk_sparse_ptr_ = new bsr_data_t<int8_t>(spns::reorder_to_bsr_group<int8_t, 4>(
      ip_chanel * 2, ip_chanel, 4, 1, reinterpret_cast<int8_t*>(qk_weight_addr_)));

  // bias merge
  fused_bias_addr_ = reinterpret_cast<void*>(aligned_allocator_t<char>::allocate(bias_bytes * 3));
  memcpy(fused_bias_addr_, q_bias_addr, bias_bytes);
  void* fused_bias_addr_offset = reinterpret_cast<void*>(reinterpret_cast<intptr_t>(fused_bias_addr_) + bias_bytes);
  memcpy(fused_bias_addr_offset, k_bias_addr, bias_bytes);
  fused_bias_addr_offset = reinterpret_cast<void*>(reinterpret_cast<intptr_t>(fused_bias_addr_) + bias_bytes * 2);
  memcpy(fused_bias_addr_offset, v_bias_addr, bias_bytes);

  // scales merge
  fused_scales_addr_ = reinterpret_cast<void*>(aligned_allocator_t<char>::allocate(scale_bytes * 3));
  memcpy(fused_scales_addr_, q_scales_addr, scale_bytes);
  void* fused_scales_addr_offset =
      reinterpret_cast<void*>(reinterpret_cast<intptr_t>(fused_scales_addr_) + scale_bytes);
  memcpy(fused_scales_addr_offset, k_scales_addr, scale_bytes);
  fused_scales_addr_offset = reinterpret_cast<void*>(reinterpret_cast<intptr_t>(fused_scales_addr_) + scale_bytes * 2);
  memcpy(fused_scales_addr_offset, v_scales_addr, scale_bytes);

  const float softmax_in_zero_point = str_to_num<float>(op_attrs["softmax_in_zero_point"]);
  const float softmax_in_scale = str_to_num<float>(op_attrs["softmax_in_scale"]);
  const float softmax_out_zero_point = str_to_num<float>(op_attrs["softmax_out_zero_point"]);
  const float softmax_out_scale = str_to_num<float>(op_attrs["softmax_out_scale"]);

  // sub-kernel 0: spmm for QK
  operator_desc spmm_qk_desc =
      operator_desc(kernel_kind::sparse_matmul, op_desc_.kernel_prop(), op_desc_.engine_kind(), op_desc_.runtime_kind(),
                    {qk_weight_desc, src_desc, qk_bias_desc, qk_dst_desc_s8, qk_scales_desc},
                    {{"weight_ptr", std::to_string(reinterpret_cast<uint64_t>(qk_weight_addr_))},
                     {"sparse_ptr", std::to_string(reinterpret_cast<uint64_t>(qk_sparse_ptr_))},
                     {"bias_addr", std::to_string(reinterpret_cast<uint64_t>(fused_bias_addr_))},
                     {"scales_addr", std::to_string(reinterpret_cast<uint64_t>(fused_scales_addr_))}});
  if (!add_kernel_desc<spmm_ref_kd_t>(spmm_qk_desc, "spmm_qk")) return false;

  // sub-kernel 1 transmatmul for Q x K & eltwise
  const tensor_desc q_out_desc({head_num, head_size, batch_size, seq_len}, data_type::s8, format_type::ab);
  const tensor_desc k_out_desc({head_num, head_size, batch_size, seq_len}, data_type::s8, format_type::ab);
  const tensor_desc qk_out_desc({batch_size, head_num, seq_len, seq_len}, data_type::s8, format_type::ab);

  postop_attr quant_s8_attr = postop_attr(data_type::s8, postop_type::eltwise, postop_alg::quantize,
                                          softmax_in_zero_point, 0, softmax_in_scale);

  const operator_desc matmul_qk_desc(kernel_kind::transpose_matmul, op_desc_.kernel_prop(), op_desc_.engine_kind(),
                                     {q_out_desc, k_out_desc, qk_out_desc, q_k_src2_desc},
                                     {{"alpha", op_attrs["out_scale"]}}, {quant_s8_attr});

  if (!add_kernel_desc<matmul_ref_kd_t>(matmul_qk_desc, "matmul_qk")) return false;

  // sub-kernel 2: softmax(s8->BF16(LUT)->u8)
  postop_attr dequantize_s8_attr(data_type::s8, postop_type::eltwise, postop_alg::dequantize, softmax_in_zero_point, 0,
                                 softmax_in_scale);
  postop_attr quant_u8_attr(data_type::u8, postop_type::eltwise, postop_alg::quantize, softmax_out_zero_point, 0,
                            softmax_out_scale);
  tensor_desc softmax_output = tensor_desc({qk_out_desc.shape(), data_type::u8, format_type::undef});
  operator_desc softmax_desc = operator_desc(
      kernel_kind::softmax, op_desc_.kernel_prop(), op_desc_.engine_kind(), {qk_out_desc, softmax_output},
      {{"spec_type", "lut"}, {"vec_len", std::to_string(seq_len)}, {"postop_list", "attention_dequantize+quantize"}},
      {dequantize_s8_attr, quant_u8_attr});
  if (!add_kernel_desc<softmax_ref_kd_t>(softmax_desc, "softmax")) return false;

  // sub-kernel 3: spmm for V
  v_sparse_ptr_ = new bsr_data_t<int8_t>(spns::reorder_to_bsr_group<int8_t, 4>(
      ip_chanel, ip_chanel, 4, 1, reinterpret_cast<const int8_t*>(v_weight_addr)));
  tensor_desc v_out_desc = tensor_desc({head_num, head_size, batch_size, seq_len}, data_type::s8, format_type::ab);
  operator_desc spmm_v_desc =
      operator_desc(kernel_kind::sparse_matmul, op_desc_.kernel_prop(), op_desc_.engine_kind(),
                        {v_weight_desc, src_desc, v_bias_desc, v_out_desc, v_scales_desc},
                        {{"weight_ptr", std::to_string(reinterpret_cast<uint64_t>(v_weight_addr))},
                         {"sparse_ptr", std::to_string(reinterpret_cast<uint64_t>(v_sparse_ptr_))},
                         {"bias_addr", std::to_string(reinterpret_cast<uint64_t>(v_bias_addr))},
                         {"scales_addr", std::to_string(reinterpret_cast<uint64_t>(v_scales_addr))}});
  if (!add_kernel_desc<spmm_ref_kd_t>(spmm_v_desc, "spmm_v")) return false;

  // sub-kernel 4: transpose matmul for QK x V
  tensor_desc src2_desc = tensor_desc({}, data_type::fp32, format_type::ab);  // binary postop not supported
  tensor_desc scale_desc = tensor_desc({1}, data_type::fp32, format_type::a);
  tensor_desc zp_desc = tensor_desc({1}, data_type::fp32, format_type::a);

  operator_desc qk_v_desc =
      operator_desc(kernel_kind::transpose_matmul, op_desc_.kernel_prop(), op_desc_.engine_kind(),
                        {softmax_output, v_out_desc, dst_desc, src2_desc, scale_desc, zp_desc}, {});
  if (!add_kernel_desc<matmul_ref_kd_t>(qk_v_desc, "qk_matmul_v")) return false;
  return true;
}

attention_ref_k_t::~attention_ref_k_t() {
  if (!mem_.empty()) {
    aligned_allocator_t<char>::deallocate(mem_[SubKernel::QK_SPMM][ssd::DST]);
  }
}

bool attention_ref_k_t::setup_kernel() {
  kernels_.resize(SubKernel::NUM);
  // QK
  const auto& qk_spmm_desc = derived_kd()->get_kernel_desc(SubKernel::QK_SPMM);
  if (!create<spmm_ref_k_t, spmm_ref_kd_t>(kernels_[SubKernel::QK_SPMM], qk_spmm_desc)) return false;

  // Q X K
  const auto& q_k_gemm_desc = derived_kd()->get_kernel_desc(SubKernel::Q_K_GEMM);
  if (!create<matmul_ref_k_t, matmul_ref_kd_t>(kernels_[SubKernel::Q_K_GEMM], q_k_gemm_desc)) return false;

  // Softmax
  const auto& softmax_desc = derived_kd()->get_kernel_desc(SubKernel::SOFTMAX);
  if (!create<softmax_ref_k_t, softmax_ref_kd_t>(kernels_[SubKernel::SOFTMAX], softmax_desc)) return false;
  // V
  const auto& v_spmm_desc = derived_kd()->get_kernel_desc(SubKernel::V_SPMM);
  if (!create<spmm_ref_k_t, spmm_ref_kd_t>(kernels_[SubKernel::V_SPMM], v_spmm_desc)) return false;

  // QK X V
  const auto& qk_v_matmul = derived_kd()->get_kernel_desc(SubKernel::QK_V_MATMUL);
  if (!create<matmul_ref_k_t, matmul_ref_kd_t>(kernels_[SubKernel::QK_V_MATMUL], qk_v_matmul)) return false;
  return true;
}
void attention_ref_k_t::setup_memory() {
  mem_.resize(SubKernel::NUM);
  std::vector<size_t> offset;  // sizes in bytes for intermediate results
  const auto tensor_bytes = [](const tensor_desc& d) { return d.size() * type_size[d.dtype()]; };

  offset.push_back(tensor_bytes(ker_opdesc(SubKernel::QK_SPMM).tensor_descs()[ssd::DST]));
  offset.push_back(tensor_bytes(ker_opdesc(SubKernel::Q_K_GEMM).tensor_descs()[matmul_io::DST0]));
  offset.push_back(tensor_bytes(ker_opdesc(SubKernel::SOFTMAX).tensor_descs()[1]));
  offset.push_back(tensor_bytes(ker_opdesc(SubKernel::V_SPMM).tensor_descs()[ssd::DST]));
  // the last kernel(QK(softmax) * V) don't need alloc memory

  char* base = aligned_allocator_t<char>::allocate(std::accumulate(offset.begin(), offset.end(), 0));

  // part0 QK merged inner product with spmm_vnni
  mem_[SubKernel::QK_SPMM].resize(ssd::SCALES + 1);
  mem_[SubKernel::QK_SPMM][ssd::WEI] =
      reinterpret_cast<char*>(str_to_num<intptr_t>(ker_opdesc(SubKernel::QK_SPMM).attrs().at("weight_ptr")));
  mem_[SubKernel::QK_SPMM][ssd::BIAS] =
      reinterpret_cast<char*>(str_to_num<intptr_t>(ker_opdesc(SubKernel::QK_SPMM).attrs().at("bias_addr")));
  mem_[SubKernel::QK_SPMM][ssd::SCALES] =
      reinterpret_cast<char*>(str_to_num<intptr_t>(ker_opdesc(SubKernel::QK_SPMM).attrs().at("scales_addr")));
  mem_[SubKernel::QK_SPMM][ssd::DST] = base;

  // part1 Q X K
  mem_[SubKernel::Q_K_GEMM].resize(4);
  mem_[SubKernel::Q_K_GEMM][matmul_io::SRC0] = mem_[SubKernel::QK_SPMM][ssd::DST];
  mem_[SubKernel::Q_K_GEMM][matmul_io::SRC1] =
      mem_[SubKernel::QK_SPMM][ssd::DST] + offset[0] / 2;  // split qk out to q and k
  mem_[SubKernel::Q_K_GEMM][matmul_io::DST0] = mem_[SubKernel::QK_SPMM][ssd::DST] + offset[0];  // dst
  mem_[SubKernel::Q_K_GEMM][matmul_io::SRC2] = nullptr;

  // part2 Softmax
  mem_[SubKernel::SOFTMAX].resize(2);
  mem_[SubKernel::SOFTMAX][0] = mem_[SubKernel::Q_K_GEMM][matmul_io::DST0];
  mem_[SubKernel::SOFTMAX][1] = mem_[SubKernel::Q_K_GEMM][matmul_io::DST0] + offset[1];

  // part5 spmm for V
  mem_[SubKernel::V_SPMM].resize(ssd::SCALES + 1);
  mem_[SubKernel::V_SPMM][ssd::WEI] =
      reinterpret_cast<char*>(str_to_num<intptr_t>(ker_opdesc(SubKernel::V_SPMM).attrs().at("weight_ptr")));
  mem_[SubKernel::V_SPMM][ssd::BIAS] =
      reinterpret_cast<char*>(str_to_num<intptr_t>(ker_opdesc(SubKernel::V_SPMM).attrs().at("bias_addr")));
  mem_[SubKernel::V_SPMM][ssd::SCALES] =
      reinterpret_cast<char*>(str_to_num<intptr_t>(ker_opdesc(SubKernel::V_SPMM).attrs().at("scales_addr")));
  mem_[SubKernel::V_SPMM][ssd::DST] = mem_[SubKernel::SOFTMAX][1] + offset[2];

  // part6  V X QK(softmax out)
  mem_[SubKernel::QK_V_MATMUL].resize(matmul_io::ZP0 + 1);
  mem_[SubKernel::QK_V_MATMUL][matmul_io::SRC0] = mem_[SubKernel::SOFTMAX][1];
  mem_[SubKernel::QK_V_MATMUL][matmul_io::SRC1] = mem_[SubKernel::V_SPMM][ssd::DST];
  mem_[SubKernel::QK_V_MATMUL][matmul_io::DST0] = nullptr;
  mem_[SubKernel::QK_V_MATMUL][matmul_io::SRC2] = nullptr;
  mem_[SubKernel::QK_V_MATMUL][matmul_io::SCALE0] = nullptr;
  mem_[SubKernel::QK_V_MATMUL][matmul_io::ZP0] = nullptr;
}
bool attention_ref_k_t::init() {
  // Create kernel
  if (!setup_kernel()) return false;

  // Create feature map memory
  setup_memory();
  return true;
}

bool attention_ref_k_t::execute(const std::vector<const void*>& rt_data) const {
  for (size_t i = 0; i < kernels_.size(); i++) {
    std::vector<const void*> data = set_input_output(i, rt_data);
    kernels_[i]->execute(data);
  }
  return true;
}

std::vector<const void*> attention_ref_k_t::set_input_output(int index, const std::vector<const void*>& rt_data) const {
  std::vector<const void*> data;
  data.assign(mem_[index].begin(), mem_[index].end());
  if (index == SubKernel::QK_SPMM || index == SubKernel::V_SPMM) {
    // part0 QK spmm_vnni and part5 V spmm_vnni
    data[ssd::SRC] = rt_data[attention_io::MERGE_SRC];
  } else if (index == SubKernel::Q_K_GEMM) {
    data[matmul_io::SRC2] = rt_data[attention_io::Q_K_SRC2];
  } else if (index == SubKernel::QK_V_MATMUL) {
    // part4 transpose matmul for QK x V
    data[matmul_io::DST0] = rt_data[attention_io::MERGE_DST];
    data[matmul_io::SCALE0] = rt_data[attention_io::QK_V_OUTPUT_SCALES];
    data[matmul_io::ZP0] = rt_data[attention_io::QK_V_OUTPUT_ZERO_POINT];
  }
  return data;
}

}  // namespace jd
