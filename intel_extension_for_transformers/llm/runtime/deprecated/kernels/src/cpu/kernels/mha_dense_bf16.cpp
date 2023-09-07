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

#include "mha_dense_bf16.hpp"

#include <algorithm>
#include <unordered_map>

#include "mha_dense_ctx.hpp"

#define KERNEL_INIT_CHECK(f)                                              \
  if (!(f)) {                                                             \
    SPARSE_LOG(ERROR) << "MHA dense bf16 kernel requires `" << #f << "`"; \
    return false;                                                         \
  }
namespace jd {

using ft = format_type;

bool mha_dense_bf16_kd_t::init() {
  if (!isa_available(avx512_core_bf16_amx_bf16)) return false;
  const auto attrs = op_desc_.attrs();
  KERNEL_INIT_CHECK(attrs.find("approx_exp") != attrs.end() && attrs.at("approx_exp") == "True");
  KERNEL_INIT_CHECK(attrs.find("stable_softmax") != attrs.end() &&
                    (attrs.at("stable_softmax") == "False" || attrs.at("stable_softmax") == "True"));
  KERNEL_INIT_CHECK(attrs.find("merged_QKV") == attrs.end() || attrs.at("merged_QKV") != "True");  // no merge support
  KERNEL_INIT_CHECK(std::all_of(attrs.cbegin(), attrs.cend(), [](auto&& kv) {  // no unrecognized attr
    return is_any_of({"approx_exp", "stable_softmax", "merged_QKV"}, [&kv](auto&& k) { return kv.first == k; });
  }))

  auto& desc = op_desc_.tensor_descs();
  const auto batch_size = desc[io::SRC_Q].shape()[0];
  const auto sl_m = desc[io::SRC_Q].shape()[1];
  const auto sl_n = desc[io::SRC_K].shape()[1];
  const auto head_num = desc[io::SRC_Q].shape()[2];
  const auto head_size = desc[io::SRC_Q].shape()[3];

  KERNEL_INIT_CHECK((desc[io::SRC_Q].shape() == std::vector<dim_t>{batch_size, sl_m, head_num, head_size}));
  KERNEL_INIT_CHECK((desc[io::SRC_K].shape() == std::vector<dim_t>{batch_size, sl_n, head_num, head_size}));
  KERNEL_INIT_CHECK((desc[io::SRC_V].shape() == std::vector<dim_t>{batch_size, sl_n, head_num, head_size}));
  KERNEL_INIT_CHECK((desc[io::DST].shape() == std::vector<dim_t>{batch_size, sl_m, head_num, head_size}));
  const auto& badd_shape = desc[io::BINARY_ADD].shape();
  if (!badd_shape.empty()) {
    KERNEL_INIT_CHECK(desc[io::BINARY_ADD].dtype() == data_type::fp32);
    KERNEL_INIT_CHECK(desc[io::BINARY_ADD].ftype() == plain_format(badd_shape.size()));
    const auto& badd_pad1 = pre_pad1(4, badd_shape);
    KERNEL_INIT_CHECK((badd_pad1 == std::vector<dim_t>{1, 1, sl_m, sl_n}));  // TODO(Yi): tmp restriction to removed
    KERNEL_INIT_CHECK(badd_pad1[0] == 1 || badd_pad1[0] == batch_size);
    KERNEL_INIT_CHECK(badd_pad1[1] == 1 || badd_pad1[1] == head_num);
    KERNEL_INIT_CHECK(badd_pad1[2] == 1 || badd_pad1[2] == sl_m);
    KERNEL_INIT_CHECK(badd_pad1[3] == 1 || badd_pad1[3] == sl_n);
  }
  KERNEL_INIT_CHECK(
      (desc[io::MASK].shape().empty() || desc[io::MASK] == tensor_desc{{batch_size}, data_type::s32, ft::a}));
  KERNEL_INIT_CHECK((desc[io::ATT_SCALE] == tensor_desc{{1}, data_type::fp32, ft::a}));

  KERNEL_INIT_CHECK(desc[io::Q_SCALE].shape().empty());
  KERNEL_INIT_CHECK(desc[io::Q_ZP].shape().empty());
  KERNEL_INIT_CHECK(desc[io::K_SCALE].shape().empty());
  KERNEL_INIT_CHECK(desc[io::K_ZP].shape().empty());
  KERNEL_INIT_CHECK(desc[io::V_SCALE].shape().empty());
  KERNEL_INIT_CHECK(desc[io::V_ZP].shape().empty());
  KERNEL_INIT_CHECK(desc[io::DST_SCALE].shape().empty());
  KERNEL_INIT_CHECK(desc[io::DST_ZP].shape().empty());

  // dtype
  KERNEL_INIT_CHECK(is_all_of(
      {
          desc[io::SRC_Q].dtype(),
          desc[io::SRC_K].dtype(),
          desc[io::SRC_V].dtype(),
          desc[io::DST].dtype(),
      },
      [&](const data_type t) { return t == data_type::bf16; }));
  return true;
}

constexpr int mha_dense_bf16_k_t::PAD_SIZE;

template <typename K, typename V, typename K2, typename V2>
inline V get_entry(const std::unordered_map<K, V>& map, const K2& k, const V2& v_default) {
  static_assert(std::is_convertible<K2, K>::value);
  static_assert(std::is_convertible<V2, V>::value);
  return map.find(k) != map.end() ? map.at(k) : v_default;
}

mha_dense_bf16_k_t::mha_dense_bf16_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      ts_descs_(derived_kd()->get_operator_desc().tensor_descs()),
      has_pmask(!ts_descs_[io::MASK].shape().empty()),
      has_badd(!ts_descs_[io::BINARY_ADD].shape().empty()),
      dt_dst(ts_descs_[io::DST].dtype()),
      bs_(ts_descs_[io::SRC_Q].shape()[0]),
      sl_m_(ts_descs_[io::SRC_Q].shape()[1]),
      sl_n_(ts_descs_[io::SRC_K].shape()[1]),
      head_num_(ts_descs_[io::SRC_Q].shape()[2]),
      head_size_(ts_descs_[io::SRC_Q].shape()[3]),
      ld_src_(head_size_ * head_num_),
      ld_dst_(head_size_ * head_num_),
      sl_n_pad_(pad_to(sl_n_, PAD_SIZE)),
      head_size_pad_(pad_to(head_size_, PAD_SIZE)),
      reo_k_size_(bs_ * head_num_ * sl_n_pad_ * head_size_pad_),
      reo_v_size_(bs_ * head_num_ * sl_n_pad_ * head_size_pad_),
      thread_reo_q_size_(32 * head_size_pad_),
      thread_softmax_size_(32 * sl_n_pad_),
      thread_dst_size_(32 * head_size_pad_),
      thread_total_bytes_(sizeof(bfloat16_t) * thread_reo_q_size_ +    //
                          sizeof(bfloat16_t) * thread_softmax_size_ +  //
                          get_data_size(dt_dst) * thread_dst_size_),
      tmp_badd_size_(bs_ * sl_m_ * sl_n_pad_),
      workspace_size_(sizeof(bfloat16_t) * reo_k_size_ +                     //
                      sizeof(bfloat16_t) * reo_v_size_ +                     //
                      omp_get_max_threads() * thread_total_bytes_ +          //
                      sizeof(float) * tmp_badd_size_ +                       //
                      sizeof(float) * pad_to(sl_m_, PAD_SIZE) * sl_n_pad_),  // extra space for badd to read
      amx_full_tile_param_(16, 16, 32, true, 2),
      amx_full_tile_cfg_(amx_full_tile_param_),
      kern_tr_k({/*.pad_n = */ 64, /*.cvt_s8u8 = */ false, 2}),
      kern_tr_v(32, sizeof(bfloat16_t)),
      kern_tr_q(),
      kern_qksoftmax(has_pmask || has_badd || sl_n_pad_ != sl_n_,  // pmask is applied via binary add
                     derived_kd()->get_operator_desc().attrs().at("stable_softmax") == "True", &amx_full_tile_param_),
      kern_mmav(&amx_full_tile_param_) {}

bool mha_dense_bf16_k_t::init() {
  if (!ker_amx_cfg_.create_kernel()) return false;
  if (!ker_amx_rls_.create_kernel()) return false;

  if (!kern_tr_k.create_kernel()) return false;
  if (!kern_tr_v.create_kernel()) return false;
  if (!kern_tr_q.create_kernel()) return false;
  if (!kern_qksoftmax.create_kernel()) return false;
  if (!kern_mmav.create_kernel()) return false;

  return true;
}

bool mha_dense_bf16_k_t::execute(const exec_context_t& ctx) const {
  void *src_data[io_src::SIZE], *dst_data[io_dst::SIZE], *workspace;
  dim_t shape_data[io_shape::SIZE];
  for (auto i = 0; i < io_src::SIZE; ++i) ctx.input(i)->get_handle(&src_data[i]);
  for (auto i = 0; i < io_dst::SIZE; ++i) ctx.output(i)->get_handle(&dst_data[i]);
  for (auto i = 0; i < io_shape::SIZE; ++i) shape_data[i] = ctx.get_dynamic_shape()[i];
  ctx.workspace()->get_handle(&workspace);

  const auto sl_m = shape_data[io_shape::M] ? shape_data[io_shape::M] : sl_m_;
  const auto sl_n = shape_data[io_shape::N] ? shape_data[io_shape::N] : sl_n_;  // TODO(Yi): sl_n affects kern_qksoftmax
  assert(shape_data[io_shape::BATCH_SIZE] <= 0);                                // dynamic BATCH_SIZE not supported
  assert(shape_data[io_shape::HEAD_SIZE] <= 0);                                 // dynamic HEAD_SIZE not supported
  assert(shape_data[io_shape::HEAD_NUM] <= 0);                                  // dynamic HEAD_NUM not supported
  const auto sl_n_pad = pad_to(sl_n, PAD_SIZE);

  int constexpr m_tile = 32;
  SPARSE_DLOG_IF(FATAL, has_badd && !src_data[io_src::BINARY_ADD]) << "Binary-add operand expected but not passed!";
  SPARSE_DLOG_IF(FATAL, has_pmask && !src_data[io_src::MASK]) << "Padding mask expected but not passed!";

  const auto src_q = reinterpret_cast<const bfloat16_t*>(src_data[io_src::SRC_Q]);
  const auto src_k = reinterpret_cast<const bfloat16_t*>(src_data[io_src::SRC_K]);
  const auto src_v = reinterpret_cast<const bfloat16_t*>(src_data[io_src::SRC_V]);
  const auto pmask = reinterpret_cast<const int32_t*>(src_data[io_src::MASK]);
  const auto badd = has_badd ? reinterpret_cast<const float*>(src_data[io_src::BINARY_ADD]) : nullptr;
  const auto dst = reinterpret_cast<bfloat16_t*>(const_cast<void*>(dst_data[io_dst::DST]));

  const auto reo_k = reinterpret_cast<bfloat16_t*>(workspace);
  const auto reo_v = reinterpret_cast<bfloat16_t*>(reo_k + reo_k_size_);
  const auto tmp_thread = reinterpret_cast<char*>(reo_v + reo_v_size_);
  const auto tmp_mask = reinterpret_cast<float*>(tmp_thread + omp_get_max_threads() * thread_total_bytes_);

  const auto att_scale = reinterpret_cast<const float*>(src_data[io_src::ATT_SCALE])[0];

// Reorder K & V on workspace
#pragma omp parallel for collapse(2)
  for (int ibat = 0; ibat < bs_ * head_num_; ibat++) {
    for (int i_n = 0; i_n < sl_n; i_n += PAD_SIZE) {
      const int ibs = ibat / head_num_;  // batch_size idx
      const int ihn = ibat % head_num_;  // head_num idx
      const int curr_pmask = has_pmask ? pmask[ibs] : sl_n;
      const int curr_pmask_pad = pad_to(curr_pmask, PAD_SIZE);
      if (i_n >= curr_pmask) continue;
      const auto bat_offset = ibs * sl_n * ld_src_ + ihn * head_size_;
      const auto bat_reo_offset = (ibs * head_num_ + ihn) * sl_n_pad * head_size_pad_;
      const int curr_nsize_ = std::min(PAD_SIZE, curr_pmask - i_n);
      {  // Reorder K
        const auto curr_src_k = src_k + bat_offset + i_n * ld_src_;
        const auto curr_reo_k = reo_k + bat_reo_offset + i_n * head_size_pad_;
        const jit_trans_AB16a4b_16x::rt_data_t rtdata_reorder_k{
            /* .src = */ reinterpret_cast<const int8_t*>(curr_src_k),
            /* .dst = */ curr_reo_k,
            /*.ld_src = */ ld_src_ * static_cast<int>(sizeof(bfloat16_t)),
            /*.M = */ curr_nsize_,                                        // row
            /*.N = */ head_size_ * static_cast<int>(sizeof(bfloat16_t)),  // col
        };
        kern_tr_k(&rtdata_reorder_k);
      }
      {  // Reorder V
        const auto curr_src_v = src_v + bat_offset + i_n * ld_src_;
        const auto curr_reo_v = reo_v + bat_reo_offset + i_n * PAD_SIZE;
        const jit_padding_interleave4b_n::rt_data_t rtdata_reorder_v{
            /*.srcptr = */ curr_src_v,
            /*.dstptr = */ curr_reo_v,
            /*.row = */ curr_nsize_,
            /*.col = */ head_size_,
            /*.rowpad = */ PAD_SIZE,
            /*.colpad = */ head_size_pad_,
            /*.srcstride = */ ld_src_ * static_cast<int>(sizeof(bfloat16_t)),
            /*.dststride = */ curr_pmask_pad * static_cast<int>(sizeof(bfloat16_t)),
        };
        kern_tr_v(&rtdata_reorder_v);
      }

      // TODO(Yi): implement real pmask to eliminate badd copying
      if (i_n == 0 && ihn == 0 && (has_pmask || sl_n_pad != sl_n)) {
        for (int ii = 0; ii < sl_m; ++ii) {
          const auto curr_tmp_mask = tmp_mask + (ibs * sl_m + ii) * sl_n_pad;
          if (has_badd) {
            const auto curr_badd = badd + ii * sl_n;
            memcpy(curr_tmp_mask, curr_badd, curr_pmask * sizeof(float));
          } else {
            memset(curr_tmp_mask, 0, curr_pmask * sizeof(float));
          }
          std::fill(curr_tmp_mask + curr_pmask, curr_tmp_mask + sl_n_pad, -INFINITY);
        }
      }
    }
  }

// QxK & softmax & AxV
#pragma omp parallel for collapse(2)
  for (int ibat = 0; ibat < bs_ * head_num_; ibat++) {
    for (int i_m = 0; i_m < sl_m; i_m += m_tile) {
      const int thread_num = omp_get_thread_num();
      const auto curr_reo_q = reinterpret_cast<bfloat16_t*>(tmp_thread + thread_num * thread_total_bytes_);
      const auto curr_softmax = reinterpret_cast<bfloat16_t*>(curr_reo_q + thread_reo_q_size_);
      assert(dt_dst == data_type::bf16);
      const auto curr_tmp_dst = reinterpret_cast<bfloat16_t*>(curr_softmax + thread_softmax_size_);

      // init amx for each omp thread
      ker_amx_cfg_(&amx_full_tile_cfg_);
      const auto ibs = ibat / head_num_;  // batch_size idx
      const auto ihn = ibat % head_num_;  // head_num idx
      const int curr_pmask = has_pmask ? pmask[ibs] : sl_n;
      const int curr_pmask_pad = pad_to(curr_pmask, PAD_SIZE);
      auto ioffset = ibs * sl_m * ld_src_ + ihn * head_size_;  // offset for Q & DST

      auto curr_q = src_q + ioffset + i_m * ld_src_;
      const jit_padding_copy2d::rt_data_t rtdata_reoder_q{
          /*.srcptr = */ curr_q,
          /*.dstptr = */ curr_reo_q,
          /*.row = */ std::min(m_tile, static_cast<int>(sl_m - i_m)),
          /*.col = */ head_size_ * static_cast<int>(sizeof(bfloat16_t)),
          /*.rowpad = */ m_tile,
          /*.colpad = */ head_size_pad_ * static_cast<int>(sizeof(bfloat16_t)),
          /*.srcstride = */ ld_src_ * static_cast<int>(sizeof(bfloat16_t)),
          /*.dststride = */ head_size_pad_ * static_cast<int>(sizeof(bfloat16_t)),
      };
      kern_tr_q(&rtdata_reoder_q);

      auto curr_k = reo_k + ibat * sl_n_pad * head_size_pad_;
      const auto&& curr_badd = badd + i_m * sl_n;
      const auto&& curr_tmp_mask = tmp_mask + (ibs * sl_m + i_m) * sl_n_pad;
      const auto qksoftmax_badd = (has_pmask || sl_n_pad != sl_n) ? curr_tmp_mask : curr_badd;
      const jit_mha_bf16_row_amx_32x32_softmax::rt_data_t rtdata_qksoftmax{
          /*.matA = */ curr_reo_q,
          /*.matB = */ curr_k,
          /*.matC = */ curr_softmax,
          /*.matD = */ qksoftmax_badd,
          /*.m = */ m_tile,
          /*.n = */ curr_pmask_pad,
          /*.k = */ head_size_pad_,
          /*.astep = */ head_size_pad_ * static_cast<int>(sizeof(bfloat16_t)),
          /*.dstep = */ static_cast<int>(sl_n_pad * sizeof(float)),
          /*.scaleAB = */ att_scale,
      };  // cstep is n * size(bf16)
      kern_qksoftmax(&rtdata_qksoftmax);

      const auto is_tail_m = i_m + m_tile > sl_m;
      const auto curr_dst = dst + ioffset + i_m * ld_dst_;
      const auto mmav_dst = (!is_tail_m) ? curr_dst : curr_tmp_dst;
      auto curr_v = reo_v + ibat * sl_n_pad * head_size_pad_;
      const jit_mha_bf16_row_amx_32x32::rt_data_t rtdata_mmav{
          /*.matA = */ curr_softmax,
          /*.matB = */ curr_v,
          /*.matC = */ mmav_dst,
          /*.m = */ m_tile,
          /*.n = */ head_size_,
          /*.k = */ curr_pmask_pad,
          /*.astep = */ curr_pmask_pad * static_cast<int>(sizeof(bfloat16_t)),
          /*.cstep = */ (is_tail_m ? head_size_pad_ : ld_dst_) * static_cast<int>(sizeof(bfloat16_t)),
          /*.alpha = */ 1.f,  // TODO(Yi): not implemented
      };
      kern_mmav(&rtdata_mmav);
      if (is_tail_m) {
        for (int i = 0; i < sl_m - i_m; ++i)
          memcpy(curr_dst + i * ld_dst_, mmav_dst + i * head_size_pad_, head_size_ * sizeof(bfloat16_t));
      }
    }
  }
  return true;
}

bool mha_dense_bf16_k_t::execute(const std::vector<const void*>& rt_data) const {
  return execute(*get_mha_dense_ctx(rt_data));
}
}  // namespace jd
