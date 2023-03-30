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

#include "kernels/mha_dense.hpp"

#ifdef WITH_GCC_FLAGS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
#pragma GCC diagnostic ignored "-Wuninitialized"
#include <immintrin.h>
#pragma GCC diagnostic pop
#else
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>

#define KERNEL_INIT_CHECK(f)                                         \
  if (!(f)) {                                                        \
    SPARSE_LOG(ERROR) << "MHA dense kernel requires `" << #f << "`"; \
    return false;                                                    \
  }
namespace jd {

bool mha_dense_kd_t::init() {
  if (!isa_available(avx512_core_amx)) return false;
  auto op_attrs = op_desc_.attrs();
  param_.QK_rescale_ = str_to_num<float>(op_attrs["QK_rescale"]);
  param_.softmax_rescale_ = str_to_num<float>(op_attrs["softmax_rescale"]);
  param_.QKV_rescale_ = str_to_num<float>(op_attrs["QKV_rescale"]);
  param_.QKV_dstzp_ = str_to_num<float>(op_attrs["QKV_dstzp"]);
  param_.Q_scale_ = str_to_num<float>(op_attrs["Q_scale"]);
  param_.K_scale_ = str_to_num<float>(op_attrs["K_scale"]);
  param_.V_scale_ = str_to_num<float>(op_attrs["V_scale"]);
  param_.DST_scale_ = str_to_num<float>(op_attrs["DST_scale"]);
  param_.QK_output_scale_ = str_to_num<float>(op_attrs["QK_output_scale"]);
  if (op_attrs.find("merged_QKV") != op_attrs.end() && op_attrs["merged_QKV"] == "True")
    param_.base_ = 3;
  else
    param_.base_ = 1;

  auto& tensor_desc = op_desc_.tensor_descs();
  auto& q_shape = tensor_desc[io::SRC_Q].shape();
  auto& k_shape = tensor_desc[io::SRC_K].shape();
  auto& dst_shape = tensor_desc[io::DST].shape();
  KERNEL_INIT_CHECK(q_shape == dst_shape)
  KERNEL_INIT_CHECK(k_shape == tensor_desc[io::SRC_V].shape())

  param_.dst_dt_ = tensor_desc[io::DST].dtype();
  param_.src_bs_ = q_shape[0];
  param_.src_sl_m_ = q_shape[1];
  param_.src_sl_n_ = k_shape[1];
  param_.head_num_ = q_shape[2];
  param_.head_size_ = q_shape[3];

  KERNEL_INIT_CHECK(param_.head_size_ == 32 || param_.head_size_ == 64 || param_.head_size_ % 64 == 0)
  KERNEL_INIT_CHECK(param_.src_sl_m_ == 1 || param_.src_sl_m_ >= 16)

  KERNEL_INIT_CHECK(is_any_of({data_type::u8, data_type::s8, data_type::fp32, data_type::bf16},
                              [dst_dt = param_.dst_dt_](auto t) { return dst_dt == t; }))

  const auto has_badd = has_binary_add();
  if (param_.src_sl_m_ == 1) {
    KERNEL_INIT_CHECK(param_.head_size_ % 64 == 0)
    KERNEL_INIT_CHECK(has_badd)
  } else {
    KERNEL_INIT_CHECK(q_shape == k_shape)
  }
  if (has_badd) {
    // only amx impl supports attention mask (binary add before softmax)
    KERNEL_INIT_CHECK(isa_available(avx512_core_amx))

    const auto& badd_shape = tensor_desc[io::BINARY_ADD].shape();
    KERNEL_INIT_CHECK(tensor_desc[io::BINARY_ADD].dtype() == data_type::fp32);
    KERNEL_INIT_CHECK(param_.src_sl_n_ == badd_shape[badd_shape.size() - 1]);
    KERNEL_INIT_CHECK(badd_shape.size() < 2 || param_.src_sl_m_ == badd_shape[badd_shape.size() - 2]);
    KERNEL_INIT_CHECK(badd_shape.size() < 3 || param_.head_num_ == badd_shape[badd_shape.size() - 3]);
    KERNEL_INIT_CHECK(badd_shape.size() < 4 || param_.src_bs_ == badd_shape[badd_shape.size() - 4]);
  }
  return true;
}

mha_dense_k_t::mha_dense_k_t(const std::shared_ptr<const kernel_desc_t>& kd)
    : kernel_t(kd),
      dst_dt_(derived_kd()->params().dst_dt_),
      src_bs_(derived_kd()->params().src_bs_),
      src_sl_m_(derived_kd()->params().src_sl_m_),
      src_sl_n_(derived_kd()->params().src_sl_n_),
      head_num_(derived_kd()->params().head_num_),
      head_size_(derived_kd()->params().head_size_),
      ld_src_(head_size_ * head_num_ * derived_kd()->params().base_),
      ld_dst_(head_size_ * head_num_ * get_data_size(dst_dt_)),
      softmax_rescale_f32_(derived_kd()->params().softmax_rescale_),
      softmax_rescale_(fp32_to_fp16(softmax_rescale_f32_)),
      QK_rescale_(derived_kd()->params().QK_rescale_),
      QKV_rescale_(derived_kd()->params().QKV_rescale_),
      QKV_dstzp_(derived_kd()->params().QKV_dstzp_),
      Q_scale_(derived_kd()->params().Q_scale_),
      K_scale_(derived_kd()->params().K_scale_),
      V_scale_(derived_kd()->params().V_scale_),
      DST_scale_(derived_kd()->params().DST_scale_),
      QK_output_scale_(derived_kd()->params().QK_output_scale_),
      ts_descs_(derived_kd()->get_operator_desc().tensor_descs()),
      has_binary_add(derived_kd()->has_binary_add()),
      thread_workspace_size_(sizeof(int8_t) * head_size_ * pad_to(src_sl_n_, 16) +  // k
                             sizeof(int8_t) * head_size_ * pad_to(src_sl_n_, 64) +  // v
                             sizeof(int32_t) * 32 * pad_to(src_sl_n_, 16) +         // qk
                             sizeof(uint8_t) * 32 * pad_to(src_sl_n_, 64)),         // softmax
      amx_full_tile_param_(16, 16, 64, false, 4),
      amx_full_tile_cfg_(amx_full_tile_param_) {
#define SET_NULL(arr) std::fill_n(reinterpret_cast<void**>(arr), sizeof(arr) / sizeof(nullptr), nullptr)
  SET_NULL(ker_trans_k_);
  SET_NULL(ker_trans_v_);
  SET_NULL(ker_softmax_);
  SET_NULL(ker_qk_gemm_16x_);
  SET_NULL(ker_qk_gemm_32x_);
  SET_NULL(ker_av_gemm_16x_);
  SET_NULL(ker_av_gemm_32x_);
#undef SET_NULL
}

bool mha_dense_k_t::init() {
  if (src_sl_m_ == 1) return true;  // TODO(Yi): Better checking for execute_tiny branch
  if (!ker_amx_cfg_.create_kernel()) return false;
  if (!ker_amx_rls_.create_kernel()) return false;

  const auto hs_max64 = std::min(head_size_, 64);

  // init ker_trans_k
  for (int i = 1; i <= 16; ++i) {
    ker_trans_k_[i] = new jit_trans_AB16a4b(
        {/*.M = */ i, /*.N = */ hs_max64, /*.ld_src = */ ld_src_, /*.pad_n = */ pad_to(hs_max64, 16)});
    if (!ker_trans_k_[i]->create_kernel()) return false;
  }

  // init ker_qk_gemm
  for (int i = 16; i <= MAX_SL_N; i += 16) {
    ker_qk_gemm_32x_[i / 16] =
        new jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b(jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::param_t{
            32,
            head_size_,
            i,
            ld_src_,
            &amx_full_tile_param_,
        });
    if (!ker_qk_gemm_32x_[i / 16]->create_kernel()) return false;
    ker_qk_gemm_16x_[i / 16] =
        new jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b(jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::param_t{
            16,
            head_size_,
            i,
            ld_src_,
            &amx_full_tile_param_,
        });
    if (!ker_qk_gemm_16x_[i / 16]->create_kernel()) return false;
  }

  // init softmax
  for (int i = 64; i <= MAX_SL_N; i += 64) {
    for (int j = 0; j < 16; j++) {
      ker_softmax_[i / 64][j] = new jit_softmax_Ab16a({j, i, has_binary_add, "u8"});
      if (!ker_softmax_[i / 64][j]->create_kernel()) return false;
    }
  }

  // init ker_trans_v
  for (int i = 0; i <= 4; i++) {
    ker_trans_v_[i] = new jit_trans_BA16b4a({/*.M = */ i, /*.N = */ hs_max64, /*.ld_src = */ ld_src_});
    if (!ker_trans_v_[i]->create_kernel()) return false;
  }

  // init ker_av_gemm
  const int ld_dst = head_size_ * head_num_;
  for (int i = 64; i <= MAX_SL_N; i += 64) {
    ker_av_gemm_32x_[i / 64] = new jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab(
        {/*.M = */ 32, /*.K_pad = */ i, /*.N = */ head_size_, /*.ld_dst = */ ld_dst, /*.dst_dt = */ dst_dt_});
    if (!ker_av_gemm_32x_[i / 64]->create_kernel()) return false;
    ker_av_gemm_16x_[i / 64] = new jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab(
        {/*.M = */ 16, /*.K_pad = */ i, /*.N = */ head_size_, /*.ld_dst = */ ld_dst, /*.dst_dt = */ dst_dt_});
    if (!ker_av_gemm_16x_[i / 64]->create_kernel()) return false;
  }
  return true;
}

inline void mha_dense_k_t::mha_per_head_32x(const jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t& rt_data_qk,
                                            const jit_softmax_Ab16a::rt_data_t& rt_data_softmax1,
                                            const jit_softmax_Ab16a::rt_data_t& rt_data_softmax2,
                                            const jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t& rt_data_av,
                                            const int att_tail, const int col_tile, const int att_tile) const {
  (*ker_qk_gemm_32x_[col_tile])(&rt_data_qk);                // matmul QxK
  (*(ker_softmax_[att_tile][att_tail]))(&rt_data_softmax1);  // softmax 1
  (*(ker_softmax_[att_tile][att_tail]))(&rt_data_softmax2);  // softmax 2
  (*ker_av_gemm_32x_[att_tile])(&rt_data_av);                // matmul AxV
}

inline void mha_dense_k_t::mha_per_head_16x(const jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t& rt_data_qk,
                                            const jit_softmax_Ab16a::rt_data_t& rt_data_softmax1,
                                            const jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t& rt_data_av,
                                            const int att_tail, const int col_tile, const int att_tile) const {
  (*ker_qk_gemm_16x_[col_tile])(&rt_data_qk);                // matmul QxK
  (*(ker_softmax_[att_tile][att_tail]))(&rt_data_softmax1);  // softmax
  (*ker_av_gemm_16x_[att_tile])(&rt_data_av);                // matmul AxV
}

bool mha_dense_k_t::execute(const std::vector<const void*>& rt_data) const {
  if (src_sl_m_ == 1) return execute_tiny(rt_data);  // TODO(Yi): Better checking for execute_tiny branch
  const int32_t bs = src_bs_;
  n_thread_t with_n_thread(bs * head_num_, true);

  const int badd_stride[] = {
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 4 ? 0 : head_num_ * src_sl_m_ * src_sl_n_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 3 ? 0 : src_sl_m_ * src_sl_n_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 2 ? 0 : src_sl_n_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 1 ? 0 : 1,
  };

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < bs; ibs++) {
    for (int ihn = 0; ihn < head_num_; ihn++) {
      const int sl_n = src_sl_n_;
      const int sl_n_pad16 = pad_to(src_sl_n_, 16);
      const int col_tile = ceil_div(src_sl_n_, 16);
      const int row_loop = ceil_div(src_sl_m_, 16) / 2;
      const bool is_even = ceil_div(src_sl_m_, 16) % 2 == 0;
      const int rollback = (src_sl_m_ % 16 != 0) ? 16 - (src_sl_m_ % 16) : 0;
      const int sl_n_pad64 = pad_to(src_sl_n_, 64);
      const int att_tile = sl_n_pad64 / 64;

      const auto thread_id = omp_get_thread_num();
      const auto thread_workspace = reinterpret_cast<char*>(const_cast<void*>(rt_data[mha_dense_io::WORKSPACE])) +
                                    thread_id * thread_workspace_size_;

      const auto k_scrach = reinterpret_cast<int8_t*>(thread_workspace);
      const auto v_scrach_p64 = reinterpret_cast<int8_t*>(k_scrach + head_size_ * sl_n_pad16);
      const auto qk_scrach = reinterpret_cast<int32_t*>(v_scrach_p64 + head_size_ * sl_n_pad64);
      const auto softmax_scrach_p64 = reinterpret_cast<uint8_t*>(qk_scrach + 32 * sl_n_pad16);
      // softmax_scrach_p64_size = 32 * sl_n_pad64

      const int src_q_offset = ibs * src_sl_m_ * ld_src_ + ihn * head_size_;
      const int src_kv_offset = ibs * src_sl_n_ * ld_src_ + ihn * head_size_;
      const int dst_offset = ibs * src_sl_m_ * ld_dst_ + ihn * head_size_ * get_data_size(dst_dt_);
      // init amx for each omp thread
      ker_amx_cfg_(&amx_full_tile_cfg_);

      const auto curr_q = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_Q]) + src_q_offset;
      const auto curr_dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[mha_dense_io::DST])) + dst_offset;
      const auto curr_k = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_K]) + src_kv_offset;
      const auto curr_v = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_V]) + src_kv_offset;

      const int badd_offset = ibs * badd_stride[0] + ihn * badd_stride[1];
      const auto badd_f32 =
          has_binary_add ? reinterpret_cast<const float*>(rt_data[io::BINARY_ADD]) + badd_offset : nullptr;

      // reorder K
      for (int i = 0; i < sl_n; i += 16)
        for (int j = 0; j < head_size_; j += 64) {
          jit_trans_AB16a4b::rt_data_t rt_data_tr_k{
              /*.src = */ curr_k + i * ld_src_ + j,
              /*.dst = */ k_scrach + i * head_size_ + j * 16,
          };
          (*ker_trans_k_[std::min(16, sl_n - i)])(&rt_data_tr_k);
        }

      // reorder V
      const auto tr_v_dst_stride = jit_trans_BA16b4a::dst_stride(sl_n_pad64);
      for (int j = 0; j < head_size_; j += 64)
        for (int i = 0; i < sl_n_pad64; i += 4) {
          jit_trans_BA16b4a::rt_data_t rt_data_tr_v{
              /*.src = */ curr_v + i * ld_src_ + j,
              /*.dst = */ v_scrach_p64 + i * 16 + j * sl_n_pad64,
              /*.ld_dst = */ tr_v_dst_stride,
          };
          (*ker_trans_v_[std::max(0, std::min(4, sl_n - i))])(&rt_data_tr_v);
        }

      const auto padding_mask = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs];
      jit_matmul_amx_s8ab_s8Ab4a_s32AB16a16b::rt_data_t rt_data_qk{
          /*.src0 = */ nullptr,
          /*.src1 = */ k_scrach,
          /*.dst = */ qk_scrach,
      };
      jit_softmax_Ab16a::rt_data_t rt_data_softmax1{
          /*.src = */ qk_scrach,
          /*.dst = */ softmax_scrach_p64,
          /*.att_tile = */ padding_mask / 16,
          /*.softmax_rescale = */ softmax_rescale_,
          /*.src_badd = */ nullptr,
          /*.ld_badd = */ badd_stride[2],
          /*.QK_rescale = */ QK_rescale_,
      };
      jit_softmax_Ab16a::rt_data_t rt_data_softmax2{
          /*.src = */ qk_scrach + sl_n_pad16 * 16,           // sl_pad_ / 16 * 16 * 16
          /*.dst = */ softmax_scrach_p64 + sl_n_pad64 * 16,  // sl_pad64_ / 64 * 16 * 64
          /*.att_tile = */ padding_mask / 16,
          /*.softmax_rescale = */ softmax_rescale_,
          /*.src_badd = */ nullptr,
          /*.ld_badd = */ badd_stride[2],
          /*.QK_rescale = */ QK_rescale_,
      };
      jit_matmul_amx_u8AB16a64b_s8BA16b4a_ab::rt_data_t rt_data_av{
          /*.src0 = */ softmax_scrach_p64,
          /*.src1 = */ v_scrach_p64,
          /*.dst = */ nullptr,
          /*.K = */ padding_mask,
          /*.rescale = */ QKV_rescale_,
          /*.zp = */ QKV_dstzp_,
      };
      const int att_tail = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs] % 16;
      int cur_r_pos = 0;

      for (int j = 0; j < row_loop - 1; j++, cur_r_pos += 32) {
        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
        if (has_binary_add) rt_data_softmax2.src_badd = badd_f32 + (cur_r_pos + 16) * badd_stride[2];
        mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);
      }

      if (is_even) {
        if (rollback == 0) {
          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
          if (has_binary_add) rt_data_softmax2.src_badd = badd_f32 + (cur_r_pos + 16) * badd_stride[2];
          mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);
        } else {
          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
          mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);

          cur_r_pos += 16 - rollback;

          rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
          rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
          if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
          mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);
        }
      } else {
        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
        if (has_binary_add) rt_data_softmax2.src_badd = badd_f32 + (cur_r_pos + 16) * badd_stride[2];
        mha_per_head_32x(rt_data_qk, rt_data_softmax1, rt_data_softmax2, rt_data_av, att_tail, col_tile, att_tile);

        cur_r_pos += 32 - rollback;

        rt_data_qk.src0 = curr_q + cur_r_pos * ld_src_;
        rt_data_av.dst = curr_dst + cur_r_pos * ld_dst_;
        if (has_binary_add) rt_data_softmax1.src_badd = badd_f32 + cur_r_pos * badd_stride[2];
        mha_per_head_16x(rt_data_qk, rt_data_softmax1, rt_data_av, att_tail, col_tile, att_tile);
      }

      // release amx for each omp thread
      ker_amx_rls_.tile_release();
    }
  }

  return true;
}

static inline __m512 snd_order_poly_exp(__m512 z, __m512 f, const float c[]) {
  const auto c0 = _mm512_set1_ps(c[0]);
  const auto c1 = _mm512_set1_ps(c[1]);
  const auto c2 = _mm512_set1_ps(c[2]);

  auto y = _mm512_fmadd_ps(_mm512_fmadd_ps(f, c0, c1), f, c2);  // auto y = (f * c0 + c1) * f + c2;
  auto exp = _mm512_scalef_ps(y, z);

  return exp;
}

static inline __m512 exp_ps_0_1(__m512 x) {
  static const float v_log2e = std::log2(std::exp(1.f));
  const auto log2e = _mm512_set1_ps(v_log2e);
  const float _c[] = {0.240226507f, 0.452920674f, 0.713483036f};

  auto x1 = _mm512_fmadd_ps(x, log2e, _mm512_set1_ps(.5f));  // auto x1 = x * log2e + _mm512_set1_ps(.5f);
  auto z = _mm512_floor_ps(x1);
  auto f = _mm512_sub_ps(x1, z);  // auto f = x1 - z;

  return snd_order_poly_exp(z, f, _c);
}

#ifdef WITH_GCC_FLAGS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
template <int tail, int off>
static inline std::array<__m512i, 16> tr_add_vnni_x64(const int8_t* a, size_t lda) {
  static_assert(tail > 0 && tail <= 16, "Unexpected tail value.");
  std::array<__m512i, 16> dst;
  __m512i tmp[16];
  for (int i = 0; i < tail; ++i) {  // TODO(Yi) : unroll trick
    dst[i] = _mm512_loadu_si512(a + i * lda);
    dst[i] = _mm512_add_epi8(dst[i], _mm512_set1_epi8(off));
  }
  // for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();  // don't care about these values...

#pragma GCC unroll(8)
  for (int i = 0; i < 8; ++i) {
    tmp[2 * i] = _mm512_unpacklo_epi32(dst[2 * i], dst[2 * i + 1]);
    tmp[2 * i + 1] = _mm512_unpackhi_epi32(dst[2 * i], dst[2 * i + 1]);
  }

#pragma GCC unroll(4)
  for (int i = 0; i < 4; ++i) {
    dst[4 * i] = _mm512_unpacklo_epi64(tmp[4 * i], tmp[4 * i + 2]);
    dst[4 * i + 1] = _mm512_unpackhi_epi64(tmp[4 * i], tmp[4 * i + 2]);
    dst[4 * i + 2] = _mm512_unpacklo_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
    dst[4 * i + 3] = _mm512_unpackhi_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
  }

#pragma GCC unroll(2)
  for (int i = 0; i < 2; ++i) {
    tmp[8 * i + 0] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0x88);
    tmp[8 * i + 1] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0x88);
    tmp[8 * i + 2] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0x88);
    tmp[8 * i + 3] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0x88);
    tmp[8 * i + 4] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0xdd);
    tmp[8 * i + 5] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0xdd);
    tmp[8 * i + 6] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0xdd);
    tmp[8 * i + 7] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0xdd);
  }

  dst[0] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0x88);
  dst[1] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0x88);
  dst[2] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0x88);
  dst[3] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0x88);
  dst[4] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0x88);
  dst[5] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0x88);
  dst[6] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0x88);
  dst[7] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0x88);
  dst[8] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0xdd);
  dst[9] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0xdd);
  dst[10] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0xdd);
  dst[11] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0xdd);
  dst[12] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0xdd);
  dst[13] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0xdd);
  dst[14] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0xdd);
  dst[15] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0xdd);
  return dst;
}
static const decltype(tr_add_vnni_x64<1, 128>)* tr_add128_vnni_x64_tbl[] = {
    tr_add_vnni_x64<1, 128>,  tr_add_vnni_x64<1, 128>,  tr_add_vnni_x64<2, 128>,  tr_add_vnni_x64<3, 128>,
    tr_add_vnni_x64<4, 128>,  tr_add_vnni_x64<5, 128>,  tr_add_vnni_x64<6, 128>,  tr_add_vnni_x64<7, 128>,
    tr_add_vnni_x64<8, 128>,  tr_add_vnni_x64<9, 128>,  tr_add_vnni_x64<10, 128>, tr_add_vnni_x64<11, 128>,
    tr_add_vnni_x64<12, 128>, tr_add_vnni_x64<13, 128>, tr_add_vnni_x64<14, 128>, tr_add_vnni_x64<15, 128>,
    tr_add_vnni_x64<16, 128>,
};

template <int tail>
static inline __m512i load_interleave_vnni(const int8_t* src, const size_t lda, const __m512i& vperm_ctl,
                                           const __m512i& vpshuf_ctl);
template <>  // TODO(Yi): better impl?
inline __m512i load_interleave_vnni<4>(const int8_t* src, const size_t lda, const __m512i& vperm_ctl,
                                       const __m512i& vpshuf_ctl) {
  __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + lda));
  __m256i ab = _mm256_mask_broadcast_i32x4(_mm256_castsi128_si256(a), 0xf0, b);
  __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + lda * 2));
  __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + lda * 3));
  __m256i cd = _mm256_mask_broadcast_i32x4(_mm256_castsi128_si256(c), 0xf0, d);
  __m512i abcd = _mm512_permutex2var_epi32(_mm512_castsi256_si512(ab), vperm_ctl, _mm512_castsi256_si512(cd));

  return _mm512_shuffle_epi8(abcd, vpshuf_ctl);
}
template <>  // TODO(Yi): better impl?
inline __m512i load_interleave_vnni<3>(const int8_t* src, const size_t lda, const __m512i& vperm_ctl,
                                       const __m512i& vpshuf_ctl) {
  __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + lda));
  __m256i ab = _mm256_mask_broadcast_i32x4(_mm256_castsi128_si256(a), 0xf0, b);
  __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + lda * 2));
  __m256i cd = _mm256_mask_broadcast_i32x4(_mm256_castsi128_si256(c), 0xf0, _mm_setzero_si128());
  __m512i abcd = _mm512_permutex2var_epi32(_mm512_castsi256_si512(ab), vperm_ctl, _mm512_castsi256_si512(cd));

  return _mm512_shuffle_epi8(abcd, vpshuf_ctl);
}
template <>  // TODO(Yi): better impl?
inline __m512i load_interleave_vnni<2>(const int8_t* src, const size_t lda, const __m512i& vperm_ctl,
                                       const __m512i& vpshuf_ctl) {
  __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + lda));
  __m256i ab = _mm256_mask_broadcast_i32x4(_mm256_castsi128_si256(a), 0xf0, b);
  __m512i abcd = _mm512_permutex2var_epi32(_mm512_castsi256_si512(ab), vperm_ctl, _mm512_setzero_si512());

  return _mm512_shuffle_epi8(abcd, vpshuf_ctl);
}
template <>  // TODO(Yi): better impl?
inline __m512i load_interleave_vnni<1>(const int8_t* src, const size_t, const __m512i& vperm_ctl,
                                       const __m512i& vpshuf_ctl) {
  __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
  __m256i ab = _mm256_mask_broadcast_i32x4(_mm256_castsi128_si256(a), 0xf0, _mm_setzero_si128());
  __m512i abcd = _mm512_permutex2var_epi32(_mm512_castsi256_si512(ab), vperm_ctl, _mm512_setzero_si512());

  return _mm512_shuffle_epi8(abcd, vpshuf_ctl);
}
static const decltype(load_interleave_vnni<1>)* load_interleave_vnni_tbl[] = {
    nullptr, load_interleave_vnni<1>, load_interleave_vnni<2>, load_interleave_vnni<3>, load_interleave_vnni<4>,
};
bool mha_dense_k_t::execute_tiny(const std::vector<const void*>& rt_data) const {
  const int32_t bs = src_bs_;
  const int badd_stride[] = {
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 4 ? 0 : head_num_ * src_sl_m_ * src_sl_n_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 3 ? 0 : src_sl_m_ * src_sl_n_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 2 ? 0 : src_sl_n_,
      !has_binary_add || ts_descs_[io::BINARY_ADD].shape().size() < 1 ? 0 : 1,
  };
  const int sl_n_pad16 = pad_to(src_sl_n_, 16);
  const int sl_n_pad64 = pad_to(src_sl_n_, 64);

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < bs; ibs++) {
    for (int ihn = 0; ihn < head_num_; ihn++) {
      const int src_q_offset = ibs * src_sl_m_ * ld_src_ + ihn * head_size_;
      const int src_kv_offset = ibs * src_sl_n_ * ld_src_ + ihn * head_size_;
      const int dst_offset = ibs * src_sl_m_ * ld_dst_ + ihn * head_size_ * get_data_size(dst_dt_);
      const int badd_offset = ibs * badd_stride[0] + ihn * badd_stride[1];

      const auto curr_q = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_Q]) + src_q_offset;
      const auto curr_k = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_K]) + src_kv_offset;
      const auto curr_v = reinterpret_cast<const int8_t*>(rt_data[mha_dense_io::SRC_V]) + src_kv_offset;
      const auto curr_dst = reinterpret_cast<char*>(const_cast<void*>(rt_data[mha_dense_io::DST])) + dst_offset;
      const auto padding_mask = reinterpret_cast<const int32_t*>(rt_data[mha_dense_io::MASK])[ibs];
      const auto pmask_floor16 = (padding_mask - 1) / 16 * 16;
      const auto pmask_tail16 = padding_mask - pmask_floor16;
      const auto pmask_floor4 = (padding_mask - 1) / 4 * 4;
      const auto pmask_tail4 = padding_mask - pmask_floor4;
      const auto badd_f32 =
          has_binary_add ? reinterpret_cast<const float*>(rt_data[io::BINARY_ADD]) + badd_offset : nullptr;

      const auto thread_id = omp_get_thread_num();
      const auto thread_workspace = reinterpret_cast<char*>(const_cast<void*>(rt_data[mha_dense_io::WORKSPACE])) +
                                    thread_id * thread_workspace_size_;
      const auto v_scrach_p64 = reinterpret_cast<int8_t*>(thread_workspace);
      const auto qk_scrach = reinterpret_cast<float*>(v_scrach_p64 + head_size_ * sl_n_pad64);
      const auto softmax_scrach_p64 = reinterpret_cast<uint8_t*>(qk_scrach + 32 * sl_n_pad16);

      constexpr int VEC = 16;
      __m512i v_128u = _mm512_set1_epi8(128);
      __mmask16 pmask_tail = (1 << pmask_tail16) - 1;

      // no m-loop, only support m == 1

      /* Q x K + deq10n + mask + get_max */
      constexpr int rn_sae = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
      auto v_scale = _mm512_set1_ps(QK_rescale_);
      auto v_max = _mm512_set1_ps(-INFINITY);
      for (int j = 0; j < pmask_floor16; j += VEC) {
        __m512i v_dst = _mm512_setzero_epi32();
        __m512i v_src0_sum128 = _mm512_setzero_epi32();
        for (int k = 0; k < head_size_; k += 4 * VEC) {
          auto v_src1 = tr_add_vnni_x64<16, 128>(curr_k + j * ld_src_ + k, ld_src_);
          v_src0_sum128 = _mm512_dpbusds_epi32(v_src0_sum128, v_128u, _mm512_loadu_si512(curr_q + k));
#pragma GCC unroll VEC
          for (int kk = 0; kk < VEC; ++kk) {
            const auto v_src0 = _mm512_set1_epi32(*reinterpret_cast<const int32_t*>(curr_q + k + kk * 4));
            v_dst = _mm512_dpbusds_epi32(v_dst, v_src1[kk], v_src0);
          }
        }
        const int32_t src0_sum128 = _mm512_reduce_add_epi32(v_src0_sum128);
        v_dst = _mm512_sub_epi32(v_dst, _mm512_set1_epi32(src0_sum128));
        auto dstf32 = _mm512_fmadd_ps(_mm512_cvt_roundepi32_ps(v_dst, rn_sae), v_scale, _mm512_loadu_ps(badd_f32 + j));
        v_max = _mm512_max_ps(v_max, dstf32);
        _mm512_store_ps(qk_scrach + j, dstf32);
      }
      {  // QxK tail
        __m512i v_dst = _mm512_setzero_epi32();
        __m512i v_src0_sum128 = _mm512_setzero_epi32();
        for (int k = 0; k < head_size_; k += 4 * VEC) {
          auto v_src1 = tr_add128_vnni_x64_tbl[pmask_tail16](curr_k + pmask_floor16 * ld_src_ + k, ld_src_);
          v_src0_sum128 = _mm512_dpbusds_epi32(v_src0_sum128, v_128u, _mm512_loadu_si512(curr_q + k));
#pragma GCC unroll VEC
          for (int kk = 0; kk < VEC; ++kk) {
            const auto v_src0 =
                _mm512_maskz_set1_epi32(pmask_tail, *reinterpret_cast<const int32_t*>(curr_q + k + kk * 4));
            v_dst = _mm512_dpbusds_epi32(v_dst, v_src1[kk], v_src0);
          }
        }
        const int32_t src0_sum128 = _mm512_reduce_add_epi32(v_src0_sum128);
        v_dst = _mm512_sub_epi32(v_dst, _mm512_set1_epi32(src0_sum128));
        auto dstf32 = _mm512_fmadd_ps(_mm512_cvt_roundepi32_ps(v_dst, rn_sae), v_scale,
                                      _mm512_maskz_loadu_ps(pmask_tail, badd_f32 + pmask_floor16));
        v_max = _mm512_mask_max_ps(v_max, pmask_tail, v_max, dstf32);
        _mm512_store_ps(qk_scrach + pmask_floor16, dstf32);
      }
      v_max = _mm512_set1_ps(_mm512_reduce_max_ps(v_max));

      /* exp */
      for (int j = 0; j < pmask_floor16; j += VEC) {
        __m512 xs = _mm512_sub_ps(_mm512_load_ps(qk_scrach + j), v_max);
        xs = exp_ps_0_1(_mm512_max_ps(xs, _mm512_set1_ps(-1000.f)));
        _mm512_store_ps(qk_scrach + j, xs);
      }
      {  // exp tail
        __m512 xs = _mm512_sub_ps(_mm512_load_ps(qk_scrach + pmask_floor16), v_max);
        xs = exp_ps_0_1(_mm512_max_ps(xs, _mm512_set1_ps(-1000.f)));
        _mm512_store_ps(qk_scrach + pmask_floor16, xs);
      }
      float exp_sum = 0.f;
#pragma omp simd
      for (int i = 0; i < padding_mask; ++i)  // should be fine?
        exp_sum += reinterpret_cast<const float*>(qk_scrach)[i];

      v_scale = _mm512_set1_ps(softmax_rescale_f32_ / exp_sum);
      for (int j = 0; j < pmask_floor16; j += VEC) {
        auto xs = _mm512_load_ps(qk_scrach + j);
        xs = _mm512_mul_ps(xs, v_scale);
        _mm512_mask_cvtepi32_storeu_epi8(softmax_scrach_p64 + j, 0xffff, _mm512_cvt_roundps_epu32(xs, rn_sae));
      }
      {
        auto xs = _mm512_maskz_load_ps(pmask_tail, qk_scrach + pmask_floor16);
        xs = _mm512_mul_ps(xs, v_scale);
        _mm512_mask_cvtepi32_storeu_epi8(softmax_scrach_p64 + pmask_floor16, 0xffff,
                                         _mm512_cvt_roundps_epu32(xs, rn_sae));
      }

      // A x V
      alignas(16) const uint8_t vpermt2d_control[16] = {0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23};
      alignas(16) const uint8_t vpshufb_control[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
      __m512i vperm_ctl = _mm512_cvtepu8_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(vpermt2d_control)));
      __m512i vpshuf_ctl = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(vpshufb_control)));
      v_scale = _mm512_set1_ps(QKV_rescale_);
      auto v_zp = _mm512_set1_ps(QKV_dstzp_);
      for (int j = 0; j < head_size_; j += VEC) {  // head_size_ must be a multiple of 16
        __m512i v_dst = _mm512_setzero_epi32();
        for (int k = 0; k < pmask_floor4; k += 4) {
          __m512i v_src0 = _mm512_set1_epi32(*reinterpret_cast<const int32_t*>(softmax_scrach_p64 + k));
          __m512i v_src1 = load_interleave_vnni<4>(curr_v + j + k * ld_src_, ld_src_, vperm_ctl, vpshuf_ctl);
          v_dst = _mm512_dpbusds_epi32(v_dst, v_src0, v_src1);
        }
        {  // tail
          __m512i v_src0 = _mm512_set1_epi32(*reinterpret_cast<const int32_t*>(softmax_scrach_p64 + pmask_floor4));
          __m512i v_src1 = load_interleave_vnni_tbl[pmask_tail4](  //
              curr_v + j + pmask_floor4 * ld_src_, ld_src_, vperm_ctl, vpshuf_ctl);
          v_dst = _mm512_dpbusds_epi32(v_dst, v_src0, v_src1);
        }
        auto xs = _mm512_fmadd_ps(_mm512_cvt_roundepi32_ps(v_dst, rn_sae), v_scale, v_zp);
        switch (dst_dt_) {
          case data_type::u8:
            _mm512_mask_cvtusepi32_storeu_epi8(
                curr_dst + j, 0xffff, _mm512_max_epi32(_mm512_cvt_roundps_epu32(xs, rn_sae), _mm512_setzero_epi32()));
            break;
          case data_type::s8:
            _mm512_mask_cvtsepi32_storeu_epi8(
                curr_dst + j, 0xffff, _mm512_max_epi32(_mm512_cvt_roundps_epu32(xs, rn_sae), _mm512_set1_epi32(-128)));
            break;
          case data_type::fp32:
            _mm512_storeu_ps(reinterpret_cast<float*>(curr_dst) + j, xs);
            break;
          case data_type::bf16:
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(reinterpret_cast<bfloat16_t*>(curr_dst) + j),
                                _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(xs), 16)));
            break;
          default:
            break;
        }
      }
    }
  }
  return true;
}
#ifdef WITH_GCC_FLAGS
#pragma GCC diagnostic pop
#endif
}  // namespace jd
