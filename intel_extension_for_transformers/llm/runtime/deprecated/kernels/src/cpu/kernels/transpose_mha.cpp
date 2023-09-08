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

#include "transpose_mha.hpp"

#define KERNEL_INIT_CHECK(f)                                                   \
  if (!(f)) {                                                                  \
    SPARSE_LOG(ERROR) << "Transpose attention kernel requires `" << #f << "`"; \
    return false;                                                              \
  }

namespace jd {
using io = ssd::transpose_mha_io::io;

bool transpose_mha_kd_t::init() {
  KERNEL_INIT_CHECK(isa_available(avx512_core_vnni))

  auto tensor_desc = op_desc_.tensor_descs();
  auto matK = tensor_desc[io::SRC_K];
  auto matQ = tensor_desc[io::SRC_Q];
  auto matMask = tensor_desc[io::MASK];
  auto matV = tensor_desc[io::SRC_V];
  auto matRet = tensor_desc[io::DST];

  KERNEL_INIT_CHECK(matK.dtype() == data_type::s8)
  KERNEL_INIT_CHECK(matQ.dtype() == data_type::s8)
  KERNEL_INIT_CHECK(matV.dtype() == data_type::s8)
  KERNEL_INIT_CHECK(matRet.dtype() == data_type::u8)
  KERNEL_INIT_CHECK(matMask.dtype() == data_type::fp32)
  // TODO(zhe1wang): checking size
  return true;
}

bool transpose_mha_k_t::init() {
  // int max_threads = std::min(32, omp_get_max_threads());
  // mTmp = (uint8_t*)aligned_alloc(64, max_threads * Size2M);

  auto& attrs = kd_->get_operator_desc().attrs();
  if (attrs.find("impl") != attrs.end() && attrs.at("impl") != "") {
    if (attrs.at("impl") == "vnni_b")
      impl_ = impl::vnni_b;
    else if (attrs.at("impl") == "vnni_w")
      impl_ = impl::vnni_w;
    else if (attrs.at("impl") == "amx")
      impl_ = impl::amx;
    else
      SPARSE_LOG(FATAL) << "Unexpected impl specification!";
  } else {
    if (isa_available(amx_int8) && isa_available(avx512_core_bf16))
      impl_ = impl::amx;
    else if (isa_available(avx512_core_vnni))
      impl_ = impl::vnni_b;
    else
      SPARSE_LOG(FATAL) << "ISA not meet requirement.";
  }

  for (auto&& ker : kernel_set)
    if (!ker->create_kernel()) return false;

  if (impl_ == impl::vnni_b) {
    ker_seq_cpy_k_.reset(new jit_seq_cpy_2x8x8(jit_seq_cpy_2x8x8::param_t{128}));
    if (!ker_seq_cpy_k_->create_kernel()) return false;
    ker_seq_cpy_q_.reset(new jit_seq_cpy_48x4(jit_seq_cpy_48x4::param_t{true, false, INT32_MAX}));
    if (!ker_seq_cpy_q_->create_kernel()) return false;
    ker_kxq_.reset(new jit_mm_exp_vnni_mxkx48_t({7, true, data_type::bf16, 48}));
    if (!ker_kxq_->create_kernel()) return false;
    ker_scale_trans.reset(new MHA_norm_quantize_reorder_vnnib_prescale_packed(48));
    if (!ker_scale_trans->create_kernel()) return false;
    ker_vxa_.reset(new MHA_Matmul_s8u8u8_vnni_byte_8x48());
    if (!ker_vxa_->create_kernel()) return false;
  }
  return true;
}

inline bool transpose_mha_k_t::execute_vnnib(const std::vector<const void*>& rt_data) const {
  const auto src_k = reinterpret_cast<const int8_t*>(rt_data[io::SRC_K]);
  const auto src_q = reinterpret_cast<const int8_t*>(rt_data[io::SRC_Q]);
  const auto src_mask = reinterpret_cast<const float*>(rt_data[io::MASK]);
  const auto src_v = reinterpret_cast<const int8_t*>(rt_data[io::SRC_V]);
  const auto dst = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[io::DST]));
  const auto mTmp = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[io::TMP2M]));
  const auto seq_pad = *reinterpret_cast<const int*>(rt_data[io::SL_PAD]);
  const auto batch_size = *reinterpret_cast<const int*>(rt_data[io::BATCH]);
  const auto head_num = *reinterpret_cast<const int*>(rt_data[io::HEAD_NUM]);
  const auto head_size = *reinterpret_cast<const int*>(rt_data[io::HEAD_SIZE]);
  const auto seq_len = *reinterpret_cast<const int*>(rt_data[io::SEQ_LEN]);
  const auto scale_q = *reinterpret_cast<const float*>(rt_data[io::SCALE_Q]);
  const auto scale_k = *reinterpret_cast<const float*>(rt_data[io::SCALE_K]);
  const auto scale_v = *reinterpret_cast<const float*>(rt_data[io::SCALE_V]);
  const auto scale_dst = *reinterpret_cast<const float*>(rt_data[io::SCALE_DST]);
  const auto zp_dst = *reinterpret_cast<const int*>(rt_data[io::ZP_DST]);
  const auto sl_pad8 = pad_to(seq_len, 8);
  const auto sl_pad48 = pad_to(seq_len, 48);

#pragma omp parallel for collapse(2)
  for (int ibs = 0; ibs < batch_size; ++ibs) {
    for (int ihn = 0; ihn < head_num; ++ihn) {
      const auto src_offset = (ibs * head_num + ihn) * head_size * seq_pad;
      const auto curr_k = src_k + src_offset;
      const auto curr_q = src_q + src_offset;
      const auto curr_v = src_v + src_offset;
      const auto curr_dst = dst + src_offset;  // src & dst shape should be identical
      const auto curr_mask = src_mask + ibs * seq_pad;

      constexpr int exp_nstep = 48;
      const int thread_idx = omp_get_thread_num();
      const auto curr_tmp = mTmp + thread_idx * Size2M;
      const auto expoutbuf = reinterpret_cast<bfloat16_t*>(curr_tmp);
      const size_t expoutbuf_size = sl_pad8 * exp_nstep;
      const auto expsumbuf = reinterpret_cast<float*>(expoutbuf + expoutbuf_size);
      const size_t expsumbuf_size = exp_nstep;
      const auto scaletrbuf = reinterpret_cast<uint8_t*>(expsumbuf + expsumbuf_size);  // quantized & transposed exp
      const size_t scaletrbuf_size = seq_pad * seq_pad;
      const auto tmp_k = reinterpret_cast<uint8_t*>(scaletrbuf + scaletrbuf_size);  // dst of trans_cpy_src0
      const size_t tmp_k_size = head_size * sl_pad8;
      const auto tmp_q = reinterpret_cast<int8_t*>(tmp_k + tmp_k_size);
      const size_t tmp_q_size = head_size * sl_pad48;
      const auto tmp_q_sum = reinterpret_cast<int32_t*>(tmp_q + tmp_q_size);
      const size_t tmp_q_sum_size = seq_pad;
      if (thread_idx == 0) {
        const auto total_size = reinterpret_cast<uint8_t*>(tmp_q_sum + tmp_q_sum_size) - curr_tmp;
        SPARSE_LOG_IF(FATAL, total_size > Size2M) << "Buffer size too samll";
      }

      // reorder K (left mat) from (head_size x seqlen) to BAb8a4
      for (int ik = 0; ik < head_size; ik += 8) {
        jit_seq_cpy_2x8x8::rt_data_t reorder_k_data;
        reorder_k_data.src = curr_k + ik * seq_pad;
        reorder_k_data.dst = tmp_k + ik * 8;
        reorder_k_data.N = seq_len;
        reorder_k_data.ld_src = seq_pad;
        reorder_k_data.ld_dst = jit_seq_cpy_2x8x8::dst_step(head_size);
        (*ker_seq_cpy_k_)(&reorder_k_data);
      }

      // reorder Q (right mat) from (head_size x seqlen) to BAb48a4
      for (int ik = 0; ik < head_size; ik += 4) {
        jit_seq_cpy_48x4::rt_data_t reorder_q_data;
        reorder_q_data.src = curr_q + ik * seq_pad;
        reorder_q_data.dst = tmp_q + ik * 48;
        reorder_q_data.dst_sum = tmp_q_sum;
        reorder_q_data.sum_append = ik != 0;
        reorder_q_data.N = seq_len;
        reorder_q_data.ld_src = seq_pad;
        reorder_q_data.ld_dst = jit_seq_cpy_48x4::dst_step(head_size);
        (*ker_seq_cpy_q_)(&reorder_q_data);
      }

      // K x Q and reorder-norm
      for (dim_t j = 0; j < seq_len; j += 48) {
        // K x Q
        jit_mm_exp_vnni_mxkx48_t::rt_data_t<bfloat16_t> rt_matmul{
            tmp_k,                  // src0
            tmp_q + j * head_size,  // src1
            tmp_q_sum + j,          // bias
            curr_mask,              // src_b0
            expoutbuf,              // dst
            expsumbuf,              // dst_scale
            sl_pad8,                // M
            head_size,              // K
            48,                     // ld_dst
            scale_k * scale_q,      // scale
        };
        (*ker_kxq_)(&rt_matmul);

        // reroder and norm to u8
        ssd::transpose_mha_step2_params rt_scale_tr{
            expoutbuf,                 // src
            scaletrbuf + j * sl_pad8,  // dst
            expsumbuf,                 // sum / scale
            48 * sizeof(bfloat16_t),   // src_stride
            48 * 4,                    // dst_stride
            sl_pad8,                   // k
        };
        (*ker_scale_trans)(&rt_scale_tr);
      }
      // 2nd matmul
      const float sotmax_scale = 1 / 255.f;
      const auto scaleAB = scale_v * sotmax_scale;
      for (int i = 0; i < head_size; i += 8) {
        MHA_Matmul_s8u8u8_vnni_byte_8x48::rt_data_t rt_data{
            curr_v + i * seq_pad,    // src0
            scaletrbuf,              // src1
            curr_dst + i * seq_pad,  // dst
            seq_len,                 // N dim
            seq_len,                 // reduction dim
            seq_pad,                 // src0 step
            seq_pad,                 // dst step
            scaleAB,
            scale_dst,
            zp_dst,
        };
        (*ker_vxa_)(&rt_data);
      }
    }
  }
  return true;
}

bool transpose_mha_k_t::execute(const std::vector<const void*>& rt_data) const {
  if (impl_ == impl::vnni_b) return execute_vnnib(rt_data);
  const auto matA = reinterpret_cast<const int8_t*>(rt_data[io::SRC_K]);
  const auto matB = reinterpret_cast<const int8_t*>(rt_data[io::SRC_Q]);
  const auto matC = reinterpret_cast<const float*>(rt_data[io::MASK]);
  const auto matD = reinterpret_cast<const int8_t*>(rt_data[io::SRC_V]);
  const auto Ret = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[io::DST]));
  const auto mTmp = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[io::TMP2M]));
  const auto seq_pad = *reinterpret_cast<const int*>(rt_data[io::SL_PAD]);
  const auto batch = *reinterpret_cast<const int*>(rt_data[io::BATCH]);
  const auto head_num = *reinterpret_cast<const int*>(rt_data[io::HEAD_NUM]);
  const auto k = *reinterpret_cast<const int*>(rt_data[io::HEAD_SIZE]);
  const auto seq_len = *reinterpret_cast<const int*>(rt_data[io::SEQ_LEN]);
  const auto scaleQ = *reinterpret_cast<const float*>(rt_data[io::SCALE_Q]);
  const auto scaleK = *reinterpret_cast<const float*>(rt_data[io::SCALE_K]);
  const auto scaleV = *reinterpret_cast<const float*>(rt_data[io::SCALE_V]);
  const auto scaleRet = *reinterpret_cast<const float*>(rt_data[io::SCALE_DST]);
  const auto zeropointRet = *reinterpret_cast<const int*>(rt_data[io::ZP_DST]);
  // Number of heads in a batch; batchk's config need to be check carefully.
  int batchk = 2;
  if (seq_len <= 384) {
    batchk = 4;
  }
  if (seq_len <= 192) {
    batchk = 8;
  }
  auto splititer = ceil_div(head_num, batchk);
  batchk = head_num / splititer;
  int m = seq_len, n = seq_len;
  int batchleft = head_num / batchk;   // number of "barchk" in a sample
  int totalbatch = batchleft * batch;  // number of "barchk" in a batch
  int EXPSUM_BW = sizeof(float);
  int vnni_cpy_inc_var = 4;

  // get amx kernel.
  MHA_kernel* transpose_cpy = kernel_set[ker_idx::trans_cpy].get();
  MHA_kernel* vnni_cpy = kernel_set[ker_idx::vnni_cpy_Nx4].get();
  MHA_kernel* MHA_step1 = k == 64 ? kernel_set[mha_amx_step1_k64].get() : kernel_set[ker_idx::mha_amx_step1_k32].get();
  MHA_kernel* MHA_step2 = kernel_set[ker_idx::mha_amx_step2].get();
  MHA_kernel* MHA_step3 = seq_len % 64 == 0 ? kernel_set[ker_idx::mha_amx_step3_ktile64].get()
                                            : kernel_set[ker_idx::mha_amx_step3_ktile32].get();
  if (impl_ == impl::vnni_w) {
    transpose_cpy = kernel_set[ker_idx::trans_cpy].get();
    vnni_cpy = kernel_set[ker_idx::vnni_cpy_Nx2].get();
    MHA_step1 = kernel_set[ker_idx::mha_vnni_step1].get();
    MHA_step2 = kernel_set[ker_idx::mha_vnni_step2].get();
    MHA_step3 = kernel_set[ker_idx::mha_vnni_step3].get();
    vnni_cpy_inc_var = 2;
  }

#pragma omp parallel for collapse(1)
  for (int ibat = 0; ibat < totalbatch; ibat++) {
#ifdef _OPENMP
    int thread_idx = omp_get_thread_num();
#else
    int thread_idx = 0;
#endif
    const auto expoutbuf = reinterpret_cast<bfloat16_t*>(mTmp + thread_idx * Size2M);
    const size_t expoutbuf_size = batchk * m * n;

    const auto expsumbuf = reinterpret_cast<float*>(expoutbuf + expoutbuf_size);
    const size_t expsumbuf_size = batchk * n;

    const auto out0buf = reinterpret_cast<uint8_t*>(expsumbuf + expsumbuf_size);  // buf for normalized & quantized exp
    const auto reAbuf = reinterpret_cast<int8_t*>(out0buf);                       // trans_copied K
    const size_t reAbuf_size = batchk * k * m;

    const auto reBbuf = reinterpret_cast<int8_t*>(reAbuf + reAbuf_size);  // trans_copied Q
    const size_t reBbuf_size = reAbuf_size;
    if (thread_idx == 0)
      SPARSE_LOG_IF(FATAL, reinterpret_cast<uint8_t*>(reBbuf + reBbuf_size) - mTmp > Size2M) << "Buffer size too samll";

    const auto abatchptr = matA + ibat * batchk * k * seq_pad;
    const auto bbatchptr = matB + ibat * batchk * k * seq_pad;
    const auto _batch = ibat / batchleft;      // sample index
    const auto cbatchptr = matC + _batch * m;  // mask tensor for current "batchk"

    // reorder K
    for (int i = 0; i < batchk * k; i += 8) {
      ssd::transpose_copy_params p{abatchptr + i * seq_pad, reAbuf + i, seq_pad, batchk * k, m};
      (*transpose_cpy)(&p);
    }

    // reorder Q
    for (int in = 0; in < batchk * k; in += vnni_cpy_inc_var) {
      ssd::seq_vnni_copy_params p{bbatchptr + in * seq_pad, reBbuf + in * vnni_cpy->NTile, seq_pad,
                                  vnni_cpy->NTile * batchk * k, n};
      (*vnni_cpy)(&p);
    }

    constexpr int EXP_BW = sizeof(bfloat16_t);
    const int expout_stride = n * EXP_BW;
    float scaleAB = scaleQ * scaleK;
    for (int j = 0; j < n; j += MHA_step1->NTile) {  // j += 32?
      // stage 0: Q*K + binary add + sum of exp(out)
      auto aptr = reAbuf;                   // trans_copied K
      auto bptr = reBbuf + j * batchk * k;  // trans_copied Q
      auto cptr = expoutbuf + j;            // bf16 exp sum
      auto dptr = cbatchptr;                // mask
      ssd::transpose_mha_step1_params p{aptr,
                                        bptr,
                                        cptr,
                                        dptr,
                                        expsumbuf + j,
                                        reinterpret_cast<uint8_t*>(&(MHA_step1->tc)),
                                        m,
                                        k,
                                        batchk,
                                        batchk * k,
                                        expout_stride,
                                        n * EXPSUM_BW,
                                        m * expout_stride,
                                        scaleAB};
      (*MHA_step1)(&p);
    }

    // stage 1: exp(out)/sum of exp(out) +  (QK)*V
    auto matD_threadptr = matD + ibat * batchk * k * seq_pad;
    auto matE_threadptr = Ret + ibat * batchk * k * seq_pad;

    for (int i = 0; i < batchk; i++) {  // for each head in batchk of heads
      auto expoutptr = expoutbuf + i * m * n;
      auto expsumptr = expsumbuf + i * n;
      auto rbptr = out0buf + i * m * n;
      // reroder and norm to u8
      for (int in = 0; in < m; in += MHA_step2->NPacked) {
        ssd::transpose_mha_step2_params p{expoutptr + in * n,
                                          rbptr + in * MHA_step2->NTile,
                                          expsumptr,
                                          n * EXP_BW,
                                          MHA_step2->NTile * m * static_cast<int>(sizeof(rbptr[0])),
                                          n};
        (*MHA_step2)(&p);
      }

      // 2nd matmul
      auto dptr = matD_threadptr + i * k * seq_pad;
      auto eptr = matE_threadptr + i * k * seq_pad;
      for (int im = 0; im < k; im += MHA_step3->MTile) {
        auto aptr = dptr + im * seq_pad;
        float sotmax_scale = 1 / 255.f;
        for (int in = 0; in < n; in += MHA_step3->NTile) {
          auto bptr = rbptr + in * m;
          auto cptr = eptr + im * seq_pad + in;
          ssd::transpose_mha_step3_params p{
              aptr,     bptr,        cptr,    reinterpret_cast<uint8_t*>(&(MHA_step3->tc)),
              m,        seq_pad,     seq_pad, scaleV * sotmax_scale,
              scaleRet, zeropointRet};
          (*MHA_step3)(&p);
        }
      }
    }
  }

  return true;
}
}  // namespace jd
