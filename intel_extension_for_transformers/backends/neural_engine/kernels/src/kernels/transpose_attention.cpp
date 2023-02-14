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

#include "kernels/transpose_attention.hpp"
namespace jd {

inline int updiv(int a, int b) { return (a + b - 1) / b; }

bool transpose_attention_kd_t::init() {
  auto tensor_desc = op_desc_.tensor_descs();
  assert(tensor_desc.size() == 5);
  auto matK = tensor_desc[0];
  auto matQ = tensor_desc[1];
  auto matMask = tensor_desc[2];
  auto matV = tensor_desc[3];
  auto matRet = tensor_desc[4];
  assert(matK.dtype() == data_type::s8 && matQ.dtype() == data_type::s8 && matV.dtype() == data_type::s8 &&
         matRet.dtype() == data_type::u8 && matMask.dtype() == data_type::fp32);
  return true;
}

bool transpose_attention_k_t::init() {
  // int max_threads = std::min(32, omp_get_max_threads());
  // mTmp = (uint8_t*)aligned_alloc(64, max_threads * Size2M);

  if (isa_available(amx_int8) && isa_available(avx512_core_bf16))
    amx_enable = true;
  else if (isa_available(avx512_core_vnni))
    vnni_enable = true;
  else
    SPARSE_LOG(FATAL) << "ISA not meet requirement.";
  for (auto&& ker : kernel_set)
    if (!ker->create_kernel()) return false;

  return true;
}

#pragma GCC push_options
#pragma GCC optimize("O0")
bool transpose_attention_k_t::execute(const std::vector<const void*>& rt_data) const {
  int8_t* matA = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[0]));   // K
  int8_t* matB = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[1]));   // Q
  float* matC = reinterpret_cast<float*>(const_cast<void*>(rt_data[2]));     // mask
  int8_t* matD = reinterpret_cast<int8_t*>(const_cast<void*>(rt_data[3]));   // V
  uint8_t* Ret = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[4]));  // Ret
  uint8_t* mTmp = reinterpret_cast<uint8_t*>(const_cast<void*>(rt_data[5]));
  int seq_pad = *reinterpret_cast<int*>(const_cast<void*>(rt_data[6]));
  int batch = *reinterpret_cast<int*>(const_cast<void*>(rt_data[7]));
  int head_num = *reinterpret_cast<int*>(const_cast<void*>(rt_data[8]));
  int k = *reinterpret_cast<int*>(const_cast<void*>(rt_data[9]));
  int seq_len = *reinterpret_cast<int*>(const_cast<void*>(rt_data[10]));
  float scaleQ = *reinterpret_cast<float*>(const_cast<void*>(rt_data[11]));
  float scaleK = *reinterpret_cast<float*>(const_cast<void*>(rt_data[12]));
  float scaleV = *reinterpret_cast<float*>(const_cast<void*>(rt_data[13]));
  float scaleRet = *reinterpret_cast<float*>(const_cast<void*>(rt_data[14]));
  int zeropointRet = *reinterpret_cast<int*>(const_cast<void*>(rt_data[15]));
  // batchk's config need to be check carefully.
  int batchk = 2;
  if (seq_len <= 384) {
    batchk = 4;
  }
  if (seq_len <= 192) {
    batchk = 8;
  }
  auto splititer = updiv(head_num, batchk);
  batchk = head_num / splititer;
  int m = seq_len, n = seq_len;
  int batchleft = head_num / batchk;
  int totalbatch = batchleft * batch;
  int EXPSUM_BW = sizeof(float);
  int EXP_BW = sizeof(uint16_t);
  int vnni_cpy_inc_var = 4;
  MHA_kernel* transpose_cpy = nullptr;
  MHA_kernel* vnni_cpy = nullptr;
  MHA_kernel* MHA_step1 = nullptr;
  MHA_kernel* MHA_step2 = nullptr;
  MHA_kernel* MHA_step3 = nullptr;

  if (amx_enable) {
    transpose_cpy = kernel_set[ker_idx::trans_cpy].get();
    vnni_cpy = kernel_set[ker_idx::vnni_cpy_Nx4].get();
    MHA_step1 = k == 64 ? kernel_set[mha_amx_step1_k64].get() : kernel_set[ker_idx::mha_amx_step1_k32].get();
    MHA_step2 = kernel_set[ker_idx::mha_amx_step2].get();
    MHA_step3 = seq_len % 64 == 0 ? kernel_set[ker_idx::mha_amx_step3_ktile64].get()
                                  : kernel_set[ker_idx::mha_amx_step3_ktile32].get();
  }
  if (vnni_enable) {
    transpose_cpy = kernel_set[ker_idx::trans_cpy].get();
    vnni_cpy = kernel_set[ker_idx::vnni_cpy_Nx2].get();
    MHA_step1 = kernel_set[ker_idx::mha_vnni_step1].get();
    MHA_step2 = kernel_set[ker_idx::mha_vnni_step2].get();
    MHA_step3 = kernel_set[ker_idx::mha_vnni_step3].get();
    vnni_cpy_inc_var = 2;
  }
#pragma omp parallel for collapse(1)
  for (int ibat = 0; ibat < totalbatch; ibat++) {
    int thread_idx = 0;
#ifdef _OPENMP
    thread_idx = omp_get_thread_num();
#endif
    uint16_t* expoutbuf = (uint16_t*)(mTmp + thread_idx * Size2M);
    int expout_batchsize = batchk * m * n;
    auto expsumbuf = (float*)(expoutbuf + expout_batchsize);

    int expsum_batchsize = batchk * n;
    auto out0buf = (uint8_t*)(expsumbuf + expsum_batchsize);
    auto reAbuf = (int8_t*)(out0buf);
    int reA_batchsize = batchk * k * m;
    auto reBbuf = (int8_t*)(reAbuf + reA_batchsize);
    int expout_stride = n * EXP_BW;
    auto abatchptr = matA + ibat * batchk * k * seq_pad;
    auto bbatchptr = matB + ibat * batchk * k * seq_pad;
    auto _batch = ibat / batchleft;
    auto cbatchptr = matC + _batch * m;

    for (int i = 0; i < batchk * k; i += 8) {
      ssd::transpose_copy_params p{abatchptr + i * seq_pad, reAbuf + i, seq_pad, batchk * k, m};
      (*transpose_cpy)(&p);
    }

    for (int in = 0; in < batchk * k; in += vnni_cpy_inc_var) {
      ssd::seq_vnni_copy_params p{bbatchptr + in * seq_pad, reBbuf + in * vnni_cpy->NTile, seq_pad,
                                  vnni_cpy->NTile * batchk * k, n};
      (*vnni_cpy)(&p);
    }
    float scaleAB = scaleQ * scaleK;
    for (int j = 0; j < n; j += MHA_step1->NTile) {
      // stage 0: Q*K + binary add + sum of exp(out)
      auto aptr = reAbuf;
      auto dptr = cbatchptr;
      auto bptr = reBbuf + j * batchk * k;
      auto cptr = expoutbuf + j;
      ssd::transpose_attention_step1_params p{
          aptr, bptr,   cptr,       dptr,          expsumbuf + j, (uint8_t*)&(MHA_step1->tc), m,
          k,    batchk, batchk * k, expout_stride, n * EXPSUM_BW, m * expout_stride,          scaleAB};
      (*MHA_step1)(&p);
    }

    // stage 1: exp(out)/sum of exp(out) +  (QK)*V
    auto matD_threadptr = matD + ibat * batchk * k * seq_pad;
    auto matE_threadptr = Ret + ibat * batchk * k * seq_pad;

    for (int i = 0; i < batchk; i++) {
      auto expoutptr = expoutbuf + i * m * n;
      auto expsumptr = expsumbuf + i * n;
      auto rbptr = out0buf + i * m * n;
      // reroder and norm to u8
      for (int in = 0; in < m; in += MHA_step2->NPacked)  // 0.44
      {
        ssd::transpose_attention_step2_params p{expoutptr + in * n,
                                                rbptr + in * MHA_step2->NTile,
                                                expsumptr,
                                                n * EXP_BW,
                                                MHA_step2->NTile * m * (int)sizeof(rbptr[0]),
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
          ssd::transpose_attention_step3_params p{aptr,     bptr,        cptr,    (uint8_t*)&(MHA_step3->tc),
                                                  m,        seq_pad,     seq_pad, scaleV * sotmax_scale,
                                                  scaleRet, zeropointRet};
          (*MHA_step3)(&p);
        }
      }
    }
  }

  return true;
}
#pragma GCC pop_options
}  // namespace jd
