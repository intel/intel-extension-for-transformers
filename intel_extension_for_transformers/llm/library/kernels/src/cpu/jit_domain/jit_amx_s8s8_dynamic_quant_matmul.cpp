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

#include "jit_amx_s8s8_dynamic_quant_matmul.hpp"

#include "regs_pool.hpp"

namespace jd {

#define GET_OFF(field) offsetof(ssd::dynamic_quant_matmul_data_t, field)

enum GEMM_ISA { amx, vnni };

#define N_DIM_IP(label_name, ip_func)      \
  xor_(reg_n_loop, reg_n_loop);            \
  if (param_.align_n_loop > 0) {           \
    L(label_prefix + label_name);          \
    ip_func(param_.align_build_block_num); \
    inc(reg_n_loop);                       \
    cmp(reg_n_loop, param_.align_n_loop);  \
    jl(label_prefix + label_name);         \
  }                                        \
  if (param_.tail_n_loop != 0) ip_func(param_.tail_n_loop);

void jit_amx_s8s8_dynamic_quant_matmul_t::generate() {
  float const_val[] = {1.f / 127.f, bit_cast<float>(0x80808080)};
  Xbyak::Label data_label;
  inLocalLabel();
  {
    const int stack_scale_offset = sizeof(const_val) + 2 * sizeof(tileconfig_t);
    auto trans_block_col = param_.k / param_.tile_k;
    const auto does_calc = param_.align_m_loop > 0 || param_.tail_m != 0;
    regs_pool rp(this, 1, {does_calc ? 12 : 6, does_calc ? 32 : 0, 0}, BYTES_ZMM, regs_pool::IgnoreWaste);
    const auto reg_m_loop = rp.reg<Reg64>();
    const auto reg_n_loop = rp.reg<Reg64>();
    const auto reg_strideA = rp.reg<Reg64>();
    const auto reg_strideB = rp.reg<Reg64>();
    const auto reg_strideC = rp.reg<Reg64>();

    auto prepare_mask = [&]() {
      const auto reg_tmp = rp.reg<Xbyak::Reg32>();
      mov(reg_tmp, 0xffff >> param_.write_mask);
      kmovd(matC_n_mask, reg_tmp);
      mov(reg_tmp, 0xffff >> (16 - param_.tail_m));
      kmovd(scaleC_mask, reg_tmp);
    };

    auto amx_ip_16x16 = [&](int block_num) {
      const auto reg_tmp = rp.reg<Reg64>();
      // build block
      {
        const auto reg_matA_addr = rp.reg<Reg64>();
        const auto reg_matB_addr = rp.reg<Reg64>();
        for (int i = 0; i < block_num; i++) tilezero(Tmm(i));
        // prepare addr & stride;
        mov(reg_matA_addr, ptr[rp.p[0] + GET_OFF(activation)]);
        mov(reg_matB_addr, ptr[rp.p[0] + GET_OFF(reordered_weight)]);
        imul(reg_tmp, reg_m_loop, 16 * param_.k);
        add(reg_matA_addr, reg_tmp);
        imul(reg_tmp, reg_n_loop, param_.align_build_block_num * trans_block_col * 64 * (param_.tile_k / 4));
        add(reg_matB_addr, reg_tmp);
        for (int k_loop = 0; k_loop < param_.k / param_.tile_k; k_loop++) {
          tileloadd(Tmm(3), ptr[reg_matA_addr + reg_strideA + k_loop * param_.tile_k]);
          for (int idx = 0; idx < block_num; idx++) {
            int offset = idx * trans_block_col * 64 * (param_.tile_k / 4) + k_loop * 64;
            tileloadd(Tmm(4 + idx), ptr[reg_matB_addr + reg_strideB + offset]);
            tdpbssd(Tmm(idx), Tmm(3), Tmm(4 + idx));
          }
        }
      }
      // store block to tmp_buf & dequant+add_bias
      {
        // store the block to tmp_buf
        imul(reg_tmp, reg_n_loop, 16 * param_.align_build_block_num * sizeof(int));
        const auto reg_tmp_buf = rp.reg<Reg64>();
        mov(reg_tmp_buf, ptr[rp.p[0] + GET_OFF(tmp_buf)]);
        add(reg_tmp_buf, reg_tmp);
        for (int idx = 0; idx < block_num; idx++)
          tilestored(ptr[reg_tmp_buf + reg_strideC + idx * 16 * sizeof(int)], Tmm(idx));
        // dequant + add_bias
        const auto zmms = rp.regs<Zmm, 4>();
        const auto reg_tmp2 = rp.reg<Reg64>();
        const auto reg_scale_w = rp.reg<Reg64>();
        const auto reg_scale_a = rp.reg<Reg64>();
        const auto reg_bias = rp.reg<Reg64>();
        mov(reg_scale_w, ptr[rp.p[0] + GET_OFF(scale_w)]);
        mov(reg_scale_a, ptr[rp.p[0] + GET_OFF(scale_a)]);
        mov(reg_bias, ptr[rp.p[0] + GET_OFF(bias)]);
        mov(reg_tmp_buf, ptr[rp.p[0] + GET_OFF(tmp_buf)]);

        imul(reg_tmp2, reg_m_loop, 16 * sizeof(float));  // offset of scale_a
        for (int idx = 0; idx < block_num; idx++) {
          vmovups(zmms[0], ptr[reg_scale_w + reg_tmp + idx * 16 * sizeof(float)]);
          if (param_.add_bias) vmovups(zmms[1], ptr[reg_bias + reg_tmp + idx * 16 * sizeof(float)]);
          for (int row_loop = 0; row_loop < 16; row_loop++) {
            vcvtdq2ps(zmms[2], ptr[reg_tmp_buf + reg_tmp + (idx * 16 + row_loop * param_.pad_n) * sizeof(float)]);
            vbroadcastss(zmms[3], dword[reg_scale_a + reg_tmp2 + row_loop * sizeof(float)]);
            vmulps(zmms[2], zmms[2], zmms[3]);
            if (param_.add_bias)
              vfmadd213ps(zmms[2], zmms[0], zmms[1]);
            else
              vmulps(zmms[2], zmms[2], zmms[0]);
            if (param_.postop_attrs.size() != 0) {
              eltwise_injector_.escape_rp_all_type(&rp);
              eltwise_injector_.vector_compute(zmms[2], param_.postop_attrs);
            }
            vmovups(ptr[reg_tmp_buf + reg_tmp + (idx * 16 + row_loop * param_.pad_n) * sizeof(float)], zmms[2]);
          }
        }
      }
    };

    auto vnni_ip_1xN = [&](int N) {
      auto neg_128_zmm = rp.reg<Zmm>();
      vbroadcastss(neg_128_zmm, ptr[rip + data_label + 4]);
      auto zmms = rp.regs<Zmm>(2 * N + 1);
      const int trans_block_col = param_.k / param_.tile_k;
      Reg64 vnni_m_loop(rp.get_idx_by_name("vnni_m_loop"));
      // vnni_gemm
      {
        const auto reg_matA_addr = rp.reg<Reg64>();
        const auto reg_matB_addr = rp.reg<Reg64>();
        const auto reg_wei_offset = rp.reg<Reg64>();
        const auto reg_activation_offset = rp.reg<Reg64>();
        const auto reg_tmp = rp.reg<Reg64>();
        mov(reg_matA_addr, ptr[rp.p[0] + GET_OFF(activation)]);
        mov(reg_matB_addr, ptr[rp.p[0] + GET_OFF(reordered_weight)]);
        for (int i = 0; i < 2 * N; i++) vxorps(zmms[1 + i], zmms[1 + i], zmms[1 + i]);
        imul(reg_tmp, reg_m_loop, 16 * param_.k);
        imul(reg_activation_offset, vnni_m_loop, param_.k);
        add(reg_activation_offset, reg_tmp);
        for (int k_loop = 0; k_loop < param_.k; k_loop += param_.tile_k) {
          imul(reg_wei_offset, reg_n_loop, param_.align_build_block_num * 16 * param_.tile_k * trans_block_col);
          auto wei_base_offset = k_loop / param_.tile_k * 64;
          for (int tile_k_loop = 0; tile_k_loop < param_.tile_k; tile_k_loop += 4) {
            auto activation_offset = k_loop + tile_k_loop;
            vbroadcastss(zmms[0], dword[reg_matA_addr + reg_activation_offset + activation_offset]);
            vpaddb(zmms[0], zmms[0], neg_128_zmm);
            for (int n_loop = 0; n_loop < N; n_loop++) {
              auto fin_wei_offset = wei_base_offset + (tile_k_loop + n_loop * param_.tile_k) * 16 * trans_block_col;
              vpdpbusds(zmms[1 + n_loop], zmms[0], ptr[reg_matB_addr + reg_wei_offset + fin_wei_offset]);
              vpdpbusds(zmms[1 + N + n_loop], neg_128_zmm, ptr[reg_matB_addr + reg_wei_offset + fin_wei_offset]);
            }
          }
        }
      }
      // store to tmpbuf
      {
        const auto reg_scale_w = rp.reg<Reg64>();
        const auto reg_scale_a = rp.reg<Reg64>();
        const auto reg_tmp = rp.reg<Reg64>();
        const auto reg_tmp2 = rp.reg<Reg64>();
        const auto reg_bias = rp.reg<Reg64>();
        const auto reg_tmp_buf = rp.reg<Reg64>();
        mov(reg_scale_w, ptr[rp.p[0] + GET_OFF(scale_w)]);
        mov(reg_scale_a, ptr[rp.p[0] + GET_OFF(scale_a)]);
        mov(reg_bias, ptr[rp.p[0] + GET_OFF(bias)]);
        mov(reg_tmp_buf, ptr[rp.p[0] + GET_OFF(tmp_buf)]);
        for (int store_loop = 0; store_loop < N; store_loop++) {
          vpsubd(zmms[1 + store_loop], zmms[1 + store_loop], zmms[1 + N + store_loop]);
          vcvtdq2ps(zmms[1 + store_loop], zmms[1 + store_loop]);
          imul(reg_tmp, reg_m_loop, 16);
          add(reg_tmp, vnni_m_loop);
          vmulps(zmms[1 + store_loop], zmms[1 + store_loop], zword_b[reg_scale_a + sizeof(float) * reg_tmp]);
          auto zmm_scale_w = rp.reg<Zmm>();
          imul(reg_tmp, reg_n_loop, param_.align_build_block_num * 16);
          vmovups(zmm_scale_w, ptr[reg_scale_w + sizeof(float) * reg_tmp + store_loop * 16 * sizeof(float)]);
          if (param_.add_bias) {
            vfmadd213ps(zmms[1 + store_loop], zmm_scale_w,
                        ptr[reg_bias + sizeof(float) * reg_tmp + store_loop * 16 * sizeof(float)]);
          } else {
            vmulps(zmms[1 + store_loop], zmms[1 + store_loop], zmm_scale_w);
          }
          if (param_.postop_attrs.size() != 0) {
            eltwise_injector_.escape_rp_all_type(&rp);
            eltwise_injector_.vector_compute(zmms[1 + store_loop], param_.postop_attrs);
          }
          imul(reg_tmp2, vnni_m_loop, param_.n);
          add(reg_tmp2, reg_tmp);
          vmovups(ptr[reg_tmp_buf + sizeof(float) * reg_tmp2 + store_loop * 16 * sizeof(float)], zmms[1 + store_loop]);
        }
      }
    };

    auto tmp_buf_load_M_row = [&](const int M, const Reg64& offset) {
      const auto reg_tmp_buf = rp.reg<Reg64>();
      mov(reg_tmp_buf, ptr[rp.p[0] + GET_OFF(tmp_buf)]);
      for (int i = 0; i < M; i++) vmovups(Zmm(i), ptr[reg_tmp_buf + offset + (i * param_.pad_n) * sizeof(int)]);
    };

    auto get_16_abs_max_zmm = [&](const std::array<Zmm, 16>& zmms, const Reg64& reg_max_abs_loop,
                                  const bool need_mask = false) {
      const auto reg_tmp = rp.reg<Reg64>();
      const auto reg_tmp_buf = rp.reg<Reg64>();
      mov(reg_tmp_buf, ptr[rp.p[0] + GET_OFF(tmp_buf)]);
      imul(reg_tmp, reg_max_abs_loop, 16 * sizeof(int));
      for (int i = 0; i < 16; i++)
        vrangeps(need_mask ? zmms[i] | matC_n_mask : zmms[i], zmms[i],
                 ptr[reg_tmp_buf + reg_tmp + i * param_.pad_n * sizeof(int)], 11U);
    };

    auto log2n_max_reduce_16x16 = [&](const std::array<Zmm, 16>& zmms) {
      int i = 8;
      while (i != 0) {
        for (int ii = 0; ii < i; ii++) vmaxps(zmms[ii], zmms[ii], zmms[ii + i]);
        i /= 2;
      }
    };

    auto write_back_scale = [&](const Zmm& scale, const int M) {
      const auto reg_tmp = rp.reg<Reg64>();
      const auto reg_scale_dst = rp.reg<Reg64>();
      mov(reg_scale_dst, ptr[rp.p[0] + GET_OFF(scale_dst)]);
      vmulps(scale, scale, zword_b[rip + data_label]);
      imul(reg_tmp, reg_m_loop, 16 * sizeof(float));
      vmovups(M == 16 ? ptr[reg_scale_dst + reg_tmp] : ptr[reg_scale_dst + reg_tmp] | scaleC_mask, scale);
      vrcp14ps(scale, scale);
      vmovups(ptr[rip + data_label + stack_scale_offset], scale);
    };

    auto calculate_scale = [&](int M, std::string label_prefix) {
      const auto zmms = rp.regs<Zmm, 16>();
      for (int i = 0; i < 16; i++) vxorps(zmms[i], zmms[i], zmms[i]);
      // calculate 16 row abs max in 16 zmms
      {
        const auto reg_max_abs_loop = rp.reg<Reg64>();
        xor_(reg_max_abs_loop, reg_max_abs_loop);
        if (param_.n / 16 > 0) {
          L(label_prefix + "max_abs_loop");
          get_16_abs_max_zmm(zmms, reg_max_abs_loop);
          inc(reg_max_abs_loop);
          cmp(reg_max_abs_loop, param_.n / 16);
          jl(label_prefix + "max_abs_loop");
        }
        if (param_.write_mask != 0) {
          get_16_abs_max_zmm(zmms, reg_max_abs_loop, true);
        }
      }

      // get scale
      transpose_16x16_ps(zmms, rp.regs<Zmm, 16>());
      log2n_max_reduce_16x16(zmms);
      write_back_scale(zmms[0], M);
    };

    auto quant_write_back_Mx16 = [&](const int M, const Reg64& store_n_loop, const bool need_mask = false) {
      const auto reg_tmp = rp.reg<Reg64>();
      const auto reg_tmp2 = rp.reg<Reg64>();
      const auto reg_dst = rp.reg<Reg64>();
      mov(reg_dst, ptr[rp.p[0] + GET_OFF(dst)]);
      imul(reg_tmp, store_n_loop, 16 * sizeof(float));
      tmp_buf_load_M_row(M, reg_tmp);
      imul(reg_tmp, reg_m_loop, 16 * param_.n);
      imul(reg_tmp2, store_n_loop, 16);
      add(reg_tmp, reg_tmp2);
      for (int i = 0; i < M; i++) {
        int quant_scale = i * sizeof(float) + stack_scale_offset;
        vmulps(Zmm(i), Zmm(i), zword_b[rip + data_label + quant_scale]);
        vcvtps2dq(Zmm(i), Zmm(i));
        vpmovsdb(
            need_mask ? ptr[reg_dst + reg_tmp + i * param_.n] | matC_n_mask : ptr[reg_dst + reg_tmp + i * param_.n],
            Zmm(i));
      }
    };

    auto build_MxN_tile = [&](int M, std::string label_prefix = ".") {
      auto gemm_isa = GEMM_ISA::amx;  // vnni show no perf benefit in SD, but we also want to keep the vnni-jit-path
                                      // cause it may useful in the future.
      if (gemm_isa == amx) {
        N_DIM_IP("amx_n_loop", amx_ip_16x16);
      } else {
        auto vnni_m_loop = rp.reg<Reg64>("vnni_m_loop");
        xor_(vnni_m_loop, vnni_m_loop);
        L(label_prefix + "vnni_m_loop");
        N_DIM_IP("vnni_n_loop", vnni_ip_1xN)
        inc(vnni_m_loop);
        cmp(vnni_m_loop, M);
        jl(label_prefix + "vnni_m_loop");
      }

      calculate_scale(M, label_prefix);

      const auto store_n_loop = rp.reg<Reg64>();
      xor_(store_n_loop, store_n_loop);
      if (param_.n / 16 > 0) {
        L(label_prefix + "store_n_loop");
        quant_write_back_Mx16(M, store_n_loop);
        inc(store_n_loop);
        cmp(store_n_loop, param_.n / 16);
        jl(label_prefix + "store_n_loop");
      }
      if (param_.write_mask) quant_write_back_Mx16(M, store_n_loop, true);
    };

    prepare_mask();
    xor_(reg_m_loop, reg_m_loop);
    mov(reg_strideA, param_.k);
    mov(reg_strideB, trans_block_col * 64);
    mov(reg_strideC, param_.pad_n * sizeof(int));  // strideC has same value as strideB.
    if (param_.align_m_loop > 0) {
      ldtilecfg(ptr[rip + data_label + static_cast<int>(sizeof(const_val))]);
      L("align_m_loop");
      build_MxN_tile(16);
      inc(reg_m_loop);
      cmp(reg_m_loop, param_.align_m_loop);
      jl("align_m_loop");
    }
    if (param_.tail_m != 0) {
      const int offset = sizeof(const_val) + sizeof(tileconfig_t);
      ldtilecfg(ptr[rip + data_label + offset]);
      build_MxN_tile(param_.tail_m, ".tail_");
    }
  }
  outLocalLabel();
  L(data_label);
  float scale_holdplace[16] = {0};
  db(reinterpret_cast<uint8_t*>(const_val), sizeof(const_val));
  db(reinterpret_cast<uint8_t*>(&param_.m_align_cfg), sizeof(tileconfig_t));
  db(reinterpret_cast<uint8_t*>(&param_.m_tail_cfg), sizeof(tileconfig_t));
  db(reinterpret_cast<uint8_t*>(scale_holdplace), 64);
  eltwise_injector_.prepare_table();
}

}  // namespace jd
