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

#include "jit_domain/jit_amx_s8s8_dynamic_quant_matmul.hpp"
namespace jd {

#define GET_OFF(field) offsetof(ssd::dynamic_quant_matmul_data_t, field)

void jit_amx_s8s8_dynamic_quant_matmul_t::generate() {
  Xbyak::Label data_label;
  inLocalLabel();
  {
    const int const_var_size = 4 + 2 * sizeof(tileconfig_t) + 64;
    const int stack_tmpbuf_offset = 4 + 2 * sizeof(tileconfig_t);
    auto trans_block_col = param_.k / param_.tile_k;
    Xbyak::util::StackFrame st(this, 1, 11, const_var_size);
    int reg_idx = 0;
    const Reg64& reg_param = st.p[0];
    const Reg64& reg_m_loop = st.t[reg_idx++];
    const Reg64& reg_n_loop = st.t[reg_idx++];
    const Reg64& reg_strideA = st.t[reg_idx++];
    const Reg64& reg_strideB = st.t[reg_idx++];
    const Reg64& reg_strideC = st.t[reg_idx++];
    auto allocate_reg = [&] {
      Reg64* reg = &const_cast<Reg64&>(st.t[reg_idx++]);
      return std::shared_ptr<Reg64>(reg, [&](...) { reg_idx--; });
    };

    auto prepare_mask = [&]() {
      auto reg_tmp = allocate_reg();
      mov(reg_tmp->cvt32(), 0xffff >> param_.write_mask);
      kmovd(matC_n_mask, reg_tmp->cvt32());
      mov(reg_tmp->cvt32(), 0xffff >> (16 - param_.tail_m));
      kmovd(scaleC_mask, reg_tmp->cvt32());
    };

    auto ip_16x16 = [&](int block_num) {
      auto reg_tmp = allocate_reg();
      // build block
      {
        auto reg_matA_addr = allocate_reg();
        auto reg_matB_addr = allocate_reg();
        for (int i = 0; i < block_num; i++) tilezero(Tmm(i));
        // prepare addr & stride;
        mov(*reg_matA_addr, ptr[reg_param + GET_OFF(activation)]);
        mov(*reg_matB_addr, ptr[reg_param + GET_OFF(reordered_weight)]);
        imul(*reg_tmp, reg_m_loop, 16 * param_.k);
        add(*reg_matA_addr, *reg_tmp);
        imul(*reg_tmp, reg_n_loop, param_.align_build_block_num * trans_block_col * 64 * (param_.tile_k / 4));
        add(*reg_matB_addr, *reg_tmp);
        for (int k_loop = 0; k_loop < param_.k / param_.tile_k; k_loop++) {
          tileloadd(Tmm(3), ptr[*reg_matA_addr + reg_strideA + k_loop * param_.tile_k]);
          for (int idx = 0; idx < block_num; idx++) {
            int offset = idx * trans_block_col * 64 * (param_.tile_k / 4) + k_loop * 64;
            tileloadd(Tmm(4 + idx), ptr[*reg_matB_addr + reg_strideB + offset]);
            tdpbssd(Tmm(idx), Tmm(3), Tmm(4 + idx));
          }
        }
      }
      // store block to tmp_buf & dequant+add_bias
      {
        // store the block to tmp_buf
        imul(*reg_tmp, reg_n_loop, 16 * param_.align_build_block_num * sizeof(int));
        auto reg_tmp_buf = allocate_reg();
        mov(*reg_tmp_buf, ptr[reg_param + GET_OFF(tmp_buf)]);
        add(*reg_tmp_buf, *reg_tmp);
        for (int idx = 0; idx < block_num; idx++)
          tilestored(ptr[*reg_tmp_buf + reg_strideC + idx * 16 * sizeof(int)], Tmm(idx));
        // dequant + add_bias
        auto zmms = regs<Zmm, 4>(0);
        auto reg_tmp2 = allocate_reg();
        auto reg_scale_w = allocate_reg();
        auto reg_scale_a = allocate_reg();
        auto reg_bias = allocate_reg();
        mov(*reg_scale_w, ptr[reg_param + GET_OFF(scale_w)]);
        mov(*reg_scale_a, ptr[reg_param + GET_OFF(scale_a)]);
        mov(*reg_bias, ptr[reg_param + GET_OFF(bias)]);
        mov(*reg_tmp_buf, ptr[reg_param + GET_OFF(tmp_buf)]);

        imul(*reg_tmp2, reg_m_loop, 16 * sizeof(float));  // offset of scale_a
        for (int idx = 0; idx < block_num; idx++) {
          vmovups(zmms[0], ptr[*reg_scale_w + *reg_tmp + idx * 16 * sizeof(float)]);
          if (param_.add_bias) vmovups(zmms[1], ptr[*reg_bias + *reg_tmp + idx * 16 * sizeof(float)]);
          for (int row_loop = 0; row_loop < 16; row_loop++) {
            vcvtdq2ps(zmms[2], ptr[*reg_tmp_buf + *reg_tmp + (idx * 16 + row_loop * param_.pad_n) * sizeof(float)]);
            vbroadcastss(zmms[3], dword[*reg_scale_a + *reg_tmp2 + row_loop * sizeof(float)]);
            vmulps(zmms[2], zmms[2], zmms[3]);
            if (param_.add_bias)
              vfmadd213ps(zmms[2], zmms[0], zmms[1]);
            else
              vmulps(zmms[2], zmms[2], zmms[0]);
            vmovups(ptr[*reg_tmp_buf + *reg_tmp + (idx * 16 + row_loop * param_.pad_n) * sizeof(float)], zmms[2]);
          }
        }
      }
    };

    auto tmp_buf_load_M_row = [&](int M, Reg64& offset) {
      auto reg_tmp_buf = allocate_reg();
      mov(*reg_tmp_buf, ptr[reg_param + GET_OFF(tmp_buf)]);
      for (int i = 0; i < M; i++) vmovups(Zmm(i), ptr[*reg_tmp_buf + offset + (i * param_.pad_n) * sizeof(int)]);
    };

    auto get_16_abs_max_zmm = [&](std::array<Zmm, 16>& zmms, Reg64& reg_max_abs_loop, bool need_mask = false) {
      auto reg_tmp = allocate_reg();
      auto reg_tmp_buf = allocate_reg();
      mov(*reg_tmp_buf, ptr[reg_param + GET_OFF(tmp_buf)]);
      imul(*reg_tmp, reg_max_abs_loop, 16 * sizeof(int));
      for (int i = 0; i < 16; i++)
        vrangeps(need_mask ? zmms[i] | matC_n_mask : zmms[i], zmms[i],
                 ptr[*reg_tmp_buf + *reg_tmp + i * param_.pad_n * sizeof(int)], 11U);
    };

    auto log2n_max_reduce_16x16 = [&](std::array<Zmm, 16>& zmms) {
      int i = 8;
      while (i != 0) {
        for (int ii = 0; ii < i; ii++) vmaxps(zmms[ii], zmms[ii], zmms[ii + i]);
        i /= 2;
      }
    };

    auto write_back_scale = [&](Zmm& scale, int M) {
      auto reg_tmp = allocate_reg();
      auto reg_scale_dst = allocate_reg();
      mov(*reg_scale_dst, ptr[reg_param + GET_OFF(scale_dst)]);
      vmulps(scale, scale, zword_b[rip + data_label]);
      imul(*reg_tmp, reg_m_loop, 16 * sizeof(float));
      vmovups(M == 16 ? ptr[*reg_scale_dst + *reg_tmp] : ptr[*reg_scale_dst + *reg_tmp] | scaleC_mask, scale);
      vrcp14ps(scale, scale);
      vmovups(ptr[rip + data_label + stack_tmpbuf_offset], scale);
    };

    auto calculate_scale = [&](int M, std::string label_prefix) {
      auto zmms = regs<Zmm, 16>(0);
      for (int i = 0; i < 16; i++) vxorps(zmms[i], zmms[i], zmms[i]);
      // calculate 16 row abs max in 16 zmms
      {
        auto reg_max_abs_loop = allocate_reg();
        xor_(*reg_max_abs_loop, *reg_max_abs_loop);
        if (param_.n / 16 > 0) {
          L(label_prefix + "max_abs_loop");
          get_16_abs_max_zmm(zmms, *reg_max_abs_loop);
          inc(*reg_max_abs_loop);
          cmp(*reg_max_abs_loop, param_.n / 16);
          jl(label_prefix + "max_abs_loop");
        }
        if (param_.write_mask != 0) {
          get_16_abs_max_zmm(zmms, *reg_max_abs_loop, true);
        }
      }

      // get scale
      transpose_16x16_ps(zmms, regs<Zmm, 16>(16));
      log2n_max_reduce_16x16(zmms);
      write_back_scale(zmms[0], M);
    };

    auto quant_write_back_Mx16 = [&](int M, Reg64& store_n_loop, bool need_mask = false) {
      auto reg_tmp = allocate_reg();
      auto reg_tmp2 = allocate_reg();
      auto reg_dst = allocate_reg();
      mov(*reg_dst, ptr[reg_param + GET_OFF(dst)]);
      imul(*reg_tmp, store_n_loop, 16 * sizeof(float));
      tmp_buf_load_M_row(M, *reg_tmp);
      imul(*reg_tmp, reg_m_loop, 16 * param_.n);
      imul(*reg_tmp2, store_n_loop, 16);
      add(*reg_tmp, *reg_tmp2);
      for (int i = 0; i < M; i++) {
        int quant_scale = i * sizeof(float) + stack_tmpbuf_offset;
        vmulps(Zmm(i), Zmm(i), zword_b[rip + data_label + quant_scale]);
        vcvtps2dq(Zmm(i), Zmm(i));
        vpmovsdb(
            need_mask ? ptr[*reg_dst + *reg_tmp + i * param_.n] | matC_n_mask : ptr[*reg_dst + *reg_tmp + i * param_.n],
            Zmm(i));
      }
    };

    auto build_MxN_tile = [&](int M, std::string label_prefix = ".") {
      xor_(reg_n_loop, reg_n_loop);
      if (param_.align_n_loop > 0) {
        L(label_prefix + "align_n_loop");
        ip_16x16(param_.align_build_block_num);
        inc(reg_n_loop);
        cmp(reg_n_loop, param_.align_n_loop);
        jl(label_prefix + "align_n_loop");
      }
      if (param_.tail_n_loop != 0) ip_16x16(param_.tail_n_loop);

      calculate_scale(M, label_prefix);

      auto store_n_loop = allocate_reg();
      xor_(*store_n_loop, *store_n_loop);
      if (param_.n / 16 > 0) {
        L(label_prefix + "store_n_loop");
        quant_write_back_Mx16(M, *store_n_loop);
        inc(*store_n_loop);
        cmp(*store_n_loop, param_.n / 16);
        jl(label_prefix + "store_n_loop");
      }
      if (param_.write_mask) quant_write_back_Mx16(M, *store_n_loop, true);
    };

    prepare_mask();
    xor_(reg_m_loop, reg_m_loop);
    mov(reg_strideA, param_.k);
    mov(reg_strideB, trans_block_col * 64);
    mov(reg_strideC, param_.pad_n * sizeof(int));  // strideC has same value as strideB.
    if (param_.align_m_loop > 0) {
      ldtilecfg(ptr[rip + data_label + 4]);
      L("align_m_loop");
      build_MxN_tile(16);
      inc(reg_m_loop);
      cmp(reg_m_loop, param_.align_m_loop);
      jl("align_m_loop");
    }
    if (param_.tail_m != 0) {
      const int offset = 4 + sizeof(tileconfig_t);
      ldtilecfg(ptr[rip + data_label + offset]);
      build_MxN_tile(param_.tail_m, ".tail_");
    }
  }
  outLocalLabel();
  L(data_label);
  float const_val[] = {1.f / 127.f};
  db(reinterpret_cast<uint8_t*>(const_val), sizeof(const_val));
  db(reinterpret_cast<uint8_t*>(&param_.m_align_cfg), sizeof(tileconfig_t));
  db(reinterpret_cast<uint8_t*>(&param_.m_tail_cfg), sizeof(tileconfig_t));
}

}  // namespace jd
