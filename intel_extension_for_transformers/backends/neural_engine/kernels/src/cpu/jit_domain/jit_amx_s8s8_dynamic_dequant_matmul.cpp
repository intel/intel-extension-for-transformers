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

#include "jit_amx_s8s8_dynamic_dequant_matmul.hpp"

#include "regs_pool.hpp"

namespace jd {

#define GET_OFF(field) offsetof(ssd::dynamic_quant_matmul_data_t, field)

void jit_amx_s8s8_dynamic_dequant_matmul_t::generate() {
  Xbyak::Label cfg_label;
  auto trans_block_col = param_.k / param_.tile_k;
  inLocalLabel();
  {
    const int cfg_size = sizeof(tileconfig_t);
    regs_pool rp(this, 1, {13, 32, 2}, 16, regs_pool::IgnoreWaste);

    const auto reg_m_loop = rp.reg<Reg64>();
    const auto reg_n_loop = rp.reg<Reg64>();
    const auto reg_strideA = rp.reg<Reg64>();
    const auto reg_strideB = rp.reg<Reg64>();
    const auto reg_strideTmpbuf = rp.reg<Reg64>();
    const auto reg_tmpbuf = rp.reg<Reg64>();
    const auto reg_scale_w = rp.reg<Reg64>();
    const auto reg_scale_a = rp.reg<Reg64>();
    const auto reg_bias = rp.reg<Reg64>();
    const auto reg_dst = rp.reg<Reg64>();
    const auto matC_n_mask = rp.reg<Opmask>();
    const auto scaleC_mask = rp.reg<Opmask>();

    auto prepare_mask = [&]() {
      const auto reg_tmp = rp.reg<Xbyak::Reg32>();
      mov(reg_tmp, 0xffff >> param_.write_mask);
      kmovd(matC_n_mask, reg_tmp);
      mov(reg_tmp, 0xffff >> (16 - param_.tail_m));
      kmovd(scaleC_mask, reg_tmp);
    };

    auto ip_Mx16 = [&](int M, int block_num, bool need_mask = false, int no_mask_tail_n = 0) {
      const auto reg_tmp = rp.reg<Reg64>();
      // build block
      {
        const auto reg_matA_addr = rp.reg<Reg64>();
        const auto reg_matB_addr = rp.reg<Reg64>();
        for (int i = 0; i < block_num; i++) tilezero(Tmm(i));
        // prepare addr & stride;
        mov(reg_matA_addr, ptr[rp.p[0] + GET_OFF(activation)]);
        mov(reg_matB_addr, ptr[rp.p[0] + GET_OFF(reordered_weight)]);
        imul(reg_tmp, reg_m_loop, align_build_M * param_.k);
        add(reg_matA_addr, reg_tmp);
        imul(reg_tmp, reg_n_loop,
             param_.align_build_block_num * trans_block_col * reorder_block_col_eltnum * (param_.tile_k / 4));
        add(reg_matB_addr, reg_tmp);
        for (int k_loop = 0; k_loop < param_.k / param_.tile_k; k_loop++) {
          tileloadd(Tmm(3), ptr[reg_matA_addr + reg_strideA + k_loop * param_.tile_k]);
          for (int idx = 0; idx < block_num; idx++) {
            int offset = (idx + no_mask_tail_n) * trans_block_col * reorder_block_col_eltnum * (param_.tile_k / 4) +
                         k_loop * reorder_block_col_eltnum;
            tileloadd(Tmm(4 + idx), ptr[reg_matB_addr + reg_strideB + offset]);
            tdpbssd(Tmm(idx), Tmm(3), Tmm(4 + idx));
          }
        }
      }
      // store block
      {
        for (int idx = 0; idx < block_num; idx++)
          tilestored(ptr[reg_tmpbuf + reg_strideTmpbuf + idx * align_build_N * sizeof(int)], Tmm(idx));
        // dequant + add_bias
        const auto zmms = rp.regs<Zmm, 4>();
        const auto reg_tmp2 = rp.reg<Reg64>();
        const auto reg_dst_offset = rp.reg<Reg64>();

        imul(reg_dst_offset, reg_m_loop, align_build_M * dst_n_dim_);
        imul(reg_tmp, reg_n_loop, param_.align_build_block_num * align_build_N);
        add(reg_dst_offset, reg_tmp);
        imul(reg_tmp, reg_n_loop,
             align_build_N * param_.align_build_block_num * sizeof(int));  // offset of scale_w & bias
        imul(reg_tmp2, reg_m_loop, align_build_M * sizeof(float));         // offset of scale_a
        for (int idx = 0; idx < block_num; idx++) {
          vmovups(zmms[0], ptr[reg_scale_w + reg_tmp + (idx + no_mask_tail_n) * 16 * sizeof(float)]);
          if (param_.add_bias)
            vmovups(zmms[1], ptr[reg_bias + reg_tmp + (idx + no_mask_tail_n) * align_build_N * sizeof(float)]);
          for (int row_loop = 0; row_loop < M; row_loop++) {
            vcvtdq2ps(
                zmms[2],
                ptr[reg_tmpbuf + (idx + row_loop * param_.align_build_block_num) * align_build_N * sizeof(float)]);
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

            // write back to 3-blocks tmp_buf for abs max calculation.
            if (calc_abs_max_)
              vmovups(ptr[reg_tmpbuf + (idx + row_loop * param_.align_build_block_num) * align_build_N * sizeof(float)],
                      zmms[2]);

            if (param_.dst_dt != data_type::fp32) {
              fp32_cvt_bf16(zmms[2]);
              RegExp write_back_addr =
                  reg_dst + reg_dst_offset * sizeof(bfloat16_t) +
                  ((no_mask_tail_n + idx) * align_build_N + row_loop * dst_n_dim_) * sizeof(bfloat16_t);
              vmovdqu16(need_mask ? ptr[write_back_addr] | matC_n_mask : ptr[write_back_addr], Ymm(zmms[2].getIdx()));
            } else {
              RegExp write_back_addr = reg_dst + reg_dst_offset * sizeof(float) +
                                       ((no_mask_tail_n + idx) * align_build_N + row_loop * dst_n_dim_) * sizeof(float);
              // append sum
              if (param_.append_sum) {
                if (param_.dst_dt == data_type::fp32) {
                  vaddps(zmms[2], zmms[2], ptr[write_back_addr]);
                } else {
                  auto append_value_zmm = rp.reg<Zmm>();
                  vmovups(Ymm(append_value_zmm.getIdx()), ptr[write_back_addr]);
                  bf16_cvt_fp32(append_value_zmm);
                  vaddps(zmms[2], zmms[2], append_value_zmm);
                }
              }
              vmovups(need_mask ? ptr[write_back_addr] | matC_n_mask : ptr[write_back_addr], zmms[2]);
            }
          }
        }
      }
      // calculate max abs.
      if (calc_abs_max_) {
        // compare with the scale_dst, store the larger one.
        const auto zmms = rp.regs<Zmm, 16>();
        for (int i = 0; i < 16; i++) vxorps(zmms[i], zmms[i], zmms[i]);
        for (int block_idx = 0; block_idx < block_num; block_idx++) {
          for (int row_idx = 0; row_idx < M; row_idx++) {
            vrangeps(
                need_mask ? zmms[row_idx] | matC_n_mask : zmms[row_idx], zmms[row_idx],
                ptr[reg_tmpbuf + align_build_N * sizeof(float) * (block_idx + row_idx * param_.align_build_block_num)],
                11U);
          }
        }
        transpose_16x16_ps(zmms, rp.regs<Zmm, 16>());
        reduce_vmms(zmms, &CodeGenerator::vmaxps);
        auto reg_scale_dst = rp.reg<Reg64>();
        mov(reg_scale_dst, ptr[rp.p[0] + GET_OFF(scale_dst)]);
        mov(reg_tmp, ptr[rsp + 8]);
        vmaxps(zmms[0], zmms[0], ptr[reg_scale_dst + reg_tmp]);
        vmovups(M == 16 ? ptr[reg_scale_dst + reg_tmp] : ptr[reg_scale_dst + reg_tmp] | scaleC_mask, zmms[0]);
      }
    };

    auto align_loop_ip_Mx16 = [&](int M, int loop_num, std::string label_prefix) {
      L(label_prefix + "align_n_loop");
      ip_Mx16(M, param_.align_build_block_num);
      inc(reg_n_loop);
      cmp(reg_n_loop, loop_num);
      jl(label_prefix + "align_n_loop");
    };

    auto build_MxN_tile = [&](int M, std::string label_prefix = ".") {
      if (calc_abs_max_) {
        auto clear_zmm = rp.reg<Zmm>();  // init abs max (equal to 0) in scale_dst.
        vxorps(clear_zmm, clear_zmm, clear_zmm);
        auto reg_scale_offset = reg_m_loop;  // alias of reg_m_loop
        mov(ptr[rsp], reg_m_loop);           // store the m_loop value to the stack.
        imul(reg_scale_offset, reg_m_loop, 16 * sizeof(float));
        auto reg_scale_dst = reg_n_loop;  // alias of reg_n_loop
        mov(reg_scale_dst, ptr[rp.p[0] + GET_OFF(scale_dst)]);
        RegExp scale_wirte_back_offset = reg_scale_dst + reg_scale_offset;
        vmovups(M == 16 ? ptr[scale_wirte_back_offset] : ptr[scale_wirte_back_offset] | scaleC_mask, clear_zmm);
        mov(ptr[rsp + 8], reg_scale_offset);  // store the scaleC offset to the stack.
        mov(reg_m_loop, ptr[rsp]);            // restore m_loop value.
      }
      xor_(reg_n_loop, reg_n_loop);
      if (param_.write_mask == 0) {
        if (param_.align_n_loop > 0) align_loop_ip_Mx16(M, param_.align_n_loop, label_prefix);
        if (param_.tail_n_loop != 0) ip_Mx16(M, param_.tail_n_loop);
      } else {
        int no_mask_align_n_loop = (param_.n - align_build_N) / (align_build_N * param_.align_build_block_num);
        int no_mask_tail_n =
            ((param_.n - align_build_N) % (align_build_N * param_.align_build_block_num)) / align_build_N;
        if (no_mask_align_n_loop > 0) align_loop_ip_Mx16(M, no_mask_align_n_loop, label_prefix);
        if (no_mask_tail_n != 0) ip_Mx16(M, no_mask_tail_n);
        ip_Mx16(M, 1, true, no_mask_tail_n);
      }
    };

    prepare_mask();
    xor_(reg_m_loop, reg_m_loop);
    mov(reg_strideA, param_.k);
    mov(reg_strideB, trans_block_col * reorder_block_col_eltnum);
    mov(reg_strideTmpbuf, align_build_N * param_.align_build_block_num * sizeof(int));
    mov(reg_tmpbuf, ptr[rp.p[0] + GET_OFF(tmp_buf)]);
    mov(reg_scale_a, ptr[rp.p[0] + GET_OFF(scale_a)]);
    mov(reg_scale_w, ptr[rp.p[0] + GET_OFF(scale_w)]);
    mov(reg_bias, ptr[rp.p[0] + GET_OFF(bias)]);
    mov(reg_dst, ptr[rp.p[0] + GET_OFF(dst)]);

    if (param_.align_m_loop > 0) {
      ldtilecfg(ptr[rip + cfg_label]);
      L("align_m_loop");
      build_MxN_tile(align_build_M);
      inc(reg_m_loop);
      cmp(reg_m_loop, param_.align_m_loop);
      jl("align_m_loop");
    }
    if (param_.tail_m != 0) {
      ldtilecfg(ptr[rip + cfg_label + cfg_size]);
      build_MxN_tile(param_.tail_m, ".tail_");
    }
  }
  outLocalLabel();
  L(cfg_label);
  db(reinterpret_cast<uint8_t*>(&param_.m_align_cfg), sizeof(tileconfig_t));
  db(reinterpret_cast<uint8_t*>(&param_.m_tail_cfg), sizeof(tileconfig_t));
  eltwise_injector_.prepare_table();
}

}  // namespace jd
