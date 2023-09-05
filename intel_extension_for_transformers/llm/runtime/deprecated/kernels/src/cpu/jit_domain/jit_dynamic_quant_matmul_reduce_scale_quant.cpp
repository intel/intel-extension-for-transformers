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

#include <string>
#include "jit_dynamic_quant_matmul_reduce_scale_quant.hpp"
#include "regs_pool.hpp"

namespace jd {
#define PARAM_OFF(x) offsetof(dynamic_quant_matmul_reduce_scale_quant_data_t, x)

void jit_dynamic_quant_matmul_reduce_scale_quant_t::generate() {
  Xbyak::Label data_label;
  inLocalLabel();
  {
    regs_pool rp(this, 1, {7, 17, 2}, BYTES_ZMM, regs_pool::IgnoreWaste);
    const auto scale_mask = rp.reg<Opmask>();
    const auto n_mask = rp.reg<Opmask>();
    const auto reg_tmp = rp.reg<Reg64>();
    const auto reg_mat_src = rp.reg<Reg64>();
    const auto reg_mat_dst = rp.reg<Reg64>();
    const auto reg_dst_scale = rp.reg<Reg64>();
    const auto reg_reduce_scale = rp.reg<Reg64>();
    const auto reg_m_loop = rp.reg<Reg64>();
    const auto reg_n_loop = rp.reg<Reg64>();

    bool scale_need_mask = (param_.quant_m % 16) != 0;
    bool n_need_mask = (param_.quant_n % 16) != 0;
    auto set_mask = [&](const Xbyak::Opmask& mask, int tail_num) {
      mov(reg_tmp.cvt32(), 0xffff >> (16 - tail_num));
      kmovd(mask, reg_tmp.cvt32());
    };

    if (scale_need_mask) set_mask(scale_mask, param_.quant_m % 16);
    if (n_need_mask) set_mask(n_mask, param_.quant_n % 16);

    mov(reg_reduce_scale, ptr[rp.p[0] + PARAM_OFF(reduce_scale)]);
    mov(reg_dst_scale, ptr[rp.p[0] + PARAM_OFF(dst_scale)]);
    mov(reg_mat_src, ptr[rp.p[0] + PARAM_OFF(mat_src)]);
    mov(reg_mat_dst, ptr[rp.p[0] + PARAM_OFF(mat_dst)]);

    auto m_loop_num = param_.quant_m / 16;
    auto reduce_scale = [&](bool scale_need_mask = false) {
      imul(reg_tmp, reg_m_loop, 64);
      auto scale_zmm = rp.reg<Zmm>();
      vxorps(scale_zmm, scale_zmm, scale_zmm);
      for (int i = 0; i < param_.n_block_num; i++)
        vmaxps(scale_zmm, scale_zmm, ptr[reg_reduce_scale + reg_tmp + i * param_.quant_m * sizeof(float)]);
      vmulps(scale_zmm, scale_zmm, zword_b[rip + data_label]);
      vmovups(scale_need_mask ? ptr[reg_dst_scale + reg_tmp] | scale_mask : ptr[reg_dst_scale + reg_tmp], scale_zmm);
      vrcp14ps(scale_zmm, scale_zmm);
      vmovups(ptr[rsp], scale_zmm);
    };

    auto n_loop_num = param_.quant_n / 16;
    auto zmms = rp.regs<Zmm>(16);

    auto quant_Mx16 = [&](int M, bool n_dim_mask) {
      for (int i = 0; i < M; i++) {
        vmovups(Ymm(zmms[i].getIdx()), ptr[reg_mat_src + i * sizeof(bfloat16_t) * param_.n]);
        bf16_cvt_fp32(zmms[i]);
        vmulps(zmms[i], zmms[i], zword_b[rsp + i * sizeof(float)]);
        vcvtps2dq(zmms[i], zmms[i]);
        vpmovsdb(n_dim_mask ? ptr[reg_mat_dst + i * param_.n] | n_mask : ptr[reg_mat_dst + i * param_.n], zmms[i]);
      }
    };

    auto quant_M_row = [&](int M, std::string label) {
      xor_(reg_n_loop, reg_n_loop);
      L(label);
      quant_Mx16(M, false);
      add(reg_mat_src, 16 * get_data_size(jd::data_type::bf16));
      add(reg_mat_dst, 16);
      inc(reg_n_loop);
      cmp(reg_n_loop, n_loop_num);
      jl(label);
      if (n_need_mask) quant_Mx16(M, true);
    };

    xor_(reg_m_loop, reg_m_loop);
    if (m_loop_num > 0) {
      L("m_loop");
      reduce_scale();
      quant_M_row(16, "n_loop");
      inc(reg_m_loop);
      mov(reg_mat_src, ptr[rp.p[0] + PARAM_OFF(mat_src)]);
      mov(reg_mat_dst, ptr[rp.p[0] + PARAM_OFF(mat_dst)]);
      imul(reg_tmp, reg_m_loop, 16 * param_.n);
      add(reg_mat_dst, reg_tmp);
      shl(reg_tmp, 1);
      add(reg_mat_src, reg_tmp);
      cmp(reg_m_loop, m_loop_num);
      jl("m_loop");
    }
    if (scale_need_mask) {
      reduce_scale(true);
      quant_M_row(param_.quant_m % 16, "tail_n_loop");
    }
  }
  outLocalLabel();
  L(data_label);
  db(bit_cast<uint32_t>(1.f / 127.f), sizeof(float));
}
}  // namespace jd
