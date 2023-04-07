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

#include "jit_domain/jit_dynamic_quant.hpp"

#include "regs_pool.hpp"
namespace jd {

#define GET_OFF(field) offsetof(dynamic_quant_data_t, field)

void jit_dynamic_quant_t::generate() {
  Xbyak::Label data_label;
  inLocalLabel();
  {
    const int stack_tmpbuf_offset = 4;
    regs_pool rp(this, 1, {7, 32, 0});
    const auto channel_loop = rp.reg<Reg64>();
    const auto reg_src = rp.reg<Reg64>();
    const auto reg_scale_dst = rp.reg<Reg64>();
    const auto reg_dst = rp.reg<Reg64>();

    auto get_Nx16_fp32 = [&](const Reg64& offset, const int n) {
      const auto fp32_zmms = rp.regs<Zmm, 16>();
      const auto srcdt_size = get_data_size(param_.input_dt);
      // load bf16 and cvt to fp32 or load fp32 directly.
      for (int i = 0; i < n; i++) {
        RegExp data_addr = reg_src + offset * srcdt_size + i * param_.quantized_dim_elt_num * srcdt_size;
        if (param_.input_dt == data_type::bf16) {
          vpmovzxwd(fp32_zmms[i], ptr[data_addr]);
          vpslld(fp32_zmms[i], fp32_zmms[i], 0x10);
        } else {
          vmovups(fp32_zmms[i], ptr[data_addr]);
        }
      }
      return fp32_zmms;
    };

    auto get_N_abs_max_zmm = [&](const std::array<Zmm, 16>& zmms, const Reg64& reg_max_abs_loop,
                                 const Reg64& channel_offset, const int n, const bool need_mask = false) {
      const auto reg_tmp = rp.reg<Reg64>();
      imul(reg_tmp, reg_max_abs_loop, 16);
      add(reg_tmp, channel_offset);
      auto fp32_zmms = get_Nx16_fp32(reg_tmp, n);
      for (int i = 0; i < n; i++) vrangeps(need_mask ? zmms[i] | dim_tail_mask : zmms[i], zmms[i], fp32_zmms[i], 11U);
    };

    auto log2n_max_reduce_16x16 = [&](const std::array<Zmm, 16>& zmms) {
      int i = 8;
      while (i != 0) {
        for (int ii = 0; ii < i; ii++) vmaxps(zmms[ii], zmms[ii], zmms[ii + i]);
        i /= 2;
      }
    };

    auto write_back_scale = [&](const Zmm& scale, const int n) {
      const auto reg_tmp = rp.reg<Reg64>();
      imul(reg_tmp, channel_loop, 16);
      vmulps(scale, scale, zword_b[rip + data_label]);
      RegExp scale_addr = reg_scale_dst + reg_tmp * sizeof(float);
      vmovups(n == 16 ? ptr[scale_addr] : ptr[scale_addr] | channel_tail_mask, scale);
      vrcp14ps(scale, scale);
      vmovups(ptr[rip + data_label + stack_tmpbuf_offset], scale);
    };

    auto quant_write_back_Nx16 = [&](const Reg64& reg_loop, const Reg64& channel_offset, const int n,
                                     const bool need_mask = false) {
      const auto reg_tmp = rp.reg<Reg64>();
      imul(reg_tmp, reg_loop, 16);
      add(reg_tmp, channel_offset);
      auto fp32_zmms = get_Nx16_fp32(reg_tmp, n);
      auto dstdt_size = get_data_size(param_.output_dt);
      for (int i = 0; i < n; i++) {
        RegExp write_back_addr = reg_dst + reg_tmp * dstdt_size + i * param_.quantized_dim_elt_num * dstdt_size;
        int quant_scale = i * sizeof(float) + stack_tmpbuf_offset;
        vmulps(fp32_zmms[i], fp32_zmms[i], zword_b[rip + data_label + quant_scale]);
        vcvtps2dq(fp32_zmms[i], fp32_zmms[i]);
        vpmovsdb(need_mask ? ptr[write_back_addr] | dim_tail_mask : ptr[write_back_addr], fp32_zmms[i]);
      }
    };

    auto s8quantize_N_channel = [&](int n = 16, std::string label_prefix = ".") {
      const auto zmms = rp.regs<Zmm, 16>();
      for (int i = 0; i < 16; i++) vxorps(zmms[i], zmms[i], zmms[i]);
      const auto channel_offset = rp.reg<Reg64>();
      imul(channel_offset, channel_loop, 16 * param_.quantized_dim_elt_num);
      int align_quantdim_loop = param_.quantized_dim_elt_num / 16;  // quantized dim.
      // calculate n row abs max in n zmms
      {
        const auto reg_max_abs_loop = rp.reg<Reg64>();
        xor_(reg_max_abs_loop, reg_max_abs_loop);
        if (align_quantdim_loop > 0) {
          L(label_prefix + "max_abs_loop");
          get_N_abs_max_zmm(zmms, reg_max_abs_loop, channel_offset, n);
          inc(reg_max_abs_loop);
          cmp(reg_max_abs_loop, align_quantdim_loop);
          jl(label_prefix + "max_abs_loop");
        }
        if (param_.quantized_dim_tail_elt_num != 0) get_N_abs_max_zmm(zmms, reg_max_abs_loop, channel_offset, n, true);
      }

      // get scale.
      transpose_16x16_ps(zmms, rp.regs<Zmm, 16>());
      log2n_max_reduce_16x16(zmms);
      write_back_scale(zmms[0], n);

      // quant N channel.
      {
        const auto reg_quantize_loop = rp.reg<Reg64>();
        xor_(reg_quantize_loop, reg_quantize_loop);
        if (align_quantdim_loop > 0) {
          L(label_prefix + "quantize_loop");
          quant_write_back_Nx16(reg_quantize_loop, channel_offset, n);
          inc(reg_quantize_loop);
          cmp(reg_quantize_loop, align_quantdim_loop);
          jl(label_prefix + "quantize_loop");
        }
        if (param_.quantized_dim_tail_elt_num != 0) quant_write_back_Nx16(reg_quantize_loop, channel_offset, n, true);
      }
    };

    auto prepare_mask = [&]() {
      const auto reg_tmp = rp.reg<Xbyak::Reg32>();
      mov(reg_tmp, 0xffff >> (16 - param_.quantized_dim_tail_elt_num));
      kmovd(dim_tail_mask, reg_tmp);
      mov(reg_tmp, 0xffff >> (16 - process_channel_ % 16));
      kmovd(channel_tail_mask, reg_tmp);
    };

    prepare_mask();
    xor_(channel_loop, channel_loop);
    mov(reg_src, ptr[rp.p[0] + GET_OFF(src)]);
    mov(reg_scale_dst, ptr[rp.p[0] + GET_OFF(scale_dst)]);
    mov(reg_dst, ptr[rp.p[0] + GET_OFF(mat_dst)]);

    int align_channel_loop = process_channel_ / 16;
    int tail_channel = process_channel_ % 16;
    if (align_channel_loop > 0) {
      L(".align_channel_loop");
      s8quantize_N_channel();
      inc(channel_loop);
      cmp(channel_loop, align_channel_loop);
      jl(".align_channel_loop");
    }
    if (tail_channel > 0) s8quantize_N_channel(tail_channel, ".tail");
  }
  outLocalLabel();
  L(data_label);
  db(bit_cast<uint32_t>(1.f / 127.f), sizeof(float));
}
}  // namespace jd
