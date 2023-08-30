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

#include "jit_dynamic_quant.hpp"

#include "regs_pool.hpp"
namespace jd {

#define GET_OFF(field) offsetof(dynamic_quant_data_t, field)

void jit_dynamic_quant_t::generate() {
  Xbyak::Label l_rcp127;
  inLocalLabel();
  {
    constexpr int SIMD_SIZE = 16;
    constexpr auto AVX512_ALIGNMENT = BYTES_ZMM;

    regs_pool rp(this, 1, {5, 32, 2}, BYTES_ZMM, regs_pool::IgnoreWaste, AVX512_ALIGNMENT);
    const auto reg_src = rp.reg<Reg64>();
    const auto reg_scale = rp.reg<Reg64>();
    const auto reg_dst = rp.reg<Reg64>();
    const auto dim_tail_mask = rp.reg<Opmask>();
    const auto channel_tail_mask = rp.reg<Opmask>();

    const auto srcdt_size = get_data_size(param_.input_dt);
    const auto dstdt_size = get_data_size(param_.output_dt);
    const int body_quantdim = pad_to_le(param_.quantized_dim_elt_num, SIMD_SIZE);
    const int tail_quantdim = param_.quantized_dim_elt_num % SIMD_SIZE;

    auto get_Nx16_fp32 = [&](const Reg64& offset, const int n) {
      const auto fp32_zmms = rp.regs<Zmm, 16>();
      // load bf16 and cvt to fp32 or load fp32 directly.
      for (int i = 0; i < n; i++) {
        const RegExp src_addr = reg_src + offset * srcdt_size + i * param_.ld_src * srcdt_size;
        if (param_.input_dt == data_type::bf16) {
          vpmovzxwd(fp32_zmms[i], yword[src_addr]);
          vpslld(fp32_zmms[i], fp32_zmms[i], 0x10);
        } else {
          vmovups(fp32_zmms[i], zword[src_addr]);
        }
      }
      return fp32_zmms;
    };

    auto get_N_abs_max_zmm = [&](const std::array<Zmm, 16>& zmms, const Reg64& reg_offset, const int n,
                                 const bool need_mask = false) {
      auto fp32_zmms = get_Nx16_fp32(reg_offset, n);
      for (int i = 0; i < n; i++) vrangeps(need_mask ? zmms[i] | dim_tail_mask : zmms[i], zmms[i], fp32_zmms[i], 11U);
    };

    auto write_back_scale = [&](const Zmm& scale, const int n) {
      vmulps(scale, scale, zword_b[rip + l_rcp127]);
      vmovups(zword[reg_scale] | (n == 16 ? k0 : channel_tail_mask), scale);
      vrcp14ps(scale, scale);
      vmovaps(ptr[rsp], scale);
    };

    auto quant_write_back_Nx16 = [&](const Reg64& reg_offset, const int n, const bool need_mask = false) {
      const auto fp32_zmms = get_Nx16_fp32(reg_offset, n);
      for (int i = 0; i < n; i++) {
        const RegExp write_back_addr = reg_dst + reg_offset * dstdt_size + i * param_.ld_dst * dstdt_size;
        vmulps(fp32_zmms[i], fp32_zmms[i], zword_b[rsp + i * sizeof(float)]);
        vcvtps2dq(fp32_zmms[i], fp32_zmms[i]);
        vpmovsdb(need_mask ? ptr[write_back_addr] | dim_tail_mask : ptr[write_back_addr], fp32_zmms[i]);
      }
    };

    auto s8quantize_N_channel = [&](int n = 16) {
      if (!scale_input_) {
        // calculate n row abs max in n zmms
        const auto zmms = rp.regs<Zmm, 16>();
        for (int i = 0; i < 16; i++) vxorps(zmms[i], zmms[i], zmms[i]);
        const auto reg_max_abs_offset = rp.reg<Reg64>();
        xor_(reg_max_abs_offset, reg_max_abs_offset);
        if (body_quantdim > 0) {
          Xbyak::Label l_max_abs;
          L(l_max_abs);
          get_N_abs_max_zmm(zmms, reg_max_abs_offset, n);
          lea(reg_max_abs_offset, ptr[reg_max_abs_offset + 16]);
          cmp(reg_max_abs_offset, body_quantdim);
          jl(l_max_abs);
        }
        if (tail_quantdim != 0) get_N_abs_max_zmm(zmms, reg_max_abs_offset, n, true);

        // get scale.
        transpose_16x16_ps(zmms, rp.regs<Zmm, 16>());
        reduce_vmms(zmms, &CodeGenerator::vmaxps);
        write_back_scale(zmms[0], n);
      }
      if (scale_input_) {  // copy scale to the stack
        const auto vreg_scale = rp.reg<Zmm>();
        vmovaps(vreg_scale | (n == 16 ? k0 : channel_tail_mask) | T_z, ptr[reg_scale]);
        vmovaps(zword[rsp], vreg_scale);
      }

      // quant N channel.
      {
        const auto reg_quantize_offset = rp.reg<Reg64>();
        xor_(reg_quantize_offset, reg_quantize_offset);
        if (body_quantdim > 0) {
          Xbyak::Label l_quant;
          L(l_quant);
          quant_write_back_Nx16(reg_quantize_offset, n);
          lea(reg_quantize_offset, ptr[reg_quantize_offset + 16]);
          cmp(reg_quantize_offset, body_quantdim);
          jl(l_quant);
        }
        if (tail_quantdim != 0) quant_write_back_Nx16(reg_quantize_offset, n, true);
      }
    };

    auto prepare_mask = [&]() {
      const auto reg_tmp = rp.reg<Xbyak::Reg32>();
      mov(reg_tmp, 0xffff >> (16 - tail_quantdim));
      kmovd(dim_tail_mask, reg_tmp);
      mov(reg_tmp, 0xffff >> (16 - process_channel_ % 16));
      kmovd(channel_tail_mask, reg_tmp);
    };

    prepare_mask();
    const auto channel_loop = rp.reg<Reg64>();
    xor_(channel_loop, channel_loop);
    mov(reg_src, ptr[rp.p[0] + GET_OFF(src)]);
    mov(reg_scale, ptr[rp.p[0] + GET_OFF(scale)]);
    mov(reg_dst, ptr[rp.p[0] + GET_OFF(mat_dst)]);

    constexpr int TILE_CH = 16;  // Tile size on the channel dim
    const int body_channel = pad_to_le(process_channel_, SIMD_SIZE);
    const int tail_channel = process_channel_ % SIMD_SIZE;
    if (body_channel > 0) {
      Xbyak::Label body_channel_loop;
      L(body_channel_loop);
      s8quantize_N_channel();
      lea(channel_loop, ptr[channel_loop + TILE_CH]);
      lea(reg_src, ptr[reg_src + TILE_CH * param_.ld_src * srcdt_size]);
      lea(reg_dst, ptr[reg_dst + TILE_CH * param_.ld_dst * dstdt_size]);
      lea(reg_scale, ptr[reg_scale + TILE_CH * sizeof(float)]);
      cmp(channel_loop, body_channel);
      jl(body_channel_loop);
    }
    if (tail_channel > 0) s8quantize_N_channel(tail_channel);
  }
  outLocalLabel();
  L(l_rcp127);
  db(bit_cast<uint32_t>(1.f / 127.f), sizeof(float));
}
}  // namespace jd
