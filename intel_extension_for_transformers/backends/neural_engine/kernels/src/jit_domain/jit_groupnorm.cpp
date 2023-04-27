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

#include "jit_domain/jit_groupnorm.hpp"
#include "regs_pool.hpp"
namespace jd {

#define SUM_GET_OFF(field) offsetof(channelwise_sum_data_t, field)
#define NORM_GET_OFF(field) offsetof(channelwise_norm_data_t, field)

void jit_channelwise_sum_t::generate() {
  Xbyak::Label data_label;
  inLocalLabel();
  {
    regs_pool rp(this, 1, {5, 26, 1});

    const auto reg_param = rp.p[0];
    const auto reg_src = rp.reg<Reg64>();
    const auto reg_sum_x = rp.reg<Reg64>();
    const auto reg_sum_powx = rp.reg<Reg64>();
    const auto reg_loop = rp.reg<Reg64>();
    const auto reg_tmp = rp.reg<Reg64>();
    auto zmm_sum_x = rp.regs<Zmm, 8>();
    auto zmm_sum_powx = rp.regs<Zmm, 8>();
    auto zmm_x = rp.regs<Zmm, 8>();
    auto sum_write_mask = rp.reg<Opmask>();
    auto zmm_bf16_one = rp.reg<Zmm>();
    auto zmm_tmp = rp.reg<Zmm>();

    auto sum_func = [&](std::function<void(int)> sum) {
      // determine unroll
      unroll = 8;
      auto elt_num_per_zmm = 64 / get_data_size(param_.dt);
      while (param_.HW % (unroll * elt_num_per_zmm) != 0) unroll /= 2;
      auto loop_num = param_.HW / (unroll * elt_num_per_zmm);
      xor_(reg_loop, reg_loop);
      for (int i = 0; i < unroll; i++) {
        vxorps(zmm_sum_powx[i], zmm_sum_powx[i], zmm_sum_powx[i]);
        vxorps(zmm_sum_x[i], zmm_sum_x[i], zmm_sum_x[i]);
      }
      vpbroadcastw(zmm_bf16_one, ptr[rip + data_label]);
      L(".sum_loop");
      for (int i = 0; i < unroll; i++) sum(i);
      add(reg_src, unroll * 64);
      inc(reg_loop);
      cmp(reg_loop, loop_num);
      jl(".sum_loop");
      // log2n reduce
      int loop = unroll / 2;
      while (loop != 0) {
        for (int i = 0; i < loop; i++) {
          vaddps(zmm_sum_x[i], zmm_sum_x[i], zmm_sum_x[loop + i]);
          vaddps(zmm_sum_powx[i], zmm_sum_powx[i], zmm_sum_powx[loop + i]);
        }
        loop /= 2;
      }
      get_horizontal_op(zmm_sum_x[0], zmm_tmp, op_t::sum);
      get_horizontal_op(zmm_sum_powx[0], zmm_tmp, op_t::sum);
      vmovups(ptr[reg_sum_x] | sum_write_mask, zmm_sum_x[0]);
      vmovups(ptr[reg_sum_powx] | sum_write_mask, zmm_sum_powx[0]);
    };

    auto prepare_mask = [&] {
      mov(reg_tmp.cvt32(), 0xffff >> 15);
      kmovd(sum_write_mask, reg_tmp.cvt32());
    };

    auto load_params = [&] {
      mov(reg_src, ptr[reg_param + SUM_GET_OFF(src)]);
      mov(reg_sum_x, ptr[reg_param + SUM_GET_OFF(sum_x_ptr)]);
      mov(reg_sum_powx, ptr[reg_param + SUM_GET_OFF(sum_powx_ptr)]);
    };

    std::function<void(int)> bf16_sum = [&](int i) {
      vmovups(zmm_x[i], ptr[reg_src + (i * 64)]);
      vdpbf16ps(zmm_sum_x[i], zmm_x[i], zmm_bf16_one);
      vdpbf16ps(zmm_sum_powx[i], zmm_x[i], zmm_x[i]);
    };

    std::function<void(int)> fp32_sum = [&](int i) {
      vmovups(zmm_x[i], ptr[reg_src + (i * 64)]);
      vaddps(zmm_sum_x[i], zmm_sum_x[i], zmm_x[i]);
      vmulps(zmm_x[i], zmm_x[i], zmm_x[i]);
      vaddps(zmm_sum_powx[i], zmm_sum_powx[i], zmm_x[i]);
    };

    auto sum_code_gen = [&] {
      switch (param_.dt) {
        case data_type::bf16:
          sum_func(bf16_sum);
          break;
        case data_type::fp32:
          sum_func(fp32_sum);
          break;
        default:
          SPARSE_LOG(FATAL) << "unsupported src data type.";
          break;
      }
    };

    prepare_mask();
    load_params();
    sum_code_gen();
  }
  outLocalLabel();
  L(data_label);
  uint16_t bf16_one[] = {0x3f80};
  db(reinterpret_cast<uint8_t*>(bf16_one), sizeof(bf16_one));
}

void jit_channelwise_norm_t::generate() {
  Xbyak::Label data_label;
  inLocalLabel();
  {
    regs_pool rp(this, 1, {7, 20, 0});
    const auto reg_param = rp.p[0];
    const auto reg_src = rp.reg<Reg64>();
    const auto reg_dst = rp.reg<Reg64>();
    const auto reg_group_sum_x = rp.reg<Reg64>();
    const auto reg_group_sum_powx = rp.reg<Reg64>();
    const auto reg_gamma = rp.reg<Reg64>();
    const auto reg_beta = rp.reg<Reg64>();
    const auto reg_loop = rp.reg<Reg64>();
    auto zmm_x_mean = rp.reg<Zmm>();
    auto zmm_alpha = rp.reg<Zmm>();
    auto zmm_powx_mean = rp.reg<Zmm>();
    auto zmm_powmean = rp.reg<Zmm>();

    auto load_params = [&] {
      mov(reg_src, ptr[reg_param + NORM_GET_OFF(src)]);
      mov(reg_dst, ptr[reg_param + NORM_GET_OFF(dst)]);
      mov(reg_group_sum_x, ptr[reg_param + NORM_GET_OFF(group_sum_x_ptr)]);
      mov(reg_group_sum_powx, ptr[reg_param + NORM_GET_OFF(group_sum_powx_ptr)]);
      mov(reg_gamma, ptr[reg_param + NORM_GET_OFF(gamma)]);
      mov(reg_beta, ptr[reg_param + NORM_GET_OFF(beta)]);
    };

    auto calculate_norm_param = [&] {
      vbroadcastss(zmm_x_mean, dword[reg_group_sum_x]);
      vbroadcastss(zmm_powx_mean, dword[reg_group_sum_powx]);
      vmulps(zmm_x_mean, zmm_x_mean, zword_b[rip + data_label]);
      vmulps(zmm_powmean, zmm_x_mean, zmm_x_mean);
      vmulps(zmm_powx_mean, zmm_powx_mean, zword_b[rip + data_label]);
      vsubps(zmm_alpha, zmm_powx_mean, zmm_powmean);
      vrsqrt14ps(zmm_alpha, zmm_alpha);
      vmulps(zmm_alpha, zmm_alpha, zword_b[reg_gamma]);
    };

    auto norm = [&] {
      int unroll = 16;
      while (param_.HW % (unroll * 16) != 0) unroll -= 1;
      auto zmms = rp.regs<Zmm,16>();
      auto loop_num = param_.HW / (unroll * 16);
      xor_(reg_loop, reg_loop);
      L(".norm_loop");
      for (int i = 0; i < unroll; i++) {
        if (param_.dt == data_type::bf16) {
          vmovups(Ymm(zmms[i].getIdx()), ptr[reg_src + i * 32]);
          bf16_cvt_fp32(zmms[i]);
        } else {
          vmovups(zmms[i], ptr[reg_src + i * 64]);
        }
        vsubps(zmms[i], zmms[i], zmm_x_mean);
        vfmadd213ps(zmms[i], zmm_alpha, zword_b[reg_beta]);
        if (param_.dt == data_type::bf16) {
          fp32_cvt_bf16(zmms[i]);
          vmovups(ptr[reg_dst + i * 32], Ymm(zmms[i].getIdx()));
        } else {
          vmovups(ptr[reg_dst + i * 64], zmms[i]);
        }
      }
      add(reg_src, unroll * 16 * get_data_size(param_.dt));
      add(reg_dst, unroll * 16 * get_data_size(param_.dt));
      inc(reg_loop);
      cmp(reg_loop, loop_num);
      jl(".norm_loop");
    };

    load_params();
    calculate_norm_param();
    norm();
  }
  outLocalLabel();
  L(data_label);
  auto channels_per_group = param_.channels / param_.groups;
  float const_val[] = {1.f / (param_.HW * channels_per_group)};
  db(reinterpret_cast<uint8_t*>(const_val), sizeof(const_val));
}

}  // namespace jd
