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
namespace jd {

#define GET_OFF(field) offsetof(groupnorm_data_t, field)

#define DEF_AND_LOAD_ALL_PARAM                                 \
  const auto reg_param = rp.p[0];                              \
  const auto reg_src = rp.reg<Reg64>();                        \
  const auto reg_dst = rp.reg<Reg64>();                        \
  const auto reg_sum_x = rp.reg<Reg64>();                      \
  const auto reg_sum_powx = rp.reg<Reg64>();                   \
  const auto reg_gamma = rp.reg<Reg64>();                      \
  const auto reg_beta = rp.reg<Reg64>();                       \
  auto load_params = [&] {                                     \
    mov(reg_src, ptr[reg_param + GET_OFF(src)]);               \
    mov(reg_dst, ptr[reg_param + GET_OFF(dst)]);               \
    mov(reg_sum_x, ptr[reg_param + GET_OFF(sum_x_ptr)]);       \
    mov(reg_sum_powx, ptr[reg_param + GET_OFF(sum_powx_ptr)]); \
    mov(reg_gamma, ptr[reg_param + GET_OFF(gamma)]);           \
    mov(reg_beta, ptr[reg_param + GET_OFF(beta)]);             \
  };

#define DEF_BF16_ONE_CONST(label) \
  L(label);                       \
  uint16_t bf16_one[] = {0x3f80}; \
  db(reinterpret_cast<uint8_t*>(bf16_one), sizeof(bf16_one));

#define DEF_DIV_COSNT(label, norm_dim)   \
  L(label);                              \
  float div_const[] = {1.0f / norm_dim}; \
  db(reinterpret_cast<uint8_t*>(div_const), sizeof(div_const));

#define DEF_EPS_CONST(label, eps) \
  L(label);                       \
  float eps_const[] = {eps};      \
  db(reinterpret_cast<uint8_t*>(eps_const), sizeof(eps_const));

void jit_groupnorm_t::sum_code_gen(regs_pool* rp, Reg64 reg_src, Reg64 reg_sum_x, Reg64 reg_sum_powx,
                                   Opmask sum_write_mask, const Xbyak::Label& data_label, size_t sum_dim) {
  const auto reg_loop = rp->reg<Reg64>();
  auto zmm_sum_x = rp->regs<Zmm, 8>();
  auto zmm_sum_powx = rp->regs<Zmm, 8>();
  auto zmm_x = rp->regs<Zmm, 8>();
  auto zmm_bf16_one = rp->reg<Zmm>();
  auto zmm_tmp = rp->reg<Zmm>();

  auto sum_func = [&](std::function<void(int, int)> sum) {
    // determine unroll
    unroll = 8;
    int process_simd_byte = jit_generator::BYTES_ZMM;
    if (!isa_available(avx512_core_bf16) && param_.dt == data_type::bf16) process_simd_byte = jit_generator::BYTES_YMM;
    auto elt_num_per_zmm = process_simd_byte / get_data_size(param_.dt);
    while (sum_dim % (unroll * elt_num_per_zmm) != 0) unroll /= 2;
    auto loop_num = sum_dim / (unroll * elt_num_per_zmm);
    xor_(reg_loop, reg_loop);
    for (int i = 0; i < unroll; i++) {
      vxorps(zmm_sum_powx[i], zmm_sum_powx[i], zmm_sum_powx[i]);
      vxorps(zmm_sum_x[i], zmm_sum_x[i], zmm_sum_x[i]);
    }
    vpbroadcastw(zmm_bf16_one, ptr[rip + data_label]);
    L(".sum_loop");
    for (int i = 0; i < unroll; i++) sum(i, process_simd_byte);
    add(reg_src, unroll * process_simd_byte);
    inc(reg_loop);
    cmp(reg_loop, loop_num);
    jl(".sum_loop");
    // log2n reduce
    jit_generator::reduce_vmms(zmm_sum_x, &CodeGenerator::vaddps);
    jit_generator::reduce_vmms(zmm_sum_powx, &CodeGenerator::vaddps);
    reduce_dwords(zmm_sum_x[0], zmm_tmp, &CodeGenerator::vaddps);
    reduce_dwords(zmm_sum_powx[0], zmm_tmp, &CodeGenerator::vaddps);
    vmovups(ptr[reg_sum_x] | sum_write_mask, zmm_sum_x[0]);
    vmovups(ptr[reg_sum_powx] | sum_write_mask, zmm_sum_powx[0]);
  };

  std::function<void(int, int)> bf16_sum = [&](int i, int process_simd_byte) {
    if (isa_available(avx512_core_bf16)) {
      vmovups(zmm_x[i], ptr[reg_src + (i * process_simd_byte)]);
      vdpbf16ps(zmm_sum_x[i], zmm_x[i], zmm_bf16_one);
      vdpbf16ps(zmm_sum_powx[i], zmm_x[i], zmm_x[i]);
    } else {
      vmovups(Ymm(zmm_x[i].getIdx()), ptr[reg_src + (i * process_simd_byte)]);
      bf16_cvt_fp32(zmm_x[i]);
      vaddps(zmm_sum_x[i], zmm_sum_x[i], zmm_x[i]);
      vmulps(zmm_x[i], zmm_x[i], zmm_x[i]);
      vaddps(zmm_sum_powx[i], zmm_sum_powx[i], zmm_x[i]);
    }
  };

  std::function<void(int, int)> fp32_sum = [&](int i, int process_simd_byte) {
    vmovups(zmm_x[i], ptr[reg_src + (i * process_simd_byte)]);
    vaddps(zmm_sum_x[i], zmm_sum_x[i], zmm_x[i]);
    vmulps(zmm_x[i], zmm_x[i], zmm_x[i]);
    vaddps(zmm_sum_powx[i], zmm_sum_powx[i], zmm_x[i]);
  };

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
}

void jit_groupnorm_t::norm(regs_pool* rp, Reg64 reg_src, Reg64 reg_dst, Reg64 reg_sum_x, Reg64 reg_sum_powx,
                           Reg64 reg_gamma, Reg64 reg_beta, const Xbyak::Label& div_const_label,
                           const Xbyak::Label& eps_label, size_t channels_per_group) {
  const auto reg_norm_loop = rp->reg<Reg64>();
  const auto reg_channel_loop = rp->reg<Reg64>();
  auto zmm_x_mean = rp->reg<Zmm>();
  auto zmm_tmp = rp->reg<Zmm>();
  auto zmm_alpha = rp->reg<Zmm>();
  auto zmm_powx_mean = rp->reg<Zmm>();
  auto zmm_powmean = rp->reg<Zmm>();
  // calculate norm param
  vbroadcastss(zmm_x_mean, dword[reg_sum_x]);
  vbroadcastss(zmm_powx_mean, dword[reg_sum_powx]);
  vmulps(zmm_x_mean, zmm_x_mean, zword_b[rip + div_const_label]);
  vmulps(zmm_powmean, zmm_x_mean, zmm_x_mean);
  vmulps(zmm_powx_mean, zmm_powx_mean, zword_b[rip + div_const_label]);
  vsubps(zmm_tmp, zmm_powx_mean, zmm_powmean);
  vaddps(zmm_tmp, zmm_tmp, zword_b[rip + eps_label]);
  vrsqrt14ps(zmm_tmp, zmm_tmp);

  int unroll = 16;
  while (param_.HW % (unroll * 16) != 0) unroll -= 1;
  auto zmms = rp->regs<Zmm, 16>();
  auto loop_num = param_.HW / (unroll * 16);
  xor_(reg_channel_loop, reg_channel_loop);
  L(".channel_loop");
  xor_(reg_norm_loop, reg_norm_loop);
  L(".norm_loop");
  vmulps(zmm_alpha, zmm_tmp, zword_b[reg_gamma + 4 * reg_channel_loop]);
  for (int i = 0; i < unroll; i++) {
    if (param_.dt == data_type::bf16) {
      vmovups(Ymm(zmms[i].getIdx()), ptr[reg_src + i * 32]);
      bf16_cvt_fp32(zmms[i]);
    } else {
      vmovups(zmms[i], ptr[reg_src + i * 64]);
    }
    vsubps(zmms[i], zmms[i], zmm_x_mean);
    vfmadd213ps(zmms[i], zmm_alpha, zword_b[reg_beta + 4 * reg_channel_loop]);
    if (param_.postop_attrs.size() != 0) {
      eltwise_injector_.escape_rp_all_type(rp);
      eltwise_injector_.vector_compute(zmms[i], param_.postop_attrs);
    }
    if (param_.dt == data_type::bf16) {
      fp32_cvt_bf16(zmms[i]);
      vmovups(ptr[reg_dst + i * 32], Ymm(zmms[i].getIdx()));
    } else {
      vmovups(ptr[reg_dst + i * 64], zmms[i]);
    }
  }
  add(reg_src, unroll * 16 * get_data_size(param_.dt));
  add(reg_dst, unroll * 16 * get_data_size(param_.dt));
  inc(reg_norm_loop);
  cmp(reg_norm_loop, loop_num);
  jl(".norm_loop");
  inc(reg_channel_loop);
  cmp(reg_channel_loop, channels_per_group);
  jl(".channel_loop");
}

void jit_groupnorm_t::prepare_mask(Reg64 reg_tmp, Opmask sum_write_mask) {
  mov(reg_tmp.cvt32(), 0xffff >> 15);
  kmovd(sum_write_mask, reg_tmp.cvt32());
}

void jit_channelwise_sum_t::generate() {
  Xbyak::Label data_label;
  inLocalLabel();
  {
    regs_pool rp(this, 1, {5, 26, 1});

    const auto reg_param = rp.p[0];
    const auto reg_src = rp.reg<Reg64>();
    const auto reg_sum_x = rp.reg<Reg64>();
    const auto reg_sum_powx = rp.reg<Reg64>();
    const auto reg_tmp = rp.reg<Reg64>();
    auto sum_write_mask = rp.reg<Opmask>();

    auto load_params = [&] {
      mov(reg_src, ptr[reg_param + GET_OFF(src)]);
      mov(reg_sum_x, ptr[reg_param + GET_OFF(sum_x_ptr)]);
      mov(reg_sum_powx, ptr[reg_param + GET_OFF(sum_powx_ptr)]);
    };

    prepare_mask(reg_tmp, sum_write_mask);
    load_params();
    sum_code_gen(&rp, reg_src, reg_sum_x, reg_sum_powx, sum_write_mask, data_label, param_.HW);
  }
  outLocalLabel();
  DEF_BF16_ONE_CONST(data_label)
}

void jit_channelwise_norm_t::generate() {
  Xbyak::Label div_const_label, eps_label;
  inLocalLabel();
  {
    regs_pool rp(this, 1, {8, 21, 0});
    DEF_AND_LOAD_ALL_PARAM
    load_params();
    norm(&rp, reg_src, reg_dst, reg_sum_x, reg_sum_powx, reg_gamma, reg_beta, div_const_label, eps_label);
  }
  outLocalLabel();
  auto norm_elt = param_.HW * (param_.channels / param_.groups);
  DEF_DIV_COSNT(div_const_label, norm_elt)
  DEF_EPS_CONST(eps_label, param_.eps)
  eltwise_injector_.prepare_table();
}

void jit_groupnorm_t::generate() {
  Xbyak::Label bf16_one_label, div_const_label, eps_label;
  auto norm_elt = param_.HW * (param_.channels / param_.groups);
  inLocalLabel();
  {
    regs_pool rp(this, 1, {9, 26, 1});
    const auto reg_tmp = rp.reg<Reg64>();
    auto sum_write_mask = rp.reg<Opmask>();
    DEF_AND_LOAD_ALL_PARAM
    load_params();
    prepare_mask(reg_tmp, sum_write_mask);
    sum_code_gen(&rp, reg_src, reg_sum_x, reg_sum_powx, sum_write_mask, bf16_one_label, norm_elt);
    mov(reg_src, ptr[reg_param + GET_OFF(src)]);
    norm(&rp, reg_src, reg_dst, reg_sum_x, reg_sum_powx, reg_gamma, reg_beta, div_const_label, eps_label,
         param_.channels / param_.groups);
  }
  outLocalLabel();
  DEF_BF16_ONE_CONST(bf16_one_label)
  DEF_DIV_COSNT(div_const_label, norm_elt)
  DEF_EPS_CONST(eps_label, param_.eps)
  eltwise_injector_.prepare_table();
}

}  // namespace jd
