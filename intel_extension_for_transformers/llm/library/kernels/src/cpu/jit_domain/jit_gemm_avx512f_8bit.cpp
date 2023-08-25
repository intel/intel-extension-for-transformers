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
#include "jit_gemm_avx512f_8bit.hpp"

namespace jd {
#define OFFSET(field) offsetof(ssd::matmul_fp8_data_t, field)

void jit_gemm_avx512f_8bit_t::generate() {
  inLocalLabel();  // use local label for multiple instance
  {
#ifdef _WIN32
    regs_pool rp(this, 1, {11, ARegCount + BRegCount + CRegCount + TmpRegCount, 0}, 16 * 10);
#else
    regs_pool rp(this, 1, {11, ARegCount + BRegCount + CRegCount + TmpRegCount, 0}, 0);
#endif
    auto parambase = rp.p[0];

    auto reg_matAptr = rp.reg<Reg64>();
    auto reg_matBptr = rp.reg<Reg64>();
    auto reg_ksize = rp.reg<Reg64>();
    auto reg_cstep = rp.reg<Reg64>();
    auto reg_astep = rp.reg<Reg64>();
    auto reg_bstep = rp.reg<Reg64>();
    auto reg_iterk = rp.reg<Reg64>();
    auto reg_tmp = rp.reg<Reg64>();
    auto reg_tmp1 = rp.reg<Reg64>();
    auto reg_tmp2 = rp.reg<Reg64>();
    auto reg_nsize = rp.reg<Reg64>();
    auto& reg_ret = rax;

    injector_.escape_regs(reg_type::reg64, reg_tmp.getIdx());
    injector_.escape_regs(reg_type::reg64, reg_tmp1.getIdx());
    injector_.escape_regs(reg_type::reg64, reg_tmp2.getIdx());
    vreg_push(rsp);
    auto zmms_a = rp.regs<Xbyak::Zmm>(ARegCount);
    auto zmms_b = rp.regs<Xbyak::Zmm>(BRegCount);
    auto zmms_c = rp.regs<Xbyak::Zmm>(CRegCount);
    auto zmms_tmp = rp.regs<Xbyak::Zmm>(TmpRegCount);
    zmms_a_ = zmms_a;
    zmms_b_ = zmms_b;
    zmms_c_ = zmms_c;
    zmms_tmp_ = zmms_tmp;

    for (int i = 0; i < CRegCount; i++) {
      vpxorq(zmms_c_[i], zmms_c_[i], zmms_c_[i]);
    }

    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
    load32(reg_astep, ptr[parambase + OFFSET(astep)]);
    load32(reg_bstep, ptr[parambase + OFFSET(bstep)]);
    imul(reg_bstep, reg_bstep, 16);
    xor_(reg_iterk, reg_iterk);

    mov(reg_tmp2, 0xaaaaaaaa);
    kmovq(k1, reg_tmp2);

    if (param_.weight_type == data_type::f8_e4m3) {
      mov(reg_tmp2, (127 - 7) << 23);
    } else if (param_.weight_type == data_type::f8_e5m2) {
      mov(reg_tmp2, (127 - 15) << 23);
    }
    vpbroadcastd(zmms_tmp_[1], reg_tmp2.cvt32());

    mov(reg_tmp2, 1 << 31);
    vpbroadcastd(zmms_tmp_[2], reg_tmp2.cvt32());

    cmp(reg_nsize, NTile);
    jb(".lastloop", T_NEAR);
    L(".kloop");
    generate_fma(MTile, NRegs, KTile, reg_matAptr, reg_matBptr, reg_astep, reg_bstep, reg_tmp, reg_tmp1);
    add(reg_matAptr, KTile * 2);
    add(reg_matBptr, KTile * 16);
    add(reg_iterk, KTile);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    alphabeta_process(MTile, NRegs, parambase, reg_tmp, reg_tmp1, reg_tmp2);
    jmp(".retl", T_NEAR);

    L(".lastloop");
    L(".k1loop");
    generate_fma(MTile, 1, KTile, reg_matAptr, reg_matBptr, reg_astep, reg_bstep, reg_tmp, reg_tmp1);
    add(reg_matAptr, KTile * 2);
    add(reg_matBptr, KTile * 16);
    add(reg_iterk, KTile);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".k1loop");
    alphabeta_process(MTile, 1, parambase, reg_tmp, reg_tmp1, reg_tmp2);

    L(".retl");
    vreg_pop(rsp);
    mov(reg_ret, 0);
  }
  outLocalLabel();  // end of local label
  injector_.prepare_table();
}

void jit_gemm_avx512f_8bit_t::load_int8_fp32(const Xbyak::Zmm& tar, const Xbyak::Address& addr) {
  vpmovsxbd(tar, addr);
  vcvtdq2ps(tar | T_rn_sae, tar);
}

void jit_gemm_avx512f_8bit_t::load_fp8_fp32(const Xbyak::Zmm& tar, const Xbyak::Address& addr) {
  auto& exp = zmms_tmp_[0];
  vpmovsxbd(tar, addr);
  vpslld(exp, tar, 25);
  if (param_.weight_type == data_type::f8_e4m3) {
    vpsrld(exp, exp, 5);
  } else if (param_.weight_type == data_type::f8_e5m2) {
    vpsrld(exp, exp, 4);
  }
  vpaddd(exp, exp, zmms_tmp_[1]);
  vandps(tar, zmms_tmp_[2]);
  vorps(tar, exp);
}

void jit_gemm_avx512f_8bit_t::generate_fma(int MTile, int _NRegs, int KTile, const Xbyak::Reg64& aptr,
                                           const Xbyak::Reg64& bptr, const Xbyak::Reg64& reg_astep,
                                           const Xbyak::Reg64& reg_bstep, const Xbyak::Reg64& reg_tmp,
                                           const Xbyak::Reg64& reg_tmp1) {
  int kk = 0;
  for (; kk < KTile; kk++) {
    int mm = 0;
    lea(reg_tmp, ptr[aptr + kk * 2]);
    mov(reg_tmp1, bptr);
    for (int nn = 0; nn < _NRegs; nn++) {
      if (param_.weight_type == data_type::s8) {
        load_int8_fp32(zmms_b_[nn], ptr[reg_tmp1 + kk * 16]);
      } else if (param_.weight_type == data_type::f8_e4m3 || param_.weight_type == data_type::f8_e5m2) {
        load_fp8_fp32(zmms_b_[nn], ptr[reg_tmp1 + kk * 16]);
      }
      add(reg_tmp1, reg_bstep);
    }
    for (; mm < MTile; mm++) {
      broadcast_bf16_fp32(zmms_a_[0], k1, ptr[reg_tmp]);
      add(reg_tmp, reg_astep);
      for (int nn = 0; nn < _NRegs; nn++) {
        vfmadd231ps(zmms_c_[mm * NRegs + nn], zmms_b_[nn], zmms_a_[0]);
      }
    }
  }
}
void jit_gemm_avx512f_8bit_t::alphabeta_process(int MTile, int _NRegs, const Xbyak::Reg64& parambase,
                                                const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_tmp1,
                                                const Xbyak::Reg64& reg_tmp2) {
  inLocalLabel();
  // load scale
  if (param_.has_scale0) {
    mov(reg_tmp, ptr[parambase + OFFSET(scale)]);
    for (int nn = 0; nn < _NRegs; nn++) {
      vmovups(zmms_b_[nn], ptr[reg_tmp + nn * 64]);
    }
  }
  cmp(dword[parambase + OFFSET(alpha)], 0x3F800000);  // 1.f
  je(".afteralpha", T_NEAR);
  vbroadcastss(zmms_a_[0], zword[parambase + OFFSET(alpha)]);
  if (param_.has_scale0) {
    for (int nn = 0; nn < _NRegs; nn++) {
      vmulps(zmms_b_[nn], zmms_a_[0]);
    }
  } else {
    for (int i = 0; i < MTile; i++) {
      for (int j = 0; j < _NRegs; j++) {
        vmulps(zmms_c_[i * NRegs + j], zmms_a_[0]);
      }
    }
  }
  L(".afteralpha");
  // scale
  if (param_.has_scale0) {
    for (int mm = 0; mm < MTile; mm++) {
      for (int nn = 0; nn < _NRegs; nn++) {
        vmulps(zmms_c_[mm * NRegs + nn], zmms_b_[nn]);
      }
    }
  }
  cmp(dword[parambase + OFFSET(kpos)], 0);
  jnz(".ntinitkl", T_NEAR);
  cmp(dword[parambase + OFFSET(beta)], 0);
  jz(".afterbeta", T_NEAR);
  mov(reg_tmp, ptr[parambase + OFFSET(matD)]);
  load32(reg_tmp1, ptr[parambase + OFFSET(dstep)]);
  vbroadcastss(zmms_a_[0], zword[parambase + OFFSET(beta)]);
  for (int i = 0; i < MTile; i++) {
    for (int j = 0; j < _NRegs; j++) {
      load_bf16_fp32(zmms_b_[j], ptr[reg_tmp + j * 32]);
      vmulps(zmms_b_[j], zmms_a_[0]);
      vaddps(zmms_c_[i * NRegs + j], zmms_b_[j]);
    }
    add(reg_tmp, reg_tmp1);
  }

  L(".afterbeta");
  mov(reg_tmp, ptr[parambase + OFFSET(matC)]);
  if (param_.has_append_sum) {
    mov(reg_tmp2, ptr[parambase + OFFSET(matE)]);
  }
  load32(reg_tmp1, ptr[parambase + OFFSET(cstep)]);
  for (int i = 0; i < MTile; i++) {
    for (int j = 0; j < _NRegs; j++) {
      if (param_.has_append_sum) {
        load_bf16_fp32(zmms_b_[j], ptr[reg_tmp2 + j * 32]);
        vaddps(zmms_c_[i * NRegs + j], zmms_b_[j]);
      }
      store_fp32_bf16(zmms_c_[i * NRegs + j], ptr[reg_tmp + j * 32]);
    }
    add(reg_tmp, reg_tmp1);
    if (param_.has_append_sum) {
      add(reg_tmp2, reg_tmp1);
    }
  }
  jmp(".retl", T_NEAR);

  L(".ntinitkl");
  mov(reg_tmp, ptr[parambase + OFFSET(matC)]);
  load32(reg_tmp1, ptr[parambase + OFFSET(cstep)]);
  injector_.escape_regs(reg_type::zmm, zmms_tmp_[0].getIdx());
  for (int i = 0; i < MTile; i++) {
    for (int j = 0; j < _NRegs; j++) {
      load_bf16_fp32(zmms_tmp_[0], ptr[reg_tmp + j * 32]);
      vaddps(zmms_c_[i * NRegs + j], zmms_tmp_[0]);
      store_fp32_bf16(zmms_c_[i * NRegs + j], ptr[reg_tmp + j * 32]);
    }
    add(reg_tmp, reg_tmp1);
  }

  L(".retl");
  outLocalLabel();
}

}  // namespace jd
