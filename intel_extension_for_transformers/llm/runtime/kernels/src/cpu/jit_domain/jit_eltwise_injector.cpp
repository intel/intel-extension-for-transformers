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

#include "jit_eltwise_injector.hpp"

namespace jd {

void jit_eltwise_injector::eltwise_injector_init(jit_generator* ptr, const std::vector<postop_attr>& postop_attrs) {
  h = ptr;
  register_table_entries(postop_attrs);
}

size_t jit_eltwise_injector::table_off(key_t key, size_t key_off_val_shift) {
  const auto it = entry_map.find(key);
  SPARSE_LOG_IF(FATAL, it == entry_map.end()) << "key is not in entry_map";
  const auto& te = (*it).second;
  const auto scale = te.bcast ? 64u : sizeof(table_entry_val_t);
  return te.off + key_off_val_shift * scale;
}

Xbyak::Address jit_eltwise_injector::table_val(key_t key, size_t key_off_val_shift) {
  auto off = table_off(key, key_off_val_shift);
  return h->ptr[p_table + off];
}

void jit_eltwise_injector::assert_check(const std::vector<postop_attr>& postop_attrs) {
  bool quant_flag = false;
  bool int_lut_flag = false;
  int chain_len = postop_attrs.size();
  for (int i = 0; i < chain_len; i++) {
    auto cur_attr = postop_attrs[i];
    auto cur_alg = cur_attr.op_alg;
    auto cur_dt = cur_attr.dt;
    if (i != chain_len - 1) SPARSE_LOG_IF(FATAL, cur_alg == postop_alg::quantize) << "quantize should be last op";
    if (i != 0) SPARSE_LOG_IF(FATAL, cur_alg == postop_alg::dequantize) << "Dequantize should be first op";
    // bit8-lut algo must be the fist op in the postop-chain.
    if (cur_alg == postop_alg::eltop_int_lut) {
      SPARSE_LOG_IF(FATAL, i != 0) << "eltop_int_lut should be first op";
      int_lut_flag = true;
    }

    if (cur_alg == postop_alg::quantize || cur_attr.op_alg == postop_alg::dequantize) {
      quant_flag = true;
      SPARSE_LOG_IF(FATAL, !(cur_dt == data_type::s8 || cur_dt == data_type::u8)) << "should quantize to s8/u8";
    }

    // we do not need to assert other affairs
    // because the remain ops's kernel version will not execute..
    if (int_lut_flag) return;

    // normal op only support fp32/bf16,once contain quant related operator,only support fp32.
    if (!quant_flag) {
      SPARSE_LOG_IF(FATAL, !(cur_dt == data_type::fp32 || cur_dt == data_type::bf16))
          << "normal op only support fp32/bf16";
    } else {
      if (cur_alg != postop_alg::dequantize && cur_alg != postop_alg::quantize)
        SPARSE_LOG_IF(FATAL, cur_dt != data_type::fp32) << "once contain quant related operator,only support fp32.";
    }
  }
}

void jit_eltwise_injector::vector_compute(const Xbyak::Zmm& zmm_src, const std::vector<postop_attr>& postop_attrs,
                                          std::vector<int> postop_idxs) {
  if (postop_idxs.size() == 0) {
    for (std::size_t i = 0; i < postop_attrs.size(); i++) postop_idxs.push_back(i);
  }

  assert_check(postop_attrs);
  init_tb_allocate_set(postop_attrs);
  assign_regs();
  load_table_addr();

  auto task_dispatch = [&](const Xbyak::Zmm& zmm_src) {
    switch (cur_postop_attr_.op_alg) {
      case postop_alg::exp:
        exp_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::tanh:
        tanh_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::gelu:
        gelu_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::relu:
        relu_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::quantize:
        quantize_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::dequantize:
        dequantize_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::linear:
        linear_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::low_precision_exp:
        low_precision_exp_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::swish:
        swish_compute_vector_fwd(zmm_src);
        break;
      case postop_alg::eltop_int_lut:
        if (cur_postop_attr_.alpha == 8) bit8_lut_compute_vector_fwd(zmm_src);
        if (cur_postop_attr_.alpha == 16) bit16_lut_compute_vector_fwd(zmm_src);
        break;
      default:
        SPARSE_LOG(FATAL) << "unsupported op in eltwise_injector";
        break;
    }
  };

  for (auto&& i : postop_idxs) {
    cur_postop_attr_ = postop_attrs[i];
    if (cur_postop_attr_.dt == data_type::bf16) {
      h->vmovups(zmm_tmp, zmm_src);

      Ymm ymm_src = Ymm(zmm_src.getIdx());
      h->vpmovzxwd(zmm_src, ymm_src);
      h->vpslld(zmm_src, zmm_src, 16);
      task_dispatch(zmm_src);
      h->vcvtneps2bf16(ymm_src, zmm_src);  // 0-255bit of zmm_src compute ans store in ymm_src.

      ymm_tmp = Ymm(zmm_tmp.getIdx());
      h->vextractf32x8(ymm_tmp, zmm_tmp, 1);  // shuffle the high 256bit to the low 256 bit.
      h->vpmovzxwd(zmm_tmp, ymm_tmp);
      h->vpslld(zmm_tmp, zmm_tmp, 16);
      task_dispatch(zmm_tmp);
      h->vcvtneps2bf16(ymm_tmp, zmm_tmp);  // 256-511bit of zmm_src compute ans store in ymm_tmp.

      // permute
      h->mov(reg64_tmp, 0xff00);
      h->kmovq(k_mask, reg64_tmp);
      h->vmovups(zmm_aux0, table_val(exchange_zmm_low256_high256));
      h->vpermt2ps(zmm_src | k_mask, zmm_aux0, zmm_tmp);
    } else {
      task_dispatch(zmm_src);
    }
    if (cur_postop_attr_.op_alg == postop_alg::eltop_int_lut) return;
  }
}

void jit_eltwise_injector::bit8_lut_compute_vector_fwd(const Zmm& zmm_src) {
  // regs renaming.
  Zmm zmm_bk = zmm_aux0;
  h->vmovups(zmm_bk, zmm_src);
  // zmm can store 64 byte data, the size of our bit8-lut is 256 byte, so we need to loop 4 times so that we can search
  // all terms
  for (int i = 0; i < 4; i++) {
    h->vmovups(zmm_tmp, zmm_bk);
    h->vpcmpub(k_mask, zmm_tmp, table_val(bit8_64), _cmp_lt_os);
    h->vpermb(zmm_src | k_mask, zmm_tmp, table_val(bit8_lut_term, i * 16));
    h->vpsubusb(zmm_bk, zmm_bk, table_val(bit8_64));
    h->vpaddusb(zmm_bk | k_mask, zmm_bk, table_val(bit8_255));
  }
}

void jit_eltwise_injector::bit16_lut_compute_vector_fwd(const Zmm& zmm_src) {
  // regs renaming.
  Zmm zmm_bk = zmm_aux0;
  if (cur_postop_attr_.dt == data_type::u8 || cur_postop_attr_.dt == data_type::s8)
    h->vpmovzxbw(zmm_src, Ymm(zmm_src.getIdx()));  // zeropadding
  h->vmovups(zmm_bk, zmm_src);
  // calculate look-up times, each zmm reg can store 32 terms, so we will execute 8 time lookup operate.
  for (int i = 0; i < 8; i++) {
    h->vmovups(zmm_tmp, zmm_bk);
    h->vpcmpuw(k_mask, zmm_tmp, table_val(bit16_32), _cmp_lt_os);
    // we will process 16 byte per instruction.
    h->vpermw(zmm_src | k_mask, zmm_tmp, table_val(bit16_lut_term, i * 16));
    if (i != 7) {
      h->vpsubusw(zmm_bk, zmm_bk, table_val(bit16_32));
      h->vpaddusw(zmm_bk | k_mask, zmm_bk, table_val(bit16_255));
    }
  }
}

void jit_eltwise_injector::linear_compute_vector_fwd(const Zmm& zmm_src) {
  auto key = get_attr_idx_key(cur_postop_attr_);
  h->vmovups(zmm_aux0, table_val(alpha, alpha_idx_map[key]));
  h->vfmadd213ps(zmm_src, zmm_aux0, table_val(beta, beta_idx_map[key]));
}

void jit_eltwise_injector::low_precision_exp_compute_vector_fwd(const Zmm& zmm_src) {
  h->exp_approx_f32(zmm_src, zmm_src, table_val(exp_log2ef), table_val(ln2f),  //
                    table_val(low_precision_exp_const_v0), table_val(low_precision_exp_const_v1),
                    table_val(low_precision_exp_const_v2), {zmm_aux1, zmm_aux2});
}

void jit_eltwise_injector::swish_compute_vector_fwd(const Zmm& zmm_src) {
  auto key = get_attr_idx_key(cur_postop_attr_);
  h->vmovups(zmm_aux0, zmm_src);
  h->vmulps(zmm_aux0, zmm_aux0, table_val(alpha, alpha_idx_map[key]));
  low_precision_exp_compute_vector_fwd(zmm_aux0);
  h->vaddps(zmm_aux0, zmm_aux0, table_val(one));
  h->vrcp14ps(zmm_aux0, zmm_aux0);
  h->vmulps(zmm_src, zmm_src, zmm_aux0);
}

void jit_eltwise_injector::quantize_compute_vector_fwd(const Zmm& zmm_src) {
  auto key = get_attr_idx_key(cur_postop_attr_);
  h->vmovups(zmm_aux0, table_val(scale, scale_idx_map[key]));
  h->vfmadd213ps(zmm_src, zmm_aux0, table_val(alpha, alpha_idx_map[key]));
  if (cur_postop_attr_.dt == data_type::u8) {
    h->vcvtps2udq(zmm_src, zmm_src);  // fp32->u32
    h->vpmaxsd(zmm_src, zmm_src, table_val(zero));
  } else if (cur_postop_attr_.dt == data_type::s8) {
    h->vcvtps2dq(zmm_src, zmm_src);  // fp32->s32
  } else {
    SPARSE_LOG(FATAL) << "quant op only support s8/u8 dt";
  }
}

void jit_eltwise_injector::dequantize_compute_vector_fwd(const Zmm& zmm_src) {
  if (cur_postop_attr_.dt == data_type::u8)
    h->vpmovzxbd(zmm_src, Xmm(zmm_src.getIdx()));  // u8->s32
  else
    h->vpmovsxbd(zmm_src, Xmm(zmm_src.getIdx()));  // s8->s32
  h->vcvtdq2ps(zmm_src, zmm_src);
  auto key = get_attr_idx_key(cur_postop_attr_);  // s32->f32
  h->vsubps(zmm_src, zmm_src, table_val(alpha, alpha_idx_map[key]));
  h->vmulps(zmm_src, zmm_src, table_val(scale, scale_idx_map[key]));
}

void jit_eltwise_injector::tanh_compute_vector_fwd(const Zmm& zmm_src) {
  // register mapping
  Zmm zmm_dst = zmm_aux1, zmm_src_shift = zmm_aux1, zmm_coeff = zmm_aux1, zmm_pol = zmm_aux2, zmm_indices = zmm_aux3,
      zmm_src_original = zmm_aux4, zmm_sign = zmm_aux4;

  const int tanh_n_polynomials = 32;

  // We split the positive domain in 33 intervals:
  // a) [0; linear_ubound]: in this interval tanh(x) = x
  // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
  //    half binade
  // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
  //    one interval for each half binade, there are 29 of those
  // d) [0x1.0p3; saturation_ubound]:
  //    This interval spans part of a half binade
  // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
  // For b-d, we need 31 polynomials and will do a table lookup for those.
  // To simplify the logic, we will also put a) in the table.
  auto coeffs_address = [&](int coeff_off, int off = 0) {
    return table_val(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
  };
  auto gather_coefficient = [&](Zmm vmm_coeff, int coeff_idx, Zmm vmm_pol_idx) {
    Zmm zmm_coeff(vmm_coeff.getIdx());
    Zmm zmm_pol_idx(vmm_pol_idx.getIdx());
    h->vmovups(zmm_coeff, coeffs_address(coeff_idx, 0));
    h->vpermt2ps(zmm_coeff, zmm_pol_idx, coeffs_address(coeff_idx, 16));
  };

  // because tanh(x) = -tanh(-x), we extract sign to make x postive
  // and reapply sign at the end
  h->vmovups(zmm_src_original, zmm_src);
  h->vpandd(zmm_src, zmm_src, table_val(positive_mask));

  // We compute the indices for the table lookup
  h->vmovups(zmm_indices, zmm_src);
  h->vpsubd(zmm_indices, zmm_indices, table_val(tanh_idx_bias));
  h->vpandd(zmm_indices, zmm_indices, table_val(tanh_idx_mask));
  h->vpsrld(zmm_indices, zmm_indices, 22);

  // we do the argument reduction
  h->vmovups(zmm_src_shift, zmm_src);
  h->vpandd(zmm_src_shift, zmm_src_shift, table_val(tanh_idx_mask));
  h->vsubps(zmm_src, zmm_src, zmm_src_shift);

  // we gather and evaluate the polynonials
  gather_coefficient(zmm_pol, 6, zmm_indices);
  for (int deg = 5; deg >= 0; --deg) {
    gather_coefficient(zmm_coeff, deg, zmm_indices);
    h->vfmadd213ps(zmm_pol, zmm_src, zmm_coeff);
  }

  // we restore src with cleared sign, and keep sign
  h->vmovups(zmm_src, zmm_src_original);
  h->vpandd(zmm_sign, zmm_sign, table_val(sign_mask));
  h->vpandd(zmm_src, zmm_src, table_val(positive_mask));

  // Now we blend the results
  // [saturation_ubound; +inf[ : we return +/- 1
  h->vmovups(zmm_dst, table_val(one));
  // [linear_ubound; saturation_lbound] : we return +/- P(x)
  h->vmovups(zmm_mask, table_val(tanh_saturation_lbound));
  h->vcmpps(k_mask, zmm_mask, zmm_src, _cmp_nle_us);
  h->vblendmps(zmm_dst | k_mask, zmm_dst, zmm_pol);
  // [0; linear_ubound]  : we return x
  h->vmovups(zmm_mask, table_val(tanh_linear_ubound));
  h->vcmpps(k_mask, zmm_mask, zmm_src, _cmp_nle_us);
  h->vblendmps(zmm_dst | k_mask, zmm_dst, zmm_src);

  // We reapply the sign and return
  h->vpxord(zmm_dst, zmm_dst, zmm_sign);
  h->vmovups(zmm_src, zmm_dst);
}

void jit_eltwise_injector::gelu_compute_vector_fwd(const Zmm& zmm_src) {
  h->vmovups(zmm_aux0, zmm_src);

  // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
  h->vmulps(zmm_src, zmm_src, zmm_src);
  h->vmovups(zmm_aux1, table_val(gelu_tanh_fitting_const));
  h->vfmadd213ps(zmm_src, zmm_aux1, table_val(one));
  h->vmulps(zmm_src, zmm_src, zmm_aux0);
  h->vmulps(zmm_src, zmm_src, table_val(gelu_tanh_sqrt_two_over_pi));

  // compute tanh(G(x))
  tanh_compute_vector_fwd(zmm_src);

  // compute 0.5 * x * (1 + tanh(G(x)))
  h->vaddps(zmm_src, zmm_src, table_val(one));
  h->vmulps(zmm_src, zmm_src, table_val(half));
  h->vmulps(zmm_src, zmm_src, zmm_aux0);
}

void jit_eltwise_injector::exp_compute_vector_fwd(const Zmm& zmm_src) {
  /* exp code */
  h->vcmpps(k_mask, zmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);
  h->vminps(zmm_src, zmm_src, table_val(exp_ln_flt_max_f));
  h->vmaxps(zmm_src, zmm_src, table_val(exp_ln_flt_min_f));
  h->vmovups(zmm_aux1, zmm_src);
  h->vmulps(zmm_src, zmm_src, table_val(exp_log2ef));
  h->vaddps(zmm_src, zmm_src, table_val(half));
  h->vrndscaleps(zmm_aux2, zmm_src, _op_floor & 0x3);

  // keep zmm_src = fx for further computations
  h->vmovups(zmm_src, zmm_aux2);

  // x = x - fx * ln2
  h->vfnmadd231ps(zmm_aux1, zmm_aux2, table_val(ln2f));

  // We do not count 2^n here, because n can reach 128 and 2^128 is not
  // representable by fp32, so to get around this problem, instead of computing
  // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
  // and 2 are numbers representable in fp32.

  // compute 2^(n-1)
  h->vsubps(zmm_src, zmm_src, table_val(one));
  h->vcvtps2dq(zmm_aux2, zmm_src);
  h->vpaddd(zmm_aux2, zmm_aux2, table_val(exponent_bias));
  h->vpslld(zmm_aux2, zmm_aux2, n_mantissa_bits);

  // use zmm_src as tmp zmm_zero when applying mask
  h->vxorps(zmm_src, zmm_src, zmm_src);

  // set zeroes at those points which were < log(FLT_MIN)
  h->vblendmps(zmm_aux2 | k_mask, zmm_aux2, zmm_src);

  // compute polynomial
  h->vmovups(zmm_src, table_val(exp_pol, 4));
  h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 3));
  h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 2));
  h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 1));
  h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 0));
  h->vfmadd213ps(zmm_src, zmm_aux1, table_val(one));

  // y = y * 2^n

  h->vmulps(zmm_src, zmm_src, zmm_aux2);
  h->vmulps(zmm_src, zmm_src, table_val(two));
}

void jit_eltwise_injector::relu_compute_vector_fwd(const Zmm& zmm_src) {
  auto key = get_attr_idx_key(cur_postop_attr_);
  h->vmovups(zmm_aux1, zmm_src);
  h->vcmpps(k_mask, zmm_src, table_val(zero), _cmp_nle_us);
  h->vmulps(zmm_src, zmm_src, table_val(alpha, alpha_idx_map[key]));  // alpha=0 by default.
  h->vblendmps(zmm_src | k_mask, zmm_src, zmm_aux1);
}

void jit_eltwise_injector::escape_regs(reg_type type, int reg_idx) {
  auto iter = used_regs.find(type);
  if (iter != used_regs.end()) {
    iter->second.insert(reg_idx);
  } else {
    used_regs.insert(std::pair<reg_type, std::set<int>>(type, {reg_idx}));
  }
}

void jit_eltwise_injector::escape_erase(reg_type type, int reg_idx) {
  auto iter = used_regs.find(type);
  if (iter != used_regs.end()) {
    if (reg_idx != -1)
      iter->second.erase(reg_idx);
    else
      iter->second.clear();
  }
}

void jit_eltwise_injector::init_tb_allocate_set(const std::vector<postop_attr>& postop_attrs) {
  reg64_tb_allocate.insert(&p_table);
  for (auto&& i : postop_attrs) {
    if (i.dt == data_type::bf16) {
      reg64_tb_allocate.insert(&reg64_tmp);
      zmm_tb_allocate.insert(&zmm_tmp);
      zmm_tb_allocate.insert(&zmm_aux0);
      mask_tb_allocate.insert(&k_mask);
    }

    if (i.op_alg == postop_alg::low_precision_exp) {
      zmm_tb_allocate.insert(&zmm_aux1);
      zmm_tb_allocate.insert(&zmm_aux2);
    }

    if (i.op_alg == postop_alg::swish) {
      zmm_tb_allocate.insert(&zmm_aux0);
      zmm_tb_allocate.insert(&zmm_aux1);
      zmm_tb_allocate.insert(&zmm_aux2);
    }

    if (i.op_alg == postop_alg::exp) {
      mask_tb_allocate.insert(&k_mask);
      zmm_tb_allocate.insert(&zmm_aux1);
      zmm_tb_allocate.insert(&zmm_aux2);
    }

    if (i.op_alg == postop_alg::gelu) {
      zmm_tb_allocate.insert(&zmm_aux0);
      zmm_tb_allocate.insert(&zmm_aux1);
      zmm_tb_allocate.insert(&zmm_aux2);
      zmm_tb_allocate.insert(&zmm_aux3);
      zmm_tb_allocate.insert(&zmm_aux4);
      zmm_tb_allocate.insert(&zmm_mask);
      mask_tb_allocate.insert(&k_mask);
    }

    if (i.op_alg == postop_alg::tanh) {
      zmm_tb_allocate.insert(&zmm_aux1);
      zmm_tb_allocate.insert(&zmm_aux2);
      zmm_tb_allocate.insert(&zmm_aux3);
      zmm_tb_allocate.insert(&zmm_aux4);
      zmm_tb_allocate.insert(&zmm_mask);
      mask_tb_allocate.insert(&k_mask);
    }

    if (i.op_alg == postop_alg::relu) {
      zmm_tb_allocate.insert(&zmm_aux1);
      mask_tb_allocate.insert(&k_mask);
    }

    if (i.op_alg == postop_alg::linear) {
      zmm_tb_allocate.insert(&zmm_aux0);
    }

    if (i.op_alg == postop_alg::quantize) {
      zmm_tb_allocate.insert(&zmm_aux0);
    }

    if (i.op_alg == postop_alg::eltop_int_lut) {
      zmm_tb_allocate.insert(&zmm_aux0);
      zmm_tb_allocate.insert(&zmm_tmp);
      mask_tb_allocate.insert(&k_mask);
      // return directly,the reason is same as the comment in compute_vector func,bit8-lut case
      return;
    }
  }
}

template <typename REG_TYPE>
void jit_eltwise_injector::escape_from_rp(reg_type type, regs_pool* rp) {
  auto idx = rp->get_next<REG_TYPE>();
  for (int i = 0; i < idx; i++) this->escape_regs(type, rp->map_reg_idx<REG_TYPE>(i));
}

void jit_eltwise_injector::escape_rp_all_type(regs_pool* rp) {
  escape_from_rp<Reg64>(reg_type::reg64, rp);
  escape_from_rp<Opmask>(reg_type::mask, rp);
  escape_from_rp<Zmm>(reg_type::zmm, rp);
}

void jit_eltwise_injector::assign_regs() {
  std::vector<Xbyak::Reg*> reg64_allocate_vec;
  std::vector<Xbyak::Reg*> mask_allocate_vec;
  std::vector<Xbyak::Reg*> zmm_allocate_vec;
  reg64_allocate_vec.assign(reg64_tb_allocate.begin(), reg64_tb_allocate.end());
  mask_allocate_vec.assign(mask_tb_allocate.begin(), mask_tb_allocate.end());
  zmm_allocate_vec.assign(zmm_tb_allocate.begin(), zmm_tb_allocate.end());

  auto allocate_regs = [&](reg_type reg_type, int max_reg_idx,
                           std::unordered_map<enum reg_type, std::set<int>>::const_iterator iter,
                           std::vector<Xbyak::Reg*> tb_allocate_regs) {
    int allocate_idx = 0;
    std::set<int> used_reg_idxs = {};
    if (iter != used_regs.end()) used_reg_idxs = iter->second;
    while (tb_allocate_regs.size() != 0) {
      while (used_reg_idxs.count(allocate_idx) != 0) allocate_idx++;
      if (allocate_idx > max_reg_idx)
        SPARSE_LOG(FATAL) << "jit_eltwise allocate_regs error:too many registers be used in front op.";

      Xbyak::Reg* reg = tb_allocate_regs.back();
      if (reg_type == reg_type::mask) {
        if (allocate_idx == 0) {
          allocate_idx++;
          continue;
        }
        *reg = Xbyak::Opmask(allocate_idx);
      } else if (reg_type == reg_type::zmm) {
        *reg = Zmm(allocate_idx);
      } else if (reg_type == reg_type::reg64) {
        // avoid allocate special usage registers such as rsp.front op dose not need to tell injector the usage
        // information of these regs.
        using Operand = Xbyak::Operand;
        if (allocate_idx == Operand::RCX || allocate_idx == Operand::RDX || allocate_idx == Operand::RSI ||
            allocate_idx == Operand::RDI || allocate_idx == Operand::RSP) {
          allocate_idx++;
          continue;
        }
        *reg = Xbyak::Reg64(allocate_idx);
      }
      tb_allocate_regs.pop_back();
      allocate_idx++;
    }
  };
  allocate_regs(reg_type::reg64, max_reg64_idx, used_regs.find(reg_type::reg64), reg64_allocate_vec);
  allocate_regs(reg_type::mask, max_mask_idx, used_regs.find(reg_type::mask), mask_allocate_vec);
  allocate_regs(reg_type::zmm, max_zmm_idx, used_regs.find(reg_type::zmm), zmm_allocate_vec);
}

void jit_eltwise_injector::prepare_table() {
  h->align(64);
  h->L(l_table);
  SPARSE_LOG_IF(FATAL, sizeof(table_entry_val_t) != 4) << "sizeof(table_entry_val_t) should be 4";

  for (auto it = entry_map.begin(); it != entry_map.end(); it++) {
    const auto& te = (*it).second;
    const auto len = te.bcast ? 64u : sizeof(table_entry_val_t);
    for (size_t d = 0; d < len; d += sizeof(table_entry_val_t)) h->dd(te.val);
  }
}

uint32_t jit_eltwise_injector::get_bit16_lut_term(int integer, const std::vector<postop_attr>& postop_attrs,
                                                  data_type output_dt) {
  SPARSE_LOG_IF(FATAL, output_dt != data_type::bf16) << "only support bf16 now";
  uint32_t ans = 0;
  bfloat16_t* u16 = new bfloat16_t;
  for (int i = 0; i < 2; i++) {
    *u16 = apply_postop_list(integer + i, postop_attrs);
    ans |= (u16->data) << (i * 16);
  }
  delete u16;
  return ans;
}

uint32_t jit_eltwise_injector::get_bit8_lut_term(int integer, const std::vector<postop_attr>& postop_attrs,
                                                 data_type output_dt) {
  uint32_t ans = 0;
  uint8_t* u8 = new uint8_t(0);
  int8_t* s8 = new int8_t(0);
  uint8_t* cvt = nullptr;
  for (int i = 0; i < 4; i++) {
    if (output_dt == data_type::s8) {
      *s8 = apply_postop_list(integer + i, postop_attrs);
      cvt = reinterpret_cast<uint8_t*>(s8);
      ans |= *cvt << (i * 8);
    } else {
      *u8 = apply_postop_list(integer + i, postop_attrs);
      ans |= *u8 << (i * 8);
    }
  }
  delete u8;
  delete s8;
  return ans;
}

std::string jit_eltwise_injector::get_attr_idx_key(const postop_attr& attr) {
  std::string result;
  switch (attr.op_alg) {
    case postop_alg::quantize:
      result += "quantize";
      break;
    case postop_alg::dequantize:
      result += "dequantize";
      break;
    case postop_alg::linear:
      result += "linear";
      break;
    case postop_alg::relu:
      result += "relu";
      break;
    case postop_alg::swish:
      result += "swish";
      break;
    default:
      std::runtime_error("this alg_type do not need alpha/beta/scale.");
  }
  result += "+" + std::to_string(attr.alpha);
  result += "+" + std::to_string(attr.beta);
  result += "+" + std::to_string(attr.scale);
  return result;
}

void jit_eltwise_injector::register_table_entries(const std::vector<postop_attr>& postop_attrs) {
  static const table_t common_values{{zero, {0x00000000, true}},      {half, {0x3f000000, true}},
                                     {one, {0x3f800000, true}},       {two, {0x40000000, true}},
                                     {minus_one, {0xbf800000, true}}, {minus_two, {0xc0000000, true}},
                                     {ln2f, {0x3f317218, true}},      {positive_mask, {0x7fffffff, true}},
                                     {sign_mask, {0x80000000, true}}, {exponent_bias, {0x0000007f, true}}};

  static const table_t low_precision_exp_consts{
      {low_precision_exp_const_v0, {bit_cast<uint32_t>(exp_approx_f32_coeff[0]), true}},
      {low_precision_exp_const_v1, {bit_cast<uint32_t>(exp_approx_f32_coeff[1]), true}},
      {low_precision_exp_const_v2, {bit_cast<uint32_t>(exp_approx_f32_coeff[2]), true}},
  };

  static const table_t bit8_lut_consts{{bit8_64, {0x40404040, true}}, {bit8_255, {0xffffffff, true}}};

  static const table_t bit16_lut_consts{{bit16_32, {0x00200020, true}}, {bit16_255, {0x00ff00ff, true}}};

  static const table_t exchange_zmm_low256_high256_const{
      {exchange_zmm_low256_high256, {0x00000000, false}}, {exchange_zmm_low256_high256, {0x00000000, false}},
      {exchange_zmm_low256_high256, {0x00000000, false}}, {exchange_zmm_low256_high256, {0x00000000, false}},
      {exchange_zmm_low256_high256, {0x00000000, false}}, {exchange_zmm_low256_high256, {0x00000000, false}},
      {exchange_zmm_low256_high256, {0x00000000, false}}, {exchange_zmm_low256_high256, {0x00000000, false}},
      {exchange_zmm_low256_high256, {0x00000010, false}}, {exchange_zmm_low256_high256, {0x00000011, false}},
      {exchange_zmm_low256_high256, {0x00000012, false}}, {exchange_zmm_low256_high256, {0x00000013, false}},
      {exchange_zmm_low256_high256, {0x00000014, false}}, {exchange_zmm_low256_high256, {0x00000015, false}},
      {exchange_zmm_low256_high256, {0x00000016, false}}, {exchange_zmm_low256_high256, {0x00000017, false}}};

  static const table_t exp_consts{
      {exp_log2ef, {0x3fb8aa3b, true}}, {exp_ln_flt_max_f, {0x42b17218, true}}, {exp_ln_flt_min_f, {0xc2aeac50, true}}};

  static const table_t exp_polynomial{
      // p0 = 1.0f
      {exp_pol, {0x3f7ffffb, true}},  // p1 = 0.999999701f
      {exp_pol, {0x3efffee3, true}},  // p2 = 0.499991506f
      {exp_pol, {0x3e2aad40, true}},  // p3 = 0.166676521f
      {exp_pol, {0x3d2b9d0d, true}},  // p4 = 0.0418978221f
      {exp_pol, {0x3c07cfce, true}}   // p5 = 0.00828929059f
  };

  static const table_t gelu_tanh_const{{gelu_tanh_fitting_const, {0x3d372713, true}},
                                       {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
                                       {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
                                       {gelu_tanh_flt_max_x, {0x4154C480, true}},
                                       {gelu_tanh_flt_min_x, {0xC154C480, true}}};

  // tanh(x) constants for four interval approximation
  static const table_t tanh_consts{{tanh_idx_bias, {0x39800000, true}},
                                   {tanh_idx_mask, {0xffc00000, true}},
                                   {tanh_linear_ubound, {0x39ddb3d7, true}},
                                   {tanh_saturation_lbound, {0x41102cb3, true}}};

  // tanh(x) polynomial approximation
  // For each coefficient, there is 32 entries
  static const table_t tanh_polynomial_table{
      // coefficients of degree 0
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0x39bfffff, false}},
      {tanh_pol_table, {0x39ffffff, false}},
      {tanh_pol_table, {0x3a3ffffe, false}},
      {tanh_pol_table, {0x3a7ffffb, false}},
      {tanh_pol_table, {0x3abffff7, false}},
      {tanh_pol_table, {0x3affffeb, false}},
      {tanh_pol_table, {0x3b3fffdc, false}},
      {tanh_pol_table, {0x3b7fffab, false}},
      {tanh_pol_table, {0x3bbfff70, false}},
      {tanh_pol_table, {0x3bfffeab, false}},
      {tanh_pol_table, {0x3c3ffdc0, false}},
      {tanh_pol_table, {0x3c7ffaab, false}},
      {tanh_pol_table, {0x3cbff701, false}},
      {tanh_pol_table, {0x3cffeaad, false}},
      {tanh_pol_table, {0x3d3fdc08, false}},
      {tanh_pol_table, {0x3d7faacd, false}},
      {tanh_pol_table, {0x3dbf7081, false}},
      {tanh_pol_table, {0x3dfeacc9, false}},
      {tanh_pol_table, {0x3e3dc7fd, false}},
      {tanh_pol_table, {0x3e7acbf5, false}},
      {tanh_pol_table, {0x3eb77a9f, false}},
      {tanh_pol_table, {0x3eec9a9f, false}},
      {tanh_pol_table, {0x3f22991f, false}},
      {tanh_pol_table, {0x3f42f7d6, false}},
      {tanh_pol_table, {0x3f67b7cc, false}},
      {tanh_pol_table, {0x3f76ca83, false}},
      {tanh_pol_table, {0x3f7ebbe9, false}},
      {tanh_pol_table, {0x3f7fd40c, false}},
      {tanh_pol_table, {0x3f7fff32, false}},
      {tanh_pol_table, {0x3f7ffffc, false}},
      {tanh_pol_table, {0x3f800000, false}},
      // coefficients of degree 1
      {tanh_pol_table, {0x3f800000, false}},
      {tanh_pol_table, {0x3f800018, false}},
      {tanh_pol_table, {0x3f7fffe8, false}},
      {tanh_pol_table, {0x3f7fffda, false}},
      {tanh_pol_table, {0x3f7fffdc, false}},
      {tanh_pol_table, {0x3f7fffdc, false}},
      {tanh_pol_table, {0x3f7fffac, false}},
      {tanh_pol_table, {0x3f7fff70, false}},
      {tanh_pol_table, {0x3f7ffeec, false}},
      {tanh_pol_table, {0x3f7ffdc0, false}},
      {tanh_pol_table, {0x3f7ffbed, false}},
      {tanh_pol_table, {0x3f7ff704, false}},
      {tanh_pol_table, {0x3f7feff5, false}},
      {tanh_pol_table, {0x3f7fdbca, false}},
      {tanh_pol_table, {0x3f7fbfff, false}},
      {tanh_pol_table, {0x3f7f7041, false}},
      {tanh_pol_table, {0x3f7f009b, false}},
      {tanh_pol_table, {0x3f7dc36c, false}},
      {tanh_pol_table, {0x3f7c0aa8, false}},
      {tanh_pol_table, {0x3f7734b8, false}},
      {tanh_pol_table, {0x3f70a4de, false}},
      {tanh_pol_table, {0x3f5f1fd8, false}},
      {tanh_pol_table, {0x3f495493, false}},
      {tanh_pol_table, {0x3f18b9ec, false}},
      {tanh_pol_table, {0x3ed706cb, false}},
      {tanh_pol_table, {0x3e390b06, false}},
      {tanh_pol_table, {0x3d90b11f, false}},
      {tanh_pol_table, {0x3c21a053, false}},
      {tanh_pol_table, {0x3aaf7fdb, false}},
      {tanh_pol_table, {0x37ccc1a3, false}},
      {tanh_pol_table, {0x355c6733, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 2
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0xbe4e0ff1, false}},
      {tanh_pol_table, {0x3d25b1b1, false}},
      {tanh_pol_table, {0x3d6b6dab, false}},
      {tanh_pol_table, {0x3c9fb1d5, false}},
      {tanh_pol_table, {0xbabff06f, false}},
      {tanh_pol_table, {0x3c07b3f6, false}},
      {tanh_pol_table, {0xbb3fc1bc, false}},
      {tanh_pol_table, {0x3a9f5921, false}},
      {tanh_pol_table, {0xbbbf06f2, false}},
      {tanh_pol_table, {0xbbb0f402, false}},
      {tanh_pol_table, {0xbc47db9e, false}},
      {tanh_pol_table, {0xbc73d5e7, false}},
      {tanh_pol_table, {0xbca25bda, false}},
      {tanh_pol_table, {0xbcfca780, false}},
      {tanh_pol_table, {0xbd40e07c, false}},
      {tanh_pol_table, {0xbd7dab03, false}},
      {tanh_pol_table, {0xbdbe4a0f, false}},
      {tanh_pol_table, {0xbdfb14a5, false}},
      {tanh_pol_table, {0xbe36cc8d, false}},
      {tanh_pol_table, {0xbe6bd102, false}},
      {tanh_pol_table, {0xbe9fe7c5, false}},
      {tanh_pol_table, {0xbeba0f10, false}},
      {tanh_pol_table, {0xbec206a8, false}},
      {tanh_pol_table, {0xbea3c388, false}},
      {tanh_pol_table, {0xbe277d62, false}},
      {tanh_pol_table, {0xbd8b7960, false}},
      {tanh_pol_table, {0xbc209f49, false}},
      {tanh_pol_table, {0xbaad44ca, false}},
      {tanh_pol_table, {0xb7c6eeac, false}},
      {tanh_pol_table, {0xb663aa41, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 3
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0x45b3ae96, false}},
      {tanh_pol_table, {0xc414eb20, false}},
      {tanh_pol_table, {0xc450e02e, false}},
      {tanh_pol_table, {0xc3152b4e, false}},
      {tanh_pol_table, {0xbead2f56, false}},
      {tanh_pol_table, {0xc2162e02, false}},
      {tanh_pol_table, {0xbeb4bd5a, false}},
      {tanh_pol_table, {0xc11a59a4, false}},
      {tanh_pol_table, {0xbed2f507, false}},
      {tanh_pol_table, {0xc020d32c, false}},
      {tanh_pol_table, {0x3dd0f506, false}},
      {tanh_pol_table, {0xbf2a75e2, false}},
      {tanh_pol_table, {0xbff950e3, false}},
      {tanh_pol_table, {0xbed47334, false}},
      {tanh_pol_table, {0xbe809b8c, false}},
      {tanh_pol_table, {0xbeb64532, false}},
      {tanh_pol_table, {0xbe961a5b, false}},
      {tanh_pol_table, {0xbe9b63ac, false}},
      {tanh_pol_table, {0xbea0d4b2, false}},
      {tanh_pol_table, {0xbe828a77, false}},
      {tanh_pol_table, {0xbe378612, false}},
      {tanh_pol_table, {0xbdc20908, false}},
      {tanh_pol_table, {0x3d2d3957, false}},
      {tanh_pol_table, {0x3dd46e89, false}},
      {tanh_pol_table, {0x3db3f629, false}},
      {tanh_pol_table, {0x3d2c5e7b, false}},
      {tanh_pol_table, {0x3bd20403, false}},
      {tanh_pol_table, {0x3a59dfae, false}},
      {tanh_pol_table, {0x3770af45, false}},
      {tanh_pol_table, {0x372cc014, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 4
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0xcc981a1b, false}},
      {tanh_pol_table, {0x4a7edd3d, false}},
      {tanh_pol_table, {0x4ab1007c, false}},
      {tanh_pol_table, {0x48fedd9c, false}},
      {tanh_pol_table, {0x41a557b5, false}},
      {tanh_pol_table, {0x477ee32a, false}},
      {tanh_pol_table, {0x422557f5, false}},
      {tanh_pol_table, {0x45ff3ce4, false}},
      {tanh_pol_table, {0x42a55641, false}},
      {tanh_pol_table, {0x446e0867, false}},
      {tanh_pol_table, {0xc33dc19a, false}},
      {tanh_pol_table, {0x42915214, false}},
      {tanh_pol_table, {0x43af4fad, false}},
      {tanh_pol_table, {0x4110fe88, false}},
      {tanh_pol_table, {0xc1099b75, false}},
      {tanh_pol_table, {0x3fc8a8dc, false}},
      {tanh_pol_table, {0xbfbeaef5, false}},
      {tanh_pol_table, {0xbe365aad, false}},
      {tanh_pol_table, {0x3f4d9652, false}},
      {tanh_pol_table, {0x3ddfa08f, false}},
      {tanh_pol_table, {0x3e34e9b8, false}},
      {tanh_pol_table, {0x3e2d07a6, false}},
      {tanh_pol_table, {0x3dc63567, false}},
      {tanh_pol_table, {0x3cdaeb78, false}},
      {tanh_pol_table, {0xbcd17537, false}},
      {tanh_pol_table, {0xbc92829c, false}},
      {tanh_pol_table, {0xbb43ab99, false}},
      {tanh_pol_table, {0xb9b471dd, false}},
      {tanh_pol_table, {0xb6baad5a, false}},
      {tanh_pol_table, {0xb78bafc7, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 5
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0x52f688d5, false}},
      {tanh_pol_table, {0xd0505c72, false}},
      {tanh_pol_table, {0xd08f98e3, false}},
      {tanh_pol_table, {0xce505cc9, false}},
      {tanh_pol_table, {0xc7162b8a, false}},
      {tanh_pol_table, {0xcc5061d6, false}},
      {tanh_pol_table, {0xc7162bdf, false}},
      {tanh_pol_table, {0xca50b37f, false}},
      {tanh_pol_table, {0xc7162a3a, false}},
      {tanh_pol_table, {0xc8422086, false}},
      {tanh_pol_table, {0x471a714e, false}},
      {tanh_pol_table, {0xc5ece1f1, false}},
      {tanh_pol_table, {0xc70e3d90, false}},
      {tanh_pol_table, {0xc3eba94a, false}},
      {tanh_pol_table, {0x43e0c424, false}},
      {tanh_pol_table, {0xc21f4552, false}},
      {tanh_pol_table, {0x42217cc8, false}},
      {tanh_pol_table, {0x405e7dc4, false}},
      {tanh_pol_table, {0xc10dd401, false}},
      {tanh_pol_table, {0x3e96b602, false}},
      {tanh_pol_table, {0xbd1a6d2f, false}},
      {tanh_pol_table, {0xbd393883, false}},
      {tanh_pol_table, {0xbd674682, false}},
      {tanh_pol_table, {0xbd310016, false}},
      {tanh_pol_table, {0xb961e269, false}},
      {tanh_pol_table, {0x3ba32495, false}},
      {tanh_pol_table, {0x3a7680d5, false}},
      {tanh_pol_table, {0x38b3173c, false}},
      {tanh_pol_table, {0x35a9deea, false}},
      {tanh_pol_table, {0x375c3f2a, false}},
      {tanh_pol_table, {0x00000000, false}},
      // coefficients of degree 6
      {tanh_pol_table, {0x00000000, false}},
      {tanh_pol_table, {0xd8995ed1, false}},
      {tanh_pol_table, {0x558285ea, false}},
      {tanh_pol_table, {0x55b2cd69, false}},
      {tanh_pol_table, {0x53028625, false}},
      {tanh_pol_table, {0x4bc9991f, false}},
      {tanh_pol_table, {0x5082898a, false}},
      {tanh_pol_table, {0x4b4999b3, false}},
      {tanh_pol_table, {0x4e02c07c, false}},
      {tanh_pol_table, {0x4ac99764, false}},
      {tanh_pol_table, {0x4b72c822, false}},
      {tanh_pol_table, {0xca40c0e1, false}},
      {tanh_pol_table, {0x489413e4, false}},
      {tanh_pol_table, {0x49b12224, false}},
      {tanh_pol_table, {0x46134c4e, false}},
      {tanh_pol_table, {0xc60c2d57, false}},
      {tanh_pol_table, {0x43c83910, false}},
      {tanh_pol_table, {0xc3c872d1, false}},
      {tanh_pol_table, {0xc186bc9e, false}},
      {tanh_pol_table, {0x42325bc3, false}},
      {tanh_pol_table, {0xbf2ffa4a, false}},
      {tanh_pol_table, {0x3d9a203c, false}},
      {tanh_pol_table, {0xbc545a43, false}},
      {tanh_pol_table, {0xbae08fee, false}},
      {tanh_pol_table, {0x3c80225d, false}},
      {tanh_pol_table, {0x3b1fd1df, false}},
      {tanh_pol_table, {0xba36b9d1, false}},
      {tanh_pol_table, {0xb91de544, false}},
      {tanh_pol_table, {0xb71f100f, false}},
      {tanh_pol_table, {0xb408e2ed, false}},
      {tanh_pol_table, {0xb685fec8, false}},
      {tanh_pol_table, {0x00000000, false}},
  };

  auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val, const bool broadcast) {
    mapped_table_entry_t te{0, val, broadcast};
    entry_map.insert(std::make_pair(key, te));
  };

  auto push_entries_of = [&](const table_t& t) {
    for (auto it = t.begin(); it != t.end(); it++) {
      auto key = it->first;
      auto te = it->second;
      push_arg_entry_of(key, te.val, te.bcast);
    }
  };

  auto set_table_term_offset = [&]() {
    size_t off = 0;
    for (auto it = entry_map.begin(); it != entry_map.end(); it++) {
      auto& te = (*it).second;
      te.off = off;
      off += te.bcast ? 64u : sizeof(table_entry_val_t);
    }
  };

  struct need_t {
    explicit need_t(const std::vector<postop_attr>& postop_attrs) {
      for (auto&& attr : postop_attrs) {
        if (attr.dt == data_type::bf16) bf16_ = true;
        if (attr.op_alg == postop_alg::exp) exp_ = true;
        if (attr.op_alg == postop_alg::tanh) tanh_ = true;
        if (attr.op_alg == postop_alg::gelu) gelu_ = true;
        if (attr.op_alg == postop_alg::swish) swish_ = true;
        if (attr.op_alg == postop_alg::low_precision_exp) low_precision_exp_ = true;
      }
    }
    bool bf16_ = false;
    bool exp_ = false;
    bool tanh_ = false;
    bool gelu_ = false;
    bool low_precision_exp_ = false;
    bool swish_ = false;

    bool bf16() const { return bf16_; }
    bool exp() const { return exp_; }
    bool tanh() const { return tanh_; }
    bool gelu() const { return gelu_; }
    bool low_precision_exp() { return low_precision_exp_; }
    bool swish() const { return swish_; }
  };

  need_t need(postop_attrs);

  push_entries_of(common_values);

  // eltop_int_lut's alpha means the bitwidth
  if (postop_attrs.size() > 0 && postop_attrs[0].op_alg == postop_alg::eltop_int_lut) {
    // lut kernel first op must be dequantize
    SPARSE_LOG_IF(FATAL, postop_attrs.size() < 2) << "postop_attrs.size() < 2";
    SPARSE_LOG_IF(FATAL, postop_attrs[1].op_alg != postop_alg::dequantize) << "First op should not be dequantize";
    auto input_dt = postop_attrs[0].dt;
    auto output_dt = postop_attrs.back().dt;
    int table_bitwidth = postop_attrs[0].alpha;
    if (table_bitwidth == 8) {
      SPARSE_LOG_IF(FATAL, postop_attrs.size() < 3) << "postop_attrs.size() < 3";
      // if first op is bit8-lut,the last op must be quantize
      SPARSE_LOG_IF(FATAL, postop_attrs.back().op_alg != postop_alg::quantize)
          << "if first op is bit8-lut,the last op must be quantize";
      push_entries_of(bit8_lut_consts);
    } else {
      push_entries_of(bit16_lut_consts);
    }

    table_t bit_lut;
    auto register_bit_lut_entries = [&](int integer, int bitwidth) {
      uint32_t term = 0;
      switch (bitwidth) {
        case 8:
          term = get_bit8_lut_term(integer, postop_attrs, output_dt);
          break;
        case 16:
          term = get_bit16_lut_term(integer, postop_attrs, output_dt);
          break;
        default:
          SPARSE_LOG(ERROR) << "Unexpected bit width for LUT: " << bitwidth;
          break;
      }
      table_entry_t tmp = {term, false};
      switch (bitwidth) {
        case 8:
          bit_lut.insert(std::make_pair(bit8_lut_term, tmp));
          break;
        case 16:
          bit_lut.insert(std::make_pair(bit16_lut_term, tmp));
          break;
        default:
          SPARSE_LOG(ERROR) << "Unexpected bit width for LUT: " << bitwidth;
          break;
      }
    };

    if (input_dt == data_type::u8) {
      for (int i = 0; i < 256; i += 32 / table_bitwidth) register_bit_lut_entries(i, table_bitwidth);
    } else if (input_dt == data_type::s8) {
      for (int i = 0; i < 128; i += 32 / table_bitwidth) register_bit_lut_entries(i, table_bitwidth);
      for (int i = -128; i < 0; i += 32 / table_bitwidth) register_bit_lut_entries(i, table_bitwidth);
    } else {
      SPARSE_LOG(FATAL) << "eltop_int_lut algo only support s8/u8 data_type as input.";
    }
    push_entries_of(bit_lut);

    set_table_term_offset();
    // once the head of postop-chain is bit8-lut, then the remain ops do not need to compute so we do not need to
    // prepare LUT entries.
    return;
  }

  auto gain_fin_const = [&](float alpha, float beta, float scale, postop_alg alg) {
    switch (alg) {
      case postop_alg::swish:
        alpha *= -1;
        break;
      case postop_alg::quantize:
        scale = 1.f / scale;
        break;
      default:
        break;
    }
    return std::vector<float>({alpha, beta, scale});
  };

  size_t alpha_idx = 0, beta_idx = 0, scale_idx = 0;
  for (auto&& attr : postop_attrs) {
    auto fin_const = gain_fin_const(attr.alpha, attr.beta, attr.scale, attr.op_alg);
    mapped_table_entry_t alpha_entry{alpha_idx, bit_cast<uint32_t, float>(fin_const[0]), true};
    mapped_table_entry_t beta_entry{beta_idx, bit_cast<uint32_t, float>(fin_const[1]), true};
    mapped_table_entry_t scale_entry{scale_idx, bit_cast<uint32_t, float>(fin_const[2]), true};
    auto key = get_attr_idx_key(attr);
    if (attr.op_alg == postop_alg::quantize || attr.op_alg == postop_alg::dequantize) {
      alpha_idx_map[key] = alpha_idx++;
      scale_idx_map[key] = scale_idx++;
      entry_map.insert(std::make_pair(alpha, alpha_entry));
      entry_map.insert(std::make_pair(scale, scale_entry));
    }
    if (attr.op_alg == postop_alg::linear) {
      alpha_idx_map[key] = alpha_idx++;
      beta_idx_map[key] = beta_idx++;
      entry_map.insert(std::make_pair(alpha, alpha_entry));
      entry_map.insert(std::make_pair(beta, beta_entry));
    }
    if (attr.op_alg == postop_alg::relu || attr.op_alg == postop_alg::swish) {
      alpha_idx_map[key] = alpha_idx++;
      entry_map.insert(std::make_pair(alpha, alpha_entry));
    }
  }

  if (need.bf16()) push_entries_of(exchange_zmm_low256_high256_const);
  if (need.exp()) {
    push_entries_of(exp_consts);
    push_entries_of(exp_polynomial);
  }
  if (need.low_precision_exp() || need.swish()) {
    push_entries_of(exp_consts);
    push_entries_of(low_precision_exp_consts);
  }
  if (need.tanh() || need.gelu()) {
    push_entries_of(tanh_consts);
    push_entries_of(tanh_polynomial_table);
  }
  if (need.gelu()) push_entries_of(gelu_tanh_const);

  set_table_term_offset();
}
}  // namespace jd
