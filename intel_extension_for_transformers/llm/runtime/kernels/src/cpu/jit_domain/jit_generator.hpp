//  Copyright (c) 2021 Intel Corporation
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

#ifndef ENGINE_SPARSELIB_SRC_CPU_JIT_GENERATOR_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_JIT_GENERATOR_HPP_

#include <array>
#include <climits>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "src/cpu/cpu_isa.hpp"
#include "src/utils.hpp"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

using Tmm = Xbyak::Tmm;
using Zmm = Xbyak::Zmm;
using Ymm = Xbyak::Ymm;
using Xmm = Xbyak::Xmm;
using Reg64 = Xbyak::Reg64;
using Reg32 = Xbyak::Reg32;
using Opmask = Xbyak::Opmask;
using RegExp = Xbyak::RegExp;

constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    // https://stackoverflow.com/questions/18024672/what-registers-are-preserved-through-a-linux-x86-64-function-call
    // r12, r13, r14, r15, rbx, rsp, rbp are the callee-saved registers - they
    // have a "Yes" in the "Preserved across
    // function calls" column.
    // usually we use r12, r13, r14 for src.
    Xbyak::Operand::RBX, Xbyak::Operand::RBP, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};

#ifdef _WIN32
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX), abi_param2(Xbyak::Operand::RDX),
    abi_param3(Xbyak::Operand::R8), abi_param4(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RDI);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI), abi_param2(Xbyak::Operand::RSI),
    abi_param3(Xbyak::Operand::RDX), abi_param4(Xbyak::Operand::RCX), abi_param5(Xbyak::Operand::R8),
    abi_param6(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RCX);
#endif

// from: https://stackoverflow.com/questions/24299543/saving-the-xmm-register-before-function-call
#ifdef _WIN32
// https://docs.microsoft.com/en-us/cpp/build/x64-software-conventions?redirectedfrom=MSDN&view=msvc-170
// xmm6:xmm15 must be preserved as needed by caller
constexpr size_t xmm_to_preserve_start = 6;
constexpr size_t xmm_to_preserve = 10;
#else
// https://github.com/hjl-tools/x86-psABI/wiki/X86-psABI: page23
// on Linux those are temporary registers, and therefore don't have to be preserved
constexpr size_t xmm_to_preserve_start = 0;
constexpr size_t xmm_to_preserve = 0;
#endif

namespace jd {
class jit_eltwise_injector;

class jit_generator : public Xbyak::CodeGenerator {
 public:
  explicit jit_generator(size_t code_size = MAX_CODE_SIZE, void* code_ptr = nullptr)
      : Xbyak::CodeGenerator(code_size, (code_ptr == nullptr) ? Xbyak::AutoGrow : code_ptr) {}
  virtual ~jit_generator() {}
  void dump_asm();  // print assembly code

  friend class jit_eltwise_injector;  //  so that injector can access utility methods

  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

  // Set to O0 to avoid undefined behavour of reinterpret_cast
  // Ref: https://stackoverflow.com/questions/25131019/reinterpret-cast-to-function-pointer
  // #pragma GCC push_options
  // #pragma GCC optimize("O0")
  template <typename... T>
  inline void operator()(T... args) const {
    using func_ptr = void (*)(const T... args);
    auto fptr = reinterpret_cast<func_ptr>(jit_ker_);
    (*fptr)(std::forward<T>(args)...);
  }
  // #pragma GCC pop_options

  virtual bool create_kernel();

  template <typename T>
  Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base, T raw_offt, bool bcast = false);

  Xbyak::Address make_safe_addr(const Xbyak::Reg64& reg_out, size_t offt, const Xbyak::Reg64& tmp_reg,
                                bool bcast = false);
  Xbyak::Address EVEX_compress_addr_safe(const Xbyak::Reg64& base, size_t raw_offt, const Xbyak::Reg64& reg_offt,
                                         bool bcast = false);

 protected:
  // derived jit_domain implementation
  virtual void generate() = 0;
  const uint8_t* get_code();
  void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Xmm& x) { movdqu(addr, x); }
  void uni_vmovdqu(const Xbyak::Xmm& x, const Xbyak::Address& addr) { vmovdqu(x, addr); }
  void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Zmm& x) { vmovdqu32(addr, x); }

  void uni_vzeroupper() {
    // TODO(hengyu): handle non-avx case
    vzeroupper();
  }

  void bf16_cvt_fp32(Zmm zmm) {
    vpmovzxwd(zmm, Ymm(zmm.getIdx()));
    vpslld(zmm, zmm, 0x10);
  }

  void fp32_cvt_bf16(Zmm zmm) {
    if (isa_available(avx512_core_bf16)) {
      vcvtneps2bf16(Ymm(zmm.getIdx()), zmm);
    } else {
      vpsrld(zmm, zmm, 16);
      vpmovdw(Ymm(zmm.getIdx()), zmm);
    }
  }

  void preserve_xmm() {
    if (xmm_to_preserve) {
      sub(rsp, xmm_to_preserve * VEC);
      for (size_t i = 1; i < xmm_to_preserve + 1; ++i)
        uni_vmovdqu(ptr[rsp + (i - 1) * VEC], Xbyak::Xmm(xmm_to_preserve_start + i - 1));
    }
  }

  void preamble() {
    preserve_xmm();
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
      push(Xbyak::Reg64(abi_save_gpr_regs[i]));
    }
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
  }

  void recover_xmm() {
    if (xmm_to_preserve) {
      for (size_t i = 1; i < xmm_to_preserve + 1; ++i)
        uni_vmovdqu(Xbyak::Xmm(xmm_to_preserve_start + i - 1), ptr[rsp + (i - 1) * VEC]);
      add(rsp, xmm_to_preserve * VEC);
    }
  }

  void postamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
      pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    recover_xmm();
    uni_vzeroupper();
    ret();
  }

  /**
   * @brief Transpose 16x16 of 32bit elements stored in 16 ZMMs
   *
   * @param src the 16 ZMMs storing the matrix to transpose
   * @param tmp 16 ZMMs for temporary use
   * @param N the second dim of 16x16, could be less than 16 to save some cycles
   */
  void transpose_16x16_ps(const std::array<Xbyak::Zmm, 16UL>& src, const std::array<Xbyak::Zmm, 16UL>& tmp,
                          const int N = 16);
  /**
   * @brief Reduce dword in an vector register inplace
   *
   * @tparam OP2 type of the 2nd operand of the reduction op
   * @tparam OP3 type of the 3rd  operand of the reduction op
   * @param src vector register to perform reduction
   * @param tmp tmp vector register to help reduction
   * @param inst the reduction instruction
   *
   * @example reduce_dwords(src, dst, &CodeGenerator::vpmaxsd);
   * @example reduce_dwords(src, dst, std::bind(&CodeGenerator::vrangeps, this, _1, _2, _3, 0b1010));
   */
  template <typename FUNC, typename = typename std::enable_if<std::is_constructible<
                               std::function<void(const Xmm&, const Xmm&, const Xmm&)>, FUNC>::value>::type>
  inline void reduce_dwords(const Ymm& src, const Ymm& tmp, const FUNC inst) {
    SPARSE_LOG_IF(FATAL, src.getBit() != tmp.getBit()) << "Operand and tmp register should be of same type!";
    if (src.isZMM()) {
      vshuff32x4(tmp, src, src, 0x4E);  // 256-bit shuffle
      inst(src, src, tmp);
    }
    vshuff32x4(tmp, src, src, 0xB1);  // 128/256-bit shuffle
    inst(src, src, tmp);
    vshufps(tmp, src, src, 0x4E);  // 64/128-bit shuffle
    inst(src, src, tmp);
    vshufps(tmp, src, src, 0xB1);  // 32/64-bit shuffle
    inst(src, src, tmp);
  }
  template <typename OP2, typename OP3, typename CG,
            typename = typename std::enable_if<std::is_base_of<Xbyak::Operand, OP2>::value>::type,
            typename = typename std::enable_if<std::is_base_of<Xbyak::Operand, OP3>::value>::type,
            typename = typename std::enable_if<std::is_base_of<Xbyak::CodeGenerator, CG>::value>::type>
  inline void reduce_dwords(const Ymm& src, const Ymm& tmp, void (CG::*inst)(const Xmm&, const OP2&, const OP3&)) {
    reduce_dwords(src, tmp, std::bind(inst, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  }

  /**
   * @brief Elementwise reduce a set of vector registers and store the result to the first one
   *
   * @tparam OP2 type of the 2nd operand of the reduction op
   * @tparam OP3 type of the 3rd  operand of the reduction op
   * @param xs vector registers to perform reduction
   * @param inst the reduction instruction
   */
  template <typename FUNC, typename = std::enable_if_t<std::is_constructible<
                               std::function<void(const Xmm&, const Xmm&, const Xmm&)>, FUNC>::value>>
  inline void reduce_vmms(const Xmm* xs, const FUNC inst, size_t n) {
    for (size_t span = 1; span < n; span *= 2)
      for (size_t i = 0; i < n; i += span * 2) inst(xs[i], xs[i], xs[i + span]);
  }
  template <typename OP2, typename OP3, typename CG,
            typename = typename std::enable_if<std::is_base_of<Xbyak::Operand, OP2>::value>::type,
            typename = typename std::enable_if<std::is_base_of<Xbyak::Operand, OP3>::value>::type,
            typename = typename std::enable_if<std::is_base_of<Xbyak::CodeGenerator, CG>::value>::type>
  inline void reduce_vmms(const Xmm* xs, void (CG::*inst)(const Xmm&, const OP2&, const OP3&), size_t n) {
    reduce_vmms(xs, std::bind(inst, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), n);
  }
  template <typename FUNC, typename VMM>
  inline void reduce_vmms(const std::vector<VMM>& xs, const FUNC inst) {
    static_assert(std::is_base_of<Xmm, VMM>::value);
    reduce_vmms(xs.data(), inst, xs.size());
  }
  template <typename FUNC, typename VMM, size_t n>
  inline void reduce_vmms(const std::array<VMM, n>& xs, const FUNC inst) {
    static_assert(std::is_base_of<Xmm, VMM>::value);
    reduce_vmms(xs.data(), inst, n);
  }

  /**
   * @brief Perform approximated exp function; src and dst can be the same; log2e and ln2 can be zword_b in ZMM limited
   * use cases.
   *
   * Use together with `exp_approx_f32_coeff`, the corresponding coefficient of the 2nd-order polynomial approximating
   * function $f(x) = exp(x)$ where $x \in (-ln2, 0]$; idea from https://arxiv.org/abs/2101.01321
   */
  void exp_approx_f32(const Zmm& dst, const Zmm& src, const Xbyak::Operand& log2e, const Xbyak::Operand& ln2,
                      const Xbyak::Operand& coeff0, const Xbyak::Operand& coeff1, const Xbyak::Operand& coeff2,
                      const std::array<Zmm, 2>& tmp) {
    vmulps(tmp[0], src, log2e);        // x / ln2
    vrndscaleps(tmp[0], tmp[0], 0x2);  // round up
    const auto& z = tmp[0];
    vmulps(tmp[1], tmp[0], ln2);
    vsubps(tmp[1], src, tmp[1]);  // x mod ln2 (can we use fmsub?)
    vmovaps(dst, coeff1);
    vfmadd231ps(dst, tmp[1], coeff0);  // dst = f * c0 + c1
    vfmadd213ps(dst, tmp[1], coeff2);  // dst = (f * c0 + c1) * f + c2
    vscalefps(dst, dst, z);            // dst = exp(f) * 2^z
  }
  void exp_approx_f32(const Zmm& dst, const Zmm& src, const Xbyak::Operand& log2e, const Xbyak::Operand& ln2,
                      const std::array<Zmm, 3>& coeff, const std::array<Zmm, 2>& tmp) {
    exp_approx_f32(dst, src, log2e, ln2, coeff[0], coeff[1], coeff[2], tmp);
  }
  /**
   * @brief refer `exp_approx_f32`
   *
   * Note that exp_approx_f16_coeff can not be constexpr without compile-time bit-cast
   */
  void exp_approx_f16(const Zmm& dst, const Zmm& src, const Xbyak::Operand& log2e, const Xbyak::Operand& ln2,
                      const Xbyak::Operand& coeff0, const Xbyak::Operand& coeff1, const Xbyak::Operand& coeff2,
                      const std::array<Zmm, 2>& tmp) {
    vmulph(tmp[0], src, log2e);        // x / ln2
    vrndscaleph(tmp[0], tmp[0], 0x2);  // round up
    const auto& z = tmp[0];
    vmulph(tmp[1], tmp[0], ln2);
    vsubph(tmp[1], src, tmp[1]);  // x mod ln2 (can we use fmsub?)
    vmovaps(dst, coeff1);
    vfmadd231ph(dst, tmp[1], coeff0);  // dst = f * c0 + c1
    vfmadd213ph(dst, tmp[1], coeff2);  // dst = (f * c0 + c1) * f + c2
    vscalefph(dst, dst, z);            // dst = exp(f) * 2^z
  }
  void exp_approx_f16(const Zmm& dst, const Zmm& src, const Xbyak::Operand& log2e, const Xbyak::Operand& ln2,
                      const std::array<Zmm, 3>& coeff, const std::array<Zmm, 2>& tmp) {
    exp_approx_f16(dst, src, log2e, ln2, coeff[0], coeff[1], coeff[2], tmp);
  }

  /**
   * @brief AMX 2x2 bf16 tile product along K
   *
   * @param reg_ksize a read-only register with k-size in it
   * @param src0 a volatile register with src0 address at the beginning
   * @param src1 a volatile register with src1 address at the beginning
   * @param src0_stride a read-only register with src0_stride in it
   * @param src1_stride a read-only register with src1_stride in it
   * @param reg_tmp0 a volatile register for intermediate use
   * @param reg_tmp1 a volatile register for intermediate use
   * @param dst_addr a function mapping tile index to the address to store it
   */
  void tile_product_amx_bf16ps(const Xbyak::Operand& reg_ksize, const Reg64& src0, const Reg64& src1,
                               const Reg64& src0_stride, const Reg64& src1_stride, const Reg64& reg_tmp0,
                               const Reg64& reg_tmp1, std::function<Xbyak::Address(int, int)> dst_addr);

  /**
   * @brief Mov 1 ~ 31 bytes with AVX2
   *
   * @param dst destination address
   * @param src source address
   * @param bytes  number of bytes to move
   * @param tmp_xmm a volatile XMM register for intermediate use
   * @param tmp_r64 a volatile Reg64 register for intermediate use
   */
  void vmov_avx2(const RegExp& dst, const RegExp& src, const int bytes, const Xmm& tmp_xmm, const Reg64& tmp_r64);

  const uint8_t* jit_ker_ = nullptr;
  static constexpr uint64_t MAX_CODE_SIZE = 128 * 1024;
  static constexpr uint64_t BYTES_ZMM = 64;
  static constexpr uint64_t BYTES_YMM = 32;
  static constexpr uint64_t BYTES_XMM = 16;
  static constexpr uint64_t TMM_MAX_ROW = 16;
  static constexpr uint64_t TMM_MAX_COL = 64;
  static constexpr uint64_t BYTES_TMM = TMM_MAX_ROW * TMM_MAX_COL;
  static constexpr int VEC = 16;  // 512 bits of ZMM register divided by S32 bits.
  int callee_functions_code_size_ = 0;
  const size_t num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);
  const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * rax.getBit() / 8 + xmm_to_preserve * VEC;

  static int dump_idx;
};

constexpr std::array<float, 3> exp_approx_f32_coeff{0.35815147f, 0.96963238f, 1.f};
extern const std::array<float16_t, 3> exp_approx_f16_coeff;
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_JIT_GENERATOR_HPP_
