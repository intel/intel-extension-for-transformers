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

#ifndef ENGINE_SPARSELIB_INCLUDE_JIT_GENERATOR_HPP_
#define ENGINE_SPARSELIB_INCLUDE_JIT_GENERATOR_HPP_
#include <climits>
#include <fstream>
#include <string>
#include <utility>

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
#include "utils.hpp"

using Zmm = Xbyak::Zmm;
using Ymm = Xbyak::Ymm;
using Xmm = Xbyak::Xmm;
using Reg64 = Xbyak::Reg64;
using Opmask = Xbyak::Opmask;
using RegExp = Xbyak::RegExp;

using Zmm = Xbyak::Zmm;
using Ymm = Xbyak::Ymm;
using Xmm = Xbyak::Xmm;
using Reg64 = Xbyak::Reg64;
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
const size_t xmm_to_preserve_start = 6;
const size_t xmm_to_preserve = 10;
#else
// https://github.com/hjl-tools/x86-psABI/wiki/X86-psABI: page23
// on Linux those are temporary registers, and therefore don't have to be preserved
const size_t xmm_to_preserve_start = 0;
const size_t xmm_to_preserve = 0;
#endif

namespace jd {
class jit_generator : public Xbyak::CodeGenerator {
 public:
  using Zmm = Xbyak::Zmm;
  using Ymm = Xbyak::Ymm;
  using Xmm = Xbyak::Xmm;
  using Reg64 = Xbyak::Reg64;
  using Opmask = Xbyak::Opmask;
  explicit jit_generator(size_t code_size = MAX_CODE_SIZE, void* code_ptr = nullptr)
      : Xbyak::CodeGenerator(code_size, (code_ptr == nullptr) ? Xbyak::AutoGrow : code_ptr) {}
  virtual ~jit_generator() {}
  void dump_asm();  // print assembly code

 public:
  const int EVEX_max_8b_offt = 0x200;
  const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

// Set to O0 to avoid undefined behavour of reinterpret_cast
// Ref: https://stackoverflow.com/questions/25131019/reinterpret-cast-to-function-pointer
#pragma GCC push_options
#pragma GCC optimize("O0")
  template <typename... T>
  inline void operator()(T... args) const {
    using func_ptr = void (*)(const T... args);
    auto fptr = reinterpret_cast<func_ptr>(jit_ker_);
    (*fptr)(std::forward<T>(args)...);
  }
#pragma GCC pop_options

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

  void preamble() {
    if (xmm_to_preserve) {
      sub(rsp, xmm_to_preserve * VEC);
      for (size_t i = 0; i < xmm_to_preserve; ++i)
        uni_vmovdqu(ptr[rsp + i * VEC], Xbyak::Xmm(xmm_to_preserve_start + i));
    }
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
      push(Xbyak::Reg64(abi_save_gpr_regs[i]));
    }
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
  }

  void postamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
      pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    if (xmm_to_preserve) {
      for (size_t i = 0; i < xmm_to_preserve; ++i)
        uni_vmovdqu(Xbyak::Xmm(xmm_to_preserve_start + i), ptr[rsp + i * VEC]);
      add(rsp, xmm_to_preserve * VEC);
    }
    uni_vzeroupper();
    ret();
  }

 protected:
  const uint8_t* jit_ker_ = nullptr;
  static constexpr uint64_t MAX_CODE_SIZE = 256 * 1024 * 1024;
  static constexpr int VEC = 16;  // 512 bits of ZMM register divided by S32 bits.
  int callee_functions_code_size_ = 0;
  const size_t num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);
  const size_t size_of_abi_save_regs = num_abi_save_gpr_regs * rax.getBit() / 8 + xmm_to_preserve * VEC;

  static int dump_idx;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_JIT_GENERATOR_HPP_
