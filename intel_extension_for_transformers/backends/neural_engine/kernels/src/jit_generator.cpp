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

#include "jit_generator.hpp"

namespace jd {

bool dump_asm_flag = false;

int jit_generator::dump_idx = 0;

bool jit_generator::create_kernel() {
  generate();
  if (std::getenv("SPARSE_LIB_DUMP") != nullptr) dump_asm_flag = true;
  if (dump_asm_flag) dump_asm();
  jit_ker_ = get_code();
  return (jit_ker_ != nullptr);
}

const uint8_t* jit_generator::get_code() {
  this->ready();
  auto code = CodeGenerator::getCode();
  if (callee_functions_code_size_ == 0) {
    return code;
  }
  return code + callee_functions_code_size_;
}

template <typename T>
Xbyak::Address jit_generator::EVEX_compress_addr(Xbyak::Reg64 base, T raw_offt, bool bcast) {
  using Xbyak::Address;
  using Xbyak::Reg64;
  using Xbyak::RegExp;
  using Xbyak::Zmm;

  SPARSE_LOG_IF(FATAL, raw_offt > INT_MAX) << "raw_offt should be int";
  auto offt = static_cast<int>(raw_offt);

  int scale = 0;

  if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
    offt = offt - 2 * EVEX_max_8b_offt;
    scale = 1;
  } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
    offt = offt - 4 * EVEX_max_8b_offt;
    scale = 2;
  }

  auto re = RegExp() + base + offt;
  if (scale) re = re + reg_EVEX_max_8b_offt * scale;

  if (bcast)
    return zword_b[re];
  else
    return zword[re];
}

Xbyak::Address jit_generator::make_safe_addr(const Xbyak::Reg64& reg_out, size_t offt, const Xbyak::Reg64& tmp_reg,
                                             bool bcast) {
  if (offt > INT_MAX) {
    mov(tmp_reg, offt);
    return bcast ? ptr_b[reg_out + tmp_reg] : ptr[reg_out + tmp_reg];
  } else {
    return bcast ? ptr_b[reg_out + offt] : ptr[reg_out + offt];
  }
}

Xbyak::Address jit_generator::EVEX_compress_addr_safe(const Xbyak::Reg64& base, size_t raw_offt,
                                                      const Xbyak::Reg64& reg_offt, bool bcast) {
  if (raw_offt > INT_MAX) {
    return make_safe_addr(base, raw_offt, reg_offt, bcast);
  } else {
    return EVEX_compress_addr(base, raw_offt, bcast);
  }
}

void jit_generator::dump_asm() {
  std::string file_name("code_" + std::to_string(dump_idx++) + ".bin");
  std::ofstream out_file(file_name, std::ios::out | std::ios::binary);
  out_file.write(reinterpret_cast<const char*>(getCode()), getSize());
  out_file.close();
}
}  // namespace jd
