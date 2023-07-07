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

#ifndef ENGINE_SPARSELIB_SRC_CPU_REGS_POOL_HPP_
#define ENGINE_SPARSELIB_SRC_CPU_REGS_POOL_HPP_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <string>

#include "jit_generator.hpp"
#include "src/utils.hpp"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace jd {
#define FOREACH_REG template <typename R>
#define FOREACH_REG_N template <typename R, size_t N>
class jit_eltwise_injector;
class regs_pool {
  friend jit_eltwise_injector;

 private:
  enum class reg_kind : size_t {
    gpr,
    xmm,
    mask,
    // tmm not implemented yet
    size,
  };
  static constexpr size_t reg_kind_size = static_cast<size_t>(reg_kind::size);

  Xbyak::util::StackFrame sf_;
  Xbyak::CodeGenerator* code_;
  const size_t stack_size_;
  const size_t stack_align_;
  const bool make_epilog_, evex_, warn_waste_;

  std::array<int, reg_kind_size> ridx_next;               // The index of next register to be allocated
  const std::array<int, reg_kind_size> ridx_max;          // max number of registers  to be allocated
  std::array<int, reg_kind_size> ridx_touched;            // (max register indexever allocated) + 1
  std::unordered_map<std::string, int> reg_name_idx_map;  // The index of specific reg_name

  template <typename Base, typename R>
  using enable_if_reg_kind_t = std::enable_if_t<std::is_base_of<Base, R>::value, reg_kind>;
  FOREACH_REG static constexpr enable_if_reg_kind_t<Xbyak::Reg32e, R> get_reg_kind() { return reg_kind::gpr; }
  FOREACH_REG static constexpr enable_if_reg_kind_t<Xbyak::Xmm, R> get_reg_kind() { return reg_kind::xmm; }
  FOREACH_REG static constexpr enable_if_reg_kind_t<Xbyak::Opmask, R> get_reg_kind() { return reg_kind::mask; }
  FOREACH_REG static constexpr size_t get_reg_kind_i() { return static_cast<size_t>(get_reg_kind<R>()); }

  FOREACH_REG inline int& get_next() { return ridx_next[get_reg_kind_i<R>()]; }
  FOREACH_REG inline int get_next() const { return ridx_next[get_reg_kind_i<R>()]; }
  FOREACH_REG inline int get_max() const { return ridx_max[get_reg_kind_i<R>()]; }
  FOREACH_REG inline int& get_touched() { return ridx_touched[get_reg_kind_i<R>()]; }
  FOREACH_REG inline int get_touched() const { return ridx_touched[get_reg_kind_i<R>()]; }

  /**
   * @brief map idx-th allocated register of kind rt to Xbyak's index used as argument of constructions.
   */
  FOREACH_REG inline int map_reg_idx(const int idx) const {
    switch (get_reg_kind<R>()) {
      case reg_kind::gpr:
        return sf_.t[idx].getIdx();
      case reg_kind::xmm:
        if (!evex_) {
          return idx;
        } else if (idx < 16) {
          return idx + 16;  // map 0-15 to zmm16-zmm31
        } else {
          return idx - 16;  // map 15-31 to zmm0-zmm15
        }
      case reg_kind::mask:
        return idx + 1;  // map 0-6 to k1-k7
      default:
        SPARSE_LOG(ERROR) << "Unexpected reg_kind";
        return -1;
    }
  }
  FOREACH_REG inline R get_next_reg() {
    SPARSE_LOG_IF(FATAL, get_next<R>() >= get_max<R>())
        << "No more registers of kind " << get_reg_kind_i<R>() << " ! "
        << "Next idx: " << get_next<R>() << " Max idx: " << get_max<R>();
    const auto ret = R{map_reg_idx<R>(get_next<R>()++)};
    get_touched<R>() = std::max(get_touched<R>(), get_next<R>());
    return ret;
  }

  FOREACH_REG class shared_reg : public R, private std::shared_ptr<void> {
    friend regs_pool;
    explicit shared_reg(regs_pool* const s)
        : R(s->get_next_reg<R>()),  //
          std::shared_ptr<void>{nullptr, [s](...) { s->get_next<R>()--; }} {}
    shared_reg(regs_pool* const s, const std::string& reg_name)
        : R(s->get_next_reg<R>()),  //
          std::shared_ptr<void>{nullptr, [s, reg_name](...) {
                                  s->get_next<R>()--;
                                  s->reg_name_idx_map.erase(reg_name);
                                }} {}
  };
  FOREACH_REG class shared_reg_vec : public std::vector<R>, private std::shared_ptr<void> {
    friend regs_pool;
    shared_reg_vec(regs_pool* const s, const size_t n)
        : std::vector<R>(),  //
          std::shared_ptr<void>{nullptr, [s, n](...) { s->get_next<R>() -= n; }} {
      for (auto _ = n; _-- > 0;) this->emplace_back(s->get_next_reg<R>());
    }
  };

  FOREACH_REG_N class shared_reg_arr : public std::array<R, N>, private std::shared_ptr<void> {
    friend regs_pool;
    using arr_t = std::array<R, N>;
    explicit shared_reg_arr(regs_pool* const s)
        : arr_t(sequence_map_([s](...) { return s->get_next_reg<R>(); }, std::make_index_sequence<N>())),
          std::shared_ptr<void>{nullptr, [s](...) { s->get_next<R>() -= N; }} {}

    template <class Func, size_t... I>
    static arr_t sequence_map_(Func f, std::integer_sequence<size_t, I...>) {
      return {f(I)...};
    }
  };

  static constexpr std::array<int, 3UL> zero_x3{0, 0, 0};

 public:
  static const int UseRCX = Xbyak::util::UseRCX;  // RCX|reg_num[0] to enable usage(reserved but not alloc) of RCX
  static const int UseRDX = Xbyak::util::UseRDX;  // RDX|reg_num[0] to enable usage(reserved but not alloc) of RDX
  const decltype(Xbyak::util::StackFrame::p)& p;  // alias of sf.p

  static constexpr int DefaultFlags = 0;        // All flags off by default
  static constexpr int DisableEpilog = 1 << 0;  // disable epilog generation during deconstruction
  static constexpr int DisableEvex = 1 << 1;    // disable the use of EVEX encoded instructions (including zmm16-31)
  static constexpr int IgnoreWaste = 1 << 2;    // surpress warnings raised when not all registers applied are used

  /**
   * @brief RAII based register pool "extended" from Xbyak::util::StackFrame
   * @param code this
   * @param pNum num of function parameter(0 <= pNum <= 4)
   * @param reg_num nums of each kind (gpr, xmm, opmask) of temporary registers;
   * UseRCX / UseRDX can apply to reg_kind::gpr to exclude them from allocation
   * @param stack_size local stack size in terms of #bytes
   * @param flags controlling flags including DisableEpilog / DisableEvex / IgnoreWaste
   * @param stack_align alignment of stack space
   */
  regs_pool(  //
      Xbyak::CodeGenerator* const code, const int pNum, const std::array<int, 3UL> reg_num = zero_x3,
      const size_t stack_size = 0, const int flags = DefaultFlags, const size_t stack_align = 8)
      : sf_(code, pNum, reg_num[get_reg_kind_i<Reg64>()], 0, false),  // stack memory and epilogue managed here
        code_(code),
        stack_size_(stack_size),
        stack_align_(stack_align),
        make_epilog_(!(flags & DisableEpilog)),
        evex_(!(flags & DisableEvex)),
        warn_waste_(!(flags & IgnoreWaste)),
        ridx_next(zero_x3),
        ridx_max({reg_num[0] & (~UseRCX) & (~UseRDX), reg_num[1], reg_num[2]}),
        ridx_touched(zero_x3),
        p(sf_.p) {
    // availability check
    SPARSE_LOG_IF(FATAL, get_max<Reg64>() + pNum > 15)  // #{pNum + num_gpr [+rcx] + [rdx]} <= 14
        << "No more GPR registers!";
    SPARSE_LOG_IF(FATAL, get_max<Zmm>() > (evex_ ? 32 : 16)) << "No more XMM registers!";
    SPARSE_LOG_IF(FATAL, get_max<Opmask>() > 7) << "No more mask registers!";
    SPARSE_LOG_IF(FATAL, stack_align < 8 || (stack_align & (stack_align - 1)) != 0)
        << "stack alignment must be a power of 2!";

    // preserve xmm on demend
#ifdef _WIN32
    static constexpr int CALLER_SAVED_XMM = 32 - xmm_to_preserve;
    const auto num_to_save = get_max<Zmm>() - CALLER_SAVED_XMM;
    if (num_to_save > 0) {
      code_->sub(code_->rsp, num_to_save * 16);
      for (int i = 0; i < num_to_save; ++i)
        code_->movdqu(code_->xword[code_->rsp + i * 16], Xbyak::Xmm(xmm_to_preserve_start + i));
    }
#endif

    // allocate stack space
    if (stack_size) {
      if (stack_align <= 8) {  // stack is aligned to 8 by default
        code_->sub(code_->rsp, pad_to(stack_size_, 8));
      } else {
        const auto tmp = reg<Reg64>();
        code_->lea(tmp, code_->ptr[code_->rsp - 8]);  // 8 for reserve rsp
        code_->and_(tmp, 0 - stack_align);            // align
        code_->mov(code_->qword[tmp], code_->rsp);
        code_->lea(code_->rsp, code_->ptr[tmp - pad_to(stack_size_, stack_align)]);
      }
    }
  }

  ~regs_pool() {
    if (make_epilog_) close();
  }

  /**
   * @brief make epilog manually
   * @param call_ret call ret() if true
   */
  inline void close(const bool call_ret = true) {
    // Usage check: do we asked more than what we actually used?
    SPARSE_DLOG_IF(WARNING, warn_waste_ && get_max<Reg64>() > get_touched<Reg64>())
        << "Asked too many GPRs! Actually used: " << get_touched<Reg64>() << "/" << get_max<Reg64>();
    SPARSE_DLOG_IF(WARNING, warn_waste_ && get_max<Zmm>() > get_touched<Zmm>())
        << "Asked too many XMMs! Actually used: " << get_touched<Zmm>() << "/" << get_max<Zmm>();
    SPARSE_DLOG_IF(WARNING, warn_waste_ && get_max<Opmask>() > get_touched<Opmask>())
        << "Asked too many masks! Actually used: " << get_touched<Opmask>() << "/" << get_max<Opmask>();

    // free stack space
    if (stack_size_) {
      if (stack_align_ <= 8) {
        code_->add(code_->rsp, pad_to(stack_size_, 8));
      } else {
        code_->mov(code_->rsp, code_->ptr[code_->rsp + pad_to(stack_size_, stack_align_)]);
      }
    }

#ifdef _WIN32
    // restore xmm on demend
    const int CALLER_SAVED_XMM = (evex_ ? 32 : 16) - xmm_to_preserve;
    const auto num_to_save = get_max<Zmm>() - CALLER_SAVED_XMM;
    if (num_to_save > 0) {
      for (int i = 0; i < num_to_save; ++i)
        code_->movdqu(Xbyak::Xmm(xmm_to_preserve_start + i), code_->xword[code_->rsp + i * 16]);
      code_->add(code_->rsp, num_to_save * 16);
    }
#endif
    sf_.close(call_ret);
  }

  /** @warning Use `auto` hold the result as the referring count will be stripped otherwise. */
  FOREACH_REG inline shared_reg<R> reg() { return shared_reg<R>{this}; }
  /** @warning Use `auto` hold the result as the referring count will be stripped otherwise. */
  FOREACH_REG inline shared_reg<R> reg(const std::string& reg_name) {
    auto reg = shared_reg<R>(this, reg_name);
    reg_name_idx_map[reg_name] = reg.getIdx();
    return reg;
  }
  /** @warning Use `auto` hold the result as the referring count will be stripped otherwise. */
  FOREACH_REG inline const shared_reg_vec<R> regs(size_t n) { return shared_reg_vec<R>{this, n}; }
  /** @warning Use `auto` hold the result as the referring count will be stripped otherwise. */
  FOREACH_REG_N inline const shared_reg_arr<R, N> regs() { return shared_reg_arr<R, N>{this}; }
  inline int get_idx_by_name(std::string reg_name) { return reg_name_idx_map[reg_name]; }
};
#undef FOREACH_REG
#undef FOREACH_REG_N
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_CPU_REGS_POOL_HPP_
