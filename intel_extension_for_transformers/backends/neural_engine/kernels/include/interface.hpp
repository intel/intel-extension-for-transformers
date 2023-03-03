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

#ifndef ENGINE_SPARSELIB_INCLUDE_INTERFACE_HPP_
#define ENGINE_SPARSELIB_INCLUDE_INTERFACE_HPP_
#include <omp.h>
#include <vector>
#include <cstdint>
#include <memory>
#include "param_types.hpp"
#include "operator_desc.hpp"
#include "engine.hpp"
#include "cpu_engine.hpp"
#include "engine_factory.hpp"
#include "kernel_desc.hpp"
#include "kernel.hpp"
#include "kernel_cache.hpp"
#include "utils.hpp"
#include "vtune_wrapper.hpp"
#include "kernels/sparse_data.hpp"

namespace jd {
/**
 * @brief Proxy pattern. The proxy could interface to anything.
 *  Similar to onednn's "struct handle". oneapi/dnnl/dnnl.hpp:136.
 */
template <typename T, typename arg_t = void>
class SPARSE_API_ proxy_base {
 public:
  proxy_base() {}
  virtual ~proxy_base() {}

 public:
  inline void reset_sp(const std::shared_ptr<const T>& sp) { data_handle_ = sp; }
  inline const std::shared_ptr<const T>& get_sp() const { return data_handle_; }

 protected:
  // internal functions of creat the proxy object.
  virtual bool create_proxy_object(std::shared_ptr<const T>& result_ref, const arg_t& arg) = 0;  // NOLINT

 private:
  std::shared_ptr<const T> data_handle_;
};

/**
 * @brief Base proxy class, interfacing to the real/cached kernel_desc_t.
 */
class SPARSE_API_ kernel_desc_proxy : public proxy_base<kernel_desc_t, operator_desc> {
 public:
  kernel_desc_proxy() {}
  explicit kernel_desc_proxy(const operator_desc& op_desc);
  virtual ~kernel_desc_proxy() {}

 protected:
  bool create_proxy_object(std::shared_ptr<const kernel_desc_t>& result_ref, const operator_desc& op_desc) override;

 public:
  inline const jd::kernel_kind& kernel_kind() const { return get_sp()->kernel_kind(); }

 protected:
  const std::vector<impl_list_item_t>* impl_list_ = nullptr;
};

/**
 * @brief Base proxy class, interfacing to the real/cached kernel_t.
 */
class SPARSE_API_ kernel_proxy : public proxy_base<kernel_t, std::shared_ptr<const kernel_desc_t>> {
 public:
  kernel_proxy() {}
  explicit kernel_proxy(const kernel_desc_proxy& kdp);
  virtual ~kernel_proxy() {}

 protected:
  bool create_proxy_object(std::shared_ptr<const kernel_t>& result_ref,
                           const std::shared_ptr<const kernel_desc_t>& kd) override;

 public:
  inline const jd::kernel_kind& kernel_kind() const { return get_sp()->kd()->kernel_kind(); }
  void execute(const std::vector<const void*>& rt_data) const;
};

//// The following paragraphs are the various derived kernels and its descriptors.
/**
 * @brief Derived proxy class, interfacing to the real/cached sparse_matmul_desc_t.
 */
class SPARSE_API_ sparse_matmul_desc : public kernel_desc_proxy {
 public:
  sparse_matmul_desc() {}
  explicit sparse_matmul_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~sparse_matmul_desc() {}
};

class SPARSE_API_ transpose_matmul_desc : public kernel_desc_proxy {
 public:
  transpose_matmul_desc() {}
  explicit transpose_matmul_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~transpose_matmul_desc() {}
};

class SPARSE_API_ eltwiseop_desc : public kernel_desc_proxy {
 public:
  eltwiseop_desc() {}
  explicit eltwiseop_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~eltwiseop_desc() {}
};

class SPARSE_API_ layernorm_ba_desc : public kernel_desc_proxy {
 public:
  layernorm_ba_desc() {}
  explicit layernorm_ba_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~layernorm_ba_desc() {}
};

class SPARSE_API_ layernormalized_spmm_desc : public kernel_desc_proxy {
 public:
  layernormalized_spmm_desc() {}
  explicit layernormalized_spmm_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~layernormalized_spmm_desc() {}
};

class gather_desc : public kernel_desc_proxy {
 public:
  gather_desc() {}
  explicit gather_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~gather_desc() {}
};

class softmax_desc : public kernel_desc_proxy {
 public:
  softmax_desc() {}
  explicit softmax_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~softmax_desc() {}
};

class attention_desc : public kernel_desc_proxy {
 public:
  attention_desc() {}
  explicit attention_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~attention_desc() {}
};

class SPARSE_API_ transpose_mha_desc : public kernel_desc_proxy {
 public:
  transpose_mha_desc() {}
  explicit transpose_mha_desc(const operator_desc& op_desc) : kernel_desc_proxy(op_desc) {}
  virtual ~transpose_mha_desc() {}
};

/**
 * @brief Derived proxy class, interfacing to the real/cached sparse_matmul_t.
 */
class SPARSE_API_ sparse_matmul : public kernel_proxy {
 public:
  sparse_matmul() {}
  explicit sparse_matmul(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~sparse_matmul() {}
};

class SPARSE_API_ transpose_matmul : public kernel_proxy {
 public:
  transpose_matmul() {}
  explicit transpose_matmul(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~transpose_matmul() {}
};

class SPARSE_API_ eltwiseop : public kernel_proxy {
 public:
  eltwiseop() {}
  explicit eltwiseop(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~eltwiseop() {}
};

class SPARSE_API_ layernorm_ba : public kernel_proxy {
 public:
  layernorm_ba() {}
  explicit layernorm_ba(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~layernorm_ba() {}
};

class SPARSE_API_ layernormalized_spmm : public kernel_proxy {
 public:
  layernormalized_spmm() {}
  explicit layernormalized_spmm(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~layernormalized_spmm() {}
};

class SPARSE_API_ gather : public kernel_proxy {
 public:
  gather() {}
  explicit gather(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~gather() {}
};

class SPARSE_API_ softmax : public kernel_proxy {
 public:
  softmax() {}
  explicit softmax(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~softmax() {}
};

class SPARSE_API_ attention : public kernel_proxy {
 public:
  attention() {}
  explicit attention(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~attention() {}
};

class SPARSE_API_ transpose_mha : public kernel_proxy {
 public:
  transpose_mha() {}
  explicit transpose_mha(const kernel_desc_proxy& kdp) : kernel_proxy(kdp) {}
  virtual ~transpose_mha() {}
};

}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_INTERFACE_HPP_
