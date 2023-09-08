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

#ifndef ENGINE_SPARSELIB_INCLUDE_OPERATOR_DESC_HPP_
#define ENGINE_SPARSELIB_INCLUDE_OPERATOR_DESC_HPP_
#include <omp.h>

#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "param_types.hpp"
#include "data_type/data_types.hpp"
#include "tensor_desc.hpp"
namespace jd {
/**
 * @brief The operator descriptor class, describing a specific kind of operator.
 */
class operator_desc {
 public:
  operator_desc()
      : ker_kind_(kernel_kind::undef),
        ker_prop_(kernel_prop::undef),
        engine_kind_(engine_kind::undef),
        runtime_kind_(runtime_kind::undef),
        impl_nthr_(0),
        ts_descs_({}),
        attrs_({}),
        apply_postops_list_({}) {}
  operator_desc(const kernel_kind& ker_kind, const kernel_prop& ker_prop, const engine_kind& eng_kind,
                const std::vector<tensor_desc>& ts_descs, const std::unordered_map<std::string, std::string>& attrs,
                const std::vector<postop_attr>& apply_postops_list = {})
      : ker_kind_(ker_kind),
        ker_prop_(ker_prop),
        engine_kind_(eng_kind),
        runtime_kind_(runtime_kind::undef),
        impl_nthr_(omp_get_max_threads()),
        ts_descs_(ts_descs),
        attrs_(attrs),
        apply_postops_list_(apply_postops_list) {}
  operator_desc(const kernel_kind& ker_kind, const kernel_prop& ker_prop, const engine_kind& eng_kind,
                const runtime_kind& runtime_kind, const std::vector<tensor_desc>& ts_descs,
                const std::unordered_map<std::string, std::string>& attrs,
                const std::vector<postop_attr>& apply_postops_list = {})
      : ker_kind_(ker_kind),
        ker_prop_(ker_prop),
        engine_kind_(eng_kind),
        runtime_kind_(runtime_kind),
        impl_nthr_(omp_get_max_threads()),
        ts_descs_(ts_descs),
        attrs_(attrs),
        apply_postops_list_(apply_postops_list) {}
  virtual ~operator_desc() {}

 public:
  bool operator==(const operator_desc& rhs) const {
    return (ker_kind_ == rhs.ker_kind_) && (ker_prop_ == rhs.ker_prop_) && (engine_kind_ == rhs.engine_kind_) &&
           (impl_nthr_ == rhs.impl_nthr_) && (ts_descs_ == rhs.ts_descs_) && (attrs_ == rhs.attrs_);
  }

  void set_binaryop_list(const std::vector<binaryop_attr>& binaryop_list) { this->binaryop_list_ = binaryop_list; }

 public:
  inline const jd::kernel_kind& kernel_kind() const { return ker_kind_; }
  inline const jd::kernel_prop& kernel_prop() const { return ker_prop_; }
  inline const jd::engine_kind& engine_kind() const { return engine_kind_; }
  inline const jd::runtime_kind& runtime_kind() const { return runtime_kind_; }
  inline const uint64_t& impl_nthr() const { return impl_nthr_; }
  inline const std::vector<tensor_desc>& tensor_descs() const { return ts_descs_; }
  inline const std::unordered_map<std::string, std::string>& attrs() const { return attrs_; }
  inline const std::vector<postop_attr>& apply_postops_list() const { return apply_postops_list_; }
  inline const std::vector<binaryop_attr>& get_binaryop_list() const { return binaryop_list_; }

  inline std::vector<std::vector<dim_t>> tensor_shapes() const {
    std::vector<std::vector<dim_t>> ret(ts_descs_.size());
    std::transform(ts_descs_.cbegin(), ts_descs_.cend(), ret.begin(), [](auto&& td) { return td.shape(); });
    return ret;
  }
  inline std::vector<data_type> tensor_dtypes() const {
    std::vector<data_type> ret(ts_descs_.size());
    std::transform(ts_descs_.cbegin(), ts_descs_.cend(), ret.begin(), [](auto&& td) { return td.dtype(); });
    return ret;
  }
  inline std::vector<format_type> tensor_ftypes() const {
    std::vector<format_type> ret(ts_descs_.size());
    std::transform(ts_descs_.cbegin(), ts_descs_.cend(), ret.begin(), [](auto&& td) { return td.ftype(); });
    return ret;
  }

 private:
  jd::kernel_kind ker_kind_;
  jd::kernel_prop ker_prop_ = kernel_prop::forward_inference;
  jd::engine_kind engine_kind_ = engine_kind::cpu;
  jd::runtime_kind runtime_kind_ = runtime_kind::undef;
  uint64_t impl_nthr_;
  std::vector<tensor_desc> ts_descs_;
  std::unordered_map<std::string, std::string> attrs_;
  std::vector<postop_attr> apply_postops_list_;
  std::vector<binaryop_attr> binaryop_list_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_OPERATOR_DESC_HPP_
