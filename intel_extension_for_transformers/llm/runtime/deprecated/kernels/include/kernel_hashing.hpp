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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNEL_HASHING_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNEL_HASHING_HPP_
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include "param_types.hpp"
#include "tensor_desc.hpp"
#include "operator_desc.hpp"
#include "engine.hpp"

namespace jd {
/**
 * @brief The hash function of a specific kernel descriptor or kernel primitive.
 */
class hash_t {
 public:
  uint64_t operator()(const operator_desc& key) const {
    uint64_t seed = 0;
    // Compute hash for primitive_kind_, attr_, impl_id_ and impl_nthr_
    hash_combine(seed, static_cast<uint64_t>(key.kernel_kind()));
    hash_combine(seed, static_cast<uint64_t>(key.kernel_prop()));
    hash_combine(seed, static_cast<uint64_t>(key.engine_kind()));
    hash_combine(seed, static_cast<uint64_t>(key.impl_nthr()));
    hash_combine(seed, get_tensor_descs_hash(key.tensor_descs()));
    hash_combine(seed, get_attr_hash(key.attrs(), key.kernel_kind()));
    return seed;
  }

 private:
  // http://boost.sourceforge.net/doc/html/boost/hash_combine.html
  template <typename T>
  static void hash_combine(size_t& seed, const T& v) {  // NOLINT
    seed ^= static_cast<size_t>(std::hash<T>()(v)) + static_cast<size_t>(0x9e3779b9) + (seed << 6) + (seed >> 2);
  }

 private:
  uint64_t get_tensor_descs_hash(const std::vector<tensor_desc>& ts_descs) const {
    uint64_t seed = 0;
    int tensor_cnt = ts_descs.size();
    for (int idx = 0; idx < tensor_cnt; ++idx) {
      if (idx == 0 || idx == 3) {
        continue;  // we don't care about shapes of src or dst while hashing
      }
      for (const auto& dim : ts_descs[idx].shape()) {
        hash_combine(seed, static_cast<uint64_t>(dim));
      }
      hash_combine(seed, static_cast<uint64_t>(ts_descs[idx].dtype()));
      hash_combine(seed, static_cast<uint64_t>(ts_descs[idx].ftype()));
    }
    return seed;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
  uint64_t get_attr_hash(const std::unordered_map<std::string, std::string>& attrs, const kernel_kind& ker_kind) const {
    auto op_attrs = attrs;
    uint64_t seed = 0;
    // if front op want to apply postop-fusion,they should add a filed named postop_list in op_attr
    // for distinguishing.
    hash_combine(seed, op_attrs["postop_list"]);
    hash_combine(seed, op_attrs["binaryop_list"]);
    switch (ker_kind) {
      case kernel_kind::undef:
        break;
      case kernel_kind::layernormalized_spmm:
        hash_combine(seed, op_attrs["split_output"]);
      case kernel_kind::sparse_matmul:
        hash_combine(seed, op_attrs["sparse_ptr"]);
        hash_combine(seed, op_attrs["micro_oc"]);
        hash_combine(seed, op_attrs["append_sum"]);
        hash_combine(seed, op_attrs["sub_func"]);
        hash_combine(seed, op_attrs["welford"]);
        break;
      case kernel_kind::attention:
        hash_combine(seed, op_attrs["q_weight_ptr"]);
        hash_combine(seed, op_attrs["k_weight_ptr"]);
        hash_combine(seed, op_attrs["v_weight_ptr"]);
        hash_combine(seed, op_attrs["q_bias_ptr"]);
        hash_combine(seed, op_attrs["k_bias_ptr"]);
        hash_combine(seed, op_attrs["v_bias_ptr"]);
        hash_combine(seed, op_attrs["q_scales_ptr"]);
        hash_combine(seed, op_attrs["k_scales_ptr"]);
        hash_combine(seed, op_attrs["v_scales_ptr"]);
        hash_combine(seed, op_attrs["alpha"]);
        hash_combine(seed, op_attrs["beta"]);
        hash_combine(seed, op_attrs["softmax_in_zero_point"]);
        hash_combine(seed, op_attrs["softmax_in_scale"]);
        hash_combine(seed, op_attrs["softmax_out_zero_point"]);
        hash_combine(seed, op_attrs["softmax_out_scale"]);
        break;
      case kernel_kind::transpose_mha:
        break;
      case kernel_kind::groupnorm:
        hash_combine(seed, op_attrs["groups"]);
        hash_combine(seed, op_attrs["eps"]);
        break;
      case kernel_kind::layernorm_ba:
        hash_combine(seed, op_attrs["split_output"]);
        hash_combine(seed, op_attrs["matrix_shape"]);
        hash_combine(seed, op_attrs["spec_type"]);
        break;
      case kernel_kind::gather:
        hash_combine(seed, op_attrs["matrix_shape"]);
        break;
      case kernel_kind::slice:
        hash_combine(seed, op_attrs["begin"]);
        hash_combine(seed, op_attrs["step"]);
        hash_combine(seed, op_attrs["axis"]);
        break;
      case kernel_kind::transpose_matmul:
        hash_combine(seed, op_attrs["alpha"]);
        hash_combine(seed, op_attrs["beta"]);
        hash_combine(seed, op_attrs["m_tile"]);
        hash_combine(seed, op_attrs["n_tile"]);
        break;
      case kernel_kind::dynamic_quant_matmul:
        hash_combine(seed, op_attrs["large_wei_threshold"]);
        hash_combine(seed, op_attrs["append_sum"]);
        break;
      case kernel_kind::softmax:
        hash_combine(seed, op_attrs["spec_type"]);
        hash_combine(seed, op_attrs["vec_len"]);
        hash_combine(seed, op_attrs["quant_factor"]);
        break;
      case kernel_kind::mha_dense:
        hash_combine(seed, op_attrs["QK_rescale"]);
        hash_combine(seed, op_attrs["softmax_rescale"]);
        hash_combine(seed, op_attrs["QKV_rescale"]);
        hash_combine(seed, op_attrs["QKV_dstzp"]);
        hash_combine(seed, op_attrs["merged_QKV"]);
        hash_combine(seed, op_attrs["is_package"]);
      case kernel_kind::dynamic_quant:
        hash_combine(seed, op_attrs["input_dt"]);
        break;
      case kernel_kind::eltwiseop:
      default:
        break;
    }
    return seed;
  }
#pragma GCC diagnostic pop
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNEL_HASHING_HPP_
