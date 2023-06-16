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

#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATOR_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATOR_HPP_

#include <algorithm>      //NOLINT
#include <string>         //NOLINT
#include <vector>         //NOLINT
#include <map>            //NOLINT
#include <unordered_map>  //NOLINT
#include <mutex>          //NOLINT
#include <memory>         //NOLINT

#include "common.hpp"
#include "operator_registry.hpp"
#include "tensor.hpp"
#include "execution_options.hpp"

using std::shared_ptr;

namespace executor {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Model.
 *
 * Operator%s must implement a Forward function, in which they take their input
 * (input) Tensor%s (if any) and compute their output Tensor%s (if any).
 */

enum KERNEL_TYPE { Unsupported = 0, Dense = 1, Sparse = 2, SparseLib = 3, Runtime = 4 };
class Model;  // forward declaration

class Operator {
 public:
  explicit Operator(const shared_ptr<OperatorConfig>& conf) : operator_conf_(conf) {
    name_ = operator_conf_->name();
    type_ = operator_conf_->type();
  }

  virtual ~Operator() {}

  virtual void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output) {}

  // use Reshape to calculate the output shape from input tensor
  virtual void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) = 0;

  virtual void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) = 0;

  // modify op attrs befor (in) and after (out) Reshape
  virtual void AdaptAttrs(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {}

  // modify tensors before (in) or after (out) Forward
  virtual void AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage) {
    if (stage == "in") {
      if (!input.empty() && !output.empty()) {
        output[0]->set_tensor_format(input[0]->tensor_format());
      }
    } else if (stage == "out") {
      return;
    } else {
      LOG(WARNING) << "Wrong stage parameter, should be in or out...";
    }
  }

  // reset op like tensor format after finishing inference iteration
  virtual void ResetOpStatus(const vector<Tensor*>& input, const vector<Tensor*>& output) {}

  // infer dst tensor's shape
  // use the shape to get hash when feed random inputs
  // shape infer process does not contain jit codegen
  virtual void ShapeInfer(const vector<Tensor*>& input, const vector<Tensor*>& output) {}

  // return inplace tensor names pairs
  // keep these pairs in order
  // <<tensor_a0 -> tensor_b0>, <tensor_a1 -> tensor_b1>, ...>
  virtual vector<vector<string>> InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
    vector<vector<string>> pairs;
    return pairs;
  }

  inline void unref_tensors(const vector<Tensor*>& input) {
    static std::mutex unref_lock;
    std::lock_guard<std::mutex> lock(unref_lock);
    for (size_t i = 0; i < input.size(); ++i) {
      auto status = input[i]->unref_data();
      // (TODO) maybe check the tensors
    }
  }

  inline void decrease_consumers(const vector<Tensor*>& input) {
    static std::mutex decrease_consumers_lock;
    std::lock_guard<std::mutex> dec_lock(decrease_consumers_lock);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i]->decrease_left_life();
    }
  }

  inline const ExecutionMode get_execution_mode() const {
    if (execution_options_ptr_ == nullptr) {
      ExecutionOptions options = ExecutionOptions();
      return options.execution_mode;
    } else {
      return execution_options_ptr_->execution_mode;
    }
  }

  friend class Dispatcher;
  inline const string& name() const { return name_; }
  inline const string& type() const { return type_; }
  const shared_ptr<OperatorConfig>& operator_conf() const { return operator_conf_; }
  // dispatch kernel may need to do reshape and receive config, like InnerProduct to Convolution
  inline void set_dispatch_from_type(const string& type) { dispatch_from_ = type; }
  inline void set_dispatch_config(const vector<string>& config = {}) { dispatch_config_ = config; }
  inline const vector<string>& dispatch_config() const { return dispatch_config_; }
  inline void set_do_shape_infer(const bool& do_shape_infer) { do_shape_infer_ = do_shape_infer; }
  inline const bool& do_shape_infer() const { return do_shape_infer_; }
  inline const bool& monopolize_dispatcher() const { return monopolize_dispatcher_; }
  // for profiling
  inline void set_post_op(const string& post_op) { post_op_ = post_op; }
  inline const string& post_op() const { return post_op_; }
  inline void set_latency(const float latency) { latency_.emplace_back(latency); }
  inline const vector<float>& latency() const { return latency_; }
  inline void set_enable_sparse(const bool enable_sparse) { enable_sparse_ = enable_sparse; }
  inline const float& enable_sparse() const { return enable_sparse_; }
  inline const KERNEL_TYPE& kernel_type() const { return kernel_type_; }
  inline void set_kernel_type(const KERNEL_TYPE& kernel_type) { kernel_type_ = kernel_type; }
  inline const float& weight_zero_ratio() const { return weight_zero_ratio_; }
  inline void set_weight_shape(const vector<int64_t>& weight_shape) { weight_shape_ = weight_shape; }
  inline const vector<int64_t>& weight_shape() const { return weight_shape_; }
  inline void set_table_id(const string& table_id) { table_id_ = table_id; }
  inline const string& table_id() const { return table_id_; }
  inline void set_perf_ratio_id(const string& perf_ratio_id) { perf_ratio_id_ = perf_ratio_id; }
  inline const string& perf_ratio_id() const { return perf_ratio_id_; }
  inline void set_it_shape(const vector<int64_t> input_shape, int index) { input_tensor_shape_[index] = (input_shape); }
  inline void append_it_shape(const vector<int64_t> input_shape) { input_tensor_shape_.emplace_back(input_shape); }
  inline void append_ot_shape(const vector<int64_t> output_shape) { output_tensor_shape_.emplace_back(output_shape); }
  inline const vector<vector<int64_t>>& get_it_shape() const { return input_tensor_shape_; }
  inline const vector<vector<int64_t>>& get_ot_shape() const { return output_tensor_shape_; }
  inline void clear_it_shape() { input_tensor_shape_.clear(); }
  inline void clear_ot_shape() { output_tensor_shape_.clear(); }
  // get executor kernel time add reshape time
  inline void set_reshape_time(const float reshape_time) { reshape_time_.emplace_back(reshape_time); }
  inline const vector<float>& get_reshape_time() const { return reshape_time_; }
  inline void set_attrs(const std::map<string, string>& input_attrs) { attrs_ = input_attrs; }
  inline const std::map<string, string>& get_attrs() const { return attrs_; }

 protected:
  /** The conf that stores the operator configurations */
  string name_;
  string type_;
  shared_ptr<OperatorConfig> operator_conf_;
  string dispatch_from_;
  vector<string> dispatch_config_;
  bool do_shape_infer_ = false;
  bool monopolize_dispatcher_ = false;
  const ExecutionOptions* execution_options_ptr_ = nullptr;
  bool adapt_attrs_ = false;
  // for profiling
  string post_op_;
  vector<float> latency_;
  float enable_sparse_ = false;
  KERNEL_TYPE kernel_type_ = Unsupported;
  float weight_zero_ratio_ = 0.0;
  vector<int64_t> weight_shape_;
  string table_id_;
  string perf_ratio_id_;
  vector<vector<int64_t>> input_tensor_shape_;
  vector<vector<int64_t>> output_tensor_shape_;
  vector<float> reshape_time_;
  std::map<string, string> attrs_;
  static std::unordered_map<string, jd::data_type> type2sparsemem_;
  const Model* model_ = nullptr;
  static std::unordered_map<string, jd::data_type> type_2_sparsemem;
};  // class Operator

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATOR_HPP_
