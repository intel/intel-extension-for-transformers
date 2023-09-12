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

#ifndef ENGINE_EXECUTOR_INCLUDE_DISPATCHER_HPP_
#define ENGINE_EXECUTOR_INCLUDE_DISPATCHER_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"
#include "dispatch_table.hpp"
#include "isa.hpp"

namespace executor {

/**
 * @brief Use dispatcher to choose a specific kernel at runtime.
 *        Every operator in model should have its own default
 *        execute kernel, however it may be not the best one in
 *        some scenarios.
 *        Dispatcher has the ability to find the best kernel before
 *        deployment if user turn on tuning mechanism.
 *        for example:
 *        InnerProduct Dispatcher --  InnerProduct Kernel
 *                                |-- Convolution Kernel
 *        1. If do tuning, dispatcher will get the info about what
 *           is the better kernel under the feed input's shape, dtype, etc.
 *           And write these info into dispatcher table file, like
 *           InnerProduct xxxxxx Convolution 4,1,256,256
 *        2. If do not tuning, dispatcher will try to load the dispatcher
 *           table file. If dispatcher find the best kernl by hash key, it
 *           will execut that kernel (maybe 1x1 conv). Otherwise, dispatcher
 *           will execute its default kernel (innerproduct there).
 *
 */

using std::shared_ptr;
using std::string;
using std::vector;
class Model;

class Dispatcher {
 public:
  // kernel implementation table
  typedef std::unordered_map<string, shared_ptr<Operator>> KernelHandler;

  explicit Dispatcher(const shared_ptr<OperatorConfig>& conf, const ExecutionOptions* e_ptr, const Model* m_ptr);

  explicit Dispatcher(const shared_ptr<Operator>& op, const ExecutionOptions* e_ptr, const Model* m_ptr);

  ~Dispatcher() {}

  // prepare all kernel when model init
  void Prepare(const vector<Tensor*>& input, const vector<Tensor*>& output);

  void Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output);

  void Forward(const vector<Tensor*>& input, const vector<Tensor*>& output);

  // modify op attrs if need.
  // e.g. SparseLib 3D gemm - BatchMatmul, prem related attrs in BatchMatMul
  // need to adjust.
  void AdaptAttrs(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage);

  // modify tensors when dispatched kernel needs different format, etc.
  // should call it before (in) or after (out) Forward
  void AdaptTensors(const vector<Tensor*>& input, const vector<Tensor*>& output, const string& stage);

  // reset op like tensor format after finishing inference iteration
  void ResetOpStatus(const vector<Tensor*>& input, const vector<Tensor*>& output);

  void GetExecuteKernel(const vector<Tensor*>& input, const vector<Tensor*>& output, const bool& reshape_model,
                        const bool& has_dispatch_table_file);

  // turn on or turn off tuning mechanism in some specific operators if need
  inline void set_tuning_mode(const bool& mode) { do_tuning_ = mode; }
  inline const bool& do_tuning() const { return do_tuning_; }
  inline const string& name() const { return name_; }
  inline const string& type() const { return type_; }
  inline const shared_ptr<OperatorConfig>& operator_conf() const { return operator_conf_; }
  inline const string& execute_kernel() const { return execute_kernel_; }
  inline const bool& no_tuning_space() const { return no_tuning_space_; }
  inline vector<vector<string>> InplacePairs(const vector<Tensor*>& input, const vector<Tensor*>& output) {
    return kernel_handler_[execute_kernel_]->InplacePairs(input, output);
  }
  inline void decrease_consumers(const vector<Tensor*>& input) {
    kernel_handler_[execute_kernel_]->decrease_consumers(input);
  }
  // for profiling
  inline void set_post_op(const string& post_op) { kernel_handler_[execute_kernel_]->set_post_op(post_op); }
  inline const string& post_op() { return kernel_handler_[execute_kernel_]->post_op(); }
  inline void set_latency(const float latency) { kernel_handler_[execute_kernel_]->set_latency(latency); }
  inline const vector<float>& latency() { return kernel_handler_[execute_kernel_]->latency(); }
  inline void set_enable_sparse(const bool enable_sparse) {
    kernel_handler_[execute_kernel_]->set_enable_sparse(enable_sparse);
  }
  inline const float& enable_sparse() { return kernel_handler_[execute_kernel_]->enable_sparse(); }
  inline const KERNEL_TYPE& kernel_type() { return kernel_handler_[execute_kernel_]->kernel_type(); }
  inline const float& weight_zero_ratio() { return kernel_handler_[execute_kernel_]->weight_zero_ratio(); }
  inline void set_weight_shape(const vector<int64_t>& weight_shape) {
    kernel_handler_[execute_kernel_]->set_weight_shape(weight_shape);
  }
  inline const vector<int64_t>& weight_shape() { return kernel_handler_[execute_kernel_]->weight_shape(); }
  inline void set_table_id(string table_id) { kernel_handler_[execute_kernel_]->set_table_id(table_id); }
  inline const string& table_id() { return kernel_handler_[execute_kernel_]->table_id(); }
  inline void set_perf_ratio_id(string perf_ratio_id) {
    kernel_handler_[execute_kernel_]->set_perf_ratio_id(perf_ratio_id);
  }
  inline const string& perf_ratio_id() { return kernel_handler_[execute_kernel_]->perf_ratio_id(); }
  inline void append_it_shape(const vector<int64_t> input_shape) {
    kernel_handler_[execute_kernel_]->append_it_shape(input_shape);
  }
  inline void append_ot_shape(const vector<int64_t> output_shape) {
    kernel_handler_[execute_kernel_]->append_ot_shape(output_shape);
  }
  inline void set_it_shape(const vector<int64_t> output_shape, int index) {
    kernel_handler_[execute_kernel_]->set_it_shape(output_shape, index);
  }
  inline const vector<vector<int64_t>>& get_it_shape() { return kernel_handler_[execute_kernel_]->get_it_shape(); }
  inline const vector<vector<int64_t>>& get_ot_shape() { return kernel_handler_[execute_kernel_]->get_ot_shape(); }
  inline void clear_it_shape() { kernel_handler_[execute_kernel_]->clear_it_shape(); }
  inline void clear_ot_shape() { kernel_handler_[execute_kernel_]->clear_ot_shape(); }
  inline void set_reshape_time(const float reshape_time_) {
    kernel_handler_[execute_kernel_]->set_reshape_time(reshape_time_);
  }
  inline const vector<float>& get_reshape_time() { return kernel_handler_[execute_kernel_]->get_reshape_time(); }
  inline void set_attrs(const std::map<string, string> input_attrs) {
    kernel_handler_[execute_kernel_]->set_attrs(input_attrs);
  }
  inline const std::map<string, string>& get_attrs() { return kernel_handler_[execute_kernel_]->get_attrs(); }

 protected:
  // get input_hash
  size_t GetHash(const vector<Tensor*>& input);

  string name_;
  string type_;
  shared_ptr<OperatorConfig> operator_conf_;
  isa cpu_isa_;
  KernelHandler kernel_handler_;
  string execute_kernel_;
  bool do_tuning_ = false;
  bool no_tuning_space_ = false;
  int64_t iter_cnt_ = 0;
  vector<bool> sparselib_available_;
  bool tune_dense_in_sparse_ = false;
  string monopoly_kernel_;
  bool dispatch_table_file_exists_ = false;
  const ExecutionOptions* execution_options_ptr_ = nullptr;
  bool adapt_action_ = true;
  const Model* model_ = nullptr;
};
}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_DISPATCHER_HPP_
