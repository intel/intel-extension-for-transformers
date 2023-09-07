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

#ifndef ENGINE_EXECUTOR_INCLUDE_EXECUTION_OPTIONS_HPP_
#define ENGINE_EXECUTOR_INCLUDE_EXECUTION_OPTIONS_HPP_

#include <string>
#include <cstdlib>
#include <cstdint>

namespace executor {

/**
 * @brief  Configuration information for executor
 *
 */

enum class ExecutionMode { INFERENCE = 0, DEBUG = 1, TUNING = 2 };

// executor options set, include some optimizations like op tuning mechanism.
// this struct is pybinded to python api.
struct ExecutionOptions {
  // execution mode for executor.
  // default execution mode is INFERENCE. INFERENCE < DEBUG < TUNIUNG.
  // INFERENCE means inference model after compile or op tuning (deployment), and input tensors /
  // output tensors = framework model input tensors / output tensors.
  // DEBUG means user want to investigate the tensors value inside model (e.g.
  // add intermediate tensors into Output node.), it's helpful in sparse model debug process.
  // in DEBUG mode, model will add implicit reorder to make all tensors have same format as framework.
  // TUNING means model will implement op tuning mechanism for getting the best configuration of kernel
  // in you local machine.
  // this member option is exposed to python user, however, it's not recommended to set this option
  // directly and just leave it to executor if you don't know what you are doning.
  ExecutionMode execution_mode = ExecutionMode::INFERENCE;

  // if set it to true, execution mode will become TUNING afterwards.
  // enable tune different kernels or kernel with different configurations.
  // set it to true if you want better performance with fp32 / bf16 Dense model or int8 Sparse model
  // however, it can not guarantee that tuning process must bring better performance (ISA, machine, shape, ... issue).
  bool enable_op_tuning = false;

  // warmup iterations, keep it for some optimizations inside.
  int64_t warmup_iter = 1;

  // set the absolute txt file path of dispatch table.
  // dispatch table will be saved in this path if enable op tuning.
  std::string dispatch_table_file_root = "./engine_dispatch_table.txt";

  // if use activation memory compression engine or not.
  bool activation_mem_compression = getenv("ENGINE_ACTIVATION_MEM_COMPRESSION") != NULL ? true : false;

  // save the activation DAG to disk or not.
  // worked only when activation_mem_compression == true.
  bool dump_activation_dag = false;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_EXECUTION_OPTIONS_HPP_
