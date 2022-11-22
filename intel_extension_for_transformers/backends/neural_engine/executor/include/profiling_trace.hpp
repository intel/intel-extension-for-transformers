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

#ifndef ENGINE_EXECUTOR_INCLUDE_PROFILING_TRACE_HPP_
#define ENGINE_EXECUTOR_INCLUDE_PROFILING_TRACE_HPP_

#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include "operator.hpp"
#include "dispatcher.hpp"
#include "tensor.hpp"

namespace executor {
class ProfilingTracer {
 public:
  ProfilingTracer() : TotalTime(0), iterations_during() {}

  void BeginTrace(const std::string& filepath = "result.json") {
    OutputStream.open(filepath);
    TracerHeader();
  }

  void EndTrace() {
    TracerFooter();
    OutputStream.close();
  }

  void WriteProfile(const vector<shared_ptr<Dispatcher>>& operators_, const vector<vector<Tensor*>>& input_tensors,
                    const vector<vector<Tensor*>>& output_tensors) {
    IterationTotalTime(operators_);
    OutputStream << "{";
    OutputStream << "\"cat\":\"inference\",";
    OutputStream << "\"dur\":" << TotalTime*1000<< ",";
    OutputStream << "\"name\":\"" << "model_inference" << "\",";
    OutputStream << "\"ph\":\"X\",";
    OutputStream << "\"pid\": 0,";
    OutputStream << "\"tid\": \"" << "inference" << "\",";
    OutputStream << "\"ts\": " << 0;
    OutputStream << "}";
    float iter_start = 0;
    for (int i = 0; i < operators_[1]->latency().size(); ++i) {
      OutputStream << ",";
      OutputStream << "{";
      OutputStream << "\"cat\":\"" << "iteration" << "\",";
      OutputStream << "\"dur\":" << iterations_during[i]*1000 << ",";
      OutputStream << "\"name\":\"" << "Iteration" << i << "\",";
      OutputStream << "\"ph\":\"X\",";
      OutputStream << "\"pid\": 0,";
      OutputStream << "\"tid\": \"" << "Iteration" << "\",";
      OutputStream << "\"ts\":" << iter_start*1000;
      OutputStream << "}";
      float op_start = 0;
      for (int j = 1; j < operators_.size()-1; ++j) {
        const shared_ptr<Dispatcher>& op = operators_[j];
        vector<Tensor*> its = input_tensors[j];
        vector<Tensor*> ots = output_tensors[j];
        OutputStream << ",";
        OutputStream << "{";
        OutputStream << "\"cat\":\"" << op->type() << "\",";
        OutputStream << "\"dur\":" << op->latency()[i]*1000 + op->get_reshape_time()[i]*1000<< ",";
        OutputStream << "\"name\":\"" << op->name() << "\",";
        OutputStream << "\"ph\":\"X\",";
        OutputStream << "\"pid\": 0,";
        OutputStream << "\"tid\": \"" << "Operator" << "\",";
        OutputStream << "\"ts\":" << (op_start + iter_start)*1000<< ",";
        OutputStream << "\"args\": {";
        if (!op->post_op().empty()) {
          OutputStream << "\"post_op\" :\"" << op->post_op() << "\",";
        }
        OutputStream << "\"reshape_time\" :\"" << op->get_reshape_time()[i] << "ms" << "\",";
        OutputStream << "\"forward_time\" :\"" << op->latency()[i] << "ms" << "\",";
        OutputStream << "\"input_tensor_name\" :\"" << TensorsName(its) << "\",";
        OutputStream << "\"input_type\" :\"" << TensorsType(its) << "\",";
        OutputStream << "\"input_shape\" : "<< TensorsShape(op->get_it_shape(), i, its.size()) << ",";
        OutputStream << "\"output_tensor_name\" :\"" << TensorsName(ots) << "\",";
        OutputStream << "\"output_type\" :\"" << TensorsType(ots) << "\",";
        OutputStream << "\"output_shape\" : "<< TensorsShape(op->get_ot_shape(), i, 1);
        OutputStream << "}";
        OutputStream << "}";
        OutputStream.flush();
        op_start += (op->latency()[i] + op->get_reshape_time()[i]);
      }
      iter_start += iterations_during[i];
    }
  }

  void TracerHeader() {
    OutputStream << "{\"otherData\": {}, \"traceEvents\": [";
    OutputStream.flush();
  }

  void TracerFooter() {
    OutputStream << "]}";
    OutputStream.flush();
  }
// get total time and per iteration's time
  void IterationTotalTime(const vector<shared_ptr<Dispatcher>>& operators_) {
    for (int i = 0; i < operators_[1]->latency().size(); ++i) {
      float PerIterTime = 0;
      for (int j = 1; j < operators_.size()-1; ++j) {
        PerIterTime += operators_[j]->get_reshape_time()[i];
        PerIterTime += operators_[j]->latency()[i];
      }
      iterations_during.emplace_back(PerIterTime);
      TotalTime += PerIterTime;
    }
  }

  std::string TensorsName(const vector<Tensor*>& Tensors) {
    std::string result = "";
    for (int i = 0; i < Tensors.size(); ++i) {
      if (i == Tensors.size() -1) {
        result += Tensors[i]->name();
      } else {
          result += Tensors[i]->name();
          result += ",";
        }
    }
    return result;
  }

  std::string TensorsType(const vector<Tensor*>& Tensors) {
    std::string result = "";
    for (int i = 0; i < Tensors.size(); ++i) {
      if (i == Tensors.size() -1) {
        result += Tensors[i]->dtype();
      } else {
        result += Tensors[i]->dtype();
        result += ",";
        }
    }
    return result;
  }

  std::string TensorsShape(const vector<vector<int64_t>>& tensor_shape,
                           int iteration_time, int tensor_size ) {
    std::string result = "\"";
    for (int i = iteration_time*tensor_size; i < (iteration_time + 1)*tensor_size; ++i) {
      if (i == (iteration_time + 1)*tensor_size -1) {
        for (int j = 0; j < tensor_shape[i].size(); ++j) {
        if (j == tensor_shape[i].size()-1) {
          result += std::to_string(tensor_shape[i][j]);
        } else {
            result += std::to_string(tensor_shape[i][j]);
            result += "*";
          }
        }
      } else {
          for (int j = 0; j < tensor_shape[i].size(); ++j) {
            if (j == tensor_shape[i].size()-1) {
              result += std::to_string(tensor_shape[i][j]);
              result += ",";
            } else {
                result += std::to_string(tensor_shape[i][j]);
                result += "*";
              }
            }
        }
    }
    result += "\"";
    return result;
  }

 protected:
  std::ofstream OutputStream;
  float TotalTime;
  vector<float> iterations_during;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_PEOFILING_TRACE_HPP_
