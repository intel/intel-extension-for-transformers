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

#ifndef ENGINE_EXECUTOR_INCLUDE_PROFILING_HPP_
#define ENGINE_EXECUTOR_INCLUDE_PROFILING_HPP_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include "memory_allocator.hpp"
#include "operator.hpp"
#include "dispatcher.hpp"
#include "tensor.hpp"
#include "common.hpp"

namespace executor {
class ProfilingTracer {
 public:
  ProfilingTracer() : TotalTime(0), iterations_during() {}

  void BeginTrace(const std::string& filepath = "result.json") {
    OutputStream.open(filepath);
    if (!OutputStream) {
      LOG(ERROR) << "Open " << filepath << " failed!";
    }
    TracerHeader();
  }

  void EndTrace() {
    TracerFooter();
    OutputStream.close();
  }

  void WriteProfileTrace(const vector<shared_ptr<Dispatcher>>& operators_, const vector<vector<Tensor*>>& input_tensors,
                         const vector<vector<Tensor*>>& output_tensors) {
    IterationTotalTime(operators_);
    OutputStream << "{";
    OutputStream << "\"cat\":\"inference\",";
    OutputStream << "\"dur\":" << TotalTime * 1000 << ",";
    OutputStream << "\"name\":\""
                 << "model_inference"
                 << "\",";
    OutputStream << "\"ph\":\"X\",";
    OutputStream << "\"pid\": 0,";
    OutputStream << "\"tid\": \""
                 << "inference"
                 << "\",";
    OutputStream << "\"ts\": " << 0;
    OutputStream << "}";
    float iter_start = 0;
    for (int i = 0; i < operators_[1]->latency().size(); ++i) {
      OutputStream << ",";
      OutputStream << "{";
      OutputStream << "\"cat\":\""
                   << "iteration"
                   << "\",";
      OutputStream << "\"dur\":" << iterations_during[i] * 1000 << ",";
      OutputStream << "\"name\":\""
                   << "Iteration" << i << "\",";
      OutputStream << "\"ph\":\"X\",";
      OutputStream << "\"pid\": 0,";
      OutputStream << "\"tid\": \""
                   << "Iteration"
                   << "\",";
      OutputStream << "\"ts\":" << iter_start * 1000;
      OutputStream << "}";
      float op_start = 0;
      for (int j = 1; j < operators_.size() - 1; ++j) {
        const shared_ptr<Dispatcher>& op = operators_[j];
        vector<Tensor*> its = input_tensors[j];
        vector<Tensor*> ots = output_tensors[j];
        OutputStream << ",";
        OutputStream << "{";
        OutputStream << "\"cat\":\"" << op->type() << "\",";
        OutputStream << "\"dur\":" << op->latency()[i] * 1000 + op->get_reshape_time()[i] * 1000 << ",";
        OutputStream << "\"name\":\"" << op->name() << "\",";
        OutputStream << "\"ph\":\"X\",";
        OutputStream << "\"pid\": 0,";
        OutputStream << "\"tid\": \""
                     << "Operator"
                     << "\",";
        OutputStream << "\"ts\":" << (op_start + iter_start) * 1000 << ",";
        OutputStream << "\"args\": {";
        OutputStream << "\"reshape_time\" :\"" << op->get_reshape_time()[i] << "ms"
                     << "\",";
        OutputStream << "\"forward_time\" :\"" << op->latency()[i] << "ms"
                     << "\",";
        OutputStream << "\"input_tensor_name\" :\"" << TensorsName(its) << "\",";
        OutputStream << "\"input_type\" :\"" << TensorsType(its) << "\",";
        OutputStream << "\"input_shape\" :\"" << TensorsShape(op->get_it_shape(), i, its.size()) << "\",";
        OutputStream << "\"output_tensor_name\" :\"" << TensorsName(ots) << "\",";
        OutputStream << "\"output_type\" :\"" << TensorsType(ots) << "\",";
        OutputStream << "\"output_shape\" :\"" << TensorsShape(op->get_ot_shape(), i, 1) << "\",";
        OutputStream << "\"attributes\" :\"" << OpAttrs(op->get_attrs()) << "\"";
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
      for (int j = 1; j < operators_.size() - 1; ++j) {
        PerIterTime += operators_[j]->get_reshape_time()[i];
        PerIterTime += operators_[j]->latency()[i];
      }
      iterations_during.emplace_back(PerIterTime);
      TotalTime += PerIterTime;
    }
  }

  std::string OpAttrs(const std::map<string, string>& attrs) {
    std::string result = "";
    int count = 0;
    for (auto iter = attrs.begin(); iter != attrs.end(); ++iter) {
      string attr;
      if (count == attrs.size() - 1) {
        attr = iter->first + ":" + iter->second;
      } else {
        attr = iter->first + ":" + iter->second + ";";
      }
      result += attr;
      count++;
    }
    return result;
  }

  std::string TensorsName(const vector<Tensor*>& Tensors) {
    std::string result = "";
    for (int i = 0; i < Tensors.size(); ++i) {
      if (i == Tensors.size() - 1) {
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
      if (i == Tensors.size() - 1) {
        result += Tensors[i]->dtype();
      } else {
        result += Tensors[i]->dtype();
        result += ",";
      }
    }
    return result;
  }

  std::string TensorsShape(const vector<vector<int64_t>>& tensor_shape, int iteration_time, int tensor_size) {
    std::string result = "";
    for (int i = iteration_time * tensor_size; i < (iteration_time + 1) * tensor_size; ++i) {
      if (i == (iteration_time + 1) * tensor_size - 1) {
        for (int j = 0; j < tensor_shape[i].size(); ++j) {
          if (j == tensor_shape[i].size() - 1) {
            result += std::to_string(tensor_shape[i][j]);
          } else {
            result += std::to_string(tensor_shape[i][j]);
            result += "*";
          }
        }
      } else {
        for (int j = 0; j < tensor_shape[i].size(); ++j) {
          if (j == tensor_shape[i].size() - 1) {
            result += std::to_string(tensor_shape[i][j]);
            result += ",";
          } else {
            result += std::to_string(tensor_shape[i][j]);
            result += "*";
          }
        }
      }
    }
    return result;
  }

 protected:
  std::ofstream OutputStream;
  float TotalTime;
  vector<float> iterations_during;
};

class Profiling_ {
 public:
  void WriteProfiling(const vector<shared_ptr<Dispatcher>>& operators_, const vector<vector<Tensor*>>& input_vecs_,
                      const vector<vector<Tensor*>>& output_vecs_) {
    // setting permission for shared memory created by boost
    ipc::permissions unrestricted_permissions;
    unrestricted_permissions.set_unrestricted();
    // in multi instance case, dump profiling for each instance
    ipc::managed_shared_memory shm(ipc::open_or_create, space_name, 1024, 0, unrestricted_permissions);
    int* inst_count = shm.find_or_construct<int>(count_name)(0);
    std::string profiling_dir = "engine_profiling";
    std::string profiling_csv_dir = profiling_dir + "/profiling_csv";
    std::string profiling_trace_dir = profiling_dir + "/profiling_trace";
#ifdef _WIN32
    mkdir(profiling_dir.c_str());  // 00070, read/write/execute for user group
    mkdir(profiling_csv_dir.c_str());
    mkdir(profiling_trace_dir.c_str());
#else
    mkdir(profiling_dir.c_str(), S_IRWXU);  // 00070, read/write/execute for user group
    mkdir(profiling_csv_dir.c_str(), S_IRWXU);
    mkdir(profiling_trace_dir.c_str(), S_IRWXU);
#endif
    ipc::interprocess_mutex* mtx = shm.find_or_construct<ipc::interprocess_mutex>(mtx_name)();
    mtx->lock();
    char ch_curr_time[256] = {0};
    time_t curr_time = time(NULL);
    auto timeinfo = localtime(&curr_time);  // NOLINT
    if (timeinfo == nullptr) {
      mtx->unlock();
      return;
    }
    strftime(ch_curr_time, sizeof(ch_curr_time), "%Y-%m-%d_%H-%M-%S", timeinfo);
    std::string csv_file =
        profiling_csv_dir + "/profiling_" + ch_curr_time + "_" + std::to_string((*inst_count)) + ".csv";
    std::string tracer_file =
        profiling_trace_dir + "/profiling_" + ch_curr_time + "_" + std::to_string((*inst_count)) + ".json";
    WriteCSV(csv_file, operators_, input_vecs_, output_vecs_);
    WriteJSON(tracer_file, operators_, input_vecs_, output_vecs_);
    (*inst_count)++;
    mtx->unlock();
    if (*inst_count >= MemoryAllocator::InstNum()) {
      ipc::shared_memory_object::remove(space_name);
    }
  }
  void WriteCSV(const std::string& csv_file, const vector<shared_ptr<Dispatcher>>& operators_,
                const vector<vector<Tensor*>>& input_vecs_, const vector<vector<Tensor*>>& output_vecs_) {
    FILE* fp = fopen(csv_file.c_str(), "w");
    if (fp) {
      ProfilingSparse(fp, operators_, input_vecs_, output_vecs_);  // for sparse performance estimation
      fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "operator type", "post op", "operator name",
              "input tensor name", "input shape", "input dtype", "output tensor name", "output shape", "output dtype",
              "weight shape", "weight sparse ratio", "sparse support", "operator latency (ms)",
              "aim to weight sparse ratio", "sparse kernel pref ratio", "aim to sparse latency(ms)");
      float total_latency = 0;
      float enable_sparse_latency = 0.;
      // skip input and output node
      for (int i = 1; i < operators_.size() - 1; ++i) {
        const shared_ptr<Dispatcher>& op = operators_[i];
        // operator type, operator name, operator attributes
        ProfilingOperator(fp, op);
        // input tensor name, shape, dtype
        vector<Tensor*> its = input_vecs_[i];
        ProfilingTensors(fp, its);
        // output tensor name, shape, dtype
        vector<Tensor*> ots = output_vecs_[i];
        ProfilingTensors(fp, ots);
        // weight shape, sparse enable, zero ratio
        ProfilingWeights(fp, op);
        // record the last infernece latency as profiling data
        float average_latency = *(op->latency().end() - 1);
        fprintf(fp, "%.3f, ", average_latency);
        // total latency
        total_latency += average_latency;
        // latency sum of ops which could to be sparse
        enable_sparse_latency += op->enable_sparse() ? average_latency : 0.;
        // for spase performance estimate
        ProfilingSparseEstimate(fp, op, average_latency);
      }
      ProfilingLatency(fp, operators_, enable_sparse_latency, total_latency);
      fclose(fp);
    } else {
      LOG(ERROR) << "Open " << csv_file << " failed!";
    }
  }
  void WriteJSON(const std::string& tracer_file, const vector<shared_ptr<Dispatcher>>& operators_,
                 const vector<vector<Tensor*>>& input_vecs_, const vector<vector<Tensor*>>& output_vecs_) {
    ProfilingTracer Tracer = ProfilingTracer();
    Tracer.BeginTrace(tracer_file);
    Tracer.WriteProfileTrace(operators_, input_vecs_, output_vecs_);
    Tracer.EndTrace();
  }
  void ProfilingLatency(FILE* fp, const vector<shared_ptr<Dispatcher>>& operators_, float enable_sparse_latency,
                        float total_latency) {
    fprintf(fp, ",,,,,,,,,,,%s,%.3f,", "total latency(ms)", total_latency);
    // sparse total latency
    string aim_to_sparse_latency_id_begin = "P" + (*(operators_.begin() + 1))->table_id();
    string aim_to_sparse_latency_id_end = "P" + (*(operators_.end() - 2))->table_id();
    fprintf(fp, ",%s,=SUM(%s:%s),", "total aim to sparse latency(ms)", aim_to_sparse_latency_id_begin.c_str(),
            aim_to_sparse_latency_id_end.c_str());
    // sparse improve
    string dense_total_id = "M" + std::to_string(std::atoi((*(operators_.end() - 2))->table_id().c_str()) + 1);
    string sparse_total_id = "P" + std::to_string(std::atoi((*(operators_.end() - 2))->table_id().c_str()) + 1);
    fprintf(fp, "%s,=%s/%s\n", "sparse improve", dense_total_id.c_str(), sparse_total_id.c_str());
    // dense matmul/ip latency
    fprintf(fp, ",,,,,,,,,,,%s,%.3f,", "sparse support latency(ms)", enable_sparse_latency);
    // sparse matmul/ip latency
    string aim_sparse_latency = "=";
    for (auto op : operators_) {
      if (op->enable_sparse()) {
        string sparse_tmp = "P" + op->table_id() + "+";
        aim_sparse_latency += sparse_tmp;
      }
    }
    aim_sparse_latency = aim_sparse_latency.substr(0, aim_sparse_latency.length() - 1) + ",";
    fprintf(fp, ",%s,%s\n", "aim to sparse support latency(ms)", aim_sparse_latency.c_str());
    // dense matmul/ip latency / totoal latency
    fprintf(fp, ",,,,,,,,,,,%s,%.3f,", "sparse support latency ratio", enable_sparse_latency / (total_latency + 1e-6));
    // sparse matmul/ip latency / totoal latency
    string totol_aim_latency_id = "P" + std::to_string(atoi((*(operators_.end() - 2))->table_id().c_str()) + 1);
    string aim_sparse_support_latency_id =
        "P" + std::to_string(atoi((*(operators_.end() - 2))->table_id().c_str()) + 2);
    fprintf(fp, ",%s,=%s/%s\n", "aim to sparse support latency ratio", aim_sparse_support_latency_id.c_str(),
            totol_aim_latency_id.c_str());
  }
  void ProfilingSparse(FILE* fp, const vector<shared_ptr<Dispatcher>>& operators_,
                       const vector<vector<Tensor*>>& input_vecs_, const vector<vector<Tensor*>>& output_vecs_) {
    // weight shape, perf ratio, others
    fprintf(fp, "%s,%s,%s,%s,%s\n", "weight shape", "90% 4x1 perf ratio", "80% 4x1 perf ratio", "70% 4x1 perf ratio",
            "others");
    map<vector<int64_t>, float> weight_map;  // need a prior table, now is constant
    for (int i = 1; i < operators_.size() - 1; ++i) {
      const shared_ptr<Dispatcher>& op = operators_[i];
      const vector<Tensor*>& tensors = input_vecs_[i];
      op->set_weight_shape(vector<int64_t>{});
      if (op->type() == "InnerProduct" || op->type() == "Matmul") {
        Tensor* weight_tensor = op->kernel_type() == SparseLib ? tensors[0] : tensors[1];
        int nCount = std::count(output_vecs_[0].begin(), output_vecs_[0].end(), weight_tensor);
        if (nCount > 0) {
          vector<int64_t> weight_shape = weight_tensor->shape();
          op->set_weight_shape(weight_shape);
          // whether could be sparse
          if (weight_shape.size() == 2 && weight_shape[0] % 4 == 0) {
            op->set_enable_sparse(true);
            // users can adjust perf ratio accroding to actual situation
            weight_map[weight_shape] = 4.f;
          }
        }
      }
    }
    map<vector<int64_t>, string> perf_ratio_id_map;
    int perf_ratio_id_count = 1;
    for (auto weight_it = weight_map.begin(); weight_it != weight_map.end(); weight_it++) {
      vector<int64_t> weight_shape = weight_it->first;
      float perf_ratio = weight_it->second;
      perf_ratio_id_map[weight_shape] = std::to_string(++perf_ratio_id_count);
      // weight shape
      for (size_t j = 0; j < weight_shape.size(); ++j) {
        if (j == weight_shape.size() - 1) {
          fprintf(fp, "%lu,", weight_shape[j]);
        } else {
          fprintf(fp, "%lux", weight_shape[j]);
        }
      }
      // perf ratio
      fprintf(fp, "%f,", perf_ratio);
      fprintf(fp, "%f,", perf_ratio);
      fprintf(fp, "%f,", perf_ratio);
      fprintf(fp, "1\n");
    }

    for (int i = 1; i < operators_.size() - 1; ++i) {
      const shared_ptr<Dispatcher>& op = operators_[i];
      vector<int64_t> weight_shape = op->weight_shape();
      string table_id = std::to_string(weight_map.size() + 2 + i);
      op->set_table_id(table_id);
      if (perf_ratio_id_map.count(weight_shape) == 0) {
        op->set_perf_ratio_id("NA");
      } else {
        op->set_perf_ratio_id(perf_ratio_id_map[weight_shape]);
      }
    }
  }

  void ProfilingOperator(FILE* fp, const shared_ptr<Dispatcher>& op) {
    // op type
    fprintf(fp, "%s,", op->type().c_str());
    // post op
    if (!op->post_op().empty()) {
      fprintf(fp, "%s,", op->post_op().c_str());
    } else {
      fprintf(fp, ",");
    }
    // op name
    fprintf(fp, "%s,", op->name().c_str());
  }

  void ProfilingTensors(FILE* fp, const vector<Tensor*>& tensors) {
    //  tensor name
    for (int i = 0; i < tensors.size(); ++i) {
      if (i == tensors.size() - 1) {
        fprintf(fp, "%s,", tensors[i]->name().c_str());
      } else {
        fprintf(fp, "%s;", tensors[i]->name().c_str());
      }
    }
    //  tensor shape
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (i == tensors.size() - 1) {
        for (size_t j = 0; j < tensors[i]->shape().size(); ++j) {
          if (j == tensors[i]->shape().size() - 1) {
            fprintf(fp, "%lu,", tensors[i]->shape()[j]);
          } else {
            fprintf(fp, "%lux", tensors[i]->shape()[j]);
          }
        }
      } else {
        for (int j = 0; j < tensors[i]->shape().size(); ++j) {
          if (j == tensors[i]->shape().size() - 1) {
            fprintf(fp, "%lu;", tensors[i]->shape()[j]);
          } else {
            fprintf(fp, "%lux", tensors[i]->shape()[j]);
          }
        }
      }
    }
    //  tensor dtype
    for (int i = 0; i < tensors.size(); ++i) {
      if (i == tensors.size() - 1) {
        fprintf(fp, "%s,", tensors[i]->dtype().c_str());
      } else {
        fprintf(fp, "%s;", tensors[i]->dtype().c_str());
      }
    }
  }

  void ProfilingWeights(FILE* fp, const shared_ptr<Dispatcher>& op) {
    if (!op->weight_shape().empty()) {
      vector<int64_t> weight_shape = op->weight_shape();
      // weight shape
      for (int j = 0; j < weight_shape.size(); ++j) {
        if (j == weight_shape.size() - 1) {
          fprintf(fp, "%lu,", weight_shape[j]);
        } else {
          fprintf(fp, "%lux", weight_shape[j]);
        }
      }
      // weight sparse ratio
      fprintf(fp, "%.2f%%,", op->weight_zero_ratio() * 100);
      // whether could be sparse
      if (op->enable_sparse()) {
        fprintf(fp, "true,");
      } else {
        op->set_enable_sparse(false);
        fprintf(fp, ",");
      }
    } else {
      op->set_enable_sparse(false);
      fprintf(fp, ",,,");
    }
  }

  void ProfilingSparseEstimate(FILE* fp, const shared_ptr<Dispatcher>& op, const float average_latency) {
    if (op->enable_sparse() && op->weight_zero_ratio() < 0.5) {
      const string aim2sparse_id = "N" + op->table_id();
      fprintf(fp, "90%%,");
      fprintf(fp, "\"=IF(%s=90%%,%s,IF(%s=80%%,%s,IF(%s=70%%,%s,%s)))\",", aim2sparse_id.c_str(),
              ("B" + op->perf_ratio_id()).c_str(), aim2sparse_id.c_str(), ("C" + op->perf_ratio_id()).c_str(),
              aim2sparse_id.c_str(), ("D" + op->perf_ratio_id()).c_str(), ("E" + op->perf_ratio_id()).c_str());
      fprintf(fp, "=%s/%s\n", ("M" + op->table_id()).c_str(), ("O" + op->table_id()).c_str());
    } else {
      fprintf(fp, ",,%.3f\n", average_latency);
    }
  }

 protected:
  char* space_name = "InstCount";
  char* count_name = "inst_count";
  char* mtx_name = "inst_mtx";
  int warm_up = 1;
};

}  // namespace executor

#endif  // ENGINE_EXECUTOR_INCLUDE_PEOFILING_HPP_
