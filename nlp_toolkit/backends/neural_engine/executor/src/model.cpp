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

#include "model.hpp"

namespace executor {

Model::Model(const ModelConfig& conf, const string& weight_root) : weight_root_(weight_root) { Init(conf); }

Model::Model(const string& conf_file, const string& weight_root) : weight_root_(weight_root) {
  ModelConfig conf = ModelConfig(conf_file);
  CHECK_EQ(conf.CheckConfig(), true) << "model config not right....";
  Init(conf);
}

Model::~Model() {
  if (engine_profiling_) {
    Profiling();
  }
  if (MemoryAllocator::SharedEnv()) {
    RemoveSharedWeight(false);
  }
}

void Model::Init(const ModelConfig& conf) {
  // Clear the whole dnnl primitive cache map when init engine
  InnerProductPrimitiveFwdFactory::ClearFactory();
  MatMulPrimitiveFwdFactory::ClearFactory();
  ConvolutionPrimitiveFwdFactory::ClearFactory();
  InitSharedWeight();
  name_ = conf.name();
  MemoryAllocator::InitStrategy();
  // For each operator, set up its input and output
  auto op_configs = conf.operators();
  input_vecs_.resize(op_configs.size());
  output_vecs_.resize(op_configs.size());
  // Basically, build all the operators and set up their connections.
  for (int operator_id = 0; operator_id < op_configs.size(); ++operator_id) {
    auto op_conf = op_configs[operator_id];
    auto operator_name = op_conf->name();
    operators_.push_back(std::make_shared<Dispatcher>(*op_conf));
    operator_names_.push_back(operator_name);
    operator_name_index_[operator_name] = operator_id;
    // handle the input/output tensors to the model
    // we will have an input operator only have output data
    // in a graph the input tensors must come from output tensors
    // so we create output tensors and input tensors all take from output
    // tensors besides, we have two operators, one is Input and the other is
    // Output Input only have output tensors and Output only have input tensors
    // we treat weight tensors as output from Input operator and other
    // operators' input
    auto op_type = op_conf->type();

    int output_size = op_conf->output_tensor_size();
    for (int output_id = 0; output_id < output_size; ++output_id) {
      SetOutput(op_configs, operator_id, output_id, &tensor_name_index_);
    }
    int input_size = op_conf->input_tensor_size();
    for (int input_id = 0; input_id < input_size; ++input_id) {
      SetInput(op_configs, operator_id, input_id, &tensor_name_index_);
    }
  }
  // for debug tensor life
  for (size_t i = 0; i < tensors_.size(); ++i) {
    LOG(INFO) << "tensor name is " << tensors_[i]->name() << " tensor life is  " << tensors_[i]->life();
  }
  // prepare the operator like cache weight
  for (int i = 0; i < operators_.size(); ++i) {
    operators_[i]->Prepare(input_vecs_[i], output_vecs_[i]);
    // for profiling post op
    auto attrs = operators_[i]->operator_conf().attributes();
    if (attrs.find("append_op") != attrs.end()) {
      operators_[i]->set_post_op(attrs["append_op"]);
    }
  }
  if (multi_stream_flag) {
    multi_stream_tasks_.clear();
    for (int i = 0; i < operators_.size(); ++i) {
      auto op_attr_map = operators_[i]->operator_conf().attributes();
      auto it = op_attr_map.find("multi_stream");
      if (it != op_attr_map.end()) {
        multi_stream_tasks_.insert({i, StringToNum<int64_t>(it->second)});
      }
    }
  }

  engine_profiling_ = (getenv("ENGINE_PROFILING") != NULL);  // profiling env
  is_dispatcher_tuning_ = (getenv("ENGINE_DISPATCHER_TUNING_ON") != NULL);
  dispatch_table_file_root_ = getenv("ENGINE_DISPATCH_TABLE_FILE_ROOT") == NULL ? \
      string(getenv("HOME")) + "/.cache/neural_engine_workspace/engine_dispatch_table.txt" : \
      getenv("ENGINE_DISPATCH_TABLE_FILE_ROOT");
  has_dispatch_table_file_ = (access(dispatch_table_file_root_.c_str(), F_OK) != -1);
  if (!has_dispatch_table_file_) LOG(INFO) << "Missing dispatch table file, " \
                                  "all operators will use their own default kernels." \
                                  "Recommend to turn on the tuning mode for better performance." \
                                  "Ignore above info if you are doing tuning...";
}

void Model::RemoveSharedWeight(bool is_begin, char* count_space_name, char* count_name, char* space_name) {
  LOG(INFO) << "Shared instance number: " << MemoryAllocator::InstNum();
  ipc::managed_shared_memory count_shm(ipc::open_or_create, count_space_name, 512);
  int* removed_count = count_shm.find_or_construct<int>(count_name)[sizeof(int)](0);
  rmutex_.lock();
  (*removed_count)++;
  if (is_begin) {  // In model init, remove shared space at the first thread
    if (*removed_count == 1) {
      ipc::shared_memory_object::remove(space_name);
    }
    if (*removed_count == MemoryAllocator::InstNum()) {
      ipc::shared_memory_object::remove(count_space_name);
    }
  } else {  // In model release, remove shared space at the last thread
    if (*removed_count == MemoryAllocator::InstNum()) {
      ipc::shared_memory_object::remove(space_name);
      ipc::shared_memory_object::remove(count_space_name);
    }
  }
  rmutex_.unlock();
}

void Model::InitSharedWeight(char* space_name) {
  if (MemoryAllocator::SharedEnv()) {
    RemoveSharedWeight(true);
    std::ifstream inFile(weight_root_, std::ios::in | std::ios::binary);
    size_t weight_size =
        inFile ? static_cast<size_t>(inFile.seekg(0, std::ios::end).tellg()) : static_cast<size_t>(weight_root_.size());
    // 2 * weight_size: an empirical value to check weight buffers could be allocated enough in shared memory
    static ipc::managed_shared_memory managed_shm(ipc::open_or_create, space_name, 2 * weight_size);
  }
}

ipc::managed_shared_memory::handle_t Model::LoadSharedWeight(const string& root, const string& type,
                                                             const vector<int64_t>& shape,
                                                             const vector<int64_t>& location) {
  int64_t size = Product(shape);
  int64_t bytes = size * type2bytes[type];
  string weight_name = std::to_string(location[0]) + std::to_string(location[1]);
  std::ifstream inFile(root, std::ios::in | std::ios::binary);
  void* shm_ptr = MemoryAllocator::ManagedShm().find_or_construct<char>(weight_name.c_str())[bytes](0);
  if (inFile) {
    inFile.seekg(location[0], std::ios::beg);
    inFile.read(reinterpret_cast<char*>(shm_ptr), location[1]);
    inFile.close();
  } else {
    std::memcpy(shm_ptr, &root[location[0]], location[1]);
  }
  const auto& handle = MemoryAllocator::ManagedShm().get_handle_from_address(shm_ptr);
  return handle;
}

void Model::SetInput(const vector<OperatorConfig*>& conf, const int operator_id, const int tensor_id,
                     map<string, int>* tensor_name_index_) {
  // model input tensor not in output tensors
  const OperatorConfig* op_conf = conf[operator_id];
  const string& tensor_name = op_conf->input_tensors(tensor_id)->name();
  if (!tensor_name_index_->count(tensor_name)) {
    LOG(FATAL) << "Unknown input tensor " << tensor_name << ", operator " << op_conf->name() << ", input index "
               << tensor_id;
  }
  const int id = (*tensor_name_index_)[tensor_name];
  // add tensor life count for memory handling
  tensors_[id]->add_tensor_life(1);
  input_vecs_[operator_id].push_back(tensors_[id]);
  // set model output tensors, it maybe a little strange as Output operator only
  // have input and the input is MODEL's output
  const string& op_type = op_conf->type();
  if (op_type == "Output") {
    model_output_tensors_.push_back(tensors_[id]);
    output_tensors_.push_back(Tensor(nullptr, tensors_[id]->shape(), tensors_[id]->dtype()));
  }
}

void Model::SetOutput(const vector<OperatorConfig*>& conf, const int operator_id, const int tensor_id,
                      map<string, int>* tensor_name_index_) {
  const OperatorConfig* op_conf = conf[operator_id];
  const string& tensor_name = op_conf->output_tensors(tensor_id)->name();
  if (tensor_name_index_->count(tensor_name)) {
    LOG(FATAL) << "duplicate output tensor name..." << tensor_name;
  }
  // start from output tensor
  auto tensor_config = op_conf->output_tensors(tensor_id);
  const int id = tensors_.size();
  Tensor* tensor_ptr(new Tensor(*tensor_config));
  tensors_.push_back(tensor_ptr);
  tensor_names_.push_back(tensor_name);
  output_vecs_[operator_id].push_back(tensor_ptr);
  (*tensor_name_index_)[tensor_name] = id;
  // set model input tensors, it maybe a little strange as Input operator only
  // have output and the output is MODEL's input
  const string& op_type = op_conf->type();
  if (op_type == "Input") {
    // parse weight here
    if (tensor_config->location().size() != 0) {
      if (MemoryAllocator::SharedEnv()) {
        auto handle =
            LoadSharedWeight(weight_root_, tensor_config->dtype(), tensor_config->shape(), tensor_config->location());
        tensor_ptr->set_shm_handle(handle);
      } else {
        void* weight_ptr =
            read_file_to_type(weight_root_, tensor_config->dtype(), tensor_config->shape(), tensor_config->location());
        tensor_ptr->set_data(weight_ptr);
      }
      return;
    }
    // set model input tensors
    model_input_tensors_.push_back(tensor_ptr);
    model_input_configs_.push_back(tensor_config);
  }
}

vector<Tensor>& Model::Forward(vector<Tensor>& input_data) {
  CHECK_EQ(input_data.size(), model_input_tensors_.size())
      << "input data size not equal with model input tensor size....";
  // if we want use dynamic input data shape at run time, we should check the
  // input data shape and get the output shape, this should be necessary in each
  // Operator's Forward function
  bool reshape_model = false;
  for (int i = 0; i < input_data.size(); ++i) {
    vector<int64_t> data_shape = input_data[i].shape();
    // here we use model input configs to get the configured shape
    vector<int64_t> model_input_shape = model_input_configs_[i]->shape();
    vector<int64_t> origin_model_input = model_input_tensors_[i]->shape();
    LOG(INFO) << "data shape is " << data_shape[0] << " model config is " << model_input_shape[0] << " origin shape is "
              << origin_model_input[0];
    CHECK_EQ(data_shape.size(), model_input_shape.size()) << "input data should have same "
                                                          << "dimensions with configured model shape....";
    for (int axis = 0; axis < data_shape.size(); ++axis) {
      if (data_shape[axis] != origin_model_input[axis]) {
        // not equal case only happen when model input axis support dynamic in
        // config which axis value should be -1
        CHECK_EQ(model_input_shape[axis], -1) << "data shape mismatch " << data_shape[axis]
                                              << " while model input shape need " << model_input_shape[axis];
        reshape_model = true;
      }
    }
  }
  for (int i = 0; i < input_data.size(); ++i) {
    // model_input_tesnor_[i]->free_data();
    model_input_tensors_[i]->set_data(input_data[i].mutable_data());
    model_input_tensors_[i]->set_shape(input_data[i].shape());
  }

  if (is_dispatcher_tuning_) {
    for (int i = 0; i < operators_.size(); ++i) {
      operators_[i]->GetExecuteKernel(input_vecs_[i], output_vecs_[i], reshape_model,
                                      dispatch_table_file_root_, has_dispatch_table_file_);
    }
  } else {
    if (reshape_model) {
      for (int i = 0; i < operators_.size(); ++i) {
        operators_[i]->GetExecuteKernel(input_vecs_[i], output_vecs_[i], reshape_model,
                                        dispatch_table_file_root_, has_dispatch_table_file_);
      }
    }
  }

  // save dispatch table file after tuniung
  if (is_dispatcher_tuning_ && DispatchTable::Size() > 0) DispatchTable::Save(dispatch_table_file_root_);

  if (!is_dispatcher_tuning_) {
    if (reshape_model) {
      for (int i = 0; i < operators_.size(); ++i) {
        LOG(INFO) << "operator " << operators_[i]->name() << " gonna reshape with type " << operators_[i]->type();
        operators_[i]->Reshape(input_vecs_[i], output_vecs_[i]);
      }
    }
    int thread_count = 1;
    if (engine_profiling_) {
      for (int i = 0; i < operators_.size(); ++i) {
        LOG(INFO) << "operator " << operators_[i]->name() << " gonna forward with type " << operators_[i]->type();
        if (multi_stream_flag && multi_stream_tasks_.find(i) != multi_stream_tasks_.end()) {
          tp.resize(thread_count);
          float start = Time("start");
          tp.commitTask(std::bind(&executor::Dispatcher::Forward, operators_[i], input_vecs_[i], output_vecs_[i]));
          float end = Time("end");
          operators_[i]->set_latency(end - start);
          LOG(INFO) << "operator: " << operators_[i]->name() << ", latency: " << end - start << " ms";
          if (thread_count >= multi_stream_tasks_[i]) {
            tp.waitAllTaskRunOver();
            tp.close();
            thread_count = 0;
          }
          thread_count++;
        } else {
          float start = Time("start");
          operators_[i]->Forward(input_vecs_[i], output_vecs_[i]);
          float end = Time("end");
          operators_[i]->set_latency(end - start);
          LOG(INFO) << "operator: " << operators_[i]->name() << ", latency: " << end - start << " ms";
        }
      }
    } else {
      for (int i = 0; i < operators_.size(); ++i) {
        LOG(INFO) << "operator " << operators_[i]->name() << " gonna forward with type " << operators_[i]->type();
        if (multi_stream_flag && multi_stream_tasks_.find(i) != multi_stream_tasks_.end()) {
          tp.resize(thread_count);
          tp.commitTask(std::bind(&executor::Dispatcher::Forward, operators_[i], input_vecs_[i], output_vecs_[i]));
          if (thread_count >= multi_stream_tasks_[i]) {
            tp.waitAllTaskRunOver();
            tp.close();
            thread_count = 0;
          }
          thread_count++;
        } else {
          operators_[i]->Forward(input_vecs_[i], output_vecs_[i]);
        }
      }
    }
  }
  return this->output_tensors();
}

void Model::Profiling(char* space_name, char* count_name, char* mtx_name, int warm_up) {
  // in multi instance case, dump profiling for each instance
  LOG(INFO) << "Neural engine profiling ...";
  ipc::managed_shared_memory shm(ipc::open_or_create, space_name, 1024);
  int* inst_count = shm.find_or_construct<int>(count_name)(0);
  std::string profiling_dir = "engine_profiling_";
  char ch_curr_time[256];
  if (*inst_count == 0) {
    time_t curr_time = time(NULL);
    strftime(ch_curr_time, sizeof(ch_curr_time), "%Y-%m-%d_%H-%M-%S", localtime(&curr_time));
    profiling_dir += ch_curr_time;
    mkdir(profiling_dir.c_str(), S_IRWXU);  // 00070, read/write/execute for user group
  }
  ipc::interprocess_mutex* mtx = shm.find_or_construct<ipc::interprocess_mutex>(mtx_name)();
  mtx->lock();
  std::string profiling_file = profiling_dir + "/profiling_" + ch_curr_time \
                               + "_" + std::to_string((*inst_count)++) + ".csv";
  FILE* fp = fopen(profiling_file.c_str(), "w");
  if (fp) {
    ProfilingSparse(fp);  // for sparse performance estimation
    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "operator type", "post op",
            "operator name", "input tensor name", "input shape", "input dtype", "output tensor name", "output shape",
            "output dtype", "weight shape", "weight sparse ratio", "sparse support", "operator latency (ms)",
            "aim to weight sparse ratio", "sparse kernel pref ratio", "aim to sparse latency(ms)");
    float total_latency = 0;
    float enable_sparse_latency = 0.;
    // skip input and output node
    for (int i = 1; i < operators_.size()-1; ++i) {
      shared_ptr<Dispatcher>& op = operators_[i];
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
      // operator average iteration latency(exclude warm up)
      float average_latency = op->latency().size() <= warm_up ? \
                        accumulate(op->latency().begin(), op->latency().end(), 0.0)
                        / (op->latency().size()) : \
                        accumulate(op->latency().begin() + warm_up,
                        op->latency().end(), 0.0) / (op->latency().size() - warm_up);
      fprintf(fp, "%.3f, ", average_latency);
      // total latency
      total_latency += average_latency;
      // latency sum of ops which could to be sparse
      enable_sparse_latency += op->enable_sparse() ? average_latency : 0.;
      // for spase performance estimate
      ProfilingSparseEstimate(fp, op, average_latency);
    }
    // dense total latency
    fprintf(fp, ",,,,,,,,,,,%s,%.3f,",
                "total latency(ms)", total_latency);
    // sparse total latency
    string aim_to_sparse_latency_id_begin = "P" + (*(operators_.begin()+1))->table_id();
    string aim_to_sparse_latency_id_end = \
                        "P" + (*(operators_.end()-2))->table_id();
    fprintf(fp, ",%s,=SUM(%s:%s),", "total aim to sparse latency(ms)",
                aim_to_sparse_latency_id_begin.c_str(), aim_to_sparse_latency_id_end.c_str());
    // sparse improve
    string dense_total_id = "M" + \
        std::to_string(std::atoi((*(operators_.end()-2))->table_id().c_str()) + 1);
    string sparse_total_id = "P" + \
        std::to_string(std::atoi((*(operators_.end()-2))->table_id().c_str()) + 1);
    fprintf(fp, "%s,=%s/%s\n", "sparse improve",
                dense_total_id.c_str(), sparse_total_id.c_str());
    // dense matmul/ip latency
    fprintf(fp, ",,,,,,,,,,,%s,%.3f,",
                "sparse support latency(ms)", enable_sparse_latency);
    // sparse matmul/ip latency
    string aim_sparse_latency = "=";
    for (auto op : operators_) {
      if (op->enable_sparse()) {
        string sparse_tmp = "P" + op->table_id() + "+";
        aim_sparse_latency += sparse_tmp;
      }
    }
    aim_sparse_latency = aim_sparse_latency.substr(0, aim_sparse_latency.length()-1) + ",";
    fprintf(fp, ",%s,%s\n", "aim to sparse support latency(ms)", aim_sparse_latency.c_str());
    // dense matmul/ip latency / totoal latency
    fprintf(fp, ",,,,,,,,,,,%s,%.3f,",
                "sparse support latency ratio", enable_sparse_latency / total_latency);
    // sparse matmul/ip latency / totoal latency
    string totol_aim_latency_id = "P" \
                + std::to_string(atoi((*(operators_.end()-2))->table_id().c_str()) + 1);
    string aim_sparse_support_latency_id = "P" \
                + std::to_string(atoi((*(operators_.end()-2))->table_id().c_str()) + 2);
    fprintf(fp, ",%s,=%s/%s\n", "aim to sparse support latency ratio",
            aim_sparse_support_latency_id.c_str(), totol_aim_latency_id.c_str());
    fclose(fp);
  } else {
    LOG(ERROR) << "Open profiling.csv failed!";
  }
  mtx->unlock();
  if (*inst_count == MemoryAllocator::InstNum()) {
    ipc::shared_memory_object::remove(space_name);
  }
}

void Model::ProfilingSparse(FILE* fp) {
  // weight shape, perf ratio, others
  fprintf(fp, "%s,%s,%s,%s,%s\n", "weight shape", "90% 4x1 perf ratio",
          "80% 4x1 perf ratio", "70% 4x1 perf ratio", "others");
  map<vector<int64_t>, float> weight_map;  // need a prior table, now is constant
  for (int i = 1; i < operators_.size()-1; ++i) {
    const shared_ptr<Dispatcher>& op = operators_[i];
    vector<Tensor*>& tensors = input_vecs_[i];
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
  for (auto weight_it=weight_map.begin(); weight_it != weight_map.end(); weight_it++) {
    vector<int64_t> weight_shape = weight_it->first;
    float perf_ratio = weight_it->second;
    perf_ratio_id_map[weight_shape] = std::to_string(++perf_ratio_id_count);
    // weight shape
    for (int j = 0; j < weight_shape.size(); ++j) {
      if (j == weight_shape.size()-1) {
        fprintf(fp, "%ld,", weight_shape[j]);
      } else {
        fprintf(fp, "%ldx", weight_shape[j]);
      }
    }
    // perf ratio
    fprintf(fp, "%f,", perf_ratio);
    fprintf(fp, "%f,", perf_ratio);
    fprintf(fp, "%f,", perf_ratio);
    fprintf(fp, "1\n");
  }

  for (int i = 1; i < operators_.size()-1; ++i) {
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

void Model::ProfilingOperator(FILE* fp, const shared_ptr<Dispatcher>& op) {
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

void Model::ProfilingTensors(FILE* fp, const vector<Tensor*>& tensors) {
  //  tensor name
  for (int i = 0; i < tensors.size(); ++i) {
    if (i == tensors.size()-1) {
      fprintf(fp, "%s,", tensors[i]->name().c_str());
    } else {
      fprintf(fp, "%s;", tensors[i]->name().c_str());
    }
  }
  //  tensor shape
  for (int i = 0; i < tensors.size(); ++i) {
    if (i == tensors.size()-1) {
      for (int j = 0; j < tensors[i]->shape().size(); ++j) {
        if (j == tensors[i]->shape().size()-1) {
          fprintf(fp, "%ld,", tensors[i]->shape()[j]);
        } else {
          fprintf(fp, "%ldx", tensors[i]->shape()[j]);
        }
      }
    } else {
      for (int j = 0; j < tensors[i]->shape().size(); ++j) {
        if (j == tensors[i]->shape().size()-1) {
          fprintf(fp, "%ld;", tensors[i]->shape()[j]);
        } else {
          fprintf(fp, "%ldx", tensors[i]->shape()[j]);
        }
      }
    }
  }
  //  tensor dtype
  for (int i = 0; i < tensors.size(); ++i) {
    if (i == tensors.size()-1) {
      fprintf(fp, "%s,", tensors[i]->dtype().c_str());
    } else {
      fprintf(fp, "%s;", tensors[i]->dtype().c_str());
    }
  }
}

void Model::ProfilingWeights(FILE* fp, const shared_ptr<Dispatcher>& op) {
  if (!op->weight_shape().empty()) {
    vector<int64_t> weight_shape = op->weight_shape();
    // weight shape
    for (int j = 0; j < weight_shape.size(); ++j) {
      if (j == weight_shape.size()-1) {
        fprintf(fp, "%ld,", weight_shape[j]);
      } else {
        fprintf(fp, "%ldx", weight_shape[j]);
      }
    }
    // weight sparse ratio
    fprintf(fp, "%.2f%,", op->weight_zero_ratio() * 100);
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

void Model::ProfilingSparseEstimate(FILE* fp, const shared_ptr<Dispatcher>& op,
                                    const float average_latency) {
  if (op->enable_sparse()) {
    const string aim2sparse_id = "N" + op->table_id();
    fprintf(fp, "90%,");
    fprintf(fp, "\"=IF(%s=90%,%s,IF(%s=80%,%s,IF(%s=70%,%s,%s)))\",",
            aim2sparse_id.c_str(), ("B" + op->perf_ratio_id()).c_str(),
            aim2sparse_id.c_str(), ("C" + op->perf_ratio_id()).c_str(),
            aim2sparse_id.c_str(), ("D" + op->perf_ratio_id()).c_str(),
            ("E" + op->perf_ratio_id()).c_str());
    fprintf(fp, "=%s/%s\n", ("M" + op->table_id()).c_str(), ("O" + op->table_id()).c_str());
  } else {
    fprintf(fp, ",,%.3f\n", average_latency);
  }
}

}  // namespace executor
