//  Copyright (c) 2023 Intel Corporation
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

#include <numeric>

#include "../../include/activation_dag.hpp"
#include "../../include/common.hpp"
#include "../../include/static_compressed_buffer.hpp"
#include "gtest/gtest.h"

using executor::ActivationDAG;
using executor::StaticCompressedBuffer;
using std::fstream;
using std::map;
using std::shared_ptr;
using std::string;
using std::unordered_map;
using std::vector;

class DagValidEngine {
 public:
  explicit DagValidEngine(const string& path) {
    dag_.Load(path);
    correct_buffer_ = new StaticCompressedBuffer(dag_, true);
    compressed_buffer_ = new StaticCompressedBuffer(dag_);
    for (const auto& op : dag_.operators())
      for (auto&& output_tensor : op->output())
        tensor_size_map_[output_tensor->name()] =
            accumulate(output_tensor->shape().begin(), output_tensor->shape().end(), 1, std::multiplies<int64_t>());
    for (auto&& tensor : tensor_size_map_) {
      int8_t* v1 = static_cast<int8_t*>(correct_buffer_->GetDataByName(tensor.first));
      executor::InitVector(v1, tensor.second, 0, 10);
    }
  }

  ~DagValidEngine() {
    delete correct_buffer_;
    delete compressed_buffer_;
  }

  bool inference() {
    for (auto&& op : dag_.operators()) {
      for (auto&& output_tensor : op->output()) {
        for (auto&& input_tensor : op->input()) {
          int8_t* correct_in_data = static_cast<int8_t*>(correct_buffer_->GetDataByName(input_tensor->name()));
          int8_t* correct_out_data = static_cast<int8_t*>(correct_buffer_->GetDataByName(output_tensor->name()));
          int8_t* compressed_in_data = static_cast<int8_t*>(compressed_buffer_->GetDataByName(input_tensor->name()));
          int8_t* compressed_out_data = static_cast<int8_t*>(compressed_buffer_->GetDataByName(output_tensor->name()));
          auto init_compressed_buffer_value = [&](auto tensor, auto dst_buffer, auto src_buffer) {
            if (appear_tensors_.count(tensor->name()) == 0) memcpy(dst_buffer, src_buffer, get_tensor_size(tensor));
            appear_tensors_[tensor->name()] = true;
          };
          init_compressed_buffer_value(output_tensor, compressed_out_data, correct_out_data);
          init_compressed_buffer_value(input_tensor, compressed_in_data, correct_in_data);
          auto add_iter = std::min(get_tensor_size(input_tensor), get_tensor_size(output_tensor));
          for (int i = 0; i < add_iter; i++) {
            correct_out_data[i] += correct_in_data[i];
            compressed_out_data[i] += compressed_in_data[i];
            if (correct_out_data[i] != compressed_out_data[i]) {
              LOG(INFO) << op->name() << " idx" << i << " val:" << correct_out_data[i] << "vs"
                        << compressed_out_data[i];
              return false;
            }
          }
        }
      }
    }
    return true;
  }

 protected:
  ActivationDAG dag_;
  map<string, size_t> tensor_size_map_;
  unordered_map<string, bool> appear_tensors_;
  StaticCompressedBuffer* correct_buffer_;
  StaticCompressedBuffer* compressed_buffer_;
  size_t get_tensor_size(const shared_ptr<executor::ActivationTensor> tensor) {
    return tensor_size_map_[tensor->name()];
  }
};

bool FileExist(const string& path) {
  fstream f(path);
  return f.good();
}

bool CheckResult(const string& name) {
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  string cwd_path = string(cwd);
  string yaml_path = cwd_path + "/" + name;
  if (!FileExist(yaml_path)) {
    LOG(ERROR) << "File " << name << " not exist, please generate the dag yaml first and copy it to " << cwd_path
               << " first. More details please refer to static_compressed_buffer.md";
    return false;
  }
  DagValidEngine engine(yaml_path);
  return engine.inference();
}

class StaticCompressedBufferTest : public testing::TestWithParam<string> {
 protected:
  StaticCompressedBufferTest() {}
  ~StaticCompressedBufferTest() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_P(StaticCompressedBufferTest, TestPostfix) {
  string t = testing::TestWithParam<string>::GetParam();
  EXPECT_TRUE(CheckResult(t));
}

static auto Cases = [] {
  std::vector<string> cases;
  cases.push_back("llama_bf16.yaml");
  cases.push_back("gptj_bf16.yaml");
  return ::testing::ValuesIn(cases);
};

INSTANTIATE_TEST_SUITE_P(Prefix, StaticCompressedBufferTest, Cases());
