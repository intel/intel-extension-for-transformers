//  Copyright (c) 2022 Intel Corporation
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
#include "verbose.hpp"
#include "utils.hpp"

namespace jd {

static setting_t<int> verbose{0};
static setting_t<bool> verbose_timestamp{false};

int get_verbose() {
  if (!verbose.initialized()) {
    // Assumes that all threads see the same environment
    int sparselib_verbose_level = 0;
    const char* val = std::getenv("SPARSE_LIB_VERBOSE");
    if (val != nullptr) {
      if (strcmp(val, "1") == 0) {
        sparselib_verbose_level = 1;
      }
    }
    verbose.set(sparselib_verbose_level);
  }

  if (verbose.get() > 0) {
    printf("sparselib_verbose,info,cpu,runtime:CPU,nthr:%d\n", omp_get_max_threads());
  }
  return verbose.get();
}

bool get_verbose_timestamp() {
  if (verbose.get() == 0) return false;

  if (!verbose_timestamp.initialized()) {
    // Assumes that all threads see the same environment
    bool sparselib_verbose_timestamp = false;
    const char* val = std::getenv("VERBOSE_TIMESTAMP");
    if (val != nullptr) {
      if (strcmp(val, "1") == 0) {
        sparselib_verbose_timestamp = true;
      }
    }
    verbose_timestamp.set(sparselib_verbose_timestamp);
  }
  return verbose_timestamp.get();
}

double get_msec() {
  return std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now())
             .time_since_epoch()
             .count() /
         1e3;
}
static std::string init_info_attention(std::vector<dim_t> shape) {
  std::stringstream ss;
  ss << "cpu"
     << ","
     << "attention"
     << ",";

  ss << "shape";
  for (auto& kd_shape_dim : shape) {
    ss << "_" << std::to_string(kd_shape_dim);
  }

  return ss.str();
}

static std::string init_info_layernorm_ba(std::vector<dim_t> shape) {
  std::stringstream ss;
  ss << "cpu"
     << ","
     << "layernorm_ba"
     << ",";

  ss << "shape";
  for (auto& kd_shape_dim : shape) {
    ss << "_" << std::to_string(kd_shape_dim);
  }

  return ss.str();
}

static std::string init_info_eltwiseop(std::vector<dim_t> shape) {
  std::stringstream ss;
  ss << "cpu"
     << ","
     << "eltwiseop"
     << ",";

  ss << "shape";
  for (auto& kd_shape_dim : shape) {
    ss << "_" << std::to_string(kd_shape_dim);
  }

  return ss.str();
}

static std::string init_info_sparse_matmul(std::vector<dim_t> shape) {
  std::stringstream ss;
  ss << "cpu"
     << ","
     << "sparse_matmul"
     << ",";

  ss << "shape";
  for (auto& kd_shape_dim : shape) {
    ss << "_" << std::to_string(kd_shape_dim);
  }

  return ss.str();
}

static std::string init_info_transpose_mha(std::vector<dim_t> shape) {
  std::stringstream ss;
  ss << "cpu"
     << ","
     << "transpose_mha"
     << ",";

  ss << "shape";
  for (auto& kd_shape_dim : shape) {
    ss << "_" << std::to_string(kd_shape_dim);
  }

  return ss.str();
}

static std::string init_info_layernormalized_spmm(std::vector<dim_t> shape) {
  std::stringstream ss;
  ss << "cpu"
     << ","
     << "layernormalized_spmm"
     << ",";

  ss << "shape";
  for (auto& kd_shape_dim : shape) {
    ss << "_" << std::to_string(kd_shape_dim);
  }

  return ss.str();
}

void kd_info_t::init(jd::kernel_kind kind, std::vector<dim_t> shape) {
  if (is_initialized_) return;

  std::call_once(initialization_flag_, [&] {
#define CASE(kind)                  \
  case kernel_kind::kind:           \
    str_ = init_info_##kind(shape); \
    break
    switch (kind) {
      CASE(attention);
      CASE(sparse_matmul);
      CASE(eltwiseop);
      CASE(layernorm_ba);
      CASE(transpose_mha);
      CASE(layernormalized_spmm);
      default:
        SPARSE_LOG(FATAL) << "unknown primitive kind";
    }
#undef CASE

    is_initialized_ = true;
  });
}

}  // namespace jd
