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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "executor.hpp"
#include "pybind_tensor.hpp"
#include "tensor.hpp"
#include "execution_options.hpp"

namespace py = pybind11;

PYBIND11_MODULE(neural_engine_py, m) {
  m.doc() = "pybind11 engine plugin";
  py::class_<executor::Model>(m, "Model")
      .def(py::init<std::string, std::string>())
      .def(py::init<executor::ModelConfig, std::string>())
      .def(py::init<executor::ModelConfig, std::string, executor::ExecutionOptions>())
      .def(py::init<std::string, std::string, executor::ExecutionOptions>())
      .def("forward", &executor::Model::Forward, py::arg("input"), py::return_value_policy::take_ownership)
      .def("activation_mem_compression", &executor::Model::ActivationMemCompression, py::arg("input_shapes"));

  py::class_<executor::TensorConfig, std::shared_ptr<executor::TensorConfig>>(m, "tensor_config")
      .def(py::init<std::string, const std::vector<int64_t>&, std::string, const std::vector<int64_t>&,
                    const std::vector<int64_t>&>());

  py::class_<executor::AttrConfig, std::shared_ptr<executor::AttrConfig>>(m, "attrs_config")
      .def(py::init<const std::map<std::string, std::string>&>());

  py::class_<executor::OperatorConfig, std::shared_ptr<executor::OperatorConfig>>(m, "op_config")
      .def(py::init<std::string, std::string, const std::vector<std::shared_ptr<executor::TensorConfig>>&,
                    const std::vector<std::shared_ptr<executor::TensorConfig>>&,
                    const std::shared_ptr<executor::AttrConfig>&>());

  py::class_<executor::ModelConfig>(m, "model_config")
      .def(py::init<std::string, const std::vector<std::shared_ptr<executor::OperatorConfig>>&>())
      .def(py::init<YAML::Node>());

  py::enum_<executor::ExecutionMode>(m, "ExecutionMode")
      .value("INFERENCE", executor::ExecutionMode::INFERENCE)
      .value("DEBUG", executor::ExecutionMode::DEBUG)
      .value("TUNING", executor::ExecutionMode::TUNING);

  py::class_<executor::ExecutionOptions>(m, "ExecutionOptions")
      .def(py::init<>())
      .def_readwrite("warmup_iter", &executor::ExecutionOptions::warmup_iter)
      .def_readwrite("dispatch_table_file_root", &executor::ExecutionOptions::dispatch_table_file_root)
      .def_readwrite("enable_op_tuning", &executor::ExecutionOptions::enable_op_tuning)
      .def_readwrite("execution_mode", &executor::ExecutionOptions::execution_mode)
      .def_readwrite("activation_mem_compression", &executor::ExecutionOptions::activation_mem_compression)
      .def_readwrite("dump_activation_dag", &executor::ExecutionOptions::dump_activation_dag);
}
