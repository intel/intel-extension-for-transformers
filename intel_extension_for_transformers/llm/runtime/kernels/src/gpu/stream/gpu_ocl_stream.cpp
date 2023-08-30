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
#include "gpu_ocl_stream.hpp"
#include "src/gpu/engine/gpu_ocl_engine.hpp"
#include "src/gpu/kernels/opencl/common.hpp"

namespace jd {
gpu_ocl_stream_t::gpu_ocl_stream_t(const engine_t* engine) : stream_t(engine) {
  const gpu_ocl_engine_t* gpu_ocl_engine = dynamic_cast<const gpu_ocl_engine_t*>(stream_t::get_engine());
  cl_int err;
  queue_ =
      clCreateCommandQueueWithProperties(gpu_ocl_engine->get_context(), gpu_ocl_engine->get_device(), nullptr, &err);

  checkError(err, __LINE__);
}

cl_command_queue gpu_ocl_stream_t::get_queue() const { return queue_; }

bool gpu_ocl_stream_t::wait() {
  clFinish(queue_);
  return true;
}
}  // namespace jd
