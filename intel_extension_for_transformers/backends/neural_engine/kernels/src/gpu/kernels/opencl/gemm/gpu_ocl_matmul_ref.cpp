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

#include "gpu_ocl_matmul_ref.hpp"
#include "src/gpu/engine/gpu_ocl_engine.hpp"
#include "src/gpu/stream/gpu_ocl_stream.hpp"
#include "src/gpu/kernels/opencl/common.hpp"

namespace jd {
inline const char* get_kernels() {
  static const char* kernels =
#include "gpu_ocl_matmul_ref.cl"
      ;  // NOLINT
  return kernels;
}

bool gpu_ocl_matmul_ref_k_t::init() { return true; }
bool gpu_ocl_matmul_ref_k_t::init(const exec_context_t& context) {
  cl_int err;
  gpu_ocl_stream_t* stream = dynamic_cast<gpu_ocl_stream_t*>(context.get_stream());
  cl_queue_ = stream->get_queue();
  const gpu_ocl_engine_t* engine = dynamic_cast<const gpu_ocl_engine_t*>(stream->get_engine());
  const char* constCode = get_kernels();
  cl_program program = clCreateProgramWithSource(engine->get_context(), 1, &constCode, nullptr, &err);
  checkError(err, __LINE__);
  err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
  checkError(err, __LINE__);
  cl_kernel_ = clCreateKernel(program, "matmul", &err);
  checkError(err, __LINE__);
  int index = 0;
  for (const auto& arg : context.inputs()) {
    handle_.emplace_back();
    arg->get_handle(&handle_.back());
    if (index < 3) {
      // arg is int
      clSetKernelArg(cl_kernel_, index, arg->get_ptr_size(), handle_.back());
    } else {
      // arg is clmem
      clSetKernelArg(cl_kernel_, index, arg->get_ptr_size(), &handle_.back());
    }
    index++;
  }
  for (const auto& arg : context.outputs()) {
    handle_.emplace_back();
    arg->get_handle(&handle_.back());
    clSetKernelArg(cl_kernel_, index, arg->get_ptr_size(), &handle_.back());
    index++;
  }
  return err == CL_SUCCESS;
}

bool gpu_ocl_matmul_ref_k_t::execute() const {
  cl_int err;
  err = clEnqueueNDRangeKernel(cl_queue_, cl_kernel_, 2, nullptr, global_, local_, 0, nullptr, nullptr);
  checkError(err, __LINE__);
  return err == CL_SUCCESS;
}
}  // namespace jd
