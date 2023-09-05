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
#ifndef ENGINE_SPARSELIB_SRC_GPU_OCL_STREAM_HPP_
#define ENGINE_SPARSELIB_SRC_GPU_OCL_STREAM_HPP_

#include <CL/cl.h>
#include "stream.hpp"

namespace jd {
class gpu_ocl_stream_t : public stream_t {
 public:
  explicit gpu_ocl_stream_t(const engine_t* engine);
  ~gpu_ocl_stream_t() {}
  cl_command_queue get_queue() const;

  bool wait() override;

 private:
  cl_command_queue queue_;
};
}  // namespace jd
#endif  // ENGINE_SPARSELIB_SRC_GPU_OCL_STREAM_HPP_
