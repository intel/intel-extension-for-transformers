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

#include "common.hpp"
#include <cstdio>
#include "src/utils.hpp"

namespace jd {
// Print an error message to screen (only if it occurs)
void checkError(cl_int error, int line) {
  if (error != CL_SUCCESS) {
    switch (error) {
      case CL_DEVICE_NOT_FOUND:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Device not found." << std::endl;
        break;
      case CL_DEVICE_NOT_AVAILABLE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Device not available" << std::endl;
        break;
      case CL_COMPILER_NOT_AVAILABLE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Compiler not available" << std::endl;
        break;
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Memory object allocation failure" << std::endl;
        break;
      case CL_OUT_OF_RESOURCES:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Out of resources" << std::endl;
        break;
      case CL_OUT_OF_HOST_MEMORY:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Out of host memory" << std::endl;
        break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Profiling information not available" << std::endl;
        break;
      case CL_MEM_COPY_OVERLAP:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Memory copy overlap" << std::endl;
        break;
      case CL_IMAGE_FORMAT_MISMATCH:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Image format mismatch" << std::endl;
        break;
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Image format not supported" << std::endl;
        break;
      case CL_BUILD_PROGRAM_FAILURE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Program build failure" << std::endl;
        break;
      case CL_MAP_FAILURE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Map failure" << std::endl;
        break;
      case CL_INVALID_VALUE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid value" << std::endl;
        break;
      case CL_INVALID_DEVICE_TYPE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid device type" << std::endl;
        break;
      case CL_INVALID_PLATFORM:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid platform" << std::endl;
        break;
      case CL_INVALID_DEVICE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid device" << std::endl;
        break;
      case CL_INVALID_CONTEXT:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid context" << std::endl;
        break;
      case CL_INVALID_QUEUE_PROPERTIES:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid queue properties" << std::endl;
        break;
      case CL_INVALID_COMMAND_QUEUE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid command queue" << std::endl;
        break;
      case CL_INVALID_HOST_PTR:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid host pointer" << std::endl;
        break;
      case CL_INVALID_MEM_OBJECT:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid memory object" << std::endl;
        break;
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid image format descriptor" << std::endl;
        break;
      case CL_INVALID_IMAGE_SIZE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid image size" << std::endl;
        break;
      case CL_INVALID_SAMPLER:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid sampler" << std::endl;
        break;
      case CL_INVALID_BINARY:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid binary" << std::endl;
        break;
      case CL_INVALID_BUILD_OPTIONS:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid build options" << std::endl;
        break;
      case CL_INVALID_PROGRAM:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid program" << std::endl;
        break;
      case CL_INVALID_PROGRAM_EXECUTABLE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid program executable" << std::endl;
        break;
      case CL_INVALID_KERNEL_NAME:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid kernel name" << std::endl;
        break;
      case CL_INVALID_KERNEL_DEFINITION:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid kernel definition" << std::endl;
        break;
      case CL_INVALID_KERNEL:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid kernel" << std::endl;
        break;
      case CL_INVALID_ARG_INDEX:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid argument index" << std::endl;
        break;
      case CL_INVALID_ARG_VALUE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid argument value" << std::endl;
        break;
      case CL_INVALID_ARG_SIZE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid argument size" << std::endl;
        break;
      case CL_INVALID_KERNEL_ARGS:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid kernel arguments" << std::endl;
        break;
      case CL_INVALID_WORK_DIMENSION:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid work dimensionsension" << std::endl;
        break;
      case CL_INVALID_WORK_GROUP_SIZE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid work group size" << std::endl;
        break;
      case CL_INVALID_WORK_ITEM_SIZE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid work item size" << std::endl;
        break;
      case CL_INVALID_GLOBAL_OFFSET:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid global offset" << std::endl;
        break;
      case CL_INVALID_EVENT_WAIT_LIST:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid event wait list" << std::endl;
        break;
      case CL_INVALID_EVENT:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid event" << std::endl;
        break;
      case CL_INVALID_OPERATION:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid operation" << std::endl;
        break;
      case CL_INVALID_GL_OBJECT:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid OpenGL object" << std::endl;
        break;
      case CL_INVALID_BUFFER_SIZE:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid buffer size" << std::endl;
        break;
      case CL_INVALID_MIP_LEVEL:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Invalid mip-map level" << std::endl;
        break;
      case -1024:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* Functionality is not implemented" << std::endl;
        break;
      case -1023:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* Library is not initialized yet" << std::endl;
        break;
      case -1022:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* Matrix A is not a valid memory object" << std::endl;
        break;
      case -1021:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* Matrix B is not a valid memory object" << std::endl;
        break;
      case -1020:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* Matrix C is not a valid memory object" << std::endl;
        break;
      case -1019:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* Vector X is not a valid memory object" << std::endl;
        break;
      case -1018:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* Vector Y is not a valid memory object" << std::endl;
        break;
      case -1017:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* An input dimension (M,N,K) is invalid" << std::endl;
        break;
      case -1016:
        SPARSE_LOG(ERROR) << "-- Error at " << line
                          << ":  *clBLAS* Leading dimension A must not be less than the size of the first dimension"
                          << std::endl;
        break;
      case -1015:
        SPARSE_LOG(ERROR) << "-- Error at " << line
                          << ":  *clBLAS* Leading dimension B must not be less than the size of the second dimension"
                          << std::endl;
        break;
      case -1014:
        SPARSE_LOG(ERROR) << "-- Error at " << line
                          << ":  *clBLAS* Leading dimension C must not be less than the size of the third dimension"
                          << std::endl;
        break;
      case -1013:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* The increment for a vector X must not be 0"
                          << std::endl;
        break;
      case -1012:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* The increment for a vector Y must not be 0"
                          << std::endl;
        break;
      case -1011:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* The memory object for Matrix A is too small"
                          << std::endl;
        break;
      case -1010:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* The memory object for Matrix B is too small"
                          << std::endl;
        break;
      case -1009:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* The memory object for Matrix C is too small"
                          << std::endl;
        break;
      case -1008:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* The memory object for Vector X is too small"
                          << std::endl;
        break;
      case -1007:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  *clBLAS* The memory object for Vector Y is too small"
                          << std::endl;
        break;
      case -1001:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Code -1001: no GPU available?" << std::endl;
        break;
      default:
        SPARSE_LOG(ERROR) << "-- Error at " << line << ":  Unknown with code " << line << std::endl;
    }
    exit(1);
  }
}
}  // namespace jd
