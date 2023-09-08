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
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// For C++ class used in C code
typedef struct parallel_context parallel_context;

enum parallel_mode {
  TENSOR_NO_CHANGE,
  TENSOR_1D_ROW,
  TENSOR_1D_COL,
  TENSOR_2D_ROW,
  TENSOR_2D_COL,

  TENSOR_3D_INPUT,
  TENSOR_3D_WEIGHT,
  TENSOR_3D_OUTPUT,
  TENSOR_3D_INPUT_X_WEIGHT,
  TENSOR_3D_OUTPUT_X_WEIGHT,

  TENSOR_2P5D_ROW,
  TENSOR_2P5D_COL,
  TENSOR_2P5D_DEP
};
parallel_context* init_parallel_context();
int get_tp_size(parallel_context* p);
int get_tp_rank(parallel_context* p);
bool is_master(parallel_context* p);
void barrier(parallel_context* p);
void broadcast(parallel_context* p, float* buffer, size_t count);
void alltoall(parallel_context* p, float* send_buffer, float* recv_buffer, size_t count);
void reduce_add(parallel_context* p, float* send_buffer, float* recv_buffer, size_t count);

#ifdef __cplusplus
}
#endif
