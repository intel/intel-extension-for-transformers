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
#include <mpi.h>
#include "oneapi/ccl.hpp"
#include "parallel_context.h"

class parallel_class {
 public:
  parallel_class(const parallel_class& obj) = delete;
  static parallel_class* get_instance() {
    if (instance_p == NULL) {
      instance_p = new parallel_class();
      return instance_p;
    } else {
      return instance_p;
    }
  }
  ~parallel_class() { delete pcomm; }

  bool is_master() { return rank == 0; }

  int get_rank() { return rank; }
  int get_size() { return world_size; }
  // int get_rank() { return 1; }
  // int get_size() { return 2; }

  // From some example code of oneCCL, inplace reducing is supported
  void reduce_add(float* sendBuf, float* recvBuf, size_t count) {
    ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *pcomm).wait();
  }
  void broadcast(float* buf, size_t count) {
    int root = 0;  // assume always broadcast from master
    ccl::broadcast(buf, count, root, *pcomm).wait();
  }
  void alltoall(const float* send_buf, float* recv_buf, size_t count) {
    ccl::alltoall(send_buf, recv_buf, count, *pcomm).wait();
  }
  void barrier() { ccl::barrier(*pcomm); }

 private:
  static parallel_class* instance_p;
  parallel_class() {
    ccl::init();
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(parallel_class::mpi_finalize);

    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      main_addr = kvs->get_address();
      MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
      MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      kvs = ccl::create_kvs(main_addr);
    }

    pcomm = new ccl::communicator(ccl::create_communicator(world_size, rank, kvs));

    rank = pcomm->rank();
    world_size = pcomm->size();
  }
  static void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
      MPI_Finalize();
    }
  }

  int world_size;
  int rank;

  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  ccl::communicator* pcomm;
};

struct parallel_context {
  parallel_class* p_ctx;
};

parallel_class* parallel_class::instance_p = NULL;

parallel_context* init_parallel_context() {
  parallel_context* p_struct = new parallel_context();
  p_struct->p_ctx = parallel_class::get_instance();
  return p_struct;
}

bool is_master(parallel_context* p) { return p->p_ctx->is_master(); }

int get_tp_rank(parallel_context* p) { return p->p_ctx->get_rank(); }

int get_tp_size(parallel_context* p) { return p->p_ctx->get_size(); }

void reduce_add(parallel_context* p, float* send_buffer, float* recv_buffer, size_t count) {
  p->p_ctx->reduce_add(send_buffer, recv_buffer, count);
}

void alltoall(parallel_context* p, float* send_buffer, float* recv_buffer, size_t count) {
  p->p_ctx->alltoall(send_buffer, recv_buffer, count);
}

void broadcast(parallel_context* p, float* buffer, size_t count) { p->p_ctx->broadcast(buffer, count); }

void barrier(parallel_context* p) { p->p_ctx->barrier(); }
