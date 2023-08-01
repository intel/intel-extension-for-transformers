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
#include <assert.h>
#include <sycl/sycl.hpp>
#include "sycl/reduction.hpp"

template <typename T, typename BinaryOperation, size_t VL = 16>
void reduce(const int n, T* s, const T* x, sycl::queue& q) {
  assert(n % VL == 0);

  sycl::buffer<float, 1> buf(const_cast<float*>(x), sycl::range<1>(n));
  sycl::buffer<float, 1> sum_buf(s, sycl::range<1>(1));
  BinaryOperation BOp;
  q.submit([&](auto& h) {
    sycl::accessor buf_acc(buf, h, sycl::read_only);
    auto retr = sycl::reduction(sum_buf, h, BOp);
    h.parallel_for(sycl::nd_range<1>{n, 32}, retr, [=](sycl::nd_item<1> item, auto& retr_arg) {
      int glob_id = item.get_global_id(0);
      retr_arg.combine(buf_acc[glob_id]);
    });
  });
}
