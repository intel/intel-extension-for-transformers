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
#ifdef GPU_BACKEND
#include <sycl/sycl.hpp>

template <size_t VL = 16, typename kernel_t, typename kernel_tail_t>
void parallel_for(sycl::queue& q, size_t size, kernel_t kernel, kernel_tail_t kernel_tail) {
  constexpr unsigned GroupSize = 1;

  sycl::range<1> GlobalRange{size / VL};
  sycl::range<1> LocalRange{GroupSize};
  sycl::nd_range<1> Range(GlobalRange, LocalRange);

  sycl::range<1> GlobalRange_tail{size % VL};
  sycl::range<1> LocalRange_tail{GroupSize};
  sycl::nd_range<1> Range_tail(GlobalRange_tail, LocalRange_tail);

  auto e = q.submit([&](sycl::handler& cgh) { cgh.parallel_for(Range, kernel); });
  auto e_tail = q.submit([&](sycl::handler& cgh) { cgh.parallel_for(Range_tail, kernel_tail); });
  e.wait();
  e_tail.wait();
}
// #endif

// Example:
//    float* input;
//    float* output;
//    size_t size = 128 + 1;
//    size_t VL = 16;
//    ...
//    Kernel kernel(input, output);
//    Kernel_tail kernel_tail(input, output);
//    parallel_for<VL>(128, kernel, kernel_tail);
template <size_t VL, typename kernel_t, typename kernel_tail_t>
void parallel_for(size_t size, kernel_t kernel, kernel_tail_t kernel_tail) {
  for (size_t i = 0; i < size; i += VL) {
    kernel(i);
  }
  for (size_t i = size / VL * VL; i < size; i++) {
    kernel_tail(i);
  }
}
#endif
