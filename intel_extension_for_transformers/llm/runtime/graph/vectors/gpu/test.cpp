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
// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <vector>

constexpr double Pi = 3.1415926535897932384626433;

template <typename F, typename RA, typename RWA, typename WA>
void reduce(F f, RA src, RWA tmp, WA dst, cl::sycl::nd_item<1> id) {
  auto g = id.get_group().get_id();
  auto bs = id.get_local_range().get(0);
  auto l = id.get_local_id().get(0);

  auto i = g * bs * 2 + l;

  tmp[l] = f(src[i], src[i + bs]);

  id.barrier(cl::sycl::access::fence_space::local_space);

  // do reduction in shared mem
  for (auto s = bs / 2; s > 0; s >>= 1) {
    if (l < s) {
      tmp[l] = f(tmp[l], tmp[l + s]);
    }
    id.barrier(cl::sycl::access::fence_space::local_space);
  }

  // write result for this block to global mem
  if (l == 0) {
    dst[g] = tmp[0];
  }
}

int main() {
  using T = double;

  // Size of vectors
  size_t n = 8192;
  // block size
  size_t local_count = 32;

  // Host vectors
  std::vector<T> h_src(n);
  std::vector<T> h_dst(n);

  // Initialize vectors on host
  for (size_t i = 0; i < n; i++) {
    auto k = n - i;
    h_src[i] = 1.0 / (k * k);
  }

  for (size_t i = 0; i < h_dst.size(); i++) {
    h_dst[i] = 0;
  }

  auto sum = [](auto const& x, auto const& y) { return x + y; };

  try {
    cl::sycl::queue queue{cl::sycl::gpu_selector()};
    std::cout << "Selected platform: " << queue.get_context().get_platform().get_info<cl::sycl::info::platform::name>()
              << "\n";
    std::cout << "Selected device:   " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

    cl::sycl::buffer<T, 1> b_src(h_src.data(), n);
    cl::sycl::buffer<T, 1> b_dst(h_dst.data(), n);

    cl::sycl::nd_range<1> r(n / 2, local_count);

    queue.submit([&](cl::sycl::handler& cgh) {
      auto a_src = b_src.get_access<cl::sycl::access::mode::read>(cgh);

      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> a_tmp(
          cl::sycl::range<1>(local_count), cgh);

      auto a_dst = b_dst.get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class Reduce>(r, [=](cl::sycl::nd_item<1> i) { reduce(sum, a_src, a_tmp, a_dst, i); });
    });
    queue.wait();
  } catch (cl::sycl::exception e) {
    std::cout << "Exception encountered in SYCL: " << e.what() << "\n";
    return -1;
  }

  T res = 0.0;
  for (size_t i = 0; i < h_dst.size(); i++) {
    res = sum(res, h_dst[i]);
  }

  std::cout.precision(16);
  std::cout << "Riemann zeta(2) approximation by explicit summing:\n";
  std::cout << "result = " << res << "\n";
  std::cout << "exact  = " << Pi * Pi / 6.0 << "\n";

  return 0;
}
