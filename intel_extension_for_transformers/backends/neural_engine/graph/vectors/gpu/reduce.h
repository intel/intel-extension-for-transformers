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
