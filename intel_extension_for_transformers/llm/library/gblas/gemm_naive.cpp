#include <time.h>

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>

#include "esimd_test_utils.hpp"
#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif
// // Group Level
// #define     I       32
// #define     J       32
// #define     K       128
// // Thread Group Level
// #define     II      2
// #define     JJ      8
// #define     KK      1

#define K 128
#define J 32
#define I 32
#define JJ 8
#define II 2
#define KK 1
#define KKK 8
#define JJJ 8
#define III 32

// Submatrix for Every Thread

#define TOTAL_I III *II *I
#define TOTAL_J JJJ *JJ *J
#define TOTAL_K KKK *KK *K
//  Size of Matrix A
#define SIZE_A TOTAL_I *TOTAL_K
#define SIZE_B TOTAL_K *TOTAL_J
#define SIZE_C TOTAL_I *TOTAL_J
double EPS = 0.000001;
using namespace cl::sycl;
double get_event_time(event e) {
  double start = e.get_profiling_info<info::event_profiling::command_start>();
  double end = e.get_profiling_info<info::event_profiling::command_end>();
  return end - start;
}
int main() {
  srand(time(NULL));
  // Initialize the device queue with the default selector
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});
  // int A_acc[SIZE_A];
  // int B_acc[SIZE_B];
  // int C_acc[SIZE_C];
  // vec<float,4>
  float *A = new float[SIZE_A];
  float *B = new float[SIZE_B];
  float *C = new float[SIZE_C];
  float *ans = new float[SIZE_C];
  auto dev = q.get_device();
  for (unsigned i = 0; i < SIZE_A; ++i) {
    A[i] = rand() % 5;
    A[i] = 1;
  }
  for (unsigned i = 0; i < SIZE_B; ++i) {
    B[i] = rand() % 5;
    B[i] = i;
  }

  for (unsigned i = 0; i < SIZE_C; ++i) {
    C[i] = 0;
    ans[i] = 0;
  }
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  range<2> GlobalRange(JJ * J, II * I);
  range<2> LocalRange(JJ, II);
  double _time;
  double ops = (long)TOTAL_I * (long)TOTAL_J * (long)TOTAL_K * 2.0;
  try {
    sycl::image<2> imgA(A, image_channel_order::rgba, image_channel_type::fp32,
                        range<2>{TOTAL_K / 4, TOTAL_I});
    sycl::image<2> imgB(B, image_channel_order::rgba, image_channel_type::fp32,
                        range<2>{TOTAL_J / 4, TOTAL_K});
    sycl::image<2> imgC(C, image_channel_order::rgba, image_channel_type::fp32,
                        range<2>{TOTAL_J / 4, TOTAL_I});
    auto e = q.submit([&](handler &cgh) {
      auto A = imgA.get_access<uint4, access::mode::read>(cgh);
      auto B = imgB.get_access<uint4, access::mode::read>(cgh);
      auto _Out = imgC.get_access<uint4, access::mode::write>(cgh);
      cgh.parallel_for<class Test>(
          nd_range{GlobalRange, LocalRange},
          [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            static const CONSTANT char FMT[] = "58: %d,59:%d\n";
            static const CONSTANT char FMT1[] = "pos=%d,value=%f\n";
            static const CONSTANT char FMT2[] = "error\n";
            using namespace sycl::ext::intel::esimd;
            int _A_extent_0 = TOTAL_K;
            int _B_extent_0 = TOTAL_J;
            int _Out_extent_0 = TOTAL_J;
            int _X_s0_i___block_id_y = ndi.get_group(1);
            int _X_s0_j___block_id_x = ndi.get_group(0);
            int ___thread_id_y = ndi.get_local_id(1);
            int ___thread_id_x = ndi.get_local_id(0);
            simd<float, 256> _Z;
            simd<float, 256> _Y;
            simd<float, 256> _X;
            simd<float, 256> _A_im_buf;
            simd<float, 64> _B_im_buf;
            _Z.select<256, 1>(0) = 0.000000;
            for (int _X_s0_k = 0; _X_s0_k < 0 + (_A_extent_0 / 8); _X_s0_k++) {
              int _54 = (_X_s0_k * 8);
              int _55 = (((_X_s0_j___block_id_x * 8) + ___thread_id_x) * 8);
              _B_im_buf.select<64, 1>(0).bit_cast_view<float, 8, 8>() =
                  media_block_load<float, 8, 8>(B, (_55 * 4), (_54 + 0));
              // for (size_t posa = 0; posa < 64; posa++) {
              //   if (_B_im_buf[posa] != 1) {
              //     sycl::ext::oneapi::experimental::printf(FMT2);
              //   }
              // }
              int _56 = (((_X_s0_i___block_id_y * 2) + ___thread_id_y) * 32);
              int _57 = (_X_s0_k * 8);
              _A_im_buf.select<256, 1>(0)
                  .bit_cast_view<float, 32, 8>()
                  .select<8, 1, 8, 1>(0, 0) =
                  media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 0));
              _A_im_buf.select<256, 1>(0)
                  .bit_cast_view<float, 32, 8>()
                  .select<8, 1, 8, 1>(8, 0) =
                  media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 8));
              _A_im_buf.select<256, 1>(0)
                  .bit_cast_view<float, 32, 8>()
                  .select<8, 1, 8, 1>(16, 0) =
                  media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 16));
              _A_im_buf.select<256, 1>(0)
                  .bit_cast_view<float, 32, 8>()
                  .select<8, 1, 8, 1>(24, 0) =
                  media_block_load<float, 8, 8>(A, (_57 * 4), (_56 + 24));
          // for (size_t posa = 0; posa < 256; posa++) {
          //   if (_A_im_buf[posa] != 1) {
          //     sycl::ext::oneapi::experimental::printf(FMT2);
          //   }
          // }

          // for (int posa = 0; posa < 256; posa++) {
          //   sycl::ext::oneapi::experimental::printf(FMT1, posa,
          //   _A_im_buf[posa]);
          // }
#pragma unroll
              for (int _X_s0_iii = 0; _X_s0_iii < 0 + 32; _X_s0_iii++) {
            // tile -> vnni_format
#pragma unroll
                for (int _X_s0_kkk = 0; _X_s0_kkk < 0 + 8; _X_s0_kkk++) {
                  _X.select<8, 1>((_X_s0_iii * 8)) =
                      _A_im_buf.select<8, 0>(((_X_s0_iii * 8) + _X_s0_kkk));
                  _Y.select<8, 1>((_X_s0_iii * 8)) =
                      _B_im_buf.select<8, 1>((_X_s0_kkk * 8));
                  _Z.select<8, 1>((_X_s0_iii * 8)) =
                      (_Z.select<8, 1>((_X_s0_iii * 8)) +
                       (_X.select<8, 1>((_X_s0_iii * 8)) *
                        _Y.select<8, 1>((_X_s0_iii * 8))));
                }  // for _X_s0_kkk
              }    // for _X_s0_iii
            }      // for _X_s0_k
            int _58 = (((_X_s0_i___block_id_y * 2) + ___thread_id_y) * 32);
            int _59 = (((_X_s0_j___block_id_x * 8) + ___thread_id_x) * 8);
            // sycl::ext::oneapi::experimental::printf(FMT, _58, _59);
            // for (int pos = 0; pos < 256; pos++) {
            //   sycl::ext::oneapi::experimental::printf(FMT1, pos, _Z[pos]);
            // }
            // for (size_t posa = 0; posa < 256; posa++) {
            //   if (_Z[posa] != 8) {
            //     sycl::ext::oneapi::experimental::printf(FMT2);
            //   }
            // }
            media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 0),
                                           _Z.select<256, 1>(0)
                                               .bit_cast_view<float, 32, 8>()
                                               .select<8, 1, 8, 1>(0, 0));
            media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 8),
                                           _Z.select<256, 1>(0)
                                               .bit_cast_view<float, 32, 8>()
                                               .select<8, 1, 8, 1>(8, 0));
            media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 16),
                                           _Z.select<256, 1>(0)
                                               .bit_cast_view<float, 32, 8>()
                                               .select<8, 1, 8, 1>(16, 0));
            media_block_store<float, 8, 8>(_Out, (_59 * 4), (_58 + 24),
                                           _Z.select<256, 1>(0)
                                               .bit_cast_view<float, 32, 8>()
                                               .select<8, 1, 8, 1>(24, 0));
          }  // kernel kernel_X
      );
    });
    e.wait();
    _time = get_event_time(e);
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  for (int i = 0; i < TOTAL_I; i++) {
    for (int j = 0; j < TOTAL_J; j++) {
      for (int k = 0; k < TOTAL_K; k++) {
        ans[i * TOTAL_J + j] += A[i * TOTAL_K + k] * B[k * TOTAL_J + j];
      }
    }
  }

  int b = 1;
  for (int i = 0; i < SIZE_C; i++) {
    if ((abs(C[i] - ans[i]) / ans[i]) > EPS) {
      b = 0;
      std::cout << "Err: _Out[" << i << "]=" << setiosflags(std::ios::fixed)
                << C[i] << ", ans[" << i << "]= " << ans[i] << "." << std::endl;
      break;  // print all errors
    }
  }
  if (b == 1) {
    printf("All is done.\n");
  }
  std::cout << "time:" << _time * 1.0e-9f << "s\n";
  std::cout << "Size of matrix A: " << TOTAL_I << " * " << TOTAL_K << "\n";
  std::cout << "Size of matrix B: " << TOTAL_K << " * " << TOTAL_J << "\n";
  printf("GFlops: %lf\n", ops / _time);
  return 0;
}
