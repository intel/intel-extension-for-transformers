#include <torch/extension.h>
#include <ipex.h>
#include "common.hpp"

template <int TILE_K, int TILE_N, int LOCAL_K, int LOCAL_N, typename DST_T>
void gpu_dequant_s4fullrange_f32_KxN(sycl::queue &q,
                                     sycl::buffer<int8_t, 2> &src,
                                     sycl::buffer<DST_T, 2> &dst,
                                     sycl::buffer<fp16, 1> &scale, int k,
                                     int n, int blksize, int k_pos, int n_pos) {
  q.submit([&](sycl::handler &h) {
    sycl::accessor s4_wei{src, h};
    sycl::accessor fp32_wei{dst, h};
    sycl::accessor s{scale, h};
    sycl::range global{TILE_K, TILE_N};
    sycl::range local{LOCAL_K, LOCAL_N};
    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
      int i = it.get_global_id(0) + k_pos;
      int s4_j = it.get_global_id(1) + n_pos / 2;
      int fp32_j = s4_j * 2;
      if (i < k && fp32_j + 1 < n) {
        int8_t s8_l = s4_wei[i][s4_j] & 0x0f;
        int8_t s8_h = (s4_wei[i][s4_j] >> 4) & 0x0f;
        fp32_wei[i][fp32_j] = (s8_l - 8) * s[i / blksize * n + fp32_j];
        fp32_wei[i][fp32_j + 1] = (s8_h - 8) * s[i / blksize * n + fp32_j + 1];
      }
    });
  });
}

template <int TILE_K, int TILE_N, int LOCAL_K, int LOCAL_N, typename DST_T>
void gpu_dequant_s4fullrange_f32_KxN(sycl::queue &q, int8_t *src, DST_T *dst,
                                     fp16 *scale, int k, int n, int blksize,
                                     int k_pos, int n_pos) {
  q.submit([&](sycl::handler &h) {
    sycl::range global{TILE_K, TILE_N};
    sycl::range local{LOCAL_K, LOCAL_N};
    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
      int i = it.get_global_id(0) + k_pos;
      int s4_j = it.get_global_id(1) + n_pos / 2;
      int fp32_j = s4_j * 2;
      if (i < k && fp32_j + 1 < n) {
        int8_t s8_l = src[i * n / 2 + s4_j] & 0x0f;
        int8_t s8_h = (src[i * n / 2 + s4_j] >> 4) & 0x0f;
        dst[i * n + fp32_j] = (s8_l - 8) * scale[i / blksize * n + fp32_j];
        dst[i * n + fp32_j + 1] =
            (s8_h - 8) * scale[i / blksize * n + fp32_j + 1];
      }
    });
  });
}

template <typename DST_T>
void gpu_dequant(sycl::queue &q, CompressWei4Bit *compress_wei,
                 DST_T *dequant_weight, bool transpose,
                 const std::string &compute_type,
                 const std::string &weight_type) {
  int8_t *bit4_wei =
      reinterpret_cast<int8_t *>(compress_wei->get_4bit_wei_ptr());
  fp16 *scale = reinterpret_cast<fp16 *>(compress_wei->get_scale_ptr());
  sycl::buffer<fp16, 1> scale_buf(
      scale, sycl::range<1>(compress_wei->_K / compress_wei->_blksize *
                            compress_wei->_N));
  sycl::buffer<int8_t, 2> src_buf(
      reinterpret_cast<int8_t *>(bit4_wei),
      sycl::range<2>(compress_wei->_K, compress_wei->_N / 2));
  constexpr int KTILE = 1024, NTILE = 1024;
  constexpr int LOCAL_K = 32, LOCAL_N = 32;
  using namespace std::chrono;
  auto m_start = high_resolution_clock::now();
  if (transpose) {
    float *tmp_buf = new float[compress_wei->_K * compress_wei->_N];
    sycl::buffer<DST_T, 2> dst_buf(
        tmp_buf, sycl::range<2>(compress_wei->_K, compress_wei->_N));
    for (int i = 0; i < compress_wei->_K; i += KTILE) {
      for (int j = 0; j < compress_wei->_N; j += NTILE) {
        gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(
            q, src_buf, dst_buf, scale_buf, compress_wei->_K, compress_wei->_N,
            compress_wei->_blksize, i, j);
      }
    }
    q.wait();
    transpose2d<float>(tmp_buf, dequant_weight, compress_wei->_K, compress_wei->_N, compress_wei->_N, compress_wei->_K);
    delete[] tmp_buf;
  }
  else {
    sycl::buffer<DST_T, 2> dst_buf(
        tmp_buf, sycl::range<2>(compress_wei->_K, compress_wei->_N));
    for (int i = 0; i < compress_wei->_K; i += KTILE) {
      for (int j = 0; j < compress_wei->_N; j += NTILE) {
        gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(
            q, src_buf, dst_buf, scale_buf, compress_wei->_K, compress_wei->_N,
            compress_wei->_blksize, i, j);
      }
    }
    q.wait();
  }
  auto m_end = high_resolution_clock::now();
  std::cout << "GPU dequant cost"
            << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms"
            << std::endl;
  return;
}

// device mem impl
template <typename DST_T>
void gpu_dequant(sycl::queue &q, int8_t *src, DST_T *dst, fp16 *scale, int k,
                 int n, int blksize) {
  constexpr int KTILE = 1024, NTILE = 1024;
  constexpr int LOCAL_K = 32, LOCAL_N = 32;
  using namespace std::chrono;
  auto m_start = high_resolution_clock::now();
  for (int i = 0; i < k; i += KTILE) {
    for (int j = 0; j < n; j += NTILE) {
      gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(
          q, src, dst, scale, k, n, blksize, i, j);
    }
  }
  q.wait();
  auto m_end = high_resolution_clock::now();
  std::cout << "GPU dequant cost"
            << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms"
            << std::endl;
  return;
}