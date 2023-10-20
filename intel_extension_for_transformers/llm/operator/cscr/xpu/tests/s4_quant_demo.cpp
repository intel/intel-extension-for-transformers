#include <sycl/sycl.hpp>
using namespace sycl;

#include <iostream>
#include <cstring>
#include <assert.h>
#include <vector>
#include <math.h>
#include <chrono>
#define MIN(a, b) ((a) < (b) ? (a) : (b))

inline float randn(float _off = -1.f) { return 2 * (rand() + 0.5f) / (RAND_MAX + 1.f) + _off; }

struct bit4x2 {
  int8_t x : 4;
  int8_t y : 4;
  bit4x2(int8_t v) : x(v), y(v) {}
  bit4x2() : x(0), y(0) {}
};

struct int4x2 : bit4x2 {
  int4x2(int8_t v) : bit4x2(v) {}
  int4x2() : bit4x2() {}
  static int8_t convert(int8_t src) {
    int32_t dst = src;
    dst = dst >= 0 ? dst + 8 : dst - 8;
    dst = dst / 16;
    dst = dst > 7 ? 7 : dst;
    dst = dst < -8 ? -8 : dst;
    return static_cast<int8_t>(dst);
  }
};

class CompressWei4Bit {
 public:
  CompressWei4Bit(int K, int N, int blksize, bool sym = false) : _K(K), _N(N), _blksize(blksize), _sym(sym) {
    assert(sym == false);
    assert((_K * _N) % 2 == 0);  // no consider padding now.
    assert(_K % blksize == 0);
    _write_buf = (char*)malloc(get_buf_size());
  }
  virtual ~CompressWei4Bit() {
    if (_write_buf != nullptr) free(_write_buf);
  }

  void serialize(void* buf) {
    size_t offset = 0;
    memcpy((char*)buf + offset, &_N, sizeof(_N));
    offset += sizeof(_N);
    memcpy((char*)buf + offset, &_K, sizeof(_K));
    offset += sizeof(_K);
    memcpy((char*)buf + offset, &_blksize, sizeof(_blksize));
    offset += sizeof(_blksize);
    memcpy((char*)buf + offset, &_sym, sizeof(_sym));
    offset += sizeof(_sym);
    memcpy((char*)buf + offset, _write_buf, get_buf_size());
  }

  void deserialize(void* buf) {
    size_t offset = 0;
    memcpy(&_N, (char*)buf + offset, sizeof(_N));
    offset += sizeof(_N);
    memcpy(&_K, (char*)buf + offset, sizeof(_K));
    offset += sizeof(_K);
    memcpy(&_blksize, (char*)buf + offset, sizeof(_blksize));
    offset += sizeof(_blksize);
    memcpy(&_sym, (char*)buf + offset, sizeof(_sym));
    offset += sizeof(_sym);
    memcpy(_write_buf, (char*)buf + offset, get_buf_size());
  }

  size_t get_serialize_size() { return get_meta_data_size() + get_buf_size(); }

  void* get_4bit_wei_ptr() { return _write_buf; }

  void* get_scale_ptr() { return _write_buf + get_4bit_wei_size(); }
  int _N, _K, _blksize;

 private:
  size_t get_4bit_wei_size() { return _N * _K / 2; }
  size_t get_scale_size() { return _K / _blksize * _N * sizeof(float); }
  size_t get_zp_size() { return 0; }
  size_t get_buf_size() { return get_4bit_wei_size() + get_scale_size() + get_zp_size(); }
  size_t get_meta_data_size() { return sizeof(_N) + sizeof(_K) + sizeof(_blksize) + sizeof(_sym); }
  bool _sym;
  char* _write_buf;
};

void s8_quant_row_blk(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst, float* scales,
                      int blocksize) {
  int raw_blocksize = blocksize;
  for (int i = 0; i < col; i++) {
    int align_row_loop = row / blocksize * blocksize;
    int j = 0;

    auto s4_fullrange_calc_store_scale_and_quantv_sym = [&](int blocksize) {
      float amax = 0.f, max = 0.f;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto v = srcptr[(j + ij) * ld_src + i];
        if (amax < std::abs(v)) {
          amax = std::abs(v);
          max = v;
        }
      }
      float scale = max / -8.f;
      float rscale = scale != 0.f ? 1.f / scale : 0.f;
      scales[j / raw_blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < blocksize; ij++) {
        auto quant_v = srcptr[(j + ij) * ld_src + i] * rscale;
        int8_t x = MIN(15, (int8_t)(quant_v + 8.5f));
        dstptr[(j + ij) * ld_dst + i] = x << 4;
      }
    };
    for (; j < align_row_loop; j += blocksize) s4_fullrange_calc_store_scale_and_quantv_sym(blocksize);
    if (j < row) s4_fullrange_calc_store_scale_and_quantv_sym(row - align_row_loop);
  }
}

void compress_s8_s4(const int8_t* srcptr, int4x2* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int j = 0; j < row; j++) {
    for (int ii = 0; ii < col; ii += 2) {
      int4x2 tmp;
      tmp.x = int4x2::convert(srcptr[j * ld_src + ii + 0]);
      tmp.y = int4x2::convert(srcptr[j * ld_src + ii + 1]);
      dstptr[j * ld_dst / 2 + ii / 2] = tmp;
    }
  }
}

void* quantize(float* weight, int k, int n, int blksize, bool transpose, std::string weight_type,
               std::string cmpt_type) {
  CompressWei4Bit compress_wei(k, n, blksize);
  void* ret = malloc(compress_wei.get_serialize_size());
  assert(!transpose);
  if (weight_type == "s4fullrange_scalef32") {
    std::vector<int8_t> s8quant_tmp(k * n);
    float* scale = reinterpret_cast<float*>(compress_wei.get_scale_ptr());
    s8_quant_row_blk(weight, s8quant_tmp.data(), k, n, n, n, scale, blksize);
    int4x2* wei = reinterpret_cast<int4x2*>(compress_wei.get_4bit_wei_ptr());
    compress_s8_s4(s8quant_tmp.data(), wei, k, n, n, n);
    compress_wei.serialize(ret);
  } else {
    assert(0);
  }
  return ret;
}

enum SIGN_INT_TYPE { S4_CLIP, S4_FULLRANGE };

template <SIGN_INT_TYPE S4_T>
inline int8_t get_s8(int8_t v) {
  switch (S4_T) {
    case S4_CLIP:
      return v << 4;
    case S4_FULLRANGE:
      v &= 0x0f;
      return v - 8;
    default:
      assert(false);
      break;
  }
  return int8_t(0);
}

template <SIGN_INT_TYPE S4_T>
void decompress_s4_s8(int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst) {
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += 2) {
      auto tmp = srcptr[i * ld_src / 2 + j / 2];
      dstptr[i * ld_dst + j + 0] = get_s8<S4_T>(tmp.x);
      dstptr[i * ld_dst + j + 1] = get_s8<S4_T>(tmp.y);
    }
  }
}

template <typename _DST_T, typename _S_T>
void decompress_kblock_s8_f32(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst, _S_T* scales,
                              int8_t* zero_points, int k_offset, int kblock, int NPad) {
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j += 1) {
      float tmp = (float)(srcptr[i * ld_src + j]);
      if (zero_points != nullptr) tmp -= (float)(zero_points[kpos * NPad + j]);
      dstptr[i * ld_dst + j] = static_cast<_DST_T>(tmp * sptr[j]);
    }
  }
}

template <int TILE_K, int TILE_N, int LOCAL_K, int LOCAL_N>
void gpu_dequant_s4fullrange_f32_KxN(queue& q, buffer<int8_t, 2>& src, buffer<float, 2>& dst, buffer<float, 1>& scale,
                                     int k, int n, int blksize, int k_pos, int n_pos) {
  q.submit([&](handler& h) {
    accessor s4_wei{src, h};
    accessor fp32_wei{dst, h};
    accessor s{scale, h};
    range global{TILE_K, TILE_N};
    range local{LOCAL_K, LOCAL_N};
    h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
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
void gpu_dequant_s4fullrange_f32_KxN(queue& q, int8_t* src, DST_T* dst, float* scale, int k, int n, int blksize,
                                     int k_pos, int n_pos) {
  q.submit([&](handler& h) {
    range global{TILE_K, TILE_N};
    range local{LOCAL_K, LOCAL_N};
    h.parallel_for(nd_range{global, local}, [=](nd_item<2> it) {
      int i = it.get_global_id(0) + k_pos;
      int s4_j = it.get_global_id(1) + n_pos / 2;
      int fp32_j = s4_j * 2;
      if (i < k && fp32_j + 1 < n) {
        int8_t s8_l = src[i * n / 2 + s4_j] & 0x0f;
        int8_t s8_h = (src[i * n / 2 + s4_j] >> 4) & 0x0f;
        dst[i * n + fp32_j] = (s8_l - 8) * scale[i / blksize * n + fp32_j];
        dst[i * n + fp32_j + 1] = (s8_h - 8) * scale[i / blksize * n + fp32_j + 1];
      }
    });
  });
}

void dequantize(CompressWei4Bit* compress_wei, float* dequant_weight, bool transpose, const std::string& compute_type,
                const std::string& weight_type) {
  using namespace std::chrono;
  auto m_start = high_resolution_clock::now();
  assert(!transpose);
  int n = compress_wei->_N;
  int k = compress_wei->_K;
  std::vector<int8_t> s8_wei(k * n);
  int4x2* raw_wei = reinterpret_cast<int4x2*>(compress_wei->get_4bit_wei_ptr());
  float* scale = reinterpret_cast<float*>(compress_wei->get_scale_ptr());
  if (weight_type == "s4fullrange_scalef32") {
    decompress_s4_s8<S4_FULLRANGE>(raw_wei, s8_wei.data(), k, n, n, n);
    decompress_kblock_s8_f32(s8_wei.data(), dequant_weight, k, n, n, n,
                             reinterpret_cast<float*>(compress_wei->get_scale_ptr()), nullptr, 0,
                             compress_wei->_blksize, n);
  } else {
    assert(0);
  }
  auto m_end = high_resolution_clock::now();
  std::cout << "CPU dequant cost" << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms" << std::endl;
}

template <typename DST_T>
void gpu_dequant(queue& q, CompressWei4Bit* compress_wei, DST_T* dequant_weight, bool transpose,
                 const std::string& compute_type, const std::string& weight_type) {
  int8_t* bit4_wei = reinterpret_cast<int8_t*>(compress_wei->get_4bit_wei_ptr());
  float* scale = reinterpret_cast<float*>(compress_wei->get_scale_ptr());
  buffer<DST_T, 2> dst_buf(dequant_weight, range<2>(compress_wei->_K, compress_wei->_N));
  buffer<float, 1> scale_buf(scale, range<1>(compress_wei->_K / compress_wei->_blksize * compress_wei->_N));
  buffer<int8_t, 2> src_buf(reinterpret_cast<int8_t*>(bit4_wei), range<2>(compress_wei->_K, compress_wei->_N / 2));
  constexpr int KTILE = 1024, NTILE = 1024;
  constexpr int LOCAL_K = 32, LOCAL_N = 32;
  using namespace std::chrono;
  auto m_start = high_resolution_clock::now();
  for (int i = 0; i < compress_wei->_K; i += KTILE) {
    for (int j = 0; j < compress_wei->_N; j += NTILE) {
      gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(
          q, src_buf, dst_buf, scale_buf, compress_wei->_K, compress_wei->_N, compress_wei->_blksize, i, j);
    }
  }
  q.wait();
  auto m_end = high_resolution_clock::now();
  std::cout << "GPU dequant cost" << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms" << std::endl;
  return;
}

// device mem impl
template <typename DST_T>
void gpu_dequant(queue& q, int8_t* src, DST_T* dst, float* scale, int k, int n, int blksize) {
  constexpr int KTILE = 1024, NTILE = 1024;
  constexpr int LOCAL_K = 32, LOCAL_N = 32;
  using namespace std::chrono;
  auto m_start = high_resolution_clock::now();
  for (int i = 0; i < k; i += KTILE) {
    for (int j = 0; j < n; j += NTILE) {
      gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(q, src, dst, scale, k, n, blksize, i, j);
    }
  }
  q.wait();
  auto m_end = high_resolution_clock::now();
  std::cout << "GPU dequant cost" << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms" << std::endl;
  return;
}

template <typename DST_T>
void dump_matrix(DST_T* mat, int k, int n) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << mat[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

void ut(int K, int N, int blksize) {
  std::cout << "K:" << K << " N:" << N << " blksize:" << blksize << std::endl;
  assert(N % 2 == 0);
  bool sym = false;
  std::vector<float> f32wei(K * N);
  for (int i = 0; i < K * N; i++) f32wei[i] = randn() * 10;
  void* s4_wei = quantize(f32wei.data(), K, N, blksize, false, "s4fullrange_scalef32", "fp32");
  std::vector<float> gpu_dq(K * N);
  std::vector<__fp16> fp16_gpu_dq(K * N);
  std::vector<float> cpu_dq(K * N);
  queue q;

  CompressWei4Bit obj(K, N, blksize, false);
  obj.deserialize(s4_wei);

  dequantize(&obj, cpu_dq.data(), false, "fp32", "s4fullrange_scalef32");
  int8_t* dev_src = malloc_device<int8_t>(K * N / 2, q);
  float* dev_scale = malloc_device<float>(K / blksize * N, q);
  float* dev_dst = malloc_device<float>(K * N, q);
  __fp16* fp16_dev_dst = malloc_device<__fp16>(K * N, q);

  q.submit([&](handler& h) { h.memcpy(dev_src, obj.get_4bit_wei_ptr(), K * N / 2); });
  q.submit([&](handler& h) { h.memcpy(dev_scale, obj.get_scale_ptr(), K / blksize * N * sizeof(float)); });
  q.wait();

  // gpu_dequant(q, &obj, gpu_dq.data(), false, "fp32", "s4fullrange_scalef32");
  gpu_dequant(q, dev_src, dev_dst, dev_scale, K, N, blksize);
  gpu_dequant(q, dev_src, fp16_dev_dst, dev_scale, K, N, blksize);

  q.submit([&](handler& h) { h.memcpy(gpu_dq.data(), dev_dst, K * N * sizeof(float)); });
  q.submit([&](handler& h) { h.memcpy(fp16_gpu_dq.data(), fp16_dev_dst, K * N * sizeof(__fp16)); });
  q.wait();

  bool ok = true;
  for (int i = 0; i < cpu_dq.size(); i++) {
    if (cpu_dq[i] != gpu_dq[i] || abs(cpu_dq[i] - fp16_gpu_dq[i]) > 0.1) ok = false;
  }
  if (ok)
    std::cout << "ok" << std::endl;
  else
    std::cout << "fail" << std::endl;

  free(s4_wei);
  free(dev_src, q);
  free(dev_scale, q);
  free(dev_dst, q);
  free(fp16_dev_dst, q);
}

int main() {
  ut(1024, 1024, 8);
  ut(1020, 1024, 30);
  ut(11008, 4096, 32);

  return 0;
}