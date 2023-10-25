#include <ipex.h>
#include <sycl/sycl.hpp>
using namespace sycl;

#include "tests/utils/utils.hpp"
#include "xetla.hpp"
#include <assert.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <math.h>
#include <vector>
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace gblas {
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
} // namespace gblas

class CompressWei4Bit {
public:
  CompressWei4Bit(int K, int N, int blksize, bool sym = false)
      : _K(K), _N(N), _blksize(blksize), _sym(sym) {
    assert(sym == false);
    assert((_K * _N) % 2 == 0); // no consider padding now.
    assert(_K % blksize == 0);
    _write_buf = (char *)malloc(get_buf_size());
  }

  virtual ~CompressWei4Bit() {
    if (_write_buf != nullptr)
      free(_write_buf);
  }

  CompressWei4Bit(void *buf) {
    if (buf != nullptr) {
      size_t offset = deserialize_field(buf);
      _write_buf = (char *)malloc(get_buf_size());
      memcpy(_write_buf, (char *)buf + offset, get_buf_size());
    }
  }

  void serialize(void *buf) {
    size_t offset = 0;
    memcpy((char *)buf + offset, &_N, sizeof(_N));
    offset += sizeof(_N);
    memcpy((char *)buf + offset, &_K, sizeof(_K));
    offset += sizeof(_K);
    memcpy((char *)buf + offset, &_blksize, sizeof(_blksize));
    offset += sizeof(_blksize);
    memcpy((char *)buf + offset, &_sym, sizeof(_sym));
    offset += sizeof(_sym);
    memcpy((char *)buf + offset, _write_buf, get_buf_size());
  }

  void deserialize(void *buf) {
    size_t offset = 0;
    memcpy(&_N, (char *)buf + offset, sizeof(_N));
    offset += sizeof(_N);
    memcpy(&_K, (char *)buf + offset, sizeof(_K));
    offset += sizeof(_K);
    memcpy(&_blksize, (char *)buf + offset, sizeof(_blksize));
    offset += sizeof(_blksize);
    memcpy(&_sym, (char *)buf + offset, sizeof(_sym));
    offset += sizeof(_sym);
    memcpy(_write_buf, (char *)buf + offset, get_buf_size());
  }

  size_t get_serialize_size() { return get_meta_data_size() + get_buf_size(); }

  void *get_4bit_wei_ptr() { return _write_buf; }

  void *get_scale_ptr() { return _write_buf + get_4bit_wei_size(); }
  int _N, _K, _blksize;

private:
  size_t deserialize_field(void *buf) {
    size_t offset = 0;
    memcpy(&_N, (char *)buf + offset, sizeof(_N));
    offset += sizeof(_N);
    memcpy(&_K, (char *)buf + offset, sizeof(_K));
    offset += sizeof(_K);
    memcpy(&_blksize, (char *)buf + offset, sizeof(_blksize));
    offset += sizeof(_blksize);
    memcpy(&_sym, (char *)buf + offset, sizeof(_sym));
    offset += sizeof(_sym);
    return offset;
  }
  size_t get_4bit_wei_size() { return _N * _K / 2; }
  size_t get_scale_size() { return _K / _blksize * _N * sizeof(float); }
  size_t get_zp_size() { return 0; }
  size_t get_buf_size() {
    return get_4bit_wei_size() + get_scale_size() + get_zp_size();
  }
  size_t get_meta_data_size() {
    return sizeof(_N) + sizeof(_K) + sizeof(_blksize) + sizeof(_sym);
  }
  bool _sym;
  char *_write_buf;
};

void s8_quant_row_blk(const float *srcptr, int8_t *dstptr, int row, int col,
                      int ld_src, int ld_dst, float *scales, int blocksize) {
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
    for (; j < align_row_loop; j += blocksize)
      s4_fullrange_calc_store_scale_and_quantv_sym(blocksize);
    if (j < row)
      s4_fullrange_calc_store_scale_and_quantv_sym(row - align_row_loop);
  }
}

void compress_s8_s4(const int8_t *srcptr, gblas::int4x2 *dstptr, int row,
                    int col, int ld_src, int ld_dst) {
  for (int j = 0; j < row; j++) {
    for (int ii = 0; ii < col; ii += 2) {
      gblas::int4x2 tmp;
      tmp.x = gblas::int4x2::convert(srcptr[j * ld_src + ii + 0]);
      tmp.y = gblas::int4x2::convert(srcptr[j * ld_src + ii + 1]);
      dstptr[j * ld_dst / 2 + ii / 2] = tmp;
    }
  }
}

torch::Tensor quantize(float *weight, int k, int n, int blksize, bool transpose,
                       std::string weight_type, std::string cmpt_type) {
  CompressWei4Bit compress_wei(k, n, blksize);
  torch::Tensor ret =
      torch::zeros(compress_wei.get_serialize_size(), torch::kInt8);
  // void* ret = malloc(compress_wei.get_serialize_size());
  assert(!transpose);
  if (weight_type == "s4fullrange_scalef32") {
    std::vector<int8_t> s8quant_tmp(k * n);
    float *scale = reinterpret_cast<float *>(compress_wei.get_scale_ptr());
    s8_quant_row_blk(weight, s8quant_tmp.data(), k, n, n, n, scale, blksize);
    gblas::int4x2 *wei =
        reinterpret_cast<gblas::int4x2 *>(compress_wei.get_4bit_wei_ptr());
    compress_s8_s4(s8quant_tmp.data(), wei, k, n, n, n);
    compress_wei.serialize(ret.data_ptr<int8_t>());
  } else {
    assert(0);
  }
  return ret;
}

template <int TILE_K, int TILE_N, int LOCAL_K, int LOCAL_N, typename DST_T>
void gpu_dequant_s4fullrange_f32_KxN(queue &q, buffer<int8_t, 2> &src,
                                     buffer<DST_T, 2> &dst,
                                     buffer<float, 1> &scale, int k, int n,
                                     int blksize, int k_pos, int n_pos) {
  q.submit([&](handler &h) {
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
void gpu_dequant_s4fullrange_f32_KxN(queue &q, int8_t *src, DST_T *dst,
                                     float *scale, int k, int n, int blksize,
                                     int k_pos, int n_pos) {
  q.submit([&](handler &h) {
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
        dst[i * n + fp32_j + 1] =
            (s8_h - 8) * scale[i / blksize * n + fp32_j + 1];
      }
    });
  });
}

template <typename DST_T>
void gpu_dequant(queue &q, CompressWei4Bit *compress_wei, DST_T *dequant_weight,
                 bool transpose, const std::string &compute_type,
                 const std::string &weight_type) {
  int8_t *bit4_wei =
      reinterpret_cast<int8_t *>(compress_wei->get_4bit_wei_ptr());
  float *scale = reinterpret_cast<float *>(compress_wei->get_scale_ptr());
  buffer<DST_T, 2> dst_buf(dequant_weight,
                           range<2>(compress_wei->_K, compress_wei->_N));
  buffer<float, 1> scale_buf(
      scale,
      range<1>(compress_wei->_K / compress_wei->_blksize * compress_wei->_N));
  buffer<int8_t, 2> src_buf(reinterpret_cast<int8_t *>(bit4_wei),
                            range<2>(compress_wei->_K, compress_wei->_N / 2));
  constexpr int KTILE = 1024, NTILE = 1024;
  constexpr int LOCAL_K = 32, LOCAL_N = 32;
  using namespace std::chrono;
  auto m_start = high_resolution_clock::now();
  for (int i = 0; i < compress_wei->_K; i += KTILE) {
    for (int j = 0; j < compress_wei->_N; j += NTILE) {
      gpu_dequant_s4fullrange_f32_KxN<KTILE, NTILE / 2, LOCAL_K, LOCAL_N>(
          q, src_buf, dst_buf, scale_buf, compress_wei->_K, compress_wei->_N,
          compress_wei->_blksize, i, j);
    }
  }
  q.wait();
  auto m_end = high_resolution_clock::now();
  std::cout << "GPU dequant cost"
            << duration_cast<nanoseconds>(m_end - m_start).count() / 1e6 << "ms"
            << std::endl;
  return;
}

// device mem impl
template <typename DST_T>
void gpu_dequant(queue &q, int8_t *src, DST_T *dst, float *scale, int k, int n,
                 int blksize) {
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

void dequant_dispatch(queue &q, CompressWei4Bit *compress_wei,
                      torch::Tensor &dequant_weight, bool transpose,
                      const std::string &compute_type,
                      const std::string &weight_type) {
  if (compute_type == "fp32") {
    gpu_dequant<float>(q, compress_wei, dequant_weight.data_ptr<float>(),
                       transpose, compute_type, weight_type);
  }
  // else if (compute_type == "bf16")  {
  //     gpu_dequant<__bf16>(q, compress_wei, dequant_weight.data_ptr<__bf16>(),
  //     transpose, compute_type, weight_type);
  // }
  else {
    gpu_dequant<__fp16>(q, compress_wei, dequant_weight.data_ptr<__fp16>(),
                        transpose, compute_type, weight_type);
  }
}

template <typename T1, typename T2>
void gpu_linear(queue &queue, const torch::Tensor &activation, const T1 *weight,
                const torch::Tensor &bias, torch::Tensor &output, int64_t ldo,
                bool with_bias, const std::string &compute_type,
                const std::string &weight_type) {

  // GEMM input size
  uint32_t matrix_m = activation.sizes()[0];
  uint32_t matrix_n = ldo;
  uint32_t matrix_k = activation.sizes()[1];

  uint32_t size_a = matrix_m * matrix_k;
  uint32_t size_b = matrix_k * matrix_n;
  uint32_t size_c = matrix_m * matrix_n;

  using data_type_a = T2;
  using data_type_b = T2;
  using data_type_c = T2;
  using data_type_acc = float;

  // Turn on the profiling property to facilitate subsequent profiling
  sycl::property_list properties{sycl::property::queue::enable_profiling()};

  // Define SYCL context and device
  auto context = queue.get_info<info::queue::context>();
  auto device = queue.get_info<info::queue::device>();

  auto A = alloc_device_and_init<data_type_a>(
      size_a,
      [&activation](data_type_a *data, size_t idx) {
        data[idx] =
            static_cast<data_type_a>(*(activation.data_ptr<T1>() + idx));
      },
      queue, device, context);
  auto B = alloc_device_and_init<data_type_b>(
      size_b,
      [&weight](data_type_b *data, size_t idx) {
        data[idx] = static_cast<data_type_b>(*(weight + idx));
      },
      queue, device, context);
  auto C = alloc_device_and_init<data_type_c>(
      size_c,
      [](data_type_c *data, size_t idx) { data[idx] = static_cast<T1>(0.0f); },
      queue, device, context);

  // Define the shape of workgroup and subgroup
  // It's tunable parameters based on different input shape and hardware for
  // better performance
  constexpr uint32_t wg_tile_m = 256;
  constexpr uint32_t wg_tile_n = 256;
  constexpr uint32_t sg_tile_m = 32;
  constexpr uint32_t sg_tile_n = 64;

  // Workload mapping, linear mapping will be used in the code
  // Suppose it is divisible.
  uint32_t group_range_m = matrix_m / wg_tile_m;
  uint32_t group_range_n = matrix_n / wg_tile_n;

  // Each subgroup will be executed in one hardware thread
  // Calculate how many threads in a workgroup
  uint32_t thread_range_m = wg_tile_m / sg_tile_m;
  uint32_t thread_range_n = wg_tile_n / sg_tile_n;

  // leading dimension
  uint32_t lda = matrix_k;
  uint32_t ldb = matrix_n;
  uint32_t ldc = matrix_n;

  // Ndrange and workgroup shape
  cl::sycl::range<3> group_range{1, group_range_m, group_range_n};
  cl::sycl::range<3> local_range{1, thread_range_m, thread_range_n};

  cl::sycl::nd_range<3> nd_range(group_range * local_range, local_range);

  long ops = 2 * static_cast<long>(matrix_m) * matrix_n * matrix_k;

  auto gpu_event = queue.submit([&](handler &cgh) {
    // GPU kernel
    cgh.parallel_for(nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      using namespace gpu::xetla;
      using namespace gpu::xetla::group;
      using namespace gpu::xetla::kernel;
      using namespace gpu::xetla::subgroup;

      // wrap the nd_range to XeTLA range
      xetla_exec_item<3> ei(item);

      // Step 1: basic computation information
      // define A, B and accumulator datatype
      // Using float as accumuator for better accuracy
      using compute_attr =
          compute_attr_t<data_type_a, data_type_b, data_type_acc>;

      // Performance tuning setting based on different shapes
      static constexpr uint32_t periodic_sync_interval = 0;
      static constexpr uint32_t prefetch_distance = 0;
      // should larger than 8
      static constexpr uint32_t k_stride = 32;
      using perf_tuning_knob = perf_tuning_knob_t<k_stride, prefetch_distance,
                                                  periodic_sync_interval>;

      // specific the computation, performance tuning and computation core
      using compute_policy =
          compute_policy_unaligned_xmx<compute_attr, perf_tuning_knob,
                                       gpu_arch::Xe>;

      // Step 2: define the memory layout & location of input/output
      // this setting could be used to optimize the data re-use in shared
      // local memory
      using mem_desc_input_a =
          mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
      using mem_desc_input_b =
          mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>;
      using mem_desc_output_c =
          mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;

      // Step 3: define mirco-kernel's configuration
      using tile_shape =
          tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
      using gemm_t = gemm_t<compute_policy, tile_shape, mem_desc_input_a,
                            mem_desc_input_b>;
      gemm_t gemm;

      // Step 4: epilogue function to overwrite the result
      using epilogue_t = epilogue_t<epilogue_policy_unaligned<gpu_arch::Xe>,
                                    tile_shape, mem_desc_output_c>;

      // Step 5: define the shared local memory usages
      // developers have the responsibility to set
      // shared loacal memory through XeTLA API
      static constexpr uint32_t barrier_count = gemm_t::barrier_count;
      static constexpr uint32_t slm_size = gemm_t::slm_size;
      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      // Step 6: ecah workgroup gets it individual index to start computation
      int start_n = ei.get_group(2) * wg_tile_n;
      int start_m = ei.get_group(1) * wg_tile_m;
      // no slicing in K direction so start from zero for all WG
      int start_k = 0;

      // Each workgroup will compute all data in K based on no k_sliciing
      // The developer can set how much data a subgroup compute by k_stride
      uint32_t wg_tile_k = matrix_k;
      uint32_t inner_loop_count = wg_tile_k / k_stride;

      // Step 7: define the workgroup start point for each workgroup
      mem_desc_input_a md_a({A}, {matrix_k, matrix_m, lda}, {start_k, start_m});
      mem_desc_input_b md_b({B}, {matrix_n, matrix_k, ldb}, {start_n, start_k});
      mem_desc_output_c md_c({C}, {matrix_n, matrix_m, ldc},
                             {start_n, start_m});

      // Step 8: real calculation with accumulator varibales which suppose
      // will be in register.
      typename gemm_t::matAcc_t matAcc;
      matAcc.init(0);

      typename gemm_t::arguments_t gemm_args(md_a, md_b, inner_loop_count);

      // the results is in the matAcc rather than real output C
      typename gemm_t::work_group_t g(ei.get_local_linear_id());
      gemm(g, matAcc, gemm_args);

      // Step 9: write the results from matACC to real output C
      epilogue_t epilogue;
      epilogue(g, matAcc, md_c);
    });
  });
  gpu_event.wait();
  queue.memcpy(output.data_ptr<T1>(), C, size_c * sizeof(T2)).wait();
  free(A, context);
  free(B, context);
  free(C, context);
}

void linear_dispatch(queue &q, const torch::Tensor &activation,
                     const torch::Tensor weight, const torch::Tensor &bias,
                     torch::Tensor &output, int64_t ldo, bool with_bias,
                     const std::string &compute_type,
                     const std::string &weight_type) {
  if (compute_type == "fp32") {
    gpu_linear<float, gpu::xetla::tf32>(q, activation, weight.data_ptr<float>(),
                                        bias, output, ldo, with_bias,
                                        compute_type, weight_type);
  } else {
    gpu_linear<__fp16, sycl::half>(q, activation, weight.data_ptr<__fp16>(),
                                   bias, output, ldo, with_bias, compute_type,
                                   weight_type);
  }
}
