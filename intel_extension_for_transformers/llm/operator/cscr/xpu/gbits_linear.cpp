#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dequant_utils.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif
using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

void linear_fp32(sycl::queue queue, float *A, float *B, float *C,
                       uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k) {
  sycl::image<2> imgA(A, sycl::image_channel_order::rgba, sycl::image_channel_type::fp32,
                      sycl::range<2>{matrix_k / 4, matrix_m});
  sycl::image<2> imgB(B, sycl::image_channel_order::rgba, sycl::image_channel_type::fp32,
                      sycl::range<2>{matrix_n / 4, matrix_k});
  sycl::image<2> imgC(C, sycl::image_channel_order::rgba, sycl::image_channel_type::fp32,
                      sycl::range<2>{matrix_n / 4, matrix_m});

  uint32_t JJ = 8;
  uint32_t II = 2;
  uint32_t JJJ = 8;
  uint32_t III = 32;
  uint32_t I = matrix_m / (II * III);
  uint32_t J = matrix_n / (JJ * JJJ);
  sycl::range<2> GlobalRange(JJ * J, II * I);
  sycl::range<2> LocalRange(JJ, II);
  auto e = queue.submit([&](sycl::handler &cgh) {
    auto A = imgA.get_access<sycl::uint4, sycl::access::mode::read>(cgh);
    auto B = imgB.get_access<sycl::uint4, sycl::access::mode::read>(cgh);
    auto _Out = imgC.get_access<sycl::uint4, sycl::access::mode::write>(cgh);
    cgh.parallel_for<class Test>(
        sycl::nd_range{GlobalRange, LocalRange},
        [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
          static const CONSTANT char FMT[] = "58: %d,59:%d\n";
          static const CONSTANT char FMT1[] = "pos=%d,value=%f\n";
          static const CONSTANT char FMT2[] = "error\n";
          using namespace sycl::ext::intel::esimd;
          int _A_extent_0 = matrix_k;
          int _B_extent_0 = matrix_n;
          int _Out_extent_0 = matrix_n;
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
}

void xetla_linear_fp16(sycl::queue queue, fp16 *A, fp16 *B, fp16 *C,
                       uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k);

template <typename DST_T>
void gpu_dequant(sycl::queue &q, CompressWei4Bit *compress_wei,
                 DST_T *dequant_weight, bool transpose,
                 const std::string &compute_type,
                 const std::string &weight_type);

static void gbits_linear(const torch::Tensor &activation,
                         const torch::Tensor weight, const torch::Tensor &bias,
                         torch::Tensor &output, int64_t ldo, bool with_bias,
                         const std::string &compute_type,
                         const std::string &weight_type) {
  // Turn on the profiling property to facilitate subsequent profiling
  sycl::property_list properties{};

  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto queue = xpu::get_queue_from_stream(c10_stream);

  torch::Tensor revert_weight;
  if (compute_type == "fp32")
    revert_weight = torch::zeros(activation.sizes()[1] * ldo,
    torch::kFloat32);
  else
    revert_weight = torch::zeros(activation.sizes()[1] * ldo,
    torch::kFloat16);
  CompressWei4Bit obj(weight.data_ptr<int8_t>());

  uint32_t matrix_m = activation.sizes()[0];
  uint32_t matrix_n = ldo;
  uint32_t matrix_k = activation.sizes()[1];

  if (compute_type == "fp32") {
    gpu_dequant<float>(queue, &obj, revert_weight.data_ptr<float>(),
                   false, compute_type, weight_type);
    float *A = activation.data_ptr<float>();
    float *B = revert_weight.data_ptr<float>();
    float *C = output.data_ptr<float>();
    linear_fp32(queue, A, B, C, matrix_m, matrix_n, matrix_k);
  }
  else {
    gpu_dequant<at::Half>(queue, &obj, revert_weight.data_ptr<at::Half>(),
                   false, compute_type, weight_type);
    uint32_t size_a = matrix_m * matrix_k;
    uint32_t size_b = matrix_k * matrix_n;
    uint32_t size_c = matrix_m * matrix_n;
    fp16 *A = (fp16 *)malloc_device(sizeof(fp16)*size_a, queue.get_device(), queue.get_context());
    queue.memcpy(A, activation.data_ptr<at::Half>(), sizeof(fp16)*size_a);
    fp16 *B = (fp16 *)malloc_device(sizeof(fp16)*size_b, queue.get_device(), queue.get_context());
    queue.memcpy(B, revert_weight.data_ptr<at::Half>(), sizeof(fp16)*size_b);
    fp16 *C = (fp16 *)malloc_device(sizeof(fp16)*size_c, queue.get_device(), queue.get_context());
    queue.memcpy(C, output.data_ptr<at::Half>(), sizeof(fp16)*size_c);
    std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    //fp16 *A = reinterpret_cast<fp16 *>(activation.data_ptr<at::Half>());
    //fp16 *B = reinterpret_cast<fp16 *>(revert_weight.data_ptr<at::Half>());
    //fp16 *C = reinterpret_cast<fp16 *>(output.data_ptr<at::Half>());
    xetla_linear_fp16(queue, A, B, C, matrix_m, matrix_n, matrix_k);
    queue.memcpy(output.data_ptr<at::Half>(), C, sizeof(fp16)*size_c).wait();
    sycl::free(A, queue.get_context());
    sycl::free(B, queue.get_context());
    sycl::free(C, queue.get_context());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbits_linear, "gbits_linear forward (XPU)");
}
