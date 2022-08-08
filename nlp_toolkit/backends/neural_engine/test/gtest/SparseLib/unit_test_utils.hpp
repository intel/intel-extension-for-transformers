#include <chrono>
#include <glog/logging.h>
#include "interface.hpp"
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <set>

#define exp_ln_flt_max_f 0x42b17218
#define exp_ln_flt_min_f 0xc2aeac50

enum memo_mode { MALLOC, MEMSET };

// set mock regs.
std::vector<std::string> string_split(const std::string& str, char delim) {
  std::stringstream ss(str);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      elems.push_back(item);
    }
  }
  return elems;
};

double ms_now() {
  auto timePointTmp = std::chrono::high_resolution_clock::now().time_since_epoch();
  return std::chrono::duration<double, std::milli>(timePointTmp).count();
}

unsigned short int fp32_2_bf16(float float_val) { return (*reinterpret_cast<unsigned int*>(&float_val)) >> 16; }

float bf16_2_fp32(unsigned short int bf16_val) {
  unsigned int ans = bf16_val << 16;
  return *reinterpret_cast<float*>(&ans);
}

int uint8_2_int32(uint8_t a) {
  int ans = a;
  return ans;
}

float rand_float_postfix() { return rand() / float(RAND_MAX); }

float get_exp(float x) {
  unsigned int max = exp_ln_flt_max_f;
  unsigned int min = exp_ln_flt_min_f;
  float fmax = *reinterpret_cast<float*>(&max);
  float fmin = *reinterpret_cast<float*>(&min);
  if (x > fmax) {
    return INFINITY;
  } else if (x < fmin) {
    return 0;
  } else {
    return expf(x);
  }
}

float get_gelu(float x) {
  // an approximate fitting function of GELU(x)
  // GELU(x)≈0.5x(1+tanh[(2/pi)^0.5)*(x+0.044715x^3)]
  // for more details,pls refer this paper:https://arxiv.org/abs/1606.08415
  // printf("gelu\n");
  return 0.5 * x * (1 + tanhf(0.7978845834732056 * (x + 0.044714998453855515 * x * x * x)));
}

float get_relu(float x, float alpha) { return x > 0 ? x : alpha * x; }

int get_quantize(float x, float alpha, float scale) {
  x = scale * (x - alpha);
  int ans = std::round(x);
  ans = ans > 255 ? 255 : ans;
  ans = ans < 0 ? 0 : ans;
  return ans;
}

float get_dequantize(float x, float alpha, float scale) {
  uint8_t tmp = x;
  float ans = tmp;
  ans -= alpha;
  ans *= scale;
  return ans;
}

int get_data_width(jd::data_type dtype) {
  int data_width = 0;
  switch (dtype) {
    case jd::data_type::fp32:
      data_width = 4;
      break;
    case jd::data_type::bf16:
      data_width = 2;
      break;
    case jd::data_type::u8:
    case jd::data_type::s8:
      data_width = 1;
      break;
    default:
      throw std::runtime_error(std::string("sparselib_ut_malloc error:unsupport data type."));
      break;
  }
  return data_width;
}

float apply_postop_list(float value, const std::vector<jd::postop_attr>& attrs) {
  for (auto&& i : attrs) {
    if (i.op_type == jd::postop_type::eltwise) {
      if (i.op_alg == jd::postop_alg::exp) value = get_exp(value);
      if (i.op_alg == jd::postop_alg::gelu) value = get_gelu(value);
      if (i.op_alg == jd::postop_alg::relu) value = get_relu(value, i.alpha);
      if (i.op_alg == jd::postop_alg::quantize) value = get_quantize(value, i.alpha, i.scale);
      if (i.op_alg == jd::postop_alg::dequantize) value = get_dequantize(value, i.alpha, i.scale);
      if (i.op_alg == jd::postop_alg::tanh) value = tanh(value);
    } else {
      std::runtime_error("do not support postop type.");
    }
  }
  return value;
}

void assign_val(void* ptr, jd::data_type dtype, float val, int idx) {
  switch (dtype) {
    case jd::data_type::fp32:
      *((float*)ptr + idx) = val;
      break;
    case jd::data_type::bf16:
      *((unsigned short int*)ptr + idx) = fp32_2_bf16(val);
      break;
    case jd::data_type::u8:
      *((uint8_t*)ptr + idx) = (uint8_t)val;
    default:
      std::runtime_error(std::string("assign_val:unsupport this dtype."));
  }
}

void* sparselib_ut_memo(void* ptr, int num, jd::data_type dtype, memo_mode mode) {
  int data_width = get_data_width(dtype);
  switch (mode) {
    case MALLOC:
      ptr = malloc(num * data_width); /* code */
      break;
    case MEMSET:
      std::memset(ptr, 0, num * data_width);
      break;
    default:
      break;
  }
  return ptr;
}

class n_thread_t {
 public:
  n_thread_t(int nthr) : prev_nthr(omp_get_max_threads()) {
    if (nthr > 0 && nthr != prev_nthr) omp_set_num_threads(nthr);
  }
  ~n_thread_t() { omp_set_num_threads(prev_nthr); }

 private:
  int prev_nthr;
};
