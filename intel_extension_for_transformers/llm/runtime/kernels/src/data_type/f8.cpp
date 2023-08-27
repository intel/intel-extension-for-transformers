//  Copyright (c) 2021 Intel Corporation
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

#include <cmath>
#include <cstring>
#include "param_types.hpp"
#include "data_type/f8.hpp"

namespace jd {
template <data_type type>
struct float8_t {
  static float fp8_to_fp32(uint8_t data) {
    uint32_t constexpr kF32_NaN = 0x7fffffff;

    uint8_t const& f8 = data;
    int sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;
    int exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
    int mantissa = f8 & FP8_MANTISSA_MASK;
    unsigned f = (sign << (FP32_NUM_BITS - 1));

    if ((type == data_type::f8_e4m3) && exp == 15 && mantissa == 0x7) {
      f = kF32_NaN;
    } else if (exp > 0 && ((type == data_type::f8_e4m3) || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))) {
      // normal
      exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS);
      f = f | (exp << FP32_NUM_MANTISSA_BITS) | (mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
    } else if (exp == 0) {
      if (mantissa) {
        // subnormal
        exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1;
        while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0) {
          mantissa <<= 1;
          exp--;
        }
        mantissa &= FP8_MANTISSA_MASK;
        f = f | (exp << FP32_NUM_MANTISSA_BITS) | (mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
      } else {
        // sign-preserving zero
      }
    } else {
      if (mantissa == 0) {
        // Sign-preserving infinity
        f = (f | 0x7f800000);
      } else {
        // Canonical NaN
        f = kF32_NaN;
      }
    }

    float flt;
    std::memcpy(&flt, &f, sizeof(flt));
    return flt;
  }

  static uint8_t fp32_to_fp8(float val) {
    // software implementation rounds toward nearest even
    uint32_t s;

    std::memcpy(&s, &val, sizeof(s));

    // Extract the bits in the FP32 type
    uint8_t sign = uint8_t((s >> 24 & 0x80));
    int8_t exp = uint8_t(((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS);
    int mantissa = s & 0x7fffff;
    uint8_t u = 0;

    uint8_t const kF8_NaN = 0x7f;

    // NaN => NaN
    if (std::isnan(val)) {
      return kF8_NaN;
    }

    // Inf => MAX_FLT (satfinite)
    if (std::isinf(val)) {
      return sign | FP8_MAX_FLT;
    }

    // Special handling
    if (exp == -128) {
      // int8 range is from -128 to 127
      // So 255(inf) - 127(bias) = 128 - will show up as -128

      // satfinite
      return (sign | FP8_MAX_FLT);
    }

    int sticky_bit = 0;

    bool skip_sign = false;
    bool may_be_nan = false;

    if ((exp >= FP8_MIN_EXPONENT) && (exp <= FP8_MAX_EXPONENT)) {
      // normal fp32 to normal fp8
      exp = uint8_t(exp + uint8_t(FP8_EXPONENT_BIAS));
      u = uint8_t(((exp & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS));
      u = uint8_t(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
    } else if (exp < FP8_MIN_EXPONENT) {
      // normal single-precision to subnormal float8-precision representation
      int rshift = (FP8_MIN_EXPONENT - exp);
      if (rshift < FP32_NUM_BITS) {
        mantissa |= (1 << FP32_NUM_MANTISSA_BITS);

        sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

        mantissa = (mantissa >> rshift);
        u = (uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
      } else {
        mantissa = 0;
        u = 0;
      }
      // Exponent > FP8_MAX_EXPONENT - this is a special case done to match HW
      // 0x4380_0000 to 0x43e0_0000 - maps from 256 to 448, and does not
      // saturate / inf.
    } else {
      if (exp == (FP8_MAX_EXPONENT + 1)) {
        uint8_t mantissa_tmp = uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
        if (mantissa_tmp < FP8_MANTISSA_MASK) {
          exp = uint8_t(exp + uint8_t(FP8_EXPONENT_BIAS));
          u = uint8_t(exp << FP8_NUM_MANTISSA_BITS) | mantissa_tmp;
          may_be_nan = (mantissa_tmp == (FP8_MANTISSA_MASK - 1));
        } else {
          // satfinite
          return (sign | FP8_MAX_FLT);
        }
      } else {
        // satfinite
        return (sign | FP8_MAX_FLT);
      }
    }

    // round to nearest even
    int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
    int round_bit = ((mantissa >> NUM_BITS_SHIFT) & 1);
    sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
      u = uint8_t(u + 1);
      if (may_be_nan) {
        skip_sign = true;
      }
    }

    if (u > FP8_MAX_FLT) {
      // satfinite
      u = (sign | FP8_MAX_FLT);
    }

    if (!skip_sign) {
      u |= sign;
    }

    return u;
  }

  static constexpr int FP32_NUM_BITS = 32;
  static constexpr int FP32_NUM_EXPONENT_BITS = 8;
  static constexpr int FP32_NUM_MANTISSA_BITS = 23;
  static constexpr uint32_t FP32_NAN = 0x7fffffff;
  static constexpr uint32_t FP32_INFINITY_MASK = 0x7f800000;
  static constexpr int FP32_MAX_EXPONENT = 127;
  static constexpr int FP32_MIN_EXPONENT = -126;
  static constexpr int FP32_EXPONENT_BIAS = 127;

  static constexpr int FP16_NUM_BITS = 16;
  static constexpr int FP16_NUM_EXPONENT_BITS = 5;
  static constexpr int FP16_NUM_MANTISSA_BITS = 10;
  static constexpr uint16_t FP16_NAN = 0x7fff;
  static constexpr uint16_t FP16_INFINITY_MASK = 0x7c00;
  static constexpr int FP16_MAX_EXPONENT = 15;
  static constexpr int FP16_MIN_EXPONENT = -14;
  static constexpr int FP16_EXPONENT_BIAS = 15;

  static constexpr int FP8_NUM_BITS = 8;
  static constexpr int FP8_NUM_EXPONENT_BITS = (type == data_type::f8_e4m3) ? 4 : 5;
  static constexpr int FP8_NUM_MANTISSA_BITS = (type == data_type::f8_e4m3) ? 3 : 2;
  static constexpr uint8_t FP8_NAN = 0x7f;  // Also F8_INF
  static constexpr uint8_t FP8_INFINITY_MASK = (type == data_type::f8_e4m3) ? 0x78 : 0x7c;
  static constexpr int FP8_MAX_EXPONENT = (type == data_type::f8_e4m3) ? 7 : 15;
  static constexpr int FP8_MIN_EXPONENT = (type == data_type::f8_e4m3) ? -6 : -14;
  static constexpr int FP8_EXPONENT_BIAS = (type == data_type::f8_e4m3) ? 7 : 15;

  static constexpr uint8_t FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
  static constexpr uint8_t FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

  static constexpr uint8_t FP8_MAX_FLT = ((type == data_type::f8_e4m3) ? 0x7e : 0x7b);

  // 256 in float
  static constexpr uint32_t FP8_SAT_VAL_FP32 = 0x43800000;
};

float8_e4m3_t::float8_e4m3_t() { (*this) = 0.f; }
float8_e4m3_t::float8_e4m3_t(int32_t val) { (*this) = static_cast<float>(val); }
float8_e4m3_t::float8_e4m3_t(float val) { (*this) = val; }
float8_e4m3_t& float8_e4m3_t::operator=(float val) {
  data = float8_t<data_type::f8_e4m3>::fp32_to_fp8(val);
  return *this;
}
float8_e4m3_t::operator float() const { return float8_t<data_type::f8_e4m3>::fp8_to_fp32(data); }

float8_e5m2_t::float8_e5m2_t() { (*this) = 0.f; }
float8_e5m2_t::float8_e5m2_t(int32_t val) { (*this) = static_cast<float>(val); }
float8_e5m2_t::float8_e5m2_t(float val) { (*this) = val; }
float8_e5m2_t& float8_e5m2_t::operator=(float val) {
  data = float8_t<data_type::f8_e5m2>::fp32_to_fp8(val);
  return *this;
}
float8_e5m2_t::operator float() const { return float8_t<data_type::f8_e5m2>::fp8_to_fp32(data); }

}  // namespace jd
