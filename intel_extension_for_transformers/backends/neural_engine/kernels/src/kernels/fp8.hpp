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

/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#ifndef ENGINE_EXECUTOR_INCLUDE_OPERATORS_FP8_HPP_
#define ENGINE_EXECUTOR_INCLUDE_OPERATORS_FP8_HPP_
#include <stdint.h>

#include <cstring>

enum class FloatEncoding { E4M3, E5M2 };

template <FloatEncoding T>
struct alignas(1) float8_base {
  static constexpr bool IS_E4M3 = (T == FloatEncoding::E4M3);
  static constexpr bool IS_E5M2 = (T == FloatEncoding::E5M2);

  // Number of Bits representing mantissa and exponents
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
  static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
  static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
  static constexpr uint8_t FP8_NAN = 0x7f;  // Also F8_INF
  static constexpr uint8_t FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;
  static constexpr int FP8_MAX_EXPONENT = IS_E4M3 ? 7 : 15;
  static constexpr int FP8_MIN_EXPONENT = IS_E4M3 ? -6 : -14;
  static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ? 7 : 15;

  static constexpr uint8_t FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
  static constexpr uint8_t FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

  static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);

  // 256 in float
  static constexpr uint32_t FP8_SAT_VAL_FP32 = 0x43800000;

  //
  // Data members
  //

  /// Data container
  uint8_t storage;

  /// Ctors.
  float8_base() : storage(0) {}

  /// Is finite implementation
  static bool isfinite(float flt) {
    uint32_t s;

    std::memcpy(&s, &flt, sizeof(s));

    return (s & 0x7f800000) < 0x7f800000;
  }

  /// Is NaN implementation
  static bool isnan(float flt) {
    uint32_t s;

    std::memcpy(&s, &flt, sizeof(s));

    return (s & 0x7fffffff) > 0x7f800000;
  }

  /// Is infinite implementation
  static bool isinf(float flt) {
    uint32_t s;

    std::memcpy(&s, &flt, sizeof(s));

    // Sign = 0 for +inf, 1 for -inf
    // Exponent = all ones
    // Mantissa = all zeros
    return (s == 0x7f800000) || (s == 0xff800000);
  }

  /// FP32 -> FP8 conversion - rounds to nearest even
  static uint8_t convert_float_to_fp8(float const& flt) {
    // software implementation rounds toward nearest even
    uint32_t s;

    std::memcpy(&s, &flt, sizeof(s));

    // Extract the bits in the FP32 type
    uint8_t sign = uint8_t((s >> 24 & 0x80));
    int8_t exp = uint8_t(((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS);
    int mantissa = s & 0x7fffff;
    uint8_t u = 0;

    uint8_t const kF8_NaN = 0x7f;

    // NaN => NaN
    if (isnan(flt)) {
      return kF8_NaN;
    }

    // Inf => MAX_FLT (satfinite)
    if (isinf(flt)) {
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

  /// Converts a fp8 value stored as a uint8_t to a float
  static float convert_fp8_to_float(uint8_t const& x) {
    uint32_t constexpr kF32_NaN = 0x7fffffff;

    uint8_t const& f8 = x;
    int sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;
    int exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
    int mantissa = f8 & FP8_MANTISSA_MASK;
    unsigned f = (sign << (FP32_NUM_BITS - 1));

    if (IS_E4M3 && exp == 15 && mantissa == 0x7) {
      f = kF32_NaN;
    } else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))) {
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
};

#endif  // ENGINE_EXECUTOR_INCLUDE_OPERATORS_FP8_HPP_
