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

#pragma once

#include <assert.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <math.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <vector>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DEVICE_MEM_ALIGNMENT (64)

using fp16 = sycl::half;

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

class Timer {
 public:
  void start() { m_start = std::chrono::high_resolution_clock::now(); }
  void stop() { m_end = std::chrono::high_resolution_clock::now(); }
  double get_elapsed_time() const { return duration_cast<std::chrono::nanoseconds>(m_end - m_start).count() / 1e6; }

 private:
  std::chrono::high_resolution_clock::time_point m_start;
  std::chrono::high_resolution_clock::time_point m_end;
};
static Timer timer;

class env_initer {
 public:
  env_initer() {
    verbose = std::getenv("GBITS_VERBOSE") != nullptr;
  }
  bool verbose;
};
static env_initer initer;

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

  CompressWei4Bit(void *buf, int k, int n, int blk) {
      _N = n;
      _K = k;
      _blksize = blk;
      if (buf != nullptr) {
          _wei = buf;
          _scale = (void *)((char *)buf + get_4bit_wei_size());
    }
  }

  void serialize(void *buf) {
    memcpy((char *)buf, _write_buf, get_buf_size());
  }

  void deserialize(void *buf) {
    memcpy(_write_buf, (char *)buf, get_buf_size());
  }

  void *get_4bit_wei_ptr() {
    return _write_buf;
  }

  void *get_scale_ptr() {
    return _write_buf + get_4bit_wei_size();
  }

  void *get_4bit_wei_ptr_device() {
    return _wei;
  }

  void *get_scale_ptr_device() {
    return _scale;
  }

  int _N, _K, _blksize;

private:
  size_t get_4bit_wei_size() { return _N * _K / 2; }
  size_t get_scale_size() { return _K / _blksize * _N * sizeof(fp16); }
  size_t get_zp_size() { return 0; }
  size_t get_buf_size() {
    return get_4bit_wei_size() + get_scale_size() + get_zp_size();
  }
  bool _sym;
  char *_write_buf = nullptr;
  void *_wei = nullptr;
  void *_scale = nullptr;
};