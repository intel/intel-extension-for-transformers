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
  size_t get_scale_size() { return _K / _blksize * _N * sizeof(fp16); }
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

template <typename data_type>
inline data_type *alloc_device(
        size_t size, sycl::device &device, sycl::context &context) {
    auto device_ptr = static_cast<data_type *>(aligned_alloc_device(
            DEVICE_MEM_ALIGNMENT, size * sizeof(data_type), device, context));
    return device_ptr;
}