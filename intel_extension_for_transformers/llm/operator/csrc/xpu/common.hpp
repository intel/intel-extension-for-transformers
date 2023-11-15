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
    if (_wei != nullptr && free_dev_mem)
      free(_wei, context);
    if (_scale != nullptr && free_dev_mem)
      free(_scale, context);
  }

  CompressWei4Bit(void *buf, sycl::queue &queue) {
    if (buf != nullptr) {
      auto device = queue.get_info<sycl::info::queue::device>();
      bool isDevicePointer = sycl::get_pointer_type(buf, context) == sycl::usm::alloc::device;
      if (isDevicePointer) {
        size_t offset = deserialize_field(buf, queue);
        _wei = (void *)((char *)buf + offset);
        _scale = (void *)((char *)buf + offset + get_4bit_wei_size() * sizeof(int8_t));
      } else {
        free_dev_mem = true;
        _wei = (void *)aligned_alloc_device(
          DEVICE_MEM_ALIGNMENT, get_4bit_wei_size() * sizeof(int8_t), device, context);
        _scale = (void *)aligned_alloc_device(
          DEVICE_MEM_ALIGNMENT, get_scale_size() * sizeof(fp16), device, context);
        size_t offset = deserialize_field(buf);
        queue.memcpy(_wei, (void *)((char *)buf + offset), get_4bit_wei_size() * sizeof(int8_t)).wait();
        queue.memcpy(_scale, (void *)((char *)buf + offset + get_4bit_wei_size() * sizeof(int8_t)), get_scale_size() * sizeof(fp16)).wait();
      }
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
    size_t deserialize_field(void *buf, sycl::queue &queue) {
    size_t offset = 0;
    queue.memcpy((void *)&_N, (void *)((char *)buf + offset), sizeof(_N)).wait();
    offset += sizeof(_N);
    queue.memcpy((void *)&_K, (void *)((char *)buf + offset), sizeof(_K)).wait();
    offset += sizeof(_K);
    queue.memcpy((void *)&_blksize, (void *)((char *)buf + offset), sizeof(_blksize)).wait();
    offset += sizeof(_blksize);
    queue.memcpy((void *)&_sym, (void *)((char *)buf + offset), sizeof(_sym)).wait();
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
  char *_write_buf = nullptr;
  void *_wei = nullptr;
  void *_scale = nullptr;
  const sycl::context context;
  bool free_dev_mem = false;
};