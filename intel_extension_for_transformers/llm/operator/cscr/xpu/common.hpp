#pragma once

#include <assert.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <math.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
//#include <torch/extension.h>

#include <vector>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DEVICE_MEM_ALIGNMENT (64)

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

  operator uint8_t() const {
    return static_cast<uint8_t>((x & 0x0F) | ((y & 0x0F) << 4));
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

