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
// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstring>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "application/common.h"
#include "core/layers/jblas_common.hpp"
#include "core/layers/mha_dense.h"
#include "core/ne_layers.h"
// #include "jblas/jblas/jit_blas_weight_compression.h"
//#include "models/model_utils/model_config.h"

#include "models/model_utils/model_files.h"
#include "models/whisper/whisper.h"
#include "models/model_utils/quant_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

quant_params_internal quant_params_to_internal(const quant_params& params) {
  return quant_params_internal{parse_bits(params.weight_dtype), parse_alg(params.alg), params.group_size,
                               parse_scale_dtype(params.scale_dtype),
                               parse_compute_type(params.compute_dtype, params.use_ggml)};
}

size_t jblas_quantize(const float* f32ptr, void* dstpr, const quant_params_internal params, int nthread, int n, int k) {
  using CompType = jblas::prologue::weight_comp::gemm_kblcok::PrologueBIDs;
  using namespace ne_jblas;
  auto cd = jblas::utils::parallel::CpuDevice::getInstance();
  auto dstbptr = (int8_t*)dstpr;
  cd->setThreads(nthread);
  if (params.scale_dtype != quant_sdtype::fp32) {
    // TODO(BesTLA): add unified scale type
    printf("Current not support none-float scale, reset to f32\n");
  }
  if (params.bits == quant_bits::q4) {
    if (params.compute_dtype == quant_comp::int8) {
      if (params.alg != quant_alg::sym) {
        printf("Current not support asymmetric int8 computation, reset to symmetric\n");
      }
      if (params.group_size == -1) {
        using Kernel = WeiS4ClipFp32PerN<GcCompInt8, JblasAVX512F>;
        using KernelRef = WeiS4ClipFp32PerN<GcCompInt8, JblasNoSIMD>;
        static Kernel kernel;
        static KernelRef kernelref;
        auto packedw = kernel.createStorage(n, k, false);
        packedw.assign(dstbptr);
        if (cd->AVX512F()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
        }
        return packedw.mSize;
      } else {
        using Kernel = WeiS4ClipFp32<GcCompInt8KBlock, JblasAVX512F>;
        using KernelRef = WeiS4ClipFp32<GcCompInt8KBlock, JblasNoSIMD>;
        static Kernel kernel;
        static KernelRef kernelref;
        auto packedw = kernel.createStorage(n, k, params.group_size);
        packedw.assign(dstbptr);
        if (cd->AVX512F()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
        }
        return packedw.mSize;
      }
    } else if (params.compute_dtype == quant_comp::fp32) {
      using Kernel = WeiS4ClipFp32<GcCompFp32, JblasAVX512F>;
      using KernelRef = WeiS4ClipFp32<GcCompFp32, JblasNoSIMD>;
      static Kernel kernel;
      static KernelRef kernelref;
      auto packedw = kernel.createStorage(n, k, params.group_size, params.alg == quant_alg::asym);
      packedw.assign(dstbptr);
      if (cd->AVX512_FP16()) {
        kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
      } else {
        kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
      }
      return packedw.mSize;
    } else if (params.compute_dtype == quant_comp::bf16) {
      using Kernel = WeiS4ClipFp32<GcCompBf16, JblasAVX512F>;
      using KernelRef = WeiS4ClipFp32<GcCompBf16, JblasNoSIMD>;
      static Kernel kernel;
      static KernelRef kernelref;
      auto packedw = kernel.createStorage(n, k, params.group_size, params.alg == quant_alg::asym);
      packedw.assign(dstbptr);
      if (cd->AMX_BF16()) {
        kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
      } else {
        kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
      }
      return packedw.mSize;
    }

  } else if (params.bits == quant_bits::q8) {
    // TODO add 8bit quantization
    if (params.compute_dtype == quant_comp::int8) {
      if (params.alg != quant_alg::sym) {
        printf("Current not support asymmetric int8 computation, reset to symmetric\n");
      }
      if (params.group_size == -1) {
        using Kernel = WeiS8Fp32PerN<GcCompInt8, JblasAVX512F>;
        using KernelRef = WeiS8Fp32PerN<GcCompInt8, JblasNoSIMD>;
        static Kernel kernel;
        static KernelRef kernelref;
        auto packedw = kernel.createStorage(n, k, false);
        packedw.assign(dstbptr);
        if (cd->AVX512F()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
        }
        return packedw.mSize;
      } else {
        using Kernel = WeiS8Fp32<GcCompInt8KBlock, JblasAVX512F>;
        using KernelRef = WeiS8Fp32<GcCompInt8KBlock, JblasNoSIMD>;
        static Kernel kernel;
        static KernelRef kernelref;
        auto packedw = kernel.createStorage(n, k, params.group_size);
        packedw.assign(dstbptr);
        if (cd->AVX512F()) {
          kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
        } else {
          kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
        }
        return packedw.mSize;
      }
    } else if (params.compute_dtype == quant_comp::fp32) {
      using Kernel = WeiS8Fp32<GcCompFp32, JblasAVX512F>;
      using KernelRef = WeiS8Fp32<GcCompFp32, JblasNoSIMD>;
      static Kernel kernel;
      static KernelRef kernelref;
      auto packedw = kernel.createStorage(n, k, params.group_size, params.alg == quant_alg::asym);
      packedw.assign(dstbptr);
      if (cd->AVX512_FP16()) {
        kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
      } else {
        kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
      }
      return packedw.mSize;
    } else if (params.compute_dtype == quant_comp::bf16) {
      using Kernel = WeiS8Fp32<GcCompBf16, JblasAVX512F>;
      using KernelRef = WeiS8Fp32<GcCompBf16, JblasNoSIMD>;
      static Kernel kernel;
      static KernelRef kernelref;
      auto packedw = kernel.createStorage(n, k, params.group_size, params.alg == quant_alg::asym);
      packedw.assign(dstbptr);
      if (cd->AMX_BF16()) {
        kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
      } else {
        kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
      }
      return packedw.mSize;
    }
  }
  return 0;
}


size_t ggml_quantize(const float* f32ptr, void* dstpr, const ne_type new_type, int nthread, size_t nelements) {
  std::vector<int64_t> hist_cur(1 << 4, 0);
  std::vector<std::thread> workers;
  std::mutex mutex;
  int chunk_size = 32 * 512;
  const int nchunk = (nelements + chunk_size - 1) / chunk_size;
  const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
  size_t new_size = 0;
  if (nthread_use < 2) {
    new_size = ne_quantize_chunk(new_type, f32ptr, dstpr, 0, nelements, hist_cur.data());
  } else {
    size_t counter = 0;
    new_size = 0;
    auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32ptr, dstpr, nelements, chunk_size]() {
      std::vector<int64_t> local_hist;
      size_t local_size = 0;
      while (true) {
        std::unique_lock<std::mutex> lock(mutex);
        size_t first = counter;
        counter += chunk_size;
        if (first >= nelements) {
          if (!local_hist.empty()) {
            for (int j = 0; j < int(local_hist.size()); ++j) {
              hist_cur[j] += local_hist[j];
            }
            new_size += local_size;
          }
          break;
        }
        lock.unlock();
        size_t last = std::min(nelements, first + chunk_size);
        if (local_hist.empty()) {
          local_hist.resize(hist_cur.size(), 0);
        }
        local_size += ne_quantize_chunk(new_type, f32ptr, dstpr, first, last - first, local_hist.data());
      }
    };
    if ((int)workers.size() < nthread_use - 1) {
      workers.resize(nthread_use - 1);
    }
    for (int it = 0; it < nthread_use - 1; ++it) {
      workers[it] = std::thread(compute);
    }
    compute();
    for (int it = 0; it < nthread_use - 1; ++it) {
      workers[it].join();
    }
  }
  return new_size;
}


void ne_common_quantize(const int nthread, const quant_params_internal& params, model_load_tensor& tensor,
                        model_file_saver& saver, size_t& size_org, size_t& size_new) {
  size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
  enum ne_type new_type = quant_params_to_type(params);
  model_buffer work;
  work.resize(nelements * 4);  // upper bound on size
  void* new_data = work.addr;
  size_t new_size = 0;
  float* f32_data = NULL;
  model_buffer f32_conv_buf;
  if (tensor.type == NE_TYPE_F32) {
    f32_data = (float*)tensor.data;
  } else if (tensor.type == NE_TYPE_F16) {
    f32_conv_buf.resize(nelements * sizeof(float));
    f32_data = (float*)f32_conv_buf.addr;
    const auto* f16_data = (const ne_fp16_t*)tensor.data;
    for (size_t i = 0; i < nelements; i++) {
      f32_data[i] = ne_fp16_to_fp32(f16_data[i]);
    }
  } else {
    throw format("type %s unsupported for integer quantization", ne_type_name(tensor.type));
  }
  printf("quantizing .. ");
  fflush(stdout);
  if (new_type == NE_TYPE_JBLAS) {
    int k_ = tensor.ne.at(0);
    int n_ = tensor.ne.at(1);
    new_size = jblas_quantize(f32_data, work.addr, params, nthread, n_, k_);
    printf("JBLAS ");
  } else if (new_type >= NE_TYPE_Q4_0 && new_type < NE_TYPE_JBLAS) {
    new_size = ggml_quantize(f32_data, work.addr, new_type, nthread, nelements);
    printf("GGML ");
  }
  printf("size = %8.2f MB -> %8.2f MB\n", tensor.size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);

__WRITE_FILE:
  size_org += tensor.size;
  size_new += new_size;
  saver.write_tensor(tensor, new_type, new_data, new_size);
  printf("\n");
}


static void model_quantize_special(const quant_params& params, std::shared_ptr<quant_layer_base> quant_layer){
  auto ftype = quant_params_to_ftype(params);
  quant_layer->set_global_config(params.nthread, quant_params_to_internal(params));
  int nthread = params.nthread;
  if (nthread <= 0) {
    nthread = std::thread::hardware_concurrency();
  }
  std::unique_ptr<model_model_loader> model_loader(new model_model_loader(params.model_file, /*use_mmap*/ false,
                                                                          /*vocab_only*/ false));
  model_file_saver file_saver(params.out_file.c_str(), model_loader->file_loaders.at(0).get(), ftype);
  size_t total_size_org = 0;
  size_t total_size_new = 0;
  size_t idx = 0;
  for (model_load_tensor& tensor : model_loader->tensors_map.tensors) {
    model_buffer read_data;
    read_data.resize(tensor.size);
    tensor.data = read_data.addr;
    model_loader->load_data_for(tensor);
    printf("[%4zu/%4zu] %36s - %16s, type = %6s, ", ++idx, model_loader->tensors_map.tensors.size(),
           tensor.name.c_str(), model_format_tensor_shape(tensor.ne).c_str(), ne_type_name(tensor.type));
    std::vector<int64_t> tmpne(tensor.ne.size());
    for (size_t i = 0; i < tmpne.size(); i++) {
      tmpne[i] = static_cast<int64_t>(tensor.ne[i]);
    }
    auto lconfig = quant_layer->get_layer_config(tensor.name, tmpne, tensor.type);
    bool quantize = lconfig.valid();
    printf("%s,", lconfig.getstr().c_str());
    if (quantize) {
      ne_common_quantize(nthread, lconfig, tensor, file_saver, total_size_org, total_size_new);
    } else {
      printf("size = %8.3f MB\n", tensor.size / 1024.0 / 1024.0);
      total_size_org += tensor.size;
      total_size_new += tensor.size;
      file_saver.write_tensor(tensor, tensor.type, tensor.data, tensor.size);
      printf("\n");
    }
  }
  printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);
};

static void model_quantize_internal(const quant_params& params, std::shared_ptr<quant_layer_base> quant_layer) {
  auto ftype = quant_params_to_ftype(params);
  quant_layer->set_global_config(params.nthread, quant_params_to_internal(params));
  int nthread = params.nthread;
  if (nthread <= 0) {
    nthread = std::thread::hardware_concurrency();
  }
  std::unique_ptr<model_model_loader> model_loader(new model_model_loader(params.model_file, /*use_mmap*/ false,
                                                                          /*vocab_only*/ false));
  model_file_saver file_saver(params.out_file.c_str(), model_loader->file_loaders.at(0).get(), ftype);
  size_t total_size_org = 0;
  size_t total_size_new = 0;
  size_t idx = 0;
  for (model_load_tensor& tensor : model_loader->tensors_map.tensors) {
    model_buffer read_data;
    read_data.resize(tensor.size);
    tensor.data = read_data.addr;
    model_loader->load_data_for(tensor);
    printf("[%4zu/%4zu] %36s - %16s, type = %6s, ", ++idx, model_loader->tensors_map.tensors.size(),
           tensor.name.c_str(), model_format_tensor_shape(tensor.ne).c_str(), ne_type_name(tensor.type));
    std::vector<int64_t> tmpne(tensor.ne.size());
    for (size_t i = 0; i < tmpne.size(); i++) {
      tmpne[i] = static_cast<int64_t>(tensor.ne[i]);
    }
    auto lconfig = quant_layer->get_layer_config(tensor.name, tmpne, tensor.type);
    bool quantize = lconfig.valid();
    printf("%s,", lconfig.getstr().c_str());
    if (quantize) {
      ne_common_quantize(nthread, lconfig, tensor, file_saver, total_size_org, total_size_new);
    } else {
      printf("size = %8.3f MB\n", tensor.size / 1024.0 / 1024.0);
      total_size_org += tensor.size;
      total_size_new += tensor.size;
      file_saver.write_tensor(tensor, tensor.type, tensor.data, tensor.size);
      printf("\n");
    }
  }
  printf("%s: model size  = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);
}



int model_quantize(const quant_params& params, std::shared_ptr<quant_layer_base> quant_layer) {
  if (params.model_arch==MODEL_WHISPER){
    model_quantize_special(params, quant_layer);
    return 0;
  }
  try {
    model_quantize_internal(params, quant_layer);
    return 0;
  } catch (const std::string& err) {
    fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
    return 1;
  }
}