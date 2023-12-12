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
  using namespace ne_jblas;  // NOLINT
  auto cd = jblas::utils::parallel::CpuDevice::getInstance();
  auto dstbptr = reinterpret_cast<int8_t*>(dstpr);
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
    // add 8bit quantization
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
            for (int j = 0; j < static_cast<int>(local_hist.size()); ++j) {
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
    if (static_cast<int>(workers.size()) < nthread_use - 1) {
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

void ne_common_quantize(const int nthread, const quant_params_internal& params, model_load_tensor& tensor,  // NOLINT
                        model_file_saver& saver, size_t& size_org, size_t& size_new) {                      // NOLINT
  size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
  enum ne_type new_type = quant_params_to_type(params);
  model_buffer work;
  work.resize(nelements * 4);  // upper bound on size
  void* new_data = work.addr;
  size_t new_size = 0;
  float* f32_data = NULL;
  model_buffer f32_conv_buf;
  if (tensor.type == NE_TYPE_F32) {
    f32_data = reinterpret_cast<float*>(tensor.data);
  } else if (tensor.type == NE_TYPE_F16) {
    f32_conv_buf.resize(nelements * sizeof(float));
    f32_data = reinterpret_cast<float*>(f32_conv_buf.addr);
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

size_t jblas_special_quantize(const float* f32ptr, void* dstpr, int group_size, int nthread, int n, int k){
  using CompType = jblas::prologue::weight_comp::gemm_kblcok::PrologueBIDs;
  using namespace ne_jblas;
  auto cd = jblas::utils::parallel::CpuDevice::getInstance();
  auto dstbptr = (int8_t*)dstpr;
  cd->setThreads(nthread);
  using Kernel = WeiS4ClipFp32<GcCompInt8KBlock, JblasAVX512F>;
  using KernelRef = WeiS4ClipFp32<GcCompInt8KBlock, JblasNoSIMD>;
  static Kernel kernel;
  static KernelRef kernelref;
  auto packedw = kernel.createStorage(n, k, group_size);
  packedw.assign(dstbptr);
  if (cd->AVX512F()) {
      kernel.packTransposeWeight(n, k, f32ptr, k, &packedw);
      } else {
        kernelref.packTransposeWeight(n, k, f32ptr, k, &packedw);
      }
  return packedw.mSize;


}
bool model_quantize_special(
        std::ifstream & finp,
        std::ofstream & fout,
        const ne_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip) {

    ne_type qtype = NE_TYPE_F32;

    switch (ftype) {
        case NE_FTYPE_MOSTLY_Q4_0: qtype = NE_TYPE_Q4_0; break;
        case NE_FTYPE_MOSTLY_Q_JBLAS: qtype = NE_TYPE_JBLAS; break;
        case NE_FTYPE_MOSTLY_F16:
                {
                    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
                    return false;
                }
    };
    if (!ne_is_quantized(qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ne_type_name(qtype));
        return false;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ne_fp16_t>   data_f16;
    std::vector<float>       data_f32;

    std::vector<int64_t> hist_all(1 << 4, 0);

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read (&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ne_type_name((ne_type) ttype));

        bool quantize = false;

        // check if we should quantize this tensor
        for (const auto & s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != NE_TYPE_F32 && ttype != NE_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ne_type_name((ne_type) ttype));
                return false;
            }

            if (ttype == NE_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ne_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ne_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
        } else {
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

            data_u8.resize(nelements*bpe);
            finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
        }

        fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout.write(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            std::vector<int64_t> hist_cur(1 << 4, 0);

            switch ((ne_type) ttype) {
                case NE_TYPE_Q4_0:
                    {
                        cur_size = ne_quantize_chunk((ne_type) ttype, data_f32.data(), work.data(), 0, nelements, hist_cur.data());
                    } break;
                case NE_TYPE_JBLAS:
                    {
                        cur_size = jblas_special_quantize(data_f32.data(), work.data(), 32, 1, ne[0], ne[1]);
                        printf("JBLAS");
                    } break;
                case NE_TYPE_F32:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ne_type_name((ne_type) ttype));
                        return false;
                    }
            }

            fout.write(reinterpret_cast<char *>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                hist_all[i] += hist_cur[i];
            }

            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                printf("%5.3f ", hist_cur[i] / (float)nelements);
            }
            printf("\n");
        } else {
            printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
            fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new/1024.0/1024.0, ftype, ne_type_name(qtype));

    {
        int64_t sum_all = 0;
        for (int i = 0; i < (int) hist_all.size(); ++i) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (int i = 0; i < (int) hist_all.size(); ++i) {
            printf("%5.3f ", hist_all[i] / (float)sum_all);
        }
        printf("\n");
    }

    return true;
}
int model_quantize(const quant_params& params, std::shared_ptr<quant_layer_base> quant_layer) {
  try {
    model_quantize_internal(params, quant_layer);
    return 0;
  } catch (const std::string& err) {
    fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
    return 1;
  }
}


