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
#ifndef QUANT_UTILS_H
#define QUANT_UTILS_H

#include "application/common.h"
#include "models/model_utils/quant_config.h"

#ifdef MODEL_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef MODEL_BUILD
#define QUANT_API __declspec(dllexport)
#else
#define QUANT_API __declspec(dllimport)
#endif
#else
#define QUANT_API __attribute__((visibility("default")))
#endif
#else
#define QUANT_API
#endif

#define MODEL_FILE_MAGIC_GGJT 0x67676a74u  // 'ggjt'
#define MODEL_FILE_MAGIC_GGLA 0x67676c61u  // 'ggla'
#define MODEL_FILE_MAGIC_GGMF 0x67676d66u  // 'ggmf'
#define MODEL_FILE_MAGIC_NE 0x67676d6cu    // 'ne'
#define MODEL_FILE_MAGIC_GGSN 0x6767736eu  // 'ggsn'

#define MODEL_FILE_VERSION 3
#define MODEL_FILE_MAGIC MODEL_FILE_MAGIC_GGJT
#define MODEL_FILE_MAGIC_UNVERSIONED MODEL_FILE_MAGIC_NE
#define MODEL_SESSION_MAGIC MODEL_FILE_MAGIC_GGSN
#define MODEL_SESSION_VERSION 1

QUANT_API int model_quantize(const quant_params& param, std::shared_ptr<quant_layer_base> quant_layer);
size_t jblas_qpack(const int8_t* src_w, const float* src_scales, const int8_t* src_zps, void* dstpr,
                   const quant_params_internal params, int nthread, int n, int k, int* g_idx);
size_t jblas_quantize(const float* f32ptr, void* dstpr, const quant_params_internal params, int nthread, size_t n,
                      size_t k);
QUANT_API bool model_quantize_special(std::ifstream& finp, std::ofstream& fout, const ne_ftype ftype,
                                      const std::vector<std::string>& to_quant,
                                      const std::vector<std::string>& to_skip);
QUANT_API bool whisper_model_quantize(const std::string& fname_inp, const std::string& fname_out, ne_ftype ftype);
#endif  // MODEL_H