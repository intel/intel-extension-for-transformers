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
#ifndef NE_CORE_GRAPH_INNER_PRODUCT_H
#define NE_CORE_GRAPH_INNER_PRODUCT_H

#ifdef __cplusplus
extern "C" {
#endif

void jblas_timer(bool _init);

int jblas_set_threads(int _nth);

void* jblas_get_thread_handle();

void jblas_init();

unsigned long long jblas_f32f32_get_workspace_size(int _m, int _n, int _k, void* wptr);

void jblas_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda, int ldo,
                          void* workspace);

bool jblas_fusion_add_f32f32_support(void* weiptr, int _m, int _n, int _k);
void jblas_fusion_add_f32f32_forward(float* activation, void* weiptr, float* bias, float* output, int _m, int _n,
                                     int _k, int lda, int ldo, bool boardcast_bias, void* workspace);

unsigned long long jblas_fusion_QKV_f32f32_get_workspace_size(int _m, int _n, int _k, void* w1ptr);

bool jblas_fusion_QKV_f32f32_support(void* wqptr, void* wkptr, void* wvptr, int _m, int _n, int _k);

void jblas_fusion_QKV_f32f32_forward(float* activation, void* wqptr, void* wkptr, void* wvptr, float* output, int _m,
                                     int _n, int _k, int lda, int ldo, void* workspace);

unsigned long long jblas_fusion_FFN_f32f32_get_workspace_size(int seq, int fin, int fmid, int fout, void* w1ptr,
                                                              void* w2ptr);

bool jblas_fusion_FFN_SiLu_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout);

void jblas_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                          float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                          void* workspace);

bool jblas_fusion_FFN_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout);
void jblas_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                          int seq, int fin, int fmid, int fout, void* workspace);

bool jblas_fusion_FFN_Add_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout);
void jblas_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                              float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                              bool boardcast_bias, void* workspace);

void jblas_unpackweight_fp32(void* wptr, int n, int k, float* fp32data, int ld);
// packweight to dstptr, copy weight attributes from srcptr
void jblas_packweight_copyattr(const float* f32ptr, void* dstpr, int n, int k, int ld, void* srcptr);
#ifdef __cplusplus
}
#endif
#endif  // NE_CORE_GRAPH_INNER_PRODUCT_H
