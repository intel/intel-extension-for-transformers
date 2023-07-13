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
#ifndef NE_GRAPH_INNER_PRODUCT_H
#define NE_GRAPH_INNER_PRODUCT_H

#ifdef __cplusplus
extern "C" {
#endif
void jblas_weights4block_f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda,
                                     int ldo);

void jblas_weightcomp_QKV_f32_forward(float* activation, void* wqptr, void* wkptr, void* wvptr, float* output, int _m,
                                      int _n, int _k, int lda, int ldo);

void jblas_weightcomp_FFN_SiLu_f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                           float* tmp2, float* output, int seq, int fin, int fmid, int fout);

void jblas_timer(bool _init);

int jblas_set_threads(int _nth);

#ifdef __cplusplus
}
#endif
#endif
