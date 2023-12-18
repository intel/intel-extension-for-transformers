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
#include "conv.h"
#include "vec_dot.h"

#define NE_UNUSED(x) (void)(x)
#define NE_TENSOR_LOCALS_1(type, prefix, pointer, array) \
  const type prefix##0 = (pointer)->array[0];            \
  NE_UNUSED(prefix##0);
#define NE_TENSOR_LOCALS_2(type, prefix, pointer, array) \
  NE_TENSOR_LOCALS_1(type, prefix, pointer, array)       \
  const type prefix##1 = (pointer)->array[1];            \
  NE_UNUSED(prefix##1);
#define NE_TENSOR_LOCALS_3(type, prefix, pointer, array) \
  NE_TENSOR_LOCALS_2(type, prefix, pointer, array)       \
  const type prefix##2 = (pointer)->array[2];            \
  NE_UNUSED(prefix##2);
#define NE_TENSOR_LOCALS(type, prefix, pointer, array) \
  NE_TENSOR_LOCALS_3(type, prefix, pointer, array)     \
  const type prefix##3 = (pointer)->array[3];          \
  NE_UNUSED(prefix##3);
#define NE_TENSOR_BINARY_OP_LOCALS          \
  NE_TENSOR_LOCALS(int64_t, ne0, src0, ne); \
  NE_TENSOR_LOCALS(size_t, nb0, src0, nb);  \
  NE_TENSOR_LOCALS(int64_t, ne1, src1, ne); \
  NE_TENSOR_LOCALS(size_t, nb1, src1, nb);  \
  NE_TENSOR_LOCALS(int64_t, ne, dst, ne);   \
  NE_TENSOR_LOCALS(size_t, nb, dst, nb);

static inline int ne_up32(int n) { return (n + 31) & ~31; }

// ne_compute_forward_conv_1d

static void ne_compute_forward_conv_1d_s1_ph_f16_f32(const struct ne_compute_params* params,
                                                     const struct ne_tensor* src0, const struct ne_tensor* src1,
                                                     struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  NE_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ne_fp16_t* const src =
              reinterpret_cast<ne_fp16_t*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          ne_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        ne_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = NE_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f16(ew0, &v, reinterpret_cast<ne_fp16_t*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_s1_ph_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                 const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F32);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  NE_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src =
              reinterpret_cast<float*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f32(ew0, &v, reinterpret_cast<float*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

void ne_compute_forward_conv_1d_s1_ph(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                      const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_conv_1d_s1_ph_f16_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_conv_1d_s1_ph_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

static void ne_compute_forward_conv_1d_s2_ph_f16_f32(const struct ne_compute_params* params,
                                                     const struct ne_tensor* src0, const struct ne_tensor* src1,
                                                     struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  NE_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ne_fp16_t* const src =
              reinterpret_cast<ne_fp16_t*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          ne_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        ne_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = NE_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f16(ew0, &v, reinterpret_cast<ne_fp16_t*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_s2_ph_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                 const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F32);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  NE_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src =
              reinterpret_cast<float*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f32(ew0, &v, reinterpret_cast<float*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

void ne_compute_forward_conv_1d_s2_ph(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                      const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_conv_1d_s2_ph_f16_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_conv_1d_s2_ph_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_conv_1d

void ne_compute_forward_conv_1d(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                const struct ne_tensor* src1, struct ne_tensor* dst) {
  const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
  const int32_t p0 = ((const int32_t*)(dst->op_params))[1];
  const int32_t d0 = ((const int32_t*)(dst->op_params))[2];
  NE_ASSERT(d0 == 1);                // dilation not supported
  NE_ASSERT(p0 == src0->ne[0] / 2);  // only half padding supported
  if (s0 == 1) {
    ne_compute_forward_conv_1d_s1_ph(params, src0, src1, dst);
  } else if (s0 == 2) {
    ne_compute_forward_conv_1d_s2_ph(params, src0, src1, dst);
  } else {
    NE_ASSERT(false);  // only stride 1 and 2 supported
  }
}

// ne_compute_forward_conv_1d_1s

static void ne_compute_forward_conv_1d_1s_f16_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                  const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ne_fp16_t* const src =
              reinterpret_cast<ne_fp16_t*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          ne_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        ne_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = NE_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f16(ew0, &v, reinterpret_cast<ne_fp16_t*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_1s_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                              const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F32);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src =
              reinterpret_cast<float*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f32(ew0, &v, reinterpret_cast<float*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

void ne_compute_forward_conv_1d_1s(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_conv_1d_1s_f16_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_conv_1d_1s_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}

// ne_compute_forward_conv_1d_2s

static void ne_compute_forward_conv_1d_2s_f16_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                                  const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F16);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(ne_fp16_t));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ne_fp16_t* const src =
              reinterpret_cast<ne_fp16_t*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          ne_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ne_fp16_t* const wdata = reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        ne_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = NE_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f16(ew0, &v, reinterpret_cast<ne_fp16_t*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<ne_fp16_t*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ne_compute_forward_conv_1d_2s_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                              const struct ne_tensor* src1, struct ne_tensor* dst) {
  NE_ASSERT(src0->type == NE_TYPE_F32);
  NE_ASSERT(src1->type == NE_TYPE_F32);
  NE_ASSERT(dst->type == NE_TYPE_F32);

  // int64_t t0 = ne_perf_time_us();
  // UNUSED(t0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  // const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  // const int64_t ne12 = src1->ne[2];
  // const int64_t ne13 = src1->ne[3];

  // const int64_t ne0  = dst->ne[0];
  // const int64_t ne1  = dst->ne[1];
  // const int64_t ne2  = dst->ne[2];
  // const int64_t ne3  = dst->ne[3];
  // const int64_t ne   = ne0*ne1*ne2*ne3;

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  // const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = src1->nb[1];
  // const size_t nb12 = src1->nb[2];
  // const size_t nb13 = src1->nb[3];

  // const size_t nb0  = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  // const size_t nb2  = dst->nb[2];
  // const size_t nb3  = dst->nb[3];

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ne_up32(ne01);

  NE_ASSERT(ne00 % 2 == 1);  // TODO(Bo): support even kernel sizes
  NE_ASSERT(nb00 == sizeof(float));
  NE_ASSERT(nb10 == sizeof(float));

  if (params->type == NE_TASK_INIT) {
    // TODO(Bo): fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src =
              reinterpret_cast<float*>(reinterpret_cast<char*>(src0->data) + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = reinterpret_cast<float*>(reinterpret_cast<char*>(src1->data) + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == NE_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ne_vec_dot_f32(ew0, &v, reinterpret_cast<float*>(params->wdata) + i1 * ew0 * ne00 + (nh + k) * ew0,
                       reinterpret_cast<float*>(params->wdata) + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

void ne_compute_forward_conv_1d_2s(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F16: {
      ne_compute_forward_conv_1d_2s_f16_f32(params, src0, src1, dst);
    } break;
    case NE_TYPE_F32: {
      ne_compute_forward_conv_1d_2s_f32(params, src0, src1, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}
