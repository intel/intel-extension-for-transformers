#pragma once

#include "core/ne.h"
#include "core/data_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void ne_compute_forward_conv_1d_s1_ph(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                      const struct ne_tensor* src1, struct ne_tensor* dst);
void ne_compute_forward_conv_1d_2s(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst);
void ne_compute_forward_conv_1d(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                const struct ne_tensor* src1, struct ne_tensor* dst);
void ne_compute_forward_conv_1d_1s(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst);
void ne_compute_forward_conv_1d_2s(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                   const struct ne_tensor* src1, struct ne_tensor* dst);
#ifdef __cplusplus
}
#endif
