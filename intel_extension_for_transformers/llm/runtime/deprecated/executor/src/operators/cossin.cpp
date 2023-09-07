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

#include "cossin.hpp"

namespace executor {

CosSinOperator::CosSinOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("algorithm");
  if (iter != attrs_map.end()) {
    algorithm_ = iter->second;
  }
}

CosSinOperator::~CosSinOperator() {}

void CosSinOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  vector<int64_t> src_shape = input[0]->shape();

  // 1.2 Get tensor's adjusted shapes
  // dst shape
  output[0]->set_shape(src_shape);
  output[0]->set_dtype(output_dtype_);
  array_size_ = input[0]->size();
}

#define PS256_CONST(Name, Val) static const auto ps256_##Name = _mm256_set1_ps(Val)
#define PI32_CONST256(Name, Val) static const auto pi256_##Name = _mm256_set1_epi32(Val)

PS256_CONST(1, 1.0f);
PS256_CONST(0p5, 0.5f);
PI32_CONST256(0, 0);
PI32_CONST256(1, 1);
PI32_CONST256(2, 2);
PI32_CONST256(4, 4);

PI32_CONST256(inv1, ~1);
PI32_CONST256(sign_mask, 0x80000000);
PI32_CONST256(inv_sign_mask, ~0x80000000);

PS256_CONST(minus_cephes_DP1, -0.78515625);
PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
PS256_CONST(sincof_p0, -1.9515295891E-4);
PS256_CONST(sincof_p1, 8.3321608736E-3);
PS256_CONST(sincof_p2, -1.6666654611E-1);
PS256_CONST(coscof_p0, 2.443315711809948E-005);
PS256_CONST(coscof_p1, -1.388731625493765E-003);
PS256_CONST(coscof_p2, 4.166664568298827E-002);
PS256_CONST(cephes_FOPI, 1.27323954473516);

PI32_CONST256(min_norm_pos, 0x00800000);
PI32_CONST256(inv_mant_mask, ~0x7f800000);

PI32_CONST256(0x7f, 0x7f);

// evaluation of 8 sines at onces using AVX intrisics
__m256 sinf(__m256 x) {
  __m256 sign_bit = x;
  // take the absolute value
  x = _mm256_and_ps(x, _mm256_castsi256_ps(pi256_inv_sign_mask));
  // extract the sign bit (upper one)
  sign_bit = _mm256_and_ps(sign_bit, _mm256_castsi256_ps(pi256_sign_mask));

  // scale by 4/Pi
  __m256 y = _mm256_mul_ps(x, ps256_cephes_FOPI);

  // store the integer part of y in mm0
  __m256i imm2 = _mm256_cvttps_epi32(y);

  // j=(j+1) & (~1) (see the cephes sources)
  imm2 = _mm256_add_epi32(imm2, pi256_1);
  imm2 = _mm256_and_si256(imm2, pi256_inv1);
  y = _mm256_cvtepi32_ps(imm2);

  // get the swap sign flag
  __m256i imm0 = _mm256_and_si256(imm2, pi256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);

  imm2 = _mm256_and_si256(imm2, pi256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, pi256_0);

  __m256 swap_sign_bit = _mm256_castsi256_ps(imm0);
  __m256 poly_mask = _mm256_castsi256_ps(imm2);
  sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

  // x = ((x - y * DP1) - y * DP2) - y * DP3;
  x = _mm256_add_ps(x, _mm256_mul_ps(y, ps256_minus_cephes_DP1));
  x = _mm256_add_ps(x, _mm256_mul_ps(y, ps256_minus_cephes_DP2));
  x = _mm256_add_ps(x, _mm256_mul_ps(y, ps256_minus_cephes_DP3));

  // Evaluate the first polynom  (0 <= x <= Pi/4)
  __m256 z = _mm256_mul_ps(x, x);

  y = _mm256_mul_ps(ps256_coscof_p0, z);
  y = _mm256_add_ps(y, ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  y = _mm256_sub_ps(y, _mm256_mul_ps(z, ps256_0p5));
  y = _mm256_add_ps(y, ps256_1);

  // Evaluate the second polynom  (Pi/4 <= x <= 0)

  __m256 y2 = _mm256_mul_ps(ps256_sincof_p0, z);
  y2 = _mm256_add_ps(y2, ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  // select the correct result from the two polynoms
  y2 = _mm256_and_ps(poly_mask, y2);
  y = _mm256_andnot_ps(poly_mask, y);
  y = _mm256_add_ps(y, y2);
  // update the sign
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

__m256 cosf(__m256 x) {
  // take the absolute value
  x = _mm256_and_ps(x, _mm256_castsi256_ps(pi256_inv_sign_mask));

  // scale by 4/Pi
  __m256 y = _mm256_mul_ps(x, ps256_cephes_FOPI);

  // store the integer part of y in mm0
  __m256i imm2 = _mm256_cvttps_epi32(y);

  // j=(j+1) & (~1) (see the cephes sources)
  imm2 = _mm256_add_epi32(imm2, pi256_1);
  imm2 = _mm256_and_si256(imm2, pi256_inv1);
  y = _mm256_cvtepi32_ps(imm2);
  imm2 = _mm256_sub_epi32(imm2, pi256_2);

  // get the swap sign flag
  __m256i imm0 = _mm256_andnot_si256(imm2, pi256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);

  // get the polynom selection mask
  imm2 = _mm256_and_si256(imm2, pi256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, pi256_0);

  __m256 sign_bit = _mm256_castsi256_ps(imm0);
  __m256 poly_mask = _mm256_castsi256_ps(imm2);

  // x = ((x - y * DP1) - y * DP2) - y * DP3
  x = _mm256_add_ps(x, _mm256_mul_ps(y, ps256_minus_cephes_DP1));
  x = _mm256_add_ps(x, _mm256_mul_ps(y, ps256_minus_cephes_DP2));
  x = _mm256_add_ps(x, _mm256_mul_ps(y, ps256_minus_cephes_DP3));

  // Evaluate the first polynom  (0 <= x <= Pi/4)
  __m256 z = _mm256_mul_ps(x, x);

  y = _mm256_mul_ps(ps256_coscof_p0, z);
  y = _mm256_add_ps(y, ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  y = _mm256_sub_ps(y, _mm256_mul_ps(z, ps256_0p5));
  y = _mm256_add_ps(y, ps256_1);

  // Evaluate the second polynom  (Pi/4 <= x <= 0)
  __m256 y2 = _mm256_mul_ps(ps256_sincof_p0, z);
  y2 = _mm256_add_ps(y2, ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  // select the correct result from the two polynoms
  y2 = _mm256_and_ps(poly_mask, y2);
  y = _mm256_andnot_ps(poly_mask, y);
  y = _mm256_add_ps(y, y2);
  // update the sign
  y = _mm256_xor_ps(y, sign_bit);
  return y;
}

// 2. inference kernel(for int8 and f32)
void CosSinOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  Tensor* src = input[0];
  Tensor* dst = output[0];
  const float* src_data = static_cast<const float*>(src->data());
  float* dst_data = static_cast<float*>(dst->mutable_data());
  // Tail should equals 0
  int tail = input[0]->size() % 8;
  int batch = input[0]->size() / 8;

#if __AVX2__
  if (algorithm_ == "sin") {
#pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
      __m256 p = _mm256_load_ps(src_data + i * 8);
      __m256 c = sinf(p);
      _mm256_storeu_ps(dst_data + i * 8, c);
    }
  } else {
#pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
      __m256 p = _mm256_load_ps(src_data + i * 8);
      __m256 c = cosf(p);
      _mm256_storeu_ps(dst_data + i * 8, c);
    }
  }
#else
  LOG(ERROR) << "No AVX2";
#endif
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(CosSin);

}  // namespace executor
