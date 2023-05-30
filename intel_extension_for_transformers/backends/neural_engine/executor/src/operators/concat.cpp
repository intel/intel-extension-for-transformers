//  Copyright (c) 2021 Intel Corporation
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

#include "concat.hpp"

#include "common.hpp"

#define AVX512_BYTES 64

namespace executor {

static unordered_map<string, dnnl::memory::data_type> type2mem{
    {"fp32", dnnl::memory::data_type::f32},
    {"s32", dnnl::memory::data_type::s32},
    {"fp16", dnnl::memory::data_type::f16},
    {"u8", dnnl::memory::data_type::u8},
    {"s8", dnnl::memory::data_type::s8},
    {"bf16", dnnl::memory::data_type::bf16}};

ConcatOperator::ConcatOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  if (iter != attrs_map.end()) {
    axis_ = StringToNum<int64_t>(attrs_map["axis"]);
  }
  iter = attrs_map.find("output_dtype");
  if (iter != attrs_map.end()) {
    output_dtype_ = attrs_map["output_dtype"];
  }
}

void ConcatOperator::Prepare(const vector<Tensor*>& input,
                             const vector<Tensor*>& output) {
  // DNNL Concat primitive is required that all source tensors are of the same
  // data type (but not necessarily matching the data type of the destination
  // tensor). src data types: f32, bf16, f16, s8/u8
  LOG_IF(FATAL, output_dtype_ == "s32") << "Unsupported dtype s32...";
  bool src_same_dtype = true;
  for (int i = 0; i < input.size(); ++i) {
    LOG_IF(FATAL, (!input[i]->dtype().empty() && input[i]->dtype() == "s32"))
        << "Unsupported dtype s32...";
    src_same_dtype =
        (!input[i]->dtype().empty() && input[i]->dtype() == input[0]->dtype())
            ? true
            : false;
    if (!src_same_dtype) {
      LOG(FATAL)
          << "Concat Op " << name_
          << " requires that all source tensors are of the same data type. "
          << "Input_tensors[" << i << "] has dtype " << input[i]->dtype()
          << ", but input_tensors[0] has dtype " << input[0]->dtype();
      break;
    }
  }
  if (output_dtype_.empty()) {
    output_dtype_ = input[0]->dtype();
  } else {
    LOG_IF(FATAL, output_dtype_ != input[0]->dtype())
        << "dst dtype should be as same as input dtype.";
  }
  output[0]->set_dtype(output_dtype_);
}

void ConcatOperator::Reshape(const vector<Tensor*>& input,
                             const vector<Tensor*>& output) {
  //// Part0: Make sure all input tensors have the same shape, except for the dimension size of the axis
  ////        to concatenate on. If there are input tensors with different shapes, we will do broadcasting,
  ////        if it's possible, to make them have the same shape indicated by the largest values of each
  ////        dimension size among all input tensors, except for the one of the axis to concatenate on.
  ////        Please note this will only work for weight tensors, which are known ahead of runtime.
  // 0.1 Get the largest values of each dimension size except for the axis to concatenate on.
  if (axis_ < 0) {
    axis_ += input[0]->shape().size();
  }
  vector<int64_t> max_dim_sizes(input[0]->shape());
  size_t dim_num = max_dim_sizes.size();
  for (size_t i = 1; i < input.size(); ++i) {
    const auto& tmp_shape = input[i]->shape();
    assert(tmp_shape.size() == dim_num);
    for (size_t j = 0; j < dim_num; ++j) {
      max_dim_sizes[j] = max(tmp_shape[j], max_dim_sizes[j]);
    }
  }
  max_dim_sizes[axis_] = 1;

  // 0.2 If there is an input tensor with a different shape, do broadcasting if each dimension size of it
  //     is a factor of the corresponding value in max_dim_sizes. Otherwise, the assertion will fail since
  //     it's not a valid input for concat op.
  for (size_t i = 0; i < input.size(); ++i) {
    bool need_broadcast = false;
    vector<int64_t> times(dim_num, 1);
    const auto& tmp_shape = input[i]->shape();
    for (size_t j = 0; j < dim_num; ++j) {
      if (j == axis_) continue;
      assert(max_dim_sizes[j] % tmp_shape[j] == 0);
      times[j] = max_dim_sizes[j] / tmp_shape[j];
      if (times[j] > 1) {
        need_broadcast = true;
      }
    }
    if (!need_broadcast) continue;
    vector<int64_t> new_shape(max_dim_sizes);
    new_shape[axis_] = tmp_shape[axis_];
    size_t type_bytes = type2bytes[input[i]->dtype()];
    // Calculate memory size before and after broadcasting
    size_t old_mem_size =
        std::accumulate(tmp_shape.begin(), tmp_shape.end(), size_t(1),
                        std::multiplies<size_t>()) *
        type_bytes;
    size_t new_mem_size =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t(1),
                        std::multiplies<size_t>()) *
        type_bytes;
    // To make sure memory address is aligned to 64 bytes, and buffer size needs
    // to be a multiple of 64.
    void* new_data = aligned_alloc(
        ALIGNMENT, (new_mem_size + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1));
    void* data_buf[2];  // For temporary data copy during broadcasting
    data_buf[0] = malloc(new_mem_size);
    data_buf[1] = malloc(new_mem_size);
    size_t buf_idx = 0;
    memcpy(data_buf[0], input[i]->data(), old_mem_size);

    // Get strides of new_shape and front_strides of tmp_shape. For an array
    // {a0, a1, a2, ..., an}, strides[i] = a(i+1) * a(i+2) * ... * an,
    // front_strides[i] = a0 * a1 * ... * a(i-1)
    vector<int64_t> new_shape_strides = GetStrides(new_shape);
    vector<int64_t> tmp_shape_front_strides(dim_num, 1);
    for (size_t j = 1; j < dim_num; ++j) {
      tmp_shape_front_strides[j] =
          tmp_shape_front_strides[j - 1] * tmp_shape[j - 1];
    }

    // Do broadcasting
    for (int64_t j = dim_num - 1; j >= 0; --j) {
      if (times[j] == 1) continue;
      size_t broadcast_size = new_shape_strides[j] * tmp_shape[j] * type_bytes;
      for (size_t k = 0; k < tmp_shape_front_strides[j]; ++k) {
        auto src_addr =
            reinterpret_cast<char*>(data_buf[buf_idx]) + k * broadcast_size;
        auto dst_addr = reinterpret_cast<char*>(data_buf[(buf_idx + 1) & 1]) +
                        k * broadcast_size * times[j];
        for (size_t l = 0; l < times[j]; ++l) {
          memcpy(dst_addr + l * broadcast_size, src_addr, broadcast_size);
        }
      }
      buf_idx = (buf_idx + 1) & 1;
    }
    memcpy(new_data, data_buf[buf_idx], new_mem_size);
    free(data_buf[0]);
    free(data_buf[1]);

    input[i]->set_shape(new_shape);
    input[i]->set_data(new_data);
  }

  //// Part1: Derive operator's user proper shape and strides
  //// 1.1: Prepare Tensor origin shape
  const auto& src_shape_origin = input[0]->shape();

  // 1.2 Get tensor's number
  const int num_src = input.size();

  // // 1.4 Prepare memory descriptors
  // std::vector<memory::desc> src_mds;
  // std::vector<memory> src_mems;
  // for (int n = 0; n < num_src; ++n) {
  //   auto md = memory::desc(input[n]->shape(), type2mem[input[n]->dtype()],
  //   GetStrides(input[n]->shape())); auto mem = memory(md, eng_);
  //   src_mds.push_back(md);
  //   src_mems.push_back(mem);
  // }

  // Part2: Derive operator's format_any memory::desc and memory.
  vector<int64_t> dst_shape = src_shape_origin;
  dst_shape[axis_] = 0;
  for (int i = 0; i < num_src; ++i) {
    dst_shape[axis_] += input[i]->shape()[axis_];
  }
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);

  // // 2.1 Prepare primitive descriptors (cached)
  // dnnl::concat::primitive_desc concat_pd(axis_, src_mds, eng_);

  // // 2.2 Prepare primitive objects (cached)
  // concat_p_ = dnnl::concat(concat_pd);

  // // 2.3 Prepare memory objects (cached)
  // src_m_ = src_mems;
  // dst_m_ = memory(concat_pd.dst_desc(), eng_);

  //// Part3: Prepare data for our own implementation of concat op
  src_concat_bytes_.clear();
  src_concat_bytes_accum_.clear();
  size_before_concat_dim_ = std::accumulate(src_shape_origin.begin(),
                                            src_shape_origin.begin() + axis_, 1,
                                            std::multiplies<int64_t>());
  int64_t size_after_concat_dim =
      std::accumulate(src_shape_origin.begin() + axis_ + 1,
                      src_shape_origin.end(), 1, std::multiplies<int64_t>());
  for (int i = 0; i < num_src; ++i) {
    src_concat_bytes_.emplace_back(input[i]->shape()[axis_] *
                                   size_after_concat_dim *
                                   type2bytes[input[i]->dtype()]);
  }
  src_concat_bytes_accum_.emplace_back(0);
  for (int i = 1; i < num_src; ++i) {
    src_concat_bytes_accum_.emplace_back(src_concat_bytes_accum_[i - 1] +
                                         src_concat_bytes_[i - 1]);
  }
  dst_concat_bytes_ =
      dst_shape[axis_] * size_after_concat_dim * type2bytes[output[0]->dtype()];

  // onednn forward results have some issues when src0 tensor has dim 1 at axis
  // for example, (a, b, 1, d) + (a, b, n, d) -> (a, b, n+1, d) gets wrong
  // output when n >= b
  forward_with_dnnl_ =
      false;  // (input[0]->shape()[axis_] == 1 && output[0]->dtype() ==
              // input[0]->dtype())? false : true;
}

void ConcatOperator::Forward(const vector<Tensor*>& input,
                             const vector<Tensor*>& output) {
  // 0. Alias variables part
  const int num_src = input.size();

  void* dst_data = output[0]->mutable_data();

  // If input tensors have a same size of concat_dim, we will use oneDNN's
  // implementation. Otherwise, we will use our own implementation to avoid
  // incorrect results.
  if (forward_with_dnnl_) {
    // 1. Prepare memory objects with data_ptr
    dnnl::stream s(eng_);
    for (int n = 0; n < num_src; ++n) {
      const auto& src_data = input[n]->data();
      src_m_[n].set_data_handle(const_cast<void*>(src_data), s);
    }
    dst_m_.set_data_handle(reinterpret_cast<void*>(dst_data), s);

    // 2. Reorder the data when the primitive memory and user memory are
    // different
    // 3. Insert memory args
    std::unordered_map<int, memory> concat_args;
    for (int n = 0; n < num_src; ++n) {
      concat_args.insert({DNNL_ARG_MULTIPLE_SRC + n, src_m_[n]});
    }
    concat_args.insert({DNNL_ARG_DST, dst_m_});

    // 4. Execute the primitive
    concat_p_.execute(s, concat_args);
  } else {
#if __AVX512F__
#pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < size_before_concat_dim_; ++i) {
      for (int n = 0; n < num_src; ++n) {
        auto dst_addr = reinterpret_cast<char*>(dst_data) +
                        i * dst_concat_bytes_ + src_concat_bytes_accum_[n];
        int64_t concat_bytes = src_concat_bytes_[n];
        const char* src_addr =
            reinterpret_cast<const char*>(input[n]->data()) + i * concat_bytes;
        // codes written below implements memcpy(dst_addr, src_addr,
        // concat_bytes); loop_size = (concat_bytes / 64) * 64
        int64_t loop_size = concat_bytes & ~(AVX512_BYTES - 1);
        // Tail part size = concat_bytes - loop_size
        // To process tail part, we need a tail_mask, whose number of bit 1
        // equals to tail part size.
        __mmask64 tail_mask = (1ULL << (concat_bytes - loop_size)) - 1;
        for (int64_t j = 0; j < loop_size; j += AVX512_BYTES) {
          __m512 reg = _mm512_loadu_ps(src_addr + j);
          _mm512_storeu_ps(dst_addr + j, reg);
        }
        __m512i reg = _mm512_maskz_loadu_epi8(tail_mask, src_addr + loop_size);
        _mm512_mask_storeu_epi8(dst_addr + loop_size, tail_mask, reg);
      }
    }
#else
    LOG(ERROR) << "AVX2 concat not implemented";
#endif
  }

  // 5. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(Concat);
}  // namespace executor
