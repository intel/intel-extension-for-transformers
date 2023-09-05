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

#include "topk.hpp"
#include "common.hpp"
#include "operator_registry.hpp"

namespace executor {


TopKOperator::TopKOperator(const shared_ptr<OperatorConfig>& conf)
    : Operator(conf),
      axis_(0),
      largest_(1),
      sorted_(1),
      k_(0) {
  auto attrs_map = operator_conf_->attributes();

  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
  iter = attrs_map.find("largest");
  largest_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
  iter = attrs_map.find("sorted");
  sorted_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
  iter = attrs_map.find("k");
  k_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
  }




void TopKOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const auto& input_shape = input[0]->shape();
  vector<int64_t> dst_shape(input_shape);
  dst_shape[axis_] = k_;
  output[0]->set_shape(dst_shape);
  output[0]->set_dtype("int32");
  }

static int64_t SizeHelper(size_t start, size_t end, const vector<int64_t>& shape) {
  // Must return 1 for an empty sequence
  size_t size = 1;  // this is used to calculate the size, which is used for memory allocations, so validate no overflow
  for (size_t i = start; i < end; i++) {
    if (shape[i] < 0) return -1;
    size *= shape[i];
  }
  return size;
  }


static int64_t SizeToDimension(size_t dimension, const vector<int64_t>& shape) {
  const size_t num_dims = shape.size();
  if (dimension > num_dims) {
              LOG(ERROR) << "Invalid dimension of " << dimension << " for SizeFromDimension. Tensor has " <<
              num_dims << " dimensions." << std::endl;
  }

  int64_t size = SizeHelper(0, dimension, shape);
  return size;
}

static int64_t SizeFromDimension(size_t dimension, const vector<int64_t>& shape) {
  const size_t num_dims = shape.size();
  if (dimension > num_dims) {
              LOG(ERROR) << "Invalid dimension of " << dimension << " for SizeFromDimension. Tensor has " <<
              num_dims << " dimensions." << std::endl;
  }

  int64_t size = SizeHelper(dimension, num_dims, shape);
  return size;
}


static int64_t Size(const vector<int64_t>& shape) {
  return SizeHelper(0, shape.size(), shape);
}

// Selects the top k elements (largest or smallest based on template parameter)
template <class Comparator>
static void SelectTopK(const Comparator& comparer,
                       int64_t row_offset, int64_t num_blocks, int64_t block_slice, int64_t inter_block_offset,
                       const unsigned k, bool sort_top_k, vector<int64_t>* p_data_holder) {
  for (int64_t l = 0; l < num_blocks; ++l) {
    (*p_data_holder)[l] = (row_offset + (l * block_slice + inter_block_offset));
  }

  // find the top k (largest or smallest) elements in the data holder - O(n) average. O(n*n) worst case.
  // See https://en.wikipedia.org/wiki/Quickselect
  std::nth_element((*p_data_holder).begin(), (*p_data_holder).begin() + (k - 1), (*p_data_holder).end(), comparer);

  // sort the top k elements if needed - O (k log k)
  if (sort_top_k) {
    std::sort((*p_data_holder).begin(), (*p_data_holder).begin() + k, comparer);
  }

  // the data_holder now contains the indices of the top k elements in the first k elements
}


template <typename T>
struct GreaterValueCmp {
  using DataType = T;
  explicit GreaterValueCmp(const T* data = nullptr) : data_(data) {
  }

  bool operator()(const int64_t lhs_idx, const int64_t rhs_idx) const {
    return (data_[lhs_idx] > data_[rhs_idx] ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (data_[lhs_idx] == data_[rhs_idx] && lhs_idx < rhs_idx));
  }

  bool CompareValueOnly(const T& lhs, const T& rhs) const {
    return lhs > rhs;
  }

 private:
  const T* data_;
};

template <typename T>
struct LesserValueCmp {
  using DataType = T;

  explicit LesserValueCmp(const T* data = nullptr) : data_(data) {
  }

  bool operator()(const int64_t lhs_idx, const int64_t rhs_idx) const {
    return (data_[lhs_idx] < data_[rhs_idx] ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (data_[lhs_idx] == data_[rhs_idx] && lhs_idx < rhs_idx));
  }

  bool CompareValueOnly(const T& lhs, const T& rhs) const {
    return lhs < rhs;
  }

 private:
  const T* data_;
};



// 2. inference kernel(for int8 and f32)
void TopKOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  const auto& src0 = input[0];
  const auto& dst0 = output[0];
  const auto& input_shape = src0->shape();
  auto& dst0_shape = dst0->shape();
  // Cache some values that will be used in the implementation below
  const int64_t rows = SizeToDimension(static_cast<size_t>(axis_), input[0]->shape());
  const int64_t cols = Size(src0->shape()) / rows;
  const int64_t reduced_cols = SizeFromDimension(static_cast<size_t>(axis_), dst0_shape);

  auto* indices_data = static_cast<int32_t*>(output[0]->mutable_data());
  void* input_data = input[0]->mutable_data();

  // This is basically the number of elements within each of the "k" rows
  const int64_t num_blocks = input_shape[axis_];
  const int64_t block_slice = reduced_cols / k_;

  GreaterValueCmp<float> comparer_greater(static_cast<float*> (input_data));
  LesserValueCmp<float> comparer_lesser(static_cast<float*> (input_data));

  #pragma omp parallel for
  for (auto i = 0; i < rows;  ++i) {
    auto row_offset = i * cols;
    std::vector<int64_t> data_holder(num_blocks);
    std::vector<int64_t>* p_data_holder = &data_holder;
    for (int64_t j = 0; j < block_slice; ++j) {
      if (largest_) {
        SelectTopK<GreaterValueCmp<float>>(comparer_greater, row_offset,
                  num_blocks, block_slice, j, k_, sorted_, p_data_holder);
      } else {
        SelectTopK<LesserValueCmp<float>>(comparer_lesser, row_offset,
                num_blocks, block_slice, j, k_, sorted_, p_data_holder);
      }

      // Insert the top 'k' (largest or smallest) elements into the final output buffers
      for (int64_t l = 0; l < k_; ++l) {
        int64_t idx = data_holder[l];
        auto col_index = l * block_slice + j;
        // convert overall index to result index. avoid the cost of the '/' is possible
        indices_data[i*reduced_cols + col_index] = block_slice == 1 ? static_cast<int32_t>((idx - row_offset - j))
                                                     : static_cast<int32_t>((idx - row_offset - j) / block_slice);
      }
    }
  }
}
REGISTER_OPERATOR_CLASS(TopK);
}  // namespace executor
