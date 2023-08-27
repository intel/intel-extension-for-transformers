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

#include "scatter_elements.hpp"
#include "common.hpp"


namespace executor {


static int64_t SizeHelper(size_t start, size_t end, const vector<int64_t>& shape) {
  // Must return 1 for an empty sequence
  int64_t size = 1;
  for (size_t i = start; i < end; i++) {
    if (shape[i] < 0) return -1;
    size *= shape[i];
  }
  return size;
}


/**
 * Return the total number of elements. Returns 1 for an empty (rank 0) TensorShape.
 */
static int64_t Size(const vector<int64_t>& shape) {
  int64_t size = SizeHelper(0, shape.size(), shape);
  // should we cache the size? as multiple operation may be expensive.
  return size;
}

template <class TIndex>
static std::vector<int64_t> GetIndices(
    const Tensor* data_input, const Tensor* indices_input, int64_t axis) {
  const auto input_data_shape = data_input->shape();
  const auto* indices_data_raw = static_cast<const TIndex*>(indices_input->raw_data());
  const auto num_indices = Size(indices_input->shape());
  const auto axis_dim_limit = input_data_shape[axis];

  std::vector<int64_t> indices_data_result;
  indices_data_result.reserve(num_indices);

  for (int64_t i = 0; i < num_indices; ++i) {
    const int64_t idx = static_cast<int64_t>(indices_data_raw[i]);

    if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
      LOG(ERROR) << "indices element out of data bounds, idx="
                 << idx << " must be within the inclusive range ["
                 << -axis_dim_limit << "," << axis_dim_limit - 1 << "]";
    }

    indices_data_result.push_back(idx < 0 ? idx + axis_dim_limit : idx);
  }
  return indices_data_result;
}


template <class Tdata>
static bool ScatterData(
    const Tensor* data_input, const std::vector<int64_t>& indices_data, const Tensor* updates_input, int64_t axis,
    Tensor* data_output) {
  const std::vector<int64_t> input_data_shape(data_input->shape());

  const auto input_elements = Size(input_data_shape);
  const auto total_input_bytes = type2bytes[data_input->dtype()] * input_elements;

  const auto num_indices = indices_data.size();

  const auto* src_base = static_cast<const Tdata*>(data_input->raw_data());
  auto dst_base =  static_cast<Tdata*>(data_output->mutable_data());




  // We allow runtime to re-use input for output. If input/output Tensor* are the same
  // we do not copy
  if (src_base != dst_base) {
      memcpy(static_cast<void*>(dst_base), static_cast<const void*>(src_base), total_input_bytes);
  }

  // Now poke updates

  const auto& upd_shape = updates_input->shape();
  const auto num_dims = input_data_shape.size();
  assert(num_dims > 0);

  // Allocate and zero out counts. The input/output is of the same rank as
  // indices/updates but the actual dimensions of indices/updates must be less or equal
  // than that of input/output because we can update no more elements than
  // the input contains. As we walk through the indices/updates
  // we maintain dimension count as we will need to use it
  // to compute output offset but using input/output dim values.
  // We treat the whole array as a number where each element having
  // different cardinality according to the upd_shape dimensions.
  // As each counter reaches its max (upd_shape) it resets to zero
  // and we carry to the more significant dim (right to left)
  std::vector<int64_t> dim_counters(num_dims);

  // This vector contains number of elements under the dimension.
  // For example, for the dimensions of [4, 2, 3] the vector
  // would contain [6, 3, 1] since for each count of dim 1 it
  // contains 3 elements of dim 2.
  // For each count of dim 0 we would have 2x3=6 elements.
  // The last value is always 1.
  // We use it to compute output element offset. For a given value of
  // counters we multiple each counter value per corresponding entry of dim_block_size value
  // and add up resulting the output element offset. However, for dimensions
  // that are equal to the specified axis value we take indices_data[index]
  // instead of the counter value.
  // E.g. for 3-dim and axis=0
  //    output[indices[i][j][k]][j][k] = updates[i][j][k]
  // for axis 1
  //    output[i][indices[i][j][k]][k] = updates[i][j][k]
  // and so on
  std::vector<int64_t> dim_block_size(num_dims);

  dim_block_size.back() = 1;
  if (num_dims > 1) {
    // We start at num_dims - 2 because we already pre-populated
    // the last element above
    for (auto i = int64_t(num_dims - 2); i >= 0; --i) {
      dim_block_size[i] = input_data_shape[i + 1] * dim_block_size[i + 1];
    }
  }


  const auto* update_data = static_cast<const Tdata*>(updates_input->raw_data());
  // For every update we compute the destination offset and copy it there
  for (int64_t index = 0; index < num_indices;) {
    const auto axis_idx = indices_data[index];

    // Compute the offset
    // See comments above for dim_block_size
    size_t dst_offset = 0;
    for (size_t i = 0; i < num_dims; ++i) {
      if (i == size_t(axis)) {
        // replace the counter with the update index for this dim
        dst_offset += axis_idx * dim_block_size[i];
      } else {
        dst_offset += dim_counters[i] * dim_block_size[i];
      }
    }

    *(dst_base + dst_offset) = *(update_data + index);

    if (++index == num_indices) {
      break;
    }
    // Increment counters
    // See comments for dim_counters above
    for (auto i = int64_t(num_dims - 1); i >= 0; --i) {
      auto v = ++dim_counters[i];
      assert(v <= upd_shape[i]);
      if (v < upd_shape[i]) {
        // No carry, done
        break;
      }
      // No carry for the most significant dim
      assert(i > 0);
      dim_counters[i] = 0;
    }
  }
  return true;
}

/*
Handle a potentially negative axis. Enforces negative axis is valid.
*/
static inline int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  // Handle negative axis
  return axis < 0 ? axis + tensor_rank : axis;
}


ScatterElementsOperator::ScatterElementsOperator(const shared_ptr<OperatorConfig>& conf) : Operator(conf) {
  auto attrs_map = operator_conf_->attributes();
  auto iter = attrs_map.find("axis");
  axis_ = (iter != attrs_map.end() && iter->second != "") ? stoi(iter->second) : 0;
}

// ScatterElements takes three inputs data, updates, and indices of the same rank r >= 1
// and an optional attribute axis that identifies an axis of data
//
void ScatterElementsOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  //// Part1: Derive operator's user proper shape and strides
  // 1.1: Prepare Tensor origin shape
  const auto& data_input_shape = input[0]->shape();
  const auto& indices_input_shape = input[1]->shape();
  const auto& update_input_shape = input[2]->shape();
  auto data_input_dtype = input[0]->dtype();

  // 1.2 Get tensor's adjusted shapes
  // dst shape = data_input_shape
  vector<int64_t> dst_shape(data_input_shape);

  // 1.4 Prepare memory descriptors
  // 1.5 Set dst tensor shape
  auto& dst_tensor_ptr = output[0];
  dst_tensor_ptr->set_shape(dst_shape);
  dst_tensor_ptr->set_dtype(data_input_dtype);
}


void ScatterElementsOperator::Forward(const vector<Tensor*>& input, const vector<Tensor*>& output) {
  // 0. Alias variables part
  const Tensor* data_input = input[0];
  const auto& input_data_shape = data_input->shape();
  const auto axis = HandleNegativeAxis(axis_, input_data_shape.size());
  const Tensor* indices_input = input[1];
  const Tensor* updates_input = input[2];
  Tensor* data_output = output[0];

  if (data_input->dtype() != updates_input->dtype()) {
    LOG(ERROR) << "data type is different from updates type" << std::endl;
  }

  auto indices_dims = indices_input->shape();
  auto updates_dims = updates_input->shape();

  if (indices_dims.size() != updates_dims.size()) {
    LOG(ERROR) << "indices and updates must have the same rank" << std::endl;
  }

  for (size_t i=0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != updates_dims[i]) {
      LOG(ERROR) << "Indices and updates dimensions differs at position=" << i << std::endl;
    }
  }

  if (data_input->shape().size() != indices_dims.size()) {
    LOG(ERROR) <<  "Indices must have the same rank as Input. Indices rank" << std::endl;
  }

  for (size_t i = 0; i < data_input->shape().size(); ++i) {
    if (static_cast<int64_t>(i) != axis && data_input->shape()[i] != indices_dims[i]) {
      LOG(ERROR) << "Indices dim at pos=" << i << "is greater than input dim" << std::endl;
    }
  }


  const auto index_type = indices_input->dtype();
  std::vector<int64_t> indices_data;
  if (index_type == "int32") {
    indices_data = GetIndices<int32_t>(data_input, indices_input, axis);
  } else if (index_type == "int64") {
    indices_data = GetIndices<int64_t>(data_input, indices_input, axis);
  } else {
    LOG(ERROR) << "unsupported datatype in GetIndices" << std::endl;
  }



  const auto data_type = data_input->dtype();

  if (data_type == "fp32") {
    ScatterData<float>(data_input, indices_data, updates_input, axis, data_output);
  } else if (data_type == "int32") {
    ScatterData<int32_t>(data_input, indices_data, updates_input, axis, data_output);
  } else {
  LOG(ERROR) << "unsupported data type" << std::endl;
  }

  // 2. unref tensors
  this->unref_tensors(input);
}

REGISTER_OPERATOR_CLASS(ScatterElements);
}  // namespace executor
