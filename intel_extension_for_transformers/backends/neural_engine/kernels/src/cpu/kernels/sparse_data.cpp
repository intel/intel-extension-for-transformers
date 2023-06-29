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

#include "kernels/sparse_data.hpp"
#include "src/utils.hpp"

namespace jd {
namespace spns {
int align_nnz(const int& a_nnz) { return ceil_div(a_nnz, ADJ) * ADJ; }

template <typename T, dim_t group>
std::vector<bsr_data_t<T>*>* reorder_to_bsr_amx(dim_t rows, dim_t cols, dim_t micro_rows, const void* uncoded_ptr) {
  const dim_t blk_row = 16;
  const dim_t blk_col = 1;
  SPARSE_LOG_IF(FATAL, rows % micro_rows != 0) << "rows should divded by micro_rows";
  dim_t num_micro_rows = rows / micro_rows;
  std::vector<bsr_data_t<T>*>* sparse_data = new std::vector<bsr_data_t<T>*>;
  for (int i = 0; i < num_micro_rows; ++i) {
    const T* uncoded_data = static_cast<const T*>(uncoded_ptr) + i * micro_rows * cols;
    const auto bsr_data = reorder_to_bsr_group<T, group>(micro_rows, cols, blk_row, blk_col, uncoded_data);
    sparse_data->push_back(new bsr_data_t<T>({blk_row, blk_col}, {rows, cols}, bsr_data.indptr(), bsr_data.indices(),
                                             bsr_data.data(), group));
  }
  return sparse_data;
}
template std::vector<bsr_data_t<int8_t>*>* reorder_to_bsr_amx<int8_t, 64>(dim_t, dim_t, dim_t, const void*);
template std::vector<bsr_data_t<bfloat16_t>*>* reorder_to_bsr_amx<bfloat16_t, 32>(dim_t, dim_t, dim_t, const void*);

template <typename T>
bsr_data_t<T> tobsr(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data) {
  std::vector<dim_t> rowptr;
  std::vector<dim_t> colidxs;
  for (dim_t b_row = 0; b_row < rows / blk_row; b_row++) {
    rowptr.push_back(colidxs.size());
    for (dim_t b_col = 0; b_col < cols / blk_col; b_col++) {
      bool is_zero = true;
      const T* dense_start = uncoded_data + b_row * blk_row * cols + b_col * blk_col;
      for (dim_t i = 0; i < blk_row; i++) {
        for (dim_t j = 0; j < blk_col; j++) {
          if (dense_start[i * cols + j] != 0) {
            is_zero = false;
            goto done_check_zero;
          }
        }
      }
    done_check_zero:
      if (!is_zero) {
        colidxs.push_back(b_col);
      }
    }
  }
  dim_t blksize = blk_row * blk_col;
  dim_t nnz = colidxs.size();
  rowptr.push_back(nnz);
  dim_t nnz_idx = 0;
  std::vector<T> data(nnz * blk_row * blk_col, static_cast<T>(0));
  for (dim_t b_row = 0; b_row < rows / blk_row; b_row++) {
    for (dim_t b_col_idx = rowptr[b_row]; b_col_idx < rowptr[b_row + 1]; b_col_idx++, nnz_idx++) {
      dim_t b_col = colidxs[b_col_idx];
      T* blkstart = data.data() + nnz_idx * blksize;
      const T* dense_start = uncoded_data + b_row * blk_row * cols + b_col * blk_col;
      for (dim_t i = 0; i < blk_row; i++) {
        for (dim_t j = 0; j < blk_col; j++) {
          blkstart[i * blk_col + j] = dense_start[i * cols + j];
        }
      }
    }
  }
  return (bsr_data_t<T>({blk_row, blk_col}, {rows, cols}, rowptr, colidxs, data));
}
template bsr_data_t<int8_t> tobsr(dim_t, dim_t, dim_t, dim_t, const int8_t*);

template <typename T, dim_t group>
bsr_data_t<T> reorder_to_bsr_group(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data) {
  bsr_data_t<T> bsr_data = tobsr<T>(rows, cols, blk_row, blk_col, uncoded_data);
  if (group == 1) return bsr_data;

  const dim_t num_row = bsr_data.indptr().size() - 1;
  std::vector<dim_t> padded_indices;
  std::vector<dim_t> padded_indptr(num_row + 1, 0);
  std::vector<T> padded_data;
  for (dim_t i = 0; i < num_row; ++i) {  // for each row of blocks
    padded_indptr[i] = padded_indices.size() / group;
    dim_t col_idx_lo = bsr_data.indptr()[i];
    dim_t col_idx_hi = bsr_data.indptr()[i + 1];

    for (dim_t col_idx = col_idx_lo; col_idx < col_idx_hi; col_idx += group) {  // for each group
      for (dim_t bi = 0; bi < blk_row; ++bi) {  // for each row (in terms of dense matrix) in a group
        for (dim_t ig = 0; ig < group; ig++) {  // for each group member
          dim_t dense_i = i * blk_row + bi;
          if (col_idx + ig < col_idx_hi) {
            dim_t dense_j = bsr_data.indices()[col_idx + ig];
            if (bi == 0)                            // colidx only need to be added once for each block
              padded_indices.push_back(dense_j);    // add col idx
            for (dim_t bj = 0; bj < blk_col; ++bj)  // for each col in a block
              padded_data.push_back(uncoded_data[dense_i * cols + dense_j]);
          } else {                                              // padding if no enough blocks left
            if (bi == 0)                                        // colidx only need to be added once for each block
              padded_indices.push_back(padded_indices.back());  // padd index with last one
            for (dim_t bj = 0; bj < blk_col; ++bj) padded_data.push_back(static_cast<T>(0));  // pad data with zeros
          }
        }
      }
    }
  }
  padded_indptr[num_row] = padded_indices.size() / group;
  return bsr_data_t<T>({blk_row, blk_col}, {rows, cols}, padded_indptr, padded_indices, padded_data, group);
}
template bsr_data_t<int8_t> reorder_to_bsr_group<int8_t, 4>(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col,
                                                            const int8_t* uncoded_data);

template <typename T>
bsc_data_t<T> tobsc(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data) {
  SPARSE_LOG_IF(FATAL, rows % blk_row != 0) << "row should devide by blk_row";
  SPARSE_LOG_IF(FATAL, cols % blk_col != 0) << "col should devide by blk_col";

  std::vector<dim_t> colptr;
  std::vector<dim_t> rowidxs;

  for (dim_t ib_col = 0; ib_col < cols / blk_col; ib_col++) {
    colptr.push_back(rowidxs.size());
    for (dim_t ib_row = 0; ib_row < rows / blk_row; ib_row++) {
      const T* blk_start = uncoded_data + ib_row * blk_row * cols + ib_col * blk_col;
      if (!all_zeros(blk_start, rows, blk_row, blk_col)) {
        rowidxs.push_back(ib_row);
      }
    }
  }

  dim_t blksize = blk_row * blk_col;
  dim_t nnz = rowidxs.size();
  colptr.push_back(nnz);

  std::vector<T> data(nnz * blksize, 0);
  T* curr_data_ptr = data.data();
  for (dim_t ib_col = 0; ib_col < cols / blk_col; ib_col++) {
    for (dim_t ib_row_idx = colptr[ib_col]; ib_row_idx < colptr[ib_col + 1]; ib_row_idx++) {
      dim_t ib_row = rowidxs[ib_row_idx];
      const T* dense_start = uncoded_data + ib_row * blk_row * cols + ib_col * blk_col;
      for (dim_t i = 0; i < blk_row; i++) {
        for (dim_t j = 0; j < blk_col; j++) {
          *(curr_data_ptr++) = dense_start[i * cols + j];
        }
      }
    }
  }
  return bsc_data_t<T>({blk_row, blk_col}, {rows, cols}, colptr, rowidxs, data);
}
template bsc_data_t<float> tobsc<float>(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col,
                                        const float* uncoded_data);
}  // namespace spns
}  // namespace jd
