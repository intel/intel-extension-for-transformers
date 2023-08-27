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

#ifndef ENGINE_SPARSELIB_INCLUDE_KERNELS_SPARSE_DATA_HPP_
#define ENGINE_SPARSELIB_INCLUDE_KERNELS_SPARSE_DATA_HPP_
#include <cassert>
#include <utility>
#include <vector>

#include "common.h"
#include "param_types.hpp"
#include "data_type/data_types.hpp"

namespace jd {
/**
 * @brief sparse_data_t class, abstraction of a pure data class. like dense
 * tensor's data member. https://matteding.github.io/2019/04/25/sparse-matrices/
 *   https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html
 *   There are two types of matrix, sparse and dense, each of which can do
 * compress or uncompress. Two concepts, that is to say, sparse/dense can do
 * compress/uncompress.
 */
template <typename T>
class sparse_data_t {
 public:
  sparse_data_t() {}
  sparse_data_t(const std::vector<int64_t>& indptr, const std::vector<int64_t>& indices, const std::vector<T>& data)
      : indptr_(indptr), indices_(indices), data_(data) {}
  virtual ~sparse_data_t() {}

 public:
  inline const std::vector<int64_t>& indptr() const { return indptr_; }
  inline const std::vector<int64_t>& indices() const { return indices_; }
  inline const std::vector<T>& data() const { return data_; }
  /**
   * @brief Get the number of non-zero elements / blocks (for blocked formats) / groups (for grouped data format) of the
   * matrix.
   */
  virtual uint64_t getnnz(int idx = -1) const {
    if (indptr_.empty()) {
      return 0;
    }
    return (idx == -1) ? (indptr_.back() - indptr_[0]) : (indptr_[idx + 1] - indptr_[idx]);
  }

 protected:
  std::vector<int64_t> indptr_;
  std::vector<int64_t> indices_;
  std::vector<T> data_;
};

template <typename T>
class csr_data_t : public sparse_data_t<T> {
 public:
  explicit csr_data_t(const format_type& encode_fmt = format_type::csr) : sparse_data_t<T>(), encode_fmt_(encode_fmt) {}
  explicit csr_data_t(const sparse_data_t<T>& spdata, const format_type& encode_fmt = format_type::csr)
      : sparse_data_t<T>(spdata), encode_fmt_(encode_fmt) {}
  virtual ~csr_data_t() {}

 public:
  inline const format_type& encode_format() const { return encode_fmt_; }

 protected:
  format_type encode_fmt_;
};

template <typename T>
class csrp_data_t : public csr_data_t<T> {
 public:
  explicit csrp_data_t(const format_type& encode_fmt = format_type::csrp) : csr_data_t<T>(encode_fmt) {}
  csrp_data_t(const sparse_data_t<T>& spdata, const format_type& encode_fmt = format_type::csrp,
              const std::vector<int64_t>& iperm = {}, const std::vector<int64_t>& xgroup = {})
      : csr_data_t<T>(spdata, encode_fmt), iperm_(iperm), xgroup_(xgroup) {}
  virtual ~csrp_data_t() {}

 public:
  inline const std::vector<int64_t>& iperm() const { return iperm_; }
  inline const std::vector<int64_t>& xgroup() const { return xgroup_; }

 protected:
  // CSRP (CSR with permutation): that rows with the same number of nonzeros are
  // grouped together. Vectorized sparse matrix multiply for compressed row
  // storage format[C].
  // https://www.climatemodeling.org/~rmills/pubs/iccs2005.pdf Learning Sparse
  // Matrix Row Permutations for Efficient SpMM on GPU Architectures[C].
  std::vector<int64_t> iperm_;   // Here iperm is the permutation vector.
  std::vector<int64_t> xgroup_;  // xgroup points to beginning indices of groups in iperm.
};

template <typename T>
class bsr_data_t : public sparse_data_t<T> {
 public:
  explicit bsr_data_t(const std::vector<dim_t> block_size, const std::vector<dim_t> shape,
                      const std::vector<dim_t>& indptr, const std::vector<dim_t>& indices, const std::vector<T>& data,
                      const dim_t group = 1)
      : sparse_data_t<T>(indptr, indices, data), shape_(shape), group_(group), block_size_(block_size) {
    nnz_group_ = indices.size() / group_;
  }
  virtual ~bsr_data_t() {}

 public:
  inline const std::vector<dim_t>& shape() const { return shape_; }
  inline const std::vector<dim_t>& block_size() const { return block_size_; }
  inline const dim_t& group() const { return group_; }
  inline const dim_t& nnz_group() const { return nnz_group_; }

 private:
  std::vector<dim_t> shape_;
  dim_t group_;
  dim_t nnz_group_;
  std::vector<dim_t> block_size_;
};

// BSC data is still row-major inside a block
template <typename T>
class bsc_data_t : public sparse_data_t<T> {
 public:
  explicit bsc_data_t(const std::vector<dim_t> block_size, const std::vector<dim_t> shape,
                      const std::vector<dim_t>& indptr, const std::vector<dim_t>& indices, const std::vector<T>& data)
      : sparse_data_t<T>(indptr, indices, data), shape_(shape), block_size_(block_size) {}
  virtual ~bsc_data_t() {}

 public:
  inline const std::vector<dim_t> shape() const { return shape_; }
  inline const std::vector<dim_t> block_size() const { return block_size_; }

 private:
  std::vector<dim_t> shape_;
  std::vector<dim_t> block_size_;
};

namespace spns {
static constexpr int ADJ = 4;  // 4 is that "Multiply groups of 4 adjacent pairs..."(vpdpbusd).

inline int align_nnz(const int& a_nnz);

template <typename T, dim_t group>
SPARSE_API_ std::vector<bsr_data_t<T>*>* reorder_to_bsr_amx(dim_t rows, dim_t cols, dim_t micro_rows,
                                                            const void* uncoded_ptr);
#ifdef _WIN32
template SPARSE_API_ std::vector<bsr_data_t<bfloat16_t>*>* reorder_to_bsr_amx<bfloat16_t, 32>(dim_t rows, dim_t cols,
                                                                                              dim_t micro_rows,
                                                                                              const void* uncoded_ptr);
#endif

template <typename T>
bsr_data_t<T> SPARSE_API_ tobsr(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data);

#ifdef _WIN32
template bsr_data_t<signed char> SPARSE_API_ tobsr<signed char>(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col,
                                                                const signed char* uncoded_data);

template bsr_data_t<float> SPARSE_API_ tobsr<float>(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col,
                                                    const float* uncoded_data);
#endif

/**
 * @brief Reorder a dense matrix to BSR encoded sparse format, grouping a certain number of blocks in a row and reorder
 * each groups in row-major, adding padding if no enough blocks left in a row. Note that colidxs will be padded with
 * last value of the same row, while data will be padded with 0.
 *
 * @tparam T data type of matrix elements
 * @tparam group the number of blocks to form a group
 * @param rows the number of rows of the original matrix
 * @param cols the number of columns of the original matrix
 * @param blk_row the numebr of rows for a BSR block
 * @param blk_col numebr of columns for a BSR block
 * @param uncoded_data pointer to the start of the original dense matrix
 * @return bsr_data_t<T> reordered and padded bsr matrix
 */
template <typename T, dim_t group>
bsr_data_t<T> reorder_to_bsr_group(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data);
#ifdef _WIN32
template bsr_data_t<signed char> SPARSE_API_ reorder_to_bsr_group<signed char, 4>(dim_t rows, dim_t cols, dim_t blk_row,
                                                                                  dim_t blk_col,
                                                                                  const signed char* uncoded_data);
#endif

template <typename T>
bsc_data_t<T> SPARSE_API_ tobsc(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col, const T* uncoded_data);

#ifdef _WIN32
template bsc_data_t<float> SPARSE_API_ tobsc<float>(dim_t rows, dim_t cols, dim_t blk_row, dim_t blk_col,
                                                    const float* uncoded_data);
#endif
}  // namespace spns
}  // namespace jd
#endif  // ENGINE_SPARSELIB_INCLUDE_KERNELS_SPARSE_DATA_HPP_
