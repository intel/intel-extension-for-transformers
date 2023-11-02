/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <common/core/core.hpp>
#include <type_traits>

namespace buff_cmp {

/// @addtogroup xetla_test_utils
/// @{

// convenient datatype to represent ulp-converted fp buffers
using ulp_vec = std::vector<size_t>;

///
///@brief Structure used to describe tensors / buffers as stdlib vectors, idx_mapping vector is used to ignore "unwanted" elements from a tensor array.
///
/// @tparam dtype Datatype of buffer elements
template <typename dtype, typename dtype_src = dtype>
struct buff_vals {
    using type = dtype;
    std::vector<dtype> buff;
    std::vector<size_t> idx_mapping;
    size_t size;

    /// @brief Initializes and empty buff_vals<dtype> structure.
    /// @tparam dtype Datatype of the output structure and input buffer.
    buff_vals() { this->size = 0; }

    /// @brief Initializes buff_vals<dtype> structure with a one-to-one mapping to an input buffer.
    /// @tparam dtype Datatype of the output structure and input buffer.
    /// @param data Pointer to buffer of input data.
    /// @param n Size of input buffer data
    buff_vals(dtype_src *data, size_t n) {
        this->size = n;
        this->buff.resize(this->size, 0);
        for (size_t i = 0; i < this->size; ++i) {
            this->buff[i] = data[i];
            this->idx_mapping.push_back(i);
        }
    }

    /// @brief Converts a buffer array into a buff_vals<dtype> structure for convenient representation in xetla_buff_cmp<Tx, Ty>.
    /// @tparam dtype Datatype of the output structure and input buffer.
    /// @param data Pointer to buffer of input data.
    /// @param Blocky By default used to define size of input buffer. If input is organized as tensor (meaning, by blocks) then defines block height.
    /// @param Blockx Default value is 1 for non-tensor buffers, otherwise describes tensor block width.
    /// @param Sizex Default value is 1 for non-tensor buffers, otherwise describes tensor pitch.
    buff_vals(dtype_src *data, size_t Blocky, size_t Blockx, size_t Sizex) {
        this->size = Blockx * Blocky;
        this->buff.resize(this->size, 0);
        this->idx_mapping.resize(this->size, 0);
        size_t idx = 0;
        for (size_t i = 0; i < Blocky; ++i) {
            for (size_t j = 0; j < Blockx; ++j) {
                this->buff[idx] = data[i * Sizex + j];
                this->idx_mapping[idx] = i * Sizex + j;
                ++idx;
            }
        }
    }

    /// @brief Adds an element to the back of buff_vals<dtype> structure.
    /// @tparam dtype Datatype of the given structure and new element.
    /// @param val Element to be added to structure.
    /// @param idx Index mapping for element being added.
    void push(dtype val, size_t idx) {
        ++this->size;
        this->buff.push_back(val);
        this->idx_mapping.push_back(idx);
    }

    ///@brief Removes the last element from the buff_vals<dtype> structure.
    ///
    ///@return dtype
    ///
    dtype pop() {
        if (this->size <= 0) exit(-1);
        --this->size;
        dtype tmp = this->buff.back();
        this->buff.pop_back();
        this->idx_mapping.pop_back();
        return tmp;
    }
};

///
/// @brief Converts a buffer array into a buff_vals<dtype> structure for convenient representation in xetla_buff_cmp<Tx, Ty>.
///
/// @tparam dtype Datatype of the output structure and input buffer.
/// @param data Pointer to buffer of input data.
/// @param Blocky By default used to define size of input buffer. If input is organized as tensor (meaning, by blocks) then defines block height.
/// @param Blockx Default value is 1 for non-tensor buffers, otherwise describes tensor block width.
/// @param Sizex Default value is 1 for non-tensor buffers, otherwise describes tensor pitch.
/// @return buff_vals<dtype> Is the output structure describing the input buffer.
///
template <typename dtype>
buff_vals<dtype> xetla_get_buff_vals(
        dtype *data, size_t Blocky, size_t Blockx = 1, size_t Sizex = 1) {
    buff_vals<dtype> res;
    res.size = Blockx * Blocky;
    res.buff.resize(res.size, 0);
    res.idx_mapping.resize(res.size, 0);

    size_t idx = 0;
    for (size_t i = 0; i < Blocky; ++i) {
        for (size_t j = 0; j < Blockx; ++j) {
            res.buff[idx] = data[i * Sizex + j];
            res.idx_mapping[idx] = i * Sizex + j;
            ++idx;
        }
    }
    return res;
}

///@brief Structure which contains the absolute (ate) and relative (rte) element-wise errors of two input buffers.
struct rel_abs_vals {
    std::vector<double> ate;
    std::vector<double> rte;
    size_t size;
};

///
///@brief Takes two buff_vals structures as input and calculates the relative and absolute element-wise errors.
///
///@tparam dtype1 Type of the first input buffer, buff_vals<dtype1> v1.
///@tparam dtype2 Type of the second input buffer, buff_vals<dtype2> v2.
///@param v1 First input buffer.
///@param v2 Second input buffer.
///@return rel_abs_vals Structure containing all ate and rte values for v1 and v2.
///
template <typename T1, typename T2>
rel_abs_vals xetla_get_rte_and_ate(T1 &v1, T2 &v2) {
    using dtype1 = typename T1::type;
    using dtype2 = typename T2::type;
    auto get_ate = [=](dtype1 a, dtype2 b) {
        return fabs(((double)a) - ((double)b));
    };
    auto get_rte = [=](dtype1 a, dtype2 b) {
        return get_ate(a, b)
                / fmax(1E-10, fmax(fabs((double)a), fabs((double)b)));
    };

    rel_abs_vals res;
    res.size = v1.size;
    res.rte.resize(res.size, 0.0);
    res.ate.resize(res.size, 0.0);

    for (size_t i = 0; i < res.size; ++i) {
        res.ate[i] = get_ate(v1.buff[i], v2.buff[i]);
        res.rte[i] = get_rte(v1.buff[i], v2.buff[i]);
    }
    return res;
}

///
///@brief Creates a buffer of ULP-converted elements from an input buffer.
///
///@tparam T Type of the input buffer, buff_vals<dtype> &v1.
///@param v1 Input buffer.
///@return ulp_vec Vector containing all ULP-converted v1 elements.
///
template <typename T>
ulp_vec xetla_get_ulp_buffer(T &v1) {
    using dtype = typename T::type;
    using uint_dtype = gpu::xetla::uint_type_t<dtype>;
    ulp_vec ulp_buff(v1.size, 0);
    for (size_t i = 0; i < ulp_buff.size(); ++i) {
        uint_dtype val = (*reinterpret_cast<uint_dtype *>(&v1.buff[i]));
        ulp_buff[i] = val;
    }
    return ulp_buff;
}

///
///@brief Internal function used to handle fp-type buffers in xetla_buff_cmp<Tx, Ty>.
///
///@tparam dtype Type of the two input buffers.
///@param data First input buffer.
///@param other Second input buffer.
///@param name Name of the buffer comparison being performed.
///@param ulp_tol ULP-comparison threshold for acceptable FP buffer differences.
///@param abs_tol Relative-comparison threshold for acceptable FP buffer differences.
///@return true
///@return false
///
template <typename dtype>
bool _handle_fp_types(buff_vals<dtype> &data, buff_vals<dtype> &other,
        std::string name, size_t ulp_tol, double abs_tol) {
    if (std::is_same<remove_const_t<dtype>, gpu::xetla::bf16>::value) {
        if (ulp_tol == 0) ulp_tol = 8;
        if (abs_tol == 0) abs_tol = 0.25;
    } else if (std::is_same<remove_const_t<dtype>, gpu::xetla::fp16>::value) {
        if (ulp_tol == 0) ulp_tol = 8;
        if (abs_tol == 0) abs_tol = 0.25;
    } else if (std::is_same<remove_const_t<dtype>, float>::value) {
        if (ulp_tol == 0) ulp_tol = 8;
        if (abs_tol == 0) abs_tol = 0.25;
    } else if (std::is_same<remove_const_t<dtype>, gpu::xetla::tf32>::value) {
        /// least 13 bit is 0
        if (ulp_tol == 0) ulp_tol = 65536;
        if (abs_tol == 0) abs_tol = 0.25;
    } else {
        std::cout << "ERROR: unknown FP data type!!\n";
        return false;
    }

    ulp_vec ulp_data = xetla_get_ulp_buffer(data);
    ulp_vec ulp_other = xetla_get_ulp_buffer(other);

    auto get_ulp_ate = [=](size_t a, size_t b) {
        if (a > b)
            return a - b;
        else
            return b - a;
    };

    ulp_vec aulpte;
    aulpte.resize(ulp_data.size(), 0);
    for (size_t i = 0; i < ulp_data.size(); ++i)
        aulpte[i] = get_ulp_ate(ulp_data[i], ulp_other[i]);

    size_t aulpidx
            = std::max_element(aulpte.begin(), aulpte.end()) - aulpte.begin();

    std::cout << "\t"
              << "max absolute ULP diff:\n";
    std::cout << "\t\t"
              << "data_idx: " << data.idx_mapping[aulpidx]
              << " gold_idx: " << other.idx_mapping[aulpidx]
              << " abserr: " << (float)aulpte[aulpidx] << std::endl;
    std::cout << "\t\t"
              << "data_val: " << ulp_data[aulpidx]
              << " gold_val: " << (float)ulp_other[aulpidx] << std::endl;

    size_t ulp_threshold = ulp_tol;
    double small_num_threshold = abs_tol;
    size_t diff_elems_count = 0;
    bool flag = true;
    for (size_t i = 0; i < ulp_data.size(); ++i) {
        float des = other.buff[i];
        float act = data.buff[i];
        size_t ulp_des = ulp_other[i];
        size_t ulp_act = ulp_data[i];
        size_t sub_ulp = ulp_act - ulp_des;
        if (ulp_des > ulp_act) sub_ulp = ulp_des - ulp_act;
        if (!((fabs(act - des) <= small_num_threshold)
                    || (sub_ulp <= ulp_threshold)
                    || (fabs((des - act) / des) <= 0.001))) {
            ++diff_elems_count;
            flag = false;
            if (diff_elems_count <= 10) {
                std::cout << "\tdata_idx: " << data.idx_mapping[i]
                          << " gold_idx: " << other.idx_mapping[i]
                          << " data_val: " << act << " gold_val: " << des
                          << " data ULP val: " << ulp_act
                          << " gold ULP val: " << ulp_des << std::endl;
            }
        }
    }

    float fail_rate = diff_elems_count / ((float)ulp_data.size()) * 100;
    float pass_rate = 100 - fail_rate;
    std::cout << "\tpass rate: " << pass_rate << "%\n";

    return flag;
}

template <typename cast_dtype, typename T1, typename T2>
bool _cast_and_handle_fp_types(T1 &data, T2 &other, std::string name,
        double diff_elems_tol, size_t ulp_tol, double abs_tol) {
    buff_vals<cast_dtype> casted_data, casted_other;
    casted_data.size = data.size;
    casted_data.buff
            = std::vector<cast_dtype>(data.buff.begin(), data.buff.end());
    casted_data.idx_mapping = data.idx_mapping;

    casted_other.size = other.size;
    casted_other.buff
            = std::vector<cast_dtype>(other.buff.begin(), other.buff.end());
    casted_other.idx_mapping = other.idx_mapping;
    return buff_cmp::_handle_fp_types<cast_dtype>(
            casted_data, casted_other, name, ulp_tol, abs_tol);
}

///
///@brief Buffer/Tensor comparison function.
///
///@tparam dtype1 Type of the first input buffer, buff_vals<dtype1> data.
///@tparam dtype2 Type of the second input buffer, buff_vals<dtype2> other.
///@param data First input buffer.
///@param other Second input buffer.
///@param name Name of the buffer/tensor comparison beinf performed.
///@param diff_elems_tol Acceptable threshold percentage of int buffer elements that vary between data and other.
///@param ulp_tol ULP-comparison threshold for acceptable FP buffer differences.
///@param abs_tol Relative-comparison threshold for acceptable FP buffer differences.
///@return true
///@return false
///
template <typename T1, typename T2>
bool xetla_buff_cmp(T1 &data, T2 &other, std::string name,
        double diff_elems_tol = 0.02, size_t ulp_tol = 0, double abs_tol = 0) {
    if (data.size != other.size) {
        std::cout << "ERROR: buffer size or shape mismatch!\n";
        return false;
    }
    using dtype1 = typename T1::type;
    using dtype2 = typename T2::type;

    rel_abs_vals diff = xetla_get_rte_and_ate(data, other);

    unsigned ridx = std::max_element(diff.rte.begin(), diff.rte.end())
            - diff.rte.begin();
    unsigned aidx = std::max_element(diff.ate.begin(), diff.ate.end())
            - diff.ate.begin();

    std::cout << name << ":\n";
    std::cout << "\t"
              << "max relative diff:\n";
    std::cout << "\t\t"
              << "data_idx: " << data.idx_mapping[ridx]
              << " gold_idx: " << other.idx_mapping[ridx]
              << " relerr: " << diff.rte[ridx] << std::endl;
    std::cout << "\t\t"
              << "data_val: " << data.buff[ridx]
              << " gold_val: " << other.buff[ridx] << std::endl;
    std::cout << "\t"
              << "max absolute diff:\n";
    std::cout << "\t\t"
              << "data_idx: " << data.idx_mapping[aidx]
              << " gold_idx: " << other.idx_mapping[aidx]
              << " abserr: " << diff.ate[aidx] << std::endl;
    std::cout << "\t\t"
              << "data_val: " << data.buff[aidx]
              << " gold_val: " << other.buff[aidx] << std::endl;

    if constexpr (std::is_floating_point_v<dtype1> != 0
            || gpu::xetla::is_internal_type<dtype1>::value
            || std::is_same<remove_const_t<dtype1>, gpu::xetla::fp16>::value) {
        if (std::is_same<remove_const_t<dtype1>, gpu::xetla::bf16>::value
                || std::is_same<remove_const_t<dtype2>,
                        gpu::xetla::bf16>::value) {
            return _cast_and_handle_fp_types<gpu::xetla::bf16>(
                    data, other, name, diff_elems_tol, ulp_tol, abs_tol);
        } else if (std::is_same<remove_const_t<dtype1>, gpu::xetla::fp16>::value
                || std::is_same<remove_const_t<dtype2>,
                        gpu::xetla::fp16>::value) {
            return _cast_and_handle_fp_types<gpu::xetla::fp16>(
                    data, other, name, diff_elems_tol, ulp_tol, abs_tol);
        } else if (std::is_same<remove_const_t<dtype1>, gpu::xetla::tf32>::value
                || std::is_same<remove_const_t<dtype2>,
                        gpu::xetla::tf32>::value) {
            return _cast_and_handle_fp_types<gpu::xetla::tf32>(
                    data, other, name, diff_elems_tol, ulp_tol, abs_tol);
        } else if (std::is_same<remove_const_t<dtype1>, float>::value
                || std::is_same<remove_const_t<dtype2>, float>::value) {
            return _cast_and_handle_fp_types<float>(
                    data, other, name, diff_elems_tol, ulp_tol, abs_tol);
        }
    } else {
        size_t diff_elems_count = 0;
        for (size_t i = 0; i < data.size; ++i)
            if (data.buff[i] != other.buff[i]) ++diff_elems_count;
        float fail_rate = diff_elems_count / ((float)data.size) * 100;
        float pass_rate = 100 - fail_rate;
        std::cout << "\tpass rate: " << pass_rate << "%\n";

        if (diff_elems_count >= data.size * diff_elems_tol) {
            std::cout << "ERROR: int Output mismatch @ " << name << "\n";
            return false;
        }
    }
    return true;
}

/// @} xetla_test_utils

} // namespace buff_cmp
