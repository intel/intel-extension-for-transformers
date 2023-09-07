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

#include "tensor_desc.hpp"

#include "utils.hpp"

namespace jd {
tensor_desc::tensor_desc(const std::vector<int64_t>& shape, const data_type& dtype, const format_type& ftype)
    : shape_(shape), dtype_(dtype), ftype_(ftype) {
  if (shape_.size() != 0) {
    SPARSE_DLOG_IF(WARNING, dtype_ == data_type::undef) << "Non-empty tensor with undefined data type";
    SPARSE_DLOG_IF(WARNING, ftype_ == format_type::undef) << "Non-empty tensor with undefined format type";
  }
}

std::ostream& operator<<(std::ostream& os, const tensor_desc& td) {
  os << "tensor_desc [";
  for (size_t i = 0; i < td.shape().size(); ++i) {
    if (i != 0) os << ' ';
    os << td.shape()[i];
  }
  os << ", data_type=";
  os << data_type_name.at(td.dtype());
  os << ", ft=";
  os << format_type_name.at(td.ftype());
  os << ']';
  return os;
}
}  // namespace jd
