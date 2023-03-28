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

#include "input.hpp"

namespace executor {

void InputOperator::Reshape(const vector<Tensor*>& input, const vector<Tensor*>& output) {
    for (int i = 0; i < input.size(); i++)
        output[i]->set_dtype(input[i]->dtype());
}

REGISTER_OPERATOR_CLASS(Input);

}  // namespace executor
