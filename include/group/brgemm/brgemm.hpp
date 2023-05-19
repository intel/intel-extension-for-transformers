
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

/// @defgroup xetla_brgemm XeTLA BRGEMM
/// This is a brgemm API to compute matAcc = matA * matB.

#include "group/brgemm/api.hpp"
#include "group/brgemm/common.hpp"
#include "group/brgemm/compute_policy.hpp"
#include "group/brgemm/impl/default_fpu_xe.hpp"
#include "group/brgemm/impl/default_xmx_xe.hpp"
#include "group/brgemm/impl/pre_processing_xe.hpp"
#include "group/brgemm/impl/selector_xe.hpp"
#include "group/tile_shape.hpp"
