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

#ifndef ENGINE_SPARSELIB_INCLUDE_COMMON_H_
#define ENGINE_SPARSELIB_INCLUDE_COMMON_H_

#ifndef SPARSE_API_
#ifdef _MSC_VER
#if SPARSE_KERNEL_BUILD
#define SPARSE_API_ __declspec(dllexport)
#else
#define SPARSE_API_ __declspec(dllimport)
#endif
#elif __GNUC__ >= 4 || defined(__clang__)
#define SPARSE_API_ __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // SPARSE_API_

#ifndef SPARSE_TEST_API_
#ifdef SPARSE_TEST
#define SPARSE_TEST_API_ SPARSE_API_
#else
#define SPARSE_TEST_API_
#endif  // SPARSE_TEST_API_
#endif  // SPARSE_TEST

#endif  // ENGINE_SPARSELIB_INCLUDE_COMMON_H_
