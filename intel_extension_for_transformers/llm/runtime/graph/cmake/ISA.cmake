#  Copyright (c) 2023 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

if (MSVC)
    if (NE_AVX512)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
        # MSVC has no compile-time flags enabling specific
        # AVX512 extensions, neither it defines the
        # macros corresponding to the extensions.
        # Do it manually.
        if (NE_AVX512_VBMI)
            add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
            add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
        endif()
        if (NE_AVX512_VNNI)
            add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
            add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
        endif()
    elseif (NE_AVX2)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    elseif (NE_AVX)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
    endif()
else()
    if (NE_F16C)
        add_compile_options(-mf16c)
    endif()
    if (NE_FMA)
        add_compile_options(-mfma)
    endif()
    if (NE_AVX)
        add_compile_options(-mavx)
    endif()
    if (NE_AVX2)
        add_compile_options(-mavx2)
    endif()
    if (NE_AVX512)
        add_compile_options(-mavx512f)
        add_compile_options(-mavx512bw)
    endif()
    if (NE_AVX512_VBMI)
        add_compile_options(-mavx512vbmi)
    endif()
    if (NE_AVX512_VNNI)
        add_compile_options(-mavx512vnni)
    endif()
    if (NE_AMX)
        add_compile_options(-mamx-tile -mamx-int8 -mamx-bf16)
    endif()
endif()
