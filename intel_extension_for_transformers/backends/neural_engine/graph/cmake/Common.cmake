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

function(warning_check TARGET)
    # TODO(hengyu): add warning check
    # if (MSVC)
    #     target_compile_definitions(${TARGET} PUBLIC -DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    #     target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:/utf-8>")
    #     target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options /sdl>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:/sdl>")
    # else()
    #     # Enable warning
    #     target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options -Wall>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Wall>")
    #     target_compile_options(${TARGET} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Wextra>")
    #     if(NOT CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
    #         target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options -Werror>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Werror>")
    #         target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:DPCPP>:SHELL:--compiler-options -Wno-error=deprecated-declarations>" "$<$<NOT:$<COMPILE_LANGUAGE:DPCPP>>:-Wno-error=deprecated-declarations>")
    #     endif()
    # endif()
endfunction()

function(add_executable_w_warning TARGET)
    add_executable(${TARGET} ${ARGN})
    warning_check(${TARGET})
endfunction()

function(add_library_w_warning TARGET)
    add_library(${TARGET} STATIC ${ARGN})
    warning_check(${TARGET})
endfunction()
