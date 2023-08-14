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

# include_directories(third_party/sentencepiece/src)
# add_subdirectory(third_party/sentencepiece)

# set(CHATGLM ne_chatglm)
# add_library(${CHATGLM} STATIC chatglm.cpp)
# target_link_libraries(${CHATGLM} PUBLIC ne_layers common sentencepiece-static ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${CHATGLM} PRIVATE cxx_std_17)

# set(TARGET main_chatglm_old)
# add_executable_w_warning(${TARGET} main_chatglm_old.cpp)
# target_link_libraries(${TARGET} PUBLIC ne_layers common ne_chatglm ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${TARGET} PRIVATE cxx_std_17)

# if(TARGET BUILD_INFO)
#   add_dependencies(${TARGET} BUILD_INFO)
# endif()

include_directories(third_party/sentencepiece/src)

set(TARGET main_chatglm)
add_executable_w_warning(${TARGET} main_chatglm.cpp)
target_link_libraries(${TARGET} PUBLIC ne_layers chatglm common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET quant_chatglm)
add_executable_w_warning(${TARGET} quant_chatglm.cpp)
target_link_libraries(${TARGET} PUBLIC ne_layers chatglm common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()
