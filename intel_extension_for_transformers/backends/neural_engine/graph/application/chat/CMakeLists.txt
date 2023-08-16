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

set(TARGET chat_falcon)
add_executable_w_warning(${TARGET} main_chat.cpp)
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="falcon")
target_link_libraries(${TARGET} PUBLIC ne_layers falcon common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET chat_gptj)
add_executable_w_warning(${TARGET} main_chat.cpp)
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="gptj")
target_link_libraries(${TARGET} PUBLIC ne_layers gptj common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET chat_gptneox)
add_executable_w_warning(${TARGET} main_chat.cpp)
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="gptneox")
target_link_libraries(${TARGET} PUBLIC ne_layers gptneox common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET chat_dolly)
add_executable_w_warning(${TARGET} main_chat.cpp)
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="dolly")
target_link_libraries(${TARGET} PUBLIC ne_layers gptneox common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET chat_llama)
add_executable_w_warning(${TARGET} main_chat.cpp)
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="llama")
target_link_libraries(${TARGET} PUBLIC ne_layers llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET chat_mpt)
add_executable_w_warning(${TARGET} main_chat.cpp)
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="mpt")
target_link_libraries(${TARGET} PUBLIC ne_layers mpt common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET chat_starcoder)
add_executable_w_warning(${TARGET} main_chat.cpp)
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="starcoder")
target_link_libraries(${TARGET} PUBLIC ne_layers starcoder common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()
