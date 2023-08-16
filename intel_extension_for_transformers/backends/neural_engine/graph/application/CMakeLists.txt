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

# common
set(TARGET common)

add_library(${TARGET} OBJECT
    common.h
    common.cpp
    )

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PUBLIC jblas::jblas)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# pybind
set(TARGET GptjPyBind)
add_library_w_warning(${TARGET} SHARED pybind_gptj.cpp)
target_link_libraries(${TARGET} PUBLIC ne_layers common gptj ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TARGET pybind_gptj)
add_executable_w_warning(${TARGET} pybind_gptj.cpp)
target_link_libraries(${TARGET} PUBLIC ne_layers common gptj ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# all models quant
set(TARGET quant_falcon)
add_executable_w_warning(${TARGET} quant_model.cpp)
warning_check(${TARGET})
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="falcon")
target_link_libraries(${TARGET} PUBLIC falcon common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET quant_gptj)
add_executable_w_warning(${TARGET} quant_model.cpp)
warning_check(${TARGET})
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="gptj")
target_link_libraries(${TARGET} PUBLIC gptj common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET quant_gptneox)
add_executable_w_warning(${TARGET} quant_model.cpp)
warning_check(${TARGET})
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="gptneox")
target_link_libraries(${TARGET} PUBLIC gptneox common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()


set(TARGET quant_dolly)
add_executable_w_warning(${TARGET} quant_model.cpp)
warning_check(${TARGET})
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="dolly")
target_link_libraries(${TARGET} PUBLIC gptneox common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET quant_llama)
add_executable_w_warning(${TARGET} quant_model.cpp)
warning_check(${TARGET})
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="llama")
target_link_libraries(${TARGET} PUBLIC llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET quant_mpt)
add_executable_w_warning(${TARGET} quant_model.cpp)
warning_check(${TARGET})
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="mpt")
target_link_libraries(${TARGET} PUBLIC mpt common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

set(TARGET quant_starcoder)
add_executable_w_warning(${TARGET} quant_model.cpp)
warning_check(${TARGET})
target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="starcoder")
target_link_libraries(${TARGET} PUBLIC starcoder common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

add_subdirectory(chat)
