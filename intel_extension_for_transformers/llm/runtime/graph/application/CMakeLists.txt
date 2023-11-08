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

add_library_w_warning(${TARGET} OBJECT
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
function(compile_quant TARGET SRC MODEL_NAME MODEL_LIB)
 add_executable_w_warning(${TARGET} ${SRC})
  warning_check(${TARGET})
  target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="${MODEL_NAME}")
  target_link_libraries(${TARGET} PUBLIC ${MODEL_LIB} common ${CMAKE_THREAD_LIBS_INIT})
  target_compile_features(${TARGET} PRIVATE cxx_std_11)
  if(TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
  endif()
endfunction()

compile_quant(quant_gptj      quant_model.cpp gptj      gptj)
compile_quant(quant_falcon    quant_model.cpp falcon    falcon)
compile_quant(quant_gptneox   quant_model.cpp gptneox   gptneox)
compile_quant(quant_dolly     quant_model.cpp dolly     gptneox)
compile_quant(quant_polyglot  quant_model.cpp polyglot  gptneox)
compile_quant(quant_llama     quant_model.cpp llama     llama)
compile_quant(quant_mpt       quant_model.cpp mpt       mpt)
compile_quant(quant_starcoder quant_model.cpp starcoder starcoder)
compile_quant(quant_opt       quant_model.cpp opt       opt)
compile_quant(quant_bloom     quant_model.cpp bloom     bloom)

compile_quant(quant_chatglm   quant_model.cpp chatglm   chatglm)
compile_quant(quant_chatglm2  quant_model.cpp chatglm2  chatglm2)
compile_quant(quant_baichuan  quant_model.cpp baichuan  baichuan)
compile_quant(quant_mistral   quant_model.cpp mistral   llama)

# all models running
if (NE_PYTHON_API)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(third_party/pybind11)
endif()

set(mymap_gptj 1)
set(mymap_falcon 2)
set(mymap_gptneox 3)
set(mymap_dolly 4)
set(mymap_llama 5)
set(mymap_mpt 6)
set(mymap_starcoder 7)
set(mymap_opt 8)
set(mymap_bloom 9)
set(mymap_chatglm2 10)
set(mymap_chatglm 11)
set(mymap_baichuan 12)
set(mymap_polyglot 13)
set(mymap_mistral 14)
set(mymap_whisper 15)


function(compile_run TARGET MAIN_CPP MAIN_PY MODEL_NAME MODEL_LIB)
  add_executable_w_warning(${TARGET} ${MAIN_CPP})
  warning_check(${TARGET})
  target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="${MODEL_NAME}" -DMODEL_NAME_ID=${mymap_${MODEL_NAME}})
  target_link_libraries(${TARGET} PUBLIC ne_layers ${MODEL_LIB} common ${CMAKE_THREAD_LIBS_INIT})
  target_compile_features(${TARGET} PRIVATE cxx_std_11)
  if(TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
  endif()

  if ((NE_PYTHON_API) AND (NOT (MAIN_PY STREQUAL "") ))
    pybind11_add_module("${MODEL_NAME}_cpp" ${MAIN_PY})
    target_link_libraries("${MODEL_NAME}_cpp" PRIVATE ne_layers ${MODEL_LIB} common)
    target_compile_definitions("${MODEL_NAME}_cpp" PUBLIC -DMODEL_NAME="${MODEL_NAME}" -DMODEL_NAME_ID=${mymap_${MODEL_NAME}})
  endif()
endfunction()

# text generation
compile_run(run_gptj      gpt_run.cpp   main_pybind.cpp gptj      gptj)
compile_run(run_falcon    gpt_run.cpp   main_pybind.cpp falcon    falcon)
compile_run(run_gptneox   gpt_run.cpp   main_pybind.cpp gptneox   gptneox)
compile_run(run_dolly     gpt_run.cpp   main_pybind.cpp dolly     gptneox)
compile_run(run_polyglot  gpt_run.cpp   main_pybind.cpp polyglot  gptneox)
compile_run(run_llama     gpt_run.cpp   main_pybind.cpp llama     llama)
compile_run(run_mpt       gpt_run.cpp   main_pybind.cpp mpt       mpt)
compile_run(run_starcoder gpt_run.cpp   main_pybind.cpp starcoder starcoder)
compile_run(run_opt       gpt_run.cpp   main_pybind.cpp opt       opt)
compile_run(run_bloom     gpt_run.cpp   main_pybind.cpp bloom     bloom)
compile_run(run_chatglm2  gpt_run.cpp   main_pybind.cpp chatglm2  chatglm2)
compile_run(run_chatglm   gpt_run.cpp   main_pybind.cpp chatglm   chatglm)
compile_run(run_baichuan  gpt_run.cpp   main_pybind.cpp baichuan  baichuan)
compile_run(run_mistral   gpt_run.cpp   main_pybind.cpp mistral   llama)

# speech recognition
compile_run(run_whisper   audio_run.cpp ""              whisper   whisper)
