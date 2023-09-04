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

function(compile_chat TARGET SRC MODEL_NAME MODEL_LIB)
 add_executable_w_warning(${TARGET} ${SRC})
  warning_check(${TARGET})
  target_compile_definitions(${TARGET} PUBLIC -DMODEL_NAME="${MODEL_NAME}")
  target_link_libraries(${TARGET} PUBLIC ne_layers ${MODEL_LIB} common ${CMAKE_THREAD_LIBS_INIT})
  target_compile_features(${TARGET} PRIVATE cxx_std_11)
  if(TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
  endif()
endfunction()

compile_chat(chat_gptj      main_chat.cpp gptj      gptj)
compile_chat(chat_falcon    main_chat.cpp falcon    falcon)
compile_chat(chat_gptneox   main_chat.cpp gptneox   gptneox)
compile_chat(chat_dolly     main_chat.cpp dolly     gptneox)
compile_chat(chat_llama     main_chat.cpp llama     llama)
compile_chat(chat_mpt       main_chat.cpp mpt       mpt)
compile_chat(chat_starcoder main_chat.cpp starcoder starcoder)
compile_chat(chat_bloom     main_chat.cpp bloom     bloom)
