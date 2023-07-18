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

find_package(Threads REQUIRED)
file(GLOB layers_srcs "layers/*.cpp")
set(sources ne_layers.c ${layers_srcs})

add_library_w_warning(ne_layers "${sources}")

target_include_directories(ne_layers PUBLIC .)
target_compile_features(ne_layers PUBLIC c_std_11) # don't bump
set_target_properties(ne_layers PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ne_layers PUBLIC Threads::Threads jblas::jblas ne_vec)
if(NOT WIN32)
  target_link_libraries(ne_layers PUBLIC rt)
endif()

