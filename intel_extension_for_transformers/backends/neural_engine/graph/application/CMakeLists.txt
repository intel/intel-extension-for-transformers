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

set(TARGET common)

add_library(${TARGET} OBJECT
    common.h
    common.cpp
    )

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PUBLIC)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ChatLLAMA)
add_subdirectory(ChatGPTNEOX)
add_subdirectory(ChatMPT)
add_subdirectory(ChatFALCON)
add_subdirectory(ChatGPTJ)
