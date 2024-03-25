##  Copyright (c) 2022 Intel Corporation
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

if(OpenMP_FOUND)
    #message(${CMAKE_CXX_FLAGS})
else()
    find_package(OpenMP REQUIRED)
    if(MSVC)
      target_compile_options(${HOST_LIBRARY_NAME} PRIVATE "-openmp:experimental")
    else()
      target_compile_options(${HOST_LIBRARY_NAME} PRIVATE "${OpenMP_CXX_FLAGS}")
    endif()
endif()
