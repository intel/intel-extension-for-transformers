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

macro(check_isa_unix isa)
  execute_process (
    COMMAND bash -c "lscpu|grep Flags"
    OUTPUT_VARIABLE outFlags
  )
  if ("${outFlags}" MATCHES "${isa}")
    set(${isa}_found TRUE CACHE BOOL "${isa} support")
    message(STATUS "${isa} support")
  else()
    set(${isa}_found FALSE CACHE BOOL "${isa} unsupported")
  endif()
endmacro()

macro(check_isa_win isa)
  find_program(WIN_COMPILER "cl")
  if (WIN_COREINFO)
    execute_process (
    COMMAND echo "$null >> empty.cpp"
    )
    execute_process (
      COMMAND cl "empty.cpp /arch:${isa}"
      OUTPUT_VARIABLE outFlags
    )
    if ("${outFlags}" MATCHES "ignoring unknown option")
      set(${isa}_found FALSE CACHE BOOL "${isa} support")
    else()
      set(${isa}_found TRUE CACHE BOOL "${isa} support")
    endif()
  else()
    set(${isa}_found FALSE CACHE BOOL "${isa} unsupported")
  endif()
endmacro()

macro(check_isa isa)
  if(WIN32)
    string(TOUPPER "${isa}" UPPER_ISA) 
    check_isa_win("${UPPER_ISA}")
  endif()
  if(UNIX) 
    check_isa_unix("${isa}")
  endif()
endmacro()
