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

add_library_w_warning(cpu_vec vec_arithmetic.cpp vec_compare.cpp vec_convert.cpp  vec_set.cpp  vec_store.cpp vec_load.cpp)
set_target_properties(cpu_vec PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET cpu_vec PROPERTY POSITION_INDEPENDENT_CODE  ON)
