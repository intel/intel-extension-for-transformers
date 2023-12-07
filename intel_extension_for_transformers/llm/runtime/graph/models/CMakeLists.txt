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

file(GLOB MODEL_UTILS_SOURCE "model_utils/*.cpp")
add_subdirectory(llama)
# add_subdirectory(gptj)
# add_subdirectory(mpt)
# add_subdirectory(gptneox)
# add_subdirectory(starcoder)
# add_subdirectory(falcon)
# add_subdirectory(opt)
# add_subdirectory(bloom)
add_subdirectory(chatglm)
# add_subdirectory(baichuan)
# add_subdirectory(qwen)
# add_subdirectory(whisper)