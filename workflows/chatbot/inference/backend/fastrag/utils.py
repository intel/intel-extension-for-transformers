
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def detect_language(query):
    is_english = all(ord(c) < 128 for c in query)
    is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)

    if is_english and not is_chinese:
        return 'English'
    elif is_chinese and not is_english:
        return 'Chinese'
    else:
        return 'Mixed'
