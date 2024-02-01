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

from fastchat import client

completion = client.ChatCompletion.create(
    model="vicuna-7b-v1.1",
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "What's your favorite food?"},
        {
            "role": "assistant",
            "content": "As an AI language model, I don't have personal preferences or emotions. However, I can provide information about food. If you have any specific cuisine or dish in mind, I can tell you more about it.",
        },
        {"role": "user", "content": "What's your recommendation?"},
    ],
)

print(completion.choices[0].message)
