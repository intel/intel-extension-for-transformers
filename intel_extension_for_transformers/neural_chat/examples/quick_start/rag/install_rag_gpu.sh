
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

#pip install intel-extension-for-transformers==1.3.2

export cwd=`pwd`
echo $cwd

pip install intel-extension-for-transformers==1.3.2
cd ~/itrex
git checkout 989671d365ce6bfd9ef2ad34c2bc1d8614dd708e

# Install neural-chat dependency for Intel GPU
cd ~/itrex/intel_extension_for_transformers/neural_chat
pip install -r requirements_xpu.txt

cd ~/itrex/intel_extension_for_transformers/neural_chat/pipeline/plugins/retrieval
pip install -r requirements.txt

git checkout main

pip install accelerate==0.28.0
pip install transformers_stream_generator==0.0.5
