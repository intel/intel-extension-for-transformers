
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

cd eval/mmmu_eval

python run_llava.py \
--output_path example_outputs/llava1.5_13b_val.json \
--model_path liuhaotian/llava-v1.5-13b \
--config_path configs/llava1.5.yaml

# evaluate the results
python main_eval_only.py --output_path example_outputs/llava1.5_13b_val.json
