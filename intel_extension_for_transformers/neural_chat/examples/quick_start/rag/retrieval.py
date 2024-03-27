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

from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig, plugins
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-retrieval', action="store_true", default=False, help="Message: Enable or Disable retrieval")
args = parser.parse_args()

print(args)

model_path="Intel/neural-chat-7b-v3-1"

plugins.retrieval.enable= False if args.no_retrieval else True

plugins.retrieval.args["embedding_model"]="BAAI/bge-base-en-v1.5"
plugins.retrieval.args["input_path"]="./text/"
plugins.retrieval.args["persist_directory"]="./output"
plugins.retrieval.args["append"]=False

config = PipelineConfig(model_name_or_path=model_path, plugins=plugins) #, optimization_config=MixedPrecisionConfig())
chatbot = build_chatbot(config)

response = chatbot.predict("Who went with Jack to seek his fortune?")
print(response)

response = chatbot.predict("What is IDM 2.0?")
print(response)
