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

import os, re
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
import time
from langchain import LLMChain
import argparse

# model_id = "/data1/lkk/llm_inference/mmlu/test/llama-7B-clm-50k-41.5"

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)


def inference(args, query, memory):
    if args.memory_type == "entity":
        prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are provided with information about entities the Human mentions, if relevant.

        Relevant entity information:
        {entities}

        Conversation:
        Human: {question}
        AI:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["entities", "question"]
        )
    else:
        prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
        AI should revise the answer according to the human feedback.

        {chat_history}

        Conversation:
        Human:{question}
        AI:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["chat_history", "question"]
        )

    start_time = time.time()
    qa = LLMChain(llm=model, memory=memory, prompt=PROMPT, verbose=False)
    result = qa.predict(question=query)
    end_time = time.time()
    print("inference cost {} seconds.".format(end_time - start_time))
    return result, memory

def is_safe_input(input_text):
    # Define a regular expression pattern to match safe input
    safe_pattern = r'^[a-zA-Z0-9\s,.!?]+$'
    return re.match(safe_pattern, input_text) is not None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to your finetuned model.', default='')
    parser.add_argument('--temperature', type=float, help='Temperature value.', default=0.3)
    parser.add_argument('--penalty', type=float, help='Penalty value.', default=1.0)
    parser.add_argument('--max_length', type=int, help='Max length for generation.',
                        default=1024)
    parser.add_argument('--memory_type', type=str, help='Select a kind of memory.', default='buffer_window')
    args = parser.parse_args()

    model = HuggingFacePipeline.from_model_id(model_id=args.model_path, task="text-generation", model_kwargs={
        #  "low_cpu_mem_usage" : True,
        "temperature": args.temperature,
        "max_length": args.max_length,
        "device_map": "auto",
        "repetition_penalty": args.penalty,
    })
    if args.memory_type == "buffer_window":
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
    elif args.memory_type == "buffer":
        memory = ConversationBufferMemory(memory_key="chat_history")
    elif args.memory_type == "entity":
        from Entity_Memory import SpacyEntityMemory
    memory = SpacyEntityMemory()

    while True:
        query = input("Enter input (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print('exit')
            break

        # Validate user input
        if not query:
            print('Input cannot be empty. Please try again.')
            continue

        # Perform input validation
        if not is_safe_input(query):
            print('Invalid characters in input. Please use only letters, numbers, and common punctuation.')
            continue

        result, memory = inference(args, query, memory)
        print("Input:" + query + '\nResponse:' + result + '\n')
