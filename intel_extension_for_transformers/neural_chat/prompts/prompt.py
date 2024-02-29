#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastchat.conversation import get_conv_template, register_conv_template, Conversation, SeparatorStyle

# neuralchat-v3-1 prompt template
register_conv_template(
    Conversation(
        name="neural-chat-7b-v3-1",
        system_message="""### System:
- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, \
short stories, and make jokes.</s>\n""",
        roles=("### User:", "### Assistant:"),
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# neuralchat-v3 prompt template
register_conv_template(
    Conversation(
        name="neural-chat-7b-v3",
        system_message="""### System:
- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, \
short stories, and make jokes.</s>\n""",
        roles=("### User:", "### Assistant:"),
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# neuralchat-v2 prompt template
register_conv_template(
    Conversation(
        name="neural-chat-7b-v2",
        system_message="""### System:
- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, \
short stories, and make jokes.</s>\n""",
        roles=("### User:", "### Assistant:"),
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# neuralchat-v1.1 prompt template
register_conv_template(
    Conversation(
        name="neural-chat-7b-v1-1",
        system_template="""<|im_start|>system
{system_message}""",
        system_message="""- You are a helpful assistant chatbot trained by Intel.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# Alpaca template without input
register_conv_template(
    Conversation(
        name="alpaca_without_input",
        system_message="Below is an instruction that describes a task. " + \
            "Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
    )
)

# Alpaca template with input
register_conv_template(
    Conversation(
        name="alpaca_with_input",
        system_message="Below is an instruction that describes a task, " + \
            "paired with an input that provides further context. " + \
            "Write a response that appropriately completes the request.",
        roles=("### Instruction", "### Input", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
    )
)

# Summarization template
register_conv_template(
    Conversation(
        name="summarization",
        system_message="",
        roles=("", "Summarize the highlights of this article.\n"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Rag with context and memory template
register_conv_template(
    Conversation(
        name="rag_with_context_memory",
        system_message="""### You are a helpful, respectful and honest assistant to help the user with questions. \
         - Please refer to the search results obtained from the local knowledge base. But be careful to not \
         incorporate the information that you think is not relevant to the question.
         - If you don't know the answer to a question, please don't share false information.\n""" ,
        roles=("### Question:", "### Search Results:", "### Chat History:", "### Response:"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Rag without context template
register_conv_template(
    Conversation(
        name="rag_without_context",
        system_message="Have a conversation with a human. " + \
            "You are required to generate suitable response to the user input.\n",
        roles=("### Input:", "### Response:"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Rag without context template
register_conv_template(
    Conversation(
        name="rag_without_context_memory",
        system_message="Have a conversation with a human. " + \
            "You are required to generate suitable response to the user input.\n",
        roles=("### Input:", "### Chat History:", "### Response:"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)


# Rag with threshold
register_conv_template(
    Conversation(
        name="rag_with_threshold",
        system_message="""### You are a helpful, respectful and honest assistant to help the user with questions. \
         - Please refer to the search results obtained from the local knowledge base. But be careful to not \
         incorporate the information that you think is not relevant to the question.
         - If you don't know the answer to a question, please don't share false information.\n""",
        roles=("### Question:", "### Search Results:", "### Chat History:", "### Response:"),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Intent template
register_conv_template(
    Conversation(
        name="intent",
        system_message="Please identify the intent of the user query." + \
            " You may only respond with \"chitchat\" or \"QA\" without explanations" + \
            " or engaging in conversation.\n",
        roles=("### User Query: ", "### Response: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# Query Polish template
register_conv_template(
    Conversation(
        name="polish",
        system_message="### Please polish the following user query to make it clear and easy to be understood.\n",
        roles=("### User Query: ", "### Polished Query: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# NER template
register_conv_template(
    Conversation(
        name="ner",
        system_message="""Please determine the precise time mentioned in the user's query.
            Your response should consist only of an accurate time in the format
            'Time: YYYY-MM-DD' or 'Period: YYYY-MM-DD to YYYY-MM-DD.'
            If the user query does not include any time reference, please reply with 'None'.\n""",
        roles=("Current Time: ", "User Query: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
    )
)

# QA template
register_conv_template(
    Conversation(
        name="question_answer",
        roles=("Question: ", "Answer: "),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n\n",
    )
)

class PromptTemplate:
    def __init__(self, name="one_shot", clear_history=False):
        self.conv = get_conv_template(name)
        self.clear_history = clear_history

    @property
    def roles(self):
        return self.conv.roles

    def append_message(self, role: str, message: str):
        self.conv.append_message(role, message)

    def get_prompt(self) -> str:
        res = self.conv.get_prompt()
        if self.clear_history:
            self.clear_messages()
        return res

    def clear_messages(self) -> str:
        self.conv.messages = []

# pylint: disable=C0301
MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

SQLCODER_PROMPT = """
Task:
Generate a SQL query to answer the following question: {qurey}

Database Schema:
The query will run on a database with the following schema: {table_metadata_string}

Answer:
Given the database schema, here is the SQL query that answers the question "{qurey}":
"""

METADATA_STRING = """
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id
"""

def generate_sqlcoder_prompt(qurey, metadata_file):
    prompt = SQLCODER_PROMPT

    if not metadata_file:
        table_metadata_string = METADATA_STRING
    else:
        with open(metadata_file, "r") as f:
            table_metadata_string = f.read()

    prompt = prompt.format(
        qurey=qurey, table_metadata_string=table_metadata_string
    )
    return prompt
