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

import sys
import requests
import json
from typing import *

class ChatModel:
  user: str
  bot: str

  endpoint: str
  ctx: List[Dict]

  def __init__(self, endpoint: str):
    self.ctx = []
    self.endpoint = endpoint

  def __headers__(self):
    return {}
  def __data__(self):
    return ""
  def __stream__(self, chunk, newctx) -> bool:
    return False

  def interact(self, msg: str):
    self.ctx.append({"role": self.user, "content": msg})
    resp = requests.post(self.endpoint, headers = self.__headers__(), data = self.__data__(), stream = True)
    newctx = {"role": None, "content": ""}
    for line in resp.iter_lines():
      if line == b'':
        continue
      if line == b'data: [DONE]':
        return
      chunk = json.loads(bytes.removeprefix(line, b'data: '))
      (word, done) = self.__stream__(chunk, newctx)
      if word is not None:
        sys.stdout.write(word)
        sys.stdout.flush()
      if done:
        break
    self.ctx.append(newctx)
    return

class OpenAIModel(ChatModel):
  key: str
  model: str
  def __init__(self, key: str, model = "gpt-3.5-turbo"):
    super().__init__('https://api.openai.com/v1/chat/completions')
    self.user = 'user'
    self.bot = 'assistant'
    self.key = key
    self.model = model
  def __headers__(self):
    return {
      'Authorization': 'Bearer ' + self.key,
      'Content-Type': 'application/json'
    }
  def __data__(self):
    return json.dumps({
      "model": self.model,
      "stream": True,
      "messages": self.ctx,
    })

  def __stream__(self, chunk, newctx) -> Tuple[str, bool]:
    choice = chunk['choices'][0]
    word = ""
    if "delta" in choice:
      if "role" in choice["delta"]:
        newctx["role"] = choice["delta"]["role"]
      if "content" in choice["delta"]:
        word = choice["delta"]["content"]
        newctx["content"] += word
    if "finish_reason" in choice and choice["finish_reason"] is not None:
      return (None, True)
    return (word, False)

class IntelModel(ChatModel):
  def __init__(self, endpoint: str, model: str, user: str, bot: str):
    super().__init__(endpoint)
    self.model = model
    self.user = user
    self.bot = bot
  
  def prompt(self):
    prompt = ""
    for ctx in self.ctx:
      prompt += ctx["role"] + ":" + ctx["content"] + "\n"
    return prompt + self.bot + ":"

  def __headers__(self):
    return {'Content-Type': 'application/json'}
  
  def __data__(self):
    return json.dumps({
      "model": self.model,
      "prompt": self.prompt(),
    })

  def __stream__(self, chunk, newctx) -> Tuple[str, bool]:
    if chunk["error_code"] != 0:
      return ("", True)

    text: str = chunk["text"]
    prompt = self.prompt()
    newctx["role"] = self.bot

    if not text.startswith(prompt):
      return ("", True)
    
    word = text.removeprefix(prompt).removeprefix(newctx["content"])
    done =  word.find('\n') != -1
    word = word[:word.find('\n')]
    newctx["content"] += word
    return (word, done)
