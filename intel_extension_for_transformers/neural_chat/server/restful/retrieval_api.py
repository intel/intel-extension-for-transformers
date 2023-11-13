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

import io
import os
import re
import csv
import datetime
from datetime import timedelta, timezone
from typing import Optional, Dict
from fastapi import APIRouter, UploadFile, File, Request, Response
from ...config import GenerationConfig
from ...cli.log import logger
from ...server.restful.request import RetrievalRequest, AskDocRequest, FeedbackRequest
from ...server.restful.response import RetrievalResponse
from fastapi.responses import StreamingResponse
from ...utils.database.mysqldb import MysqlDb
from ...utils.record_request import record_request
from ...plugins import plugins, is_plugin_enabled


def check_retrieval_params(request: RetrievalRequest) -> Optional[str]:
    if request.params is not None and (not isinstance(request.params, Dict)):
        return f'Param Error: request.params {request.params} is not in the type of Dict'

    return None


def get_current_beijing_time():
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_time = utc_now.astimezone(SHA_TZ).strftime("%Y-%m-%d %H:%M:%S")
    return beijing_time


class RetrievalAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, bot) -> None:
        self.chatbot = bot

    def get_chatbot(self):
        if self.chatbot is None:
            raise RuntimeError("Retrievalbot instance has not been set.")
        return self.chatbot
    
    def handle_retrieval_request(self, request: RetrievalRequest) -> RetrievalResponse:
        bot = self.get_chatbot()
        # TODO: NeuralChatBot.retrieve_model()
        result = bot.predict(request)
        return RetrievalResponse(content=result)
    

router = RetrievalAPIRouter()


@router.post("/v1/aiphotos/askdoc/upload_link")
async def retrieval_upload_link(request: Request):
    global plugins
    params = await request.json()
    link_list = params['link_list']
    try:
        print("[askdoc - upload_link] starting to append local db...")
        instance = plugins['retrieval']["instance"]
        instance.append_localdb(append_path=link_list)
        print(f"[askdoc - upload_link] kb appended successfully")
    except Exception as e:
        logger.info(f"[askdoc - upload_link] create knowledge base failes! {e}")
        return Response(content="Error occurred while uploading links.", status_code=500)
    return {"knowledge_base_id": "local_kb_id"}


@router.post("/v1/aiphotos/askdoc/create_kb")
async def retrieval_create_kb(file: UploadFile = File(...)):
    global plugins
    filename = file.filename
    print(f"[askdoc - create_kb] received file: {filename}")

    upload_path = f"/home/tme/letong/askdoc_upload/enterprise_docs"
    cur_time = get_current_beijing_time()
    cur_time = cur_time.replace(' ', '-')
    print(f"[askdoc - create_kb] upload path: {upload_path}")
    if '/' in filename:
        filename = filename.split('/')[-1]

    # save file to local path
    save_file_name = upload_path + '/' + cur_time + '-' + filename
    with open(save_file_name, 'wb') as fout:
        content = await file.read()
        fout.write(content)
    print(f"[askdoc - create_kb] file saved to local path: {save_file_name}")

    try:
        # get retrieval instance and reload db with new knowledge base
        print("[askdoc - create_kb] starting to create local db...")
        instance = plugins['retrieval']["instance"]
        instance.append_localdb(append_path=save_file_name)
        print(f"[askdoc - create_kb] kb created successfully")
    except Exception as e:
        logger.info(f"[askdoc - create_kb] create knowledge base failes! {e}")
        return "Error occurred while uploading files."
    return {"knowledge_base_id": "local_kb_id"}


@router.post("/v1/aiphotos/askdoc/chat")
async def retrieval_chat(request: AskDocRequest):
    chatbot = router.get_chatbot()
    plugins['tts']['enable'] = False
    res = is_plugin_enabled('tts')
    print(f"tts plugin enable status: {res}")
    
    logger.info(f"[askdoc - chat] Predicting chat completion using kb '{request.knowledge_base_id}'")
    logger.info(f"[askdoc - chat] Predicting chat completion using prompt '{request.query}'")
    config = GenerationConfig(max_new_tokens=request.max_new_tokens)
    # Set attributes of the config object from the request
    for attr, value in request.__dict__.items():
        if attr == "stream":
            continue
        setattr(config, attr, value)
    # non-stream mode
    if not request.stream:
        response = chatbot.predict(query=request.query, config=config)
        formatted_response = response.replace('\n', '<br/>')
        return formatted_response
    # stream mode
    generator, link = chatbot.predict_stream(query=request.query, config=config)
    logger.info(f"[askdoc - chat] chatbot predicted: {generator}")
    if isinstance(generator, str):
        def stream_generator():
            yield f"data: {generator}\n\n"
            yield f"data: [DONE]\n\n"
    else: 
        def stream_generator():
            for output in generator:
                ret = {
                    "text": output,
                    "error_code": 0,
                }
                if '<' in output and '>' in output:
                    output = output.replace('<', '').replace('>', '').replace(' ', '')
                    if output.endswith('.') or output.endswith('\n'):
                        output = output[:-1]
                if '](' in output:
                    output = output.split('](')[-1].replace(')', '')
                    if output.endswith('.') or output.endswith('\n'):
                        output = output[:-1]
                res = re.match("(http|https|ftp)://[^\s]+", output)
                if res != None:
                    formatted_link = f'''<a style="color: blue; text-decoration: \
                        underline;"   href="{res.group()}"> {res.group()} </a>'''
                    logger.info(f"[askdoc - chat] in-line link: {formatted_link}")
                    yield f"data: {formatted_link}\n\n"
                else:
                    formatted_str = ret['text'].replace('\n', '<br/>')
                    logger.info(f"[askdoc - chat] formatted: {formatted_str}")
                    yield f"data: {formatted_str}\n\n"
            yield f"data: [DONE]\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


