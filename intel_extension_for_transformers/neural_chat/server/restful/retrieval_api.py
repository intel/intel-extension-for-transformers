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
import re
import csv
import datetime
from typing import Optional, Dict
from fastapi import APIRouter, UploadFile, File
from ...config import GenerationConfig
from ...cli.log import logger
from ...server.restful.request import RetrievalRequest, AskDocRequest, FeedbackRequest
from ...server.restful.response import RetrievalResponse
from fastapi.responses import StreamingResponse
from ...utils.database.mysqldb import MysqlDb
from ...utils.record_request import record_request
from ...plugins import plugins


def check_retrieval_params(request: RetrievalRequest) -> Optional[str]:
    if request.params is not None and (not isinstance(request.params, Dict)):
        return f'Param Error: request.params {request.params} is not in the type of Dict'

    return None


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


@router.post("/v1/retrieval")
async def retrieval_endpoint(request: RetrievalRequest) -> RetrievalResponse:
    ret = check_retrieval_params(request)
    if ret is not None:
        raise RuntimeError("Invalid parametery.")
    return await router.handle_retrieval_request(request)


@router.post("/v1/askdoc/upload")
async def retrieval_upload(file: UploadFile = File(...)):
    global plugins
    filename = file.filename
    try:
        record_request(request_url="/v1/askdoc/upload",
                    request_body={'filename': filename},
                    user_id='default')
    except Exception as e:
        logger.error(f"[askdoc - upload] Fail to record request into db. {e}")
    path_prefix = "./enterprise_docs/"
    print(f"[askdoc - upload] filename: {filename}")
    if '/' in filename:
        filename = filename.split('/')[-1]
    with open(f"{path_prefix+filename}", 'wb') as fout:
        content = await file.read()
        fout.write(content),
    print("[askdoc - upload] file saved to local path.")

    try:
        print("[askdoc - upload] starting to append local db...")
        instance = plugins['retrieval']["instance"]
        instance.append_localdb(append_path=path_prefix)
        print(f"[askdoc - upload] kb appended successfully")
    except Exception as e:
        logger.info(f"[askdoc - upload] create knowledge base failes! {e}")
        return "Error occurred while uploading files."
    fake_kb_id = "fake_knowledge_base_id"
    return {"knowledge_base_id": fake_kb_id}


@router.post("/v1/askdoc/chat")
async def retrieval_chat(request: AskDocRequest):
    try:
        record_request(request_url="/v1/askdoc/chat",
                    request_body=request,
                    user_id='default')
    except Exception as e:
        logger.error(f"[askdoc - chat] Fail to record request into db. {e}")

    chatbot = router.get_chatbot()
    
    logger.info(f"[askdoc - chat] Predicting chat completion using kb '{request.knowledge_base_id}'")
    logger.info(f"[askdoc - chat] Predicting chat completion using prompt '{request.query}'")
    config = GenerationConfig()
    # Set attributes of the config object from the request
    for attr, value in request.__dict__.items():
        if attr == "stream":
            continue
        setattr(config, attr, value)
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
                logger.info(f"[askdoc - chat] {ret}")
                res = re.match("(http|https|ftp)://[^\s]+", output)
                if res != None:
                    formatted_link = f'<a style="color: blue; text-decoration: underline;"   href="{res.group()}" />'
                    yield f"data: {formatted_link}\n\n"
                else:
                    formatted_str = ret['text'].replace('\n', '<br/>')
                    yield f"data: {formatted_str}\n\n"
            if link != []:
                yield f"data: <hr style='border: 1px solid white; margin:0.5rem 0; '>\n\n"
                for single_link in link:
                    if single_link == None:
                        continue
                    raw_link = single_link["source"]
                    formatted_link = f"""<a style="color: blue; text-decoration: underline;"   
                                        href="{raw_link}">{raw_link}</a><br/>"""
                    yield f"data: {formatted_link}\n\n"
            yield f"data: [DONE]\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.post("/v1/askdoc/feedback")
def save_chat_feedback_to_db(request: FeedbackRequest) -> None:
    logger.info(f'[askdoc - feedback] fastrag feedback received.')
    try:
        record_request(request_url="/v1/askdoc/feedback",
                    request_body=request,
                    user_id='default')
    except Exception as e:
        logger.error(f"[askdoc - feedback] Fail to record request into db. {e}")
    mysql_db = MysqlDb()
    question, answer, feedback = request.question, request.answer, request.feedback
    feedback_str = 'dislike' if int(feedback) else 'like'
    logger.info(f'''[askdoc - feedback] feedback question: [{question}], 
                answer: [{answer}], feedback: [{feedback_str}]''')
    sql = f"INSERT INTO feedback VALUES(null, '{question}', '{answer}', {feedback})"
    logger.info(f'[askdoc - feedback] sql: {sql}')
    try:
        with mysql_db.transaction():
            mysql_db.insert(sql, None)
    except:
        raise Exception("""Exception occurred when inserting data into MySQL, 
                        please check the db session and your syntax.""")
    else:
        logger.info('[askdoc - feedback] feedback inserted.')
        mysql_db._close()
        return "Succeed"
    

@router.get("/v1/askdoc/downloadFeedback")
def get_feedback_from_db():
    try:
        record_request(request_url="/v1/askdoc/downloadFeedback",
                    request_body={},
                    user_id='default')
    except Exception as e:
        logger.error(f"[askdoc - download] Fail to record request into db. {e}")
    
    mysql_db = MysqlDb()
    sql = f"SELECT * FROM feedback ;"
    try:
        feedback_list = mysql_db.fetch_all(sql)
            
    except:
        raise Exception("""Exception occurred when querying data from MySQL, 
                        please check the db session and your syntax.""")
    else:
        mysql_db._close()
        def data_generator():
            output = io.StringIO()
            writer = csv.DictWriter(
                output, 
                fieldnames=['feedback_id', 'question', 'answer', 'feedback_result']
            )
            writer.writeheader()
            for row in feedback_list:
                writer.writerow(row)
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)

        cur_time = datetime.datetime.now()
        cur_time_str = cur_time.strftime("%Y%m%d")
        return StreamingResponse(
            data_generator(), 
            media_type='text/csv', 
            headers={"Content-Disposition": f"attachment;filename=feedback{cur_time_str}.csv"})
    
