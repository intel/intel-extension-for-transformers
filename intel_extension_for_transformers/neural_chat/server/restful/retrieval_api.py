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
from pathlib import Path
from datetime import timedelta, timezone
from typing import Optional, Dict
from fastapi import APIRouter, UploadFile, File, Request, Response, Form
from ...config import GenerationConfig
from ...cli.log import logger
from ...server.restful.request import RetrievalRequest, FeedbackRequest
from ...server.restful.response import RetrievalResponse
from fastapi.responses import StreamingResponse
from ...utils.database.mysqldb import MysqlDb
from ...plugins import plugins


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
    beijing_time = utc_now.astimezone(SHA_TZ).strftime("%Y-%m-%d-%H:%M:%S")
    return beijing_time


class RetrievalAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, bot, use_deepspeed=False, world_size=1, host="0.0.0.0", port="80") -> None:
        self.chatbot = bot
        self.use_deepspeed = use_deepspeed
        self.world_size = world_size
        self.host = host
        self.port = port

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
RETRIEVAL_FILE_PATH = os.getenv("RETRIEVAL_FILE_PATH", default="./photoai_retrieval_docs")+'/'


@router.post("/v1/aiphotos/askdoc/upload_link")
async def retrieval_upload_link(request: Request):
    global plugins
    params = await request.json()
    link_list = params['link_list']

    user_id = request.client.host
    logger.info(f'[askdoc - upload_link] user id is: {user_id}')

    # append link into existed kb
    if 'knowledge_base_id' in params.keys():
        print(f"[askdoc - upload_link] append")
        knowledge_base_id = params['knowledge_base_id']
        persist_path = RETRIEVAL_FILE_PATH+user_id+'-'+knowledge_base_id + '/persist_dir'
        if not os.path.exists(persist_path):
            return f"Knowledge base id [{knowledge_base_id}] does not exist for user {user_id}, \
                Please check kb_id and save path again."

        try:
            print("[askdoc - upload_link] starting to append local db...")
            instance = plugins['retrieval']["instance"]
            instance.append_localdb(append_path=link_list, persist_path=persist_path)
            print(f"[askdoc - upload_link] kb appended successfully")
        except Exception as e:  # pragma: no cover
            logger.info(f"[askdoc - upload_link] create knowledge base failes! {e}")
            return Response(content="Error occurred while uploading links.", status_code=500)
        return {"Succeed"}
    # create new kb with link
    else:
        print(f"[askdoc - upload_link] create")
        import uuid
        kb_id = f"kb_{str(uuid.uuid1())[:8]}"
        path_prefix = RETRIEVAL_FILE_PATH

        # create new upload path dir
        cur_path = Path(path_prefix) / f"{user_id}-{kb_id}"
        os.makedirs(path_prefix, exist_ok=True)
        cur_path.mkdir(parents=True, exist_ok=True)

        user_upload_dir = Path(path_prefix) / f"{user_id}-{kb_id}/upload_dir"
        user_persist_dir = Path(path_prefix) / f"{user_id}-{kb_id}/persist_dir"
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        user_persist_dir.mkdir(parents=True, exist_ok=True)
        print(f"[askdoc - upload_link] upload path: {user_upload_dir}")
        
        try:
            # get retrieval instance and reload db with new knowledge base
            print("[askdoc - upload_link] starting to create local db...")
            instance = plugins['retrieval']["instance"]
            instance.create(input_path=link_list, persist_dir=str(user_persist_dir))
            print(f"[askdoc - upload_link] kb created successfully")
        except Exception as e:  # pragma: no cover
            logger.info(f"[askdoc - upload_link] create knowledge base failes! {e}")
            return "Error occurred while uploading files."
        return {"knowledge_base_id": kb_id}

@router.post("/v1/aiphotos/askdoc/create")
async def retrieval_create(request: Request,
                           file: UploadFile = File(...)):
    global plugins
    filename = file.filename
    if '/' in filename:
        filename = filename.split('/')[-1]
    print(f"[askdoc - create] received file: {filename}")

    user_id = request.client.host
    logger.info(f'[askdoc - create] user id is: {user_id}')

    import uuid
    kb_id = f"kb_{str(uuid.uuid1())[:8]}"
    path_prefix = RETRIEVAL_FILE_PATH

    cur_path = Path(path_prefix) / f"{user_id}-{kb_id}"
    os.makedirs(path_prefix, exist_ok=True)
    cur_path.mkdir(parents=True, exist_ok=True)
    
    user_upload_dir = Path(path_prefix) / f"{user_id}-{kb_id}/upload_dir"
    user_persist_dir = Path(path_prefix) / f"{user_id}-{kb_id}/persist_dir"
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    user_persist_dir.mkdir(parents=True, exist_ok=True)
    cur_time = get_current_beijing_time()
    print(f"[askdoc - create] upload path: {user_upload_dir}")
    
    # save file to local path
    save_file_name = str(user_upload_dir) + '/' + cur_time + '-' + filename
    with open(save_file_name, 'wb') as fout:
        content = await file.read()
        fout.write(content)
    print(f"[askdoc - create] file saved to local path: {save_file_name}")

    try:
        # get retrieval instance and reload db with new knowledge base
        print("[askdoc - create] starting to create local db...")
        instance = plugins['retrieval']["instance"]
        instance.create(input_path=str(user_upload_dir), persist_dir=str(user_persist_dir))
        print(f"[askdoc - create] kb created successfully")
    except Exception as e:  # pragma: no cover
        logger.info(f"[askdoc - create] create knowledge base failed! {e}")
        return "Error occurred while uploading files."
    return {"knowledge_base_id": kb_id}


@router.post("/v1/aiphotos/askdoc/append")
async def retrieval_append(request: Request,
                           file: UploadFile = File(...),
                           knowledge_base_id: str = Form(...)):
    global plugins
    filename = file.filename
    if '/' in filename:
        filename = filename.split('/')[-1]
    print(f"[askdoc - append] received file: {filename}, kb_id: {knowledge_base_id}")

    user_id = request.client.host
    logger.info(f'[askdoc - append] user id is: {user_id}')
    path_prefix = RETRIEVAL_FILE_PATH+user_id+'-'+knowledge_base_id
    upload_path = path_prefix + '/upload_dir'
    persist_path = path_prefix + '/persist_dir'
    if ( not os.path.exists(upload_path) ) or ( not os.path.exists(persist_path) ):
        return f"Knowledge base id [{knowledge_base_id}] does not exist for user {user_id}, \
            Please check kb_id and save path again."
    cur_time = get_current_beijing_time()
    print(f"[askdoc - append] upload path: {upload_path}")

    # save file to local path
    save_file_name = upload_path + '/' + cur_time + '-' + filename
    with open(save_file_name, 'wb') as fout:
        content = await file.read()
        fout.write(content)
    print(f"[askdoc - append] file saved to local path: {save_file_name}")

    try:
        # get retrieval instance and reload db with new knowledge base
        print("[askdoc - append] starting to append to local db...")
        instance = plugins['retrieval']["instance"]
        instance.append_localdb(append_path=save_file_name, persist_path=persist_path)
        print(f"[askdoc - append] new file successfully appended to kb")
    except Exception as e:  # pragma: no cover
        logger.info(f"[askdoc - append] create knowledge base failes! {e}")
        return "Error occurred while uploading files."
    return "Succeed"


@router.post("/v1/aiphotos/askdoc/chat")
async def retrieval_chat(request: Request):
    chatbot = router.get_chatbot()
    plugins['tts']['enable'] = False
    plugins['retrieval']['enable'] = True

    user_id = request.client.host
    logger.info(f'[askdoc - chat] user id is: {user_id}')

    # parse parameters
    params = await request.json()
    query = params['query']
    origin_query = params['translated']
    kb_id = params['knowledge_base_id']
    stream = params['stream']
    max_new_tokens = params['max_new_tokens']
    logger.info(f"[askdoc - chat] kb_id: '{kb_id}', query: '{query}', \
                stream mode: '{stream}', max_new_tokens: '{max_new_tokens}'")
    config = GenerationConfig(max_new_tokens=max_new_tokens)

    path_prefix = RETRIEVAL_FILE_PATH
    cur_path = Path(path_prefix) / "default" / "persist_dir"
    os.makedirs(path_prefix, exist_ok=True)
    cur_path.mkdir(parents=True, exist_ok=True)

    if kb_id == 'default':
        persist_dir = RETRIEVAL_FILE_PATH+"default/persist_dir"
    else:
        persist_dir = RETRIEVAL_FILE_PATH+user_id+'-'+kb_id+'/persist_dir'
    if not os.path.exists(persist_dir):
        return f"Knowledge base id [{kb_id}] does not exist, please check again."

    # reload retrieval instance with specific knowledge base
    if kb_id != 'default':
        print("[askdoc - chat] starting to reload local db...")
        instance = plugins['retrieval']["instance"]
        instance.reload_localdb(local_persist_dir = persist_dir)

    # non-stream mode
    if not stream:
        response = chatbot.predict(query=query, origin_query=origin_query, config=config)
        formatted_response = response.replace('\n', '<br/>')
        return formatted_response
    # stream mode
    generator, link = chatbot.predict_stream(query=query, origin_query=origin_query, config=config)
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


@router.post("/v1/askdoc/feedback")
def save_chat_feedback_to_db(request: FeedbackRequest) -> None:
    logger.info(f'[askdoc - feedback] fastrag feedback received.')
    mysql_db = MysqlDb()
    mysql_db._set_db("fastrag")
    question, answer, feedback = request.question, request.answer, request.feedback
    feedback_str = 'dislike' if int(feedback) else 'like'
    logger.info(f'''[askdoc - feedback] feedback question: [{question}], 
                answer: [{answer}], feedback: [{feedback_str}]''')
    question = question.replace('"', "'")
    answer = answer.replace('"', "'")
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_time = utc_now.astimezone(SHA_TZ).strftime("%Y-%m-%d %H:%M:%S")
    sql = f'INSERT INTO feedback VALUES(null, "' + question + '", "' + \
            answer + '", ' + str(feedback) + ', "' + beijing_time + '")'
    logger.info(f"""[askdoc - feedback] sql: {sql}""")
    try:
        with mysql_db.transaction():
            mysql_db.insert(sql, None)
    except:  # pragma: no cover
        raise Exception("""Exception occurred when inserting data into MySQL, 
                        please check the db session and your syntax.""")
    else:
        logger.info('[askdoc - feedback] feedback inserted.')
        mysql_db._close()
        return "Succeed"


@router.get("/v1/askdoc/downloadFeedback")
def get_feedback_from_db():
    mysql_db = MysqlDb()
    mysql_db._set_db("fastrag")
    sql = f"SELECT * FROM feedback ;"
    try:
        feedback_list = mysql_db.fetch_all(sql)
    except:  # pragma: no cover
        raise Exception("""Exception occurred when querying data from MySQL, \
                        please check the db session and your syntax.""")
    else:
        mysql_db._close()
        def data_generator():
            output = io.StringIO()
            writer = csv.DictWriter(
                output, 
                fieldnames=[
                    'feedback_id', 
                    'question', 
                    'answer', 
                    'feedback_result', 
                    'feedback_time']
            )
            writer.writeheader()
            for row in feedback_list:
                row['feedback_result'] = 'like' if ( row['feedback_result'] == 0 ) else 'dislike'
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
