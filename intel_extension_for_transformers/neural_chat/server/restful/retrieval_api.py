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
import json
import shutil
import datetime
import requests
from pathlib import Path
from datetime import timedelta, timezone
from typing import Optional, Dict, List, Union
from fastapi import APIRouter, UploadFile, File, Request, Response, Form, HTTPException
from ...config import GenerationConfig
from ...cli.log import logger
from ...server.restful.request import RetrievalRequest, FeedbackRequest
from ...server.restful.response import RetrievalResponse
from fastapi.responses import StreamingResponse, JSONResponse
from ...utils.database.mysqldb import MysqlDb
from ...plugins import plugins
from ...pipeline.plugins.retrieval.parser.context_utils import clean_filename


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


def language_detect(text: str):
    url = "https://translation.googleapis.com/language/translate/v2/detect"
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        logger.info(f"[ language_detect ] GOOGLE_API_KEY: {api_key}")
    except Exception as e:
        logger.info(f"No GOOGLE_API_KEY found. {e}")
    params = {
        'key': api_key,
        'q': text
    }

    response = requests.post(url, params=params)
    if response.status_code == 200:
        res = response.json()
        return res["data"]["detections"][0][0]
    else:
        print("Error status:", response.status_code)
        print("Error content:", response.json())
        return None


def language_translate(text: str, target: str='en'):
    url = "https://translation.googleapis.com/language/translate/v2"
    api_key = os.getenv("GOOGLE_API_KEY")
    logger.info(f"[ language_translate ] GOOGLE_API_KEY: {api_key}")
    params = {
        'key': api_key,
        'q': text,
        'target': target
    }

    response = requests.post(url, params=params)
    if response.status_code == 200:
        res = response.json()
        return res["data"]["translations"][0]
    else:
        print("Error status:", response.status_code)
        print("Error content:", response.json())
        return None


def get_path_prefix(knowledge_base_id: str, user_id: str):
    if knowledge_base_id == 'default':
        path_prefix = RETRIEVAL_FILE_PATH + 'default'
    else:
        path_prefix = RETRIEVAL_FILE_PATH+user_id+'-'+knowledge_base_id

    return path_prefix


def check_path(path: str):
    new_path = Path(path)
    if new_path.exists():
        logger.info(f"Path {path} exists.")
        return True
    else:
        try:
            new_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Path {path} created now.")
            return True
        except Exception as e:
            logger.info(f"fail to mkdir: {e}")
            return False


def remove_folder_with_ignore(folder_path: str, except_patterns=None):
    """
    Remove the specific folder, and ignore some files/folders

    :param folder_path: file path to delete
    :param except_patterns: files/folder name to ignore
    """
    if except_patterns is None:
        except_patterns = []
    print(f"except patterns: {except_patterns}")
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            # delete files except ones that match patterns
            file_path = os.path.join(root, name)
            if any(pattern in file_path for pattern in except_patterns):
                continue
            os.remove(file_path)

        # delete empty folder
        for name in dirs:
            dir_path = os.path.join(root, name)
            # delete folders except ones that match patterns
            if any(pattern in dir_path for pattern in except_patterns):
                continue
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


async def save_file_to_local_disk(save_path: str, file):
    with save_path.open("wb") as fout:
        try:
            content = await file.read()
            fout.write(content)
        except Exception as e:
            logger.info(f"[ save_file_to_local_disk ] write file failed. Exception: {e}")
            raise HTTPException(
                status_code=500,
                detail=f'Write file {save_path} failed. Exception: {e}'
            )


def get_file_structure(root_path: str, parent_path: str="") -> List[Dict[str, Union[str, List]]]:
    result = []
    for path in os.listdir(root_path):
        complete_path = parent_path + '/' + path if parent_path else path
        file_path = root_path+'/'+path
        if any(pattern in file_path for pattern in EXCEPT_PATTERNS):
            continue
        p = Path(file_path)
        # append file into result
        if p.is_file():
            file_dict = {
                "name": path,
                "id": complete_path,
                "type": "File",
                "parent": ""
            }
            result.append(file_dict)
        else:
            # append folder and inner files/folders into result using recursive function
            folder_dict = {
                "name": path,
                "id": complete_path,
                "type": "Directory",
                "children": get_file_structure(file_path, complete_path),
                "parent": ""
            }
            result.append(folder_dict)

    return result


def save_link_content(link_list: List, link_content: List, upload_path):
    file_names = []
    for link, content in zip(link_list, link_content):
        logger.info(f"= save link = link: {link}, content: {content}")
        file_name = clean_filename(link)+".jsonl"
        file_path = os.path.join(upload_path, file_name)
        file_content = content[0]
        data = {'content':file_content, 'link':link}
        with open(file_path, 'w') as file:
            json.dump(data, file)
        file_names.append(file_name)
    return file_names


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
        result = bot.predict(request)
        return RetrievalResponse(content=result)


router = RetrievalAPIRouter()
RETRIEVAL_FILE_PATH = os.getenv("RETRIEVAL_FILE_PATH", default="./retrieval_docs")+'/'
EXCEPT_PATTERNS = ["/xuhui_doc", "default/persist_dir"]


@router.post("/v1/askdoc/upload_link")
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
        path_prefix = get_path_prefix(knowledge_base_id, user_id)
        persist_path = path_prefix + '/persist_dir'
        upload_path = path_prefix + '/upload_dir'
        assert check_path(persist_path) == True

        try:
            print("[askdoc - upload_link] starting to append local db...")
            instance = plugins['retrieval']["instance"]
            print(f"[askdoc - upload_link] persist_path: {persist_path}")
            link_content = instance.append_localdb(append_path=link_list, persist_directory=persist_path)
            print(f"[askdoc - upload_link] kb appended successfully")
            file_names = save_link_content(link_list, link_content, upload_path)
            print(f"[askdoc - upload_link] link content saved to local path: {file_names}")
        except Exception as e:  # pragma: no cover
            logger.info(f"[askdoc - upload_link] create knowledge base fails! {e}")
            return Response(content="Error occurred while uploading links.", status_code=500)
        return {"status": True}
    # create new kb with link
    else:
        print(f"[askdoc - upload_link] create")
        import uuid
        kb_id = f"kb_{str(uuid.uuid1())[:8]}"
        path_prefix = get_path_prefix(knowledge_base_id, user_id)
        user_upload_dir = path_prefix + "/upload_dir"
        user_persist_dir = path_prefix + "/persist_dir"
        # create new upload path dir
        assert check_path(user_upload_dir) == True
        assert check_path(user_persist_dir) == True
        logger.info(f"[askdoc - upload_link] upload path: {user_upload_dir}")

        try:
            # get retrieval instance and reload db with new knowledge base
            logger.info("[askdoc - upload_link] starting to create local db...")
            instance = plugins['retrieval']["instance"]
            link_content = instance.create(input_path=link_list, persist_directory=str(user_persist_dir))
            logger.info(f"[askdoc - upload_link] kb created successfully")
            file_names = save_link_content(link_list, link_content, upload_path)
            print(f"[askdoc - upload_link] link content saved to local path: {file_names}")
        except Exception as e:  # pragma: no cover
            logger.info(f"[askdoc - upload_link] create knowledge base fails! {e}")
            return "Error occurred while uploading files."
        return {"knowledge_base_id": kb_id}


@router.post("/v1/askdoc/upload_files")
async def retrieval_add_files(request: Request,
                           files: List[UploadFile] = File(...),
                           file_paths: List[str] = Form(...),
                           knowledge_base_id: str = Form(...)):
    global plugins
    if knowledge_base_id == 'null':
        import uuid
        kb_id = f"kb_{str(uuid.uuid1())[:8]}"
        logger.info(f"[askdoc - upload_files] create new kb: {kb_id}")
    else:
        kb_id = knowledge_base_id

    for file_path, file in zip(file_paths, files):
        filename = file.filename
        if '/' in filename:
            filename = filename.split('/')[-1]
        logger.info(f"[askdoc - upload_files] received file: {filename}, kb_id: {kb_id}")
        user_id = request.client.host
        logger.info(f'[askdoc - upload_files] user id: {user_id}')

        path_prefix = get_path_prefix(kb_id, user_id)
        upload_path = path_prefix + '/upload_dir'
        persist_path = path_prefix + '/persist_dir'
        save_path = Path(upload_path) / file_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # save file content to local disk
        await save_file_to_local_disk(save_path, file)
        logger.info(f"[askdoc - upload_files] file saved to path: {save_path}")

        # append to retieval kb
        try:
            # instance = plugins['retrieval']["instance"]
            # instance.create(input_path=upload_dir, persist_directory=persist_dir)
            instance = plugins['retrieval']["instance"]
            if knowledge_base_id == 'null':
                instance.create(input_path=str(save_path), persist_directory=persist_path)
            else:
                instance.append_localdb(append_path=str(save_path), persist_directory=persist_path)
            logger.info(f"[askdoc - upload_files] new file {save_path} successfully appended to kb")
        except Exception as e:  # pragma: no cover
            logger.info(f"[askdoc - upload_files] create knowledge base fails! {e}")
            return "Error occurred while uploading files."

    if knowledge_base_id == 'null':
        return {"knowledge_base_id": kb_id}
    else:
        return "Succeed"


@router.post("/v1/askdoc/chat")
async def retrieval_chat(request: Request):
    chatbot = router.get_chatbot()
    plugins['tts']['enable'] = False
    plugins['retrieval']['enable'] = True

    user_id = request.client.host
    logger.info(f'[askdoc - chat] user id is: {user_id}')

    # parse parameters
    params = await request.json()
    origin_query = params['query']
    kb_id = params['knowledge_base_id']
    stream = params['stream']
    max_new_tokens = params['max_new_tokens']
    return_link = params['return_link']
    logger.info(f"[askdoc - chat] kb_id: '{kb_id}', \
                origin_query: '{origin_query}', stream mode: '{stream}', \
                max_new_tokens: '{max_new_tokens}', \
                return_link: '{return_link}'")
    config = GenerationConfig(max_new_tokens=max_new_tokens)

    # detect and translate query
    detect_res = language_detect(origin_query)
    if detect_res['language'] == 'en':
        query = origin_query
    else:
        query = language_translate(origin_query)['translatedText']

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
                flag = False
                if '<' in output and '>' in output:
                    output = output.replace('<', '').replace('>', '').replace(' ', '')
                    if output.endswith('\n'):
                        print(f"[!!!] found link endswith \\n")
                        flag = True
                    if output.endswith('.') or output.endswith('\n'):
                        output = output[:-1]

                if '](' in output:
                    output = output.split('](')[-1].replace(')', '')
                    if output.endswith('\n'):
                        flag = True
                    if output.endswith('.') or output.endswith('\n'):
                        output = output[:-1]

                res = re.match("(http|https|ftp)://[^\s]+", output)
                if res != None:
                    formatted_link = f'''<a style="color: blue; text-decoration: \
                        underline;"   href="{res.group()}"> {res.group()} </a>'''
                    logger.info(f"[askdoc - chat] in-line link: {formatted_link}")
                    if flag:
                        formatted_link += '<br/><br/>'
                    yield f"data: {formatted_link}\n\n"
                else:
                    formatted_str = ret['text'].replace('\n', '<br/><br/>')
                    formatted_str = formatted_str.replace('**:', '</b>:').replace('**', '<b>')
                    logger.info(f"[askdoc - chat] formatted: {formatted_str}")
                    yield f"data: {formatted_str}\n\n"
            if return_link and link != []:
                yield f"data: <hr style='border: 1px solid white; margin:0.5rem 0; '>\n\n"
                for single_link in link:
                    # skip empty link
                    if single_link == None:
                        continue
                    logger.info(f"[askdoc - chat] single link: {single_link}")
                    if isinstance(single_link, str):
                        raw_link = single_link
                    elif isinstance(single_link, dict):
                        raw_link = single_link["source"]
                    else:
                        logger.info(f"[askdoc - chat] wrong link format")
                        continue
                    # skip local file link
                    if not raw_link.startswith("http"):
                        continue
                    formatted_link = f"""<div style="margin: 0.4rem; padding: 8px 0; \
                        margin: 8px 0; font-size: 0.7rem;">  <a style="color: blue; \
                            border: 1px solid #0068B5;padding: 8px; border-radius: 20px;\
                            background: #fff; white-space: nowrap; width: 10rem;  color: #0077FF;"   \
                            href="{raw_link}" target="_blank"> {raw_link} </a></div>"""
                    logger.info(f"[askdoc - chat] link below: {formatted_link}")
                    yield f"data: {formatted_link}\n\n"
            yield f"data: [DONE]\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.post("/v1/askdoc/translate")
async def retrieval_translate(request: Request):
    user_id = request.client.host
    logger.info(f'[askdoc - translate] user id is: {user_id}')

    # parse parameters
    params = await request.json()
    content = params['content']
    logger.info(f'[askdoc - translate] origin content: {content}')

    detect_res = language_detect(content)
    logger.info(f'[askdoc - translate] detected language: {detect_res["language"]}')
    if detect_res['language'] == 'en':
        translate_res = language_translate(content, target='zh-CN')['translatedText']
    else:
        translate_res = language_translate(content, target='en')['translatedText']

    logger.info(f'[askdoc - translate] translated result: {translate_res}')
    return {"tranlated_content": translate_res}


@router.post("/v1/askdoc/feedback")
def save_chat_feedback_to_db(request: FeedbackRequest) -> None:
    logger.info(f'[askdoc - feedback] fastrag feedback received.')
    mysql_db = MysqlDb()
    mysql_db._set_db("fastrag")
    question, answer, feedback, comments = request.question, request.answer, request.feedback, request.comments
    feedback_str = 'dislike' if int(feedback) else 'like'
    logger.info(f'''[askdoc - feedback] feedback question: [{question}],
                answer: [{answer}], feedback: [{feedback_str}], comments: [{comments}]''')
    question = question.replace('"', "'")
    answer = answer.replace('"', "'")
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_time = utc_now.astimezone(SHA_TZ).strftime("%Y-%m-%d %H:%M:%S")
    sql = f'INSERT INTO feedback VALUES(null, "' + question + \
        '", "' + answer + '", ' + str(feedback) + ', "' + beijing_time + '", "' + comments + '")'
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
        csv_fields = ['feedback_id', 'question', 'answer', 'feedback_result', \
                            'feedback_time', 'comments']
        check_boxes = ['This is harmful / unsafe', "This isn't true", \
                       "This isn't helpful", "The link is invalid"]
        csv_fields.extend(check_boxes)
        def data_generator():
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                csv_fields
            )
            writer.writeheader()
            for row in feedback_list:
                if '^' in row['question']:
                    row['question'] = row['question'].replace('^', "'")
                if '^' in row['answer']:
                    row['answer'] = row['answer'].replace('^', "'")
                row['feedback_result'] = 'like' if ( row['feedback_result'] == 0 ) else 'dislike'
                comments = row['comments']
                # clip real comments
                row['comments'] = comments.split('#%#')[0]
                # save check box items
                for item in check_boxes:
                    if item in comments:
                        row[item] = 'True'
                    else:
                        row[item] = 'False'
                # write into csv file
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


@router.delete("/v1/askdoc/delete_all")
async def delete_all_files():
    """
        Delete all files and knowledge bases of current user.
        Re-create retriever using default plugin configs.
    """
    delete_path = RETRIEVAL_FILE_PATH
    if not os.path.exists(delete_path):
        logger.info(f'[askdoc - delete_all] No file/link uploaded. Clear.')
    else:
        # delete folder and files
        try:
            remove_folder_with_ignore(delete_path, except_patterns=EXCEPT_PATTERNS)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f'Failed to delete {delete_path}. Exception: {e}'
            )

    return {"status": True}


@router.delete("/v1/askdoc/delete_file")
async def delete_single_file(request: Request):
    """
        Delete file according to `del_path` and `knowledge_base_id`.
        `del_path`:
            - specific file path(e.g. /path/to/file.txt)
            - folder path(e.g. /path/to/folder)
            - "all_files": delete all files of this knowledge base
    """
    params = await request.json()
    del_path = params['del_path']
    knowledge_base_id = params['knowledge_base_id']
    user_id = request.client.host
    logger.info(f'[askdoc - delete_file] user id is: {user_id}')

    path_prefix = get_path_prefix(knowledge_base_id, user_id)

    # delete all uploaded files of kb and persist file of vector db
    if del_path == 'all_files':
        logger.info(f"[askdoc - delete_file] delete all files and persist files of kb {knowledge_base_id}")
        remove_folder_with_ignore(path_prefix, except_patterns=EXCEPT_PATTERNS)
        logger.info(f"[askdoc - delete_file] successfully delete kb {knowledge_base_id}")
        return {"status": True}

    delete_path = Path(path_prefix) / "upload_dir" / del_path
    logger.info(f'[askdoc - delete_file] delete_path: {delete_path}')

    # partially delete files/folders from the kb
    if delete_path.exists():
        # delete file
        if delete_path.is_file():
            try:
                delete_path.unlink()
            except Exception as e:
                logger.info(f"[askdoc - delete_file] fail to delete file {delete_path}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f'Failed to delete file {delete_path}. Exception: {e}'
                )
        # delete folder
        else:
            try:
                shutil.rmtree(delete_path)
            except Exception as e:
                logger.info(f"[askdoc - delete_file] fail to delete folder {delete_path}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f'Failed to delete folder {delete_path}. Exception: {e}'
                )
        return {"status": True}
    else:
        raise HTTPException(status_code=404, detail="File/folder not found. Please check del_path.")


@router.post("/v1/askdoc/get_file_structure")
async def rag_get_file_structure(request: Request):
    params = await request.json()
    knowledge_base_id = params['knowledge_base_id']
    user_id = request.client.host
    logger.info(f'[askdoc - get_file_structure] user id is: {user_id}')

    path_prefix = get_path_prefix(knowledge_base_id, user_id)
    upload_dir = path_prefix + '/upload_dir'

    if not Path(upload_dir).exists():
        raise HTTPException(
            status_code=404,
            detail=f'Knowledge base {knowledge_base_id} does not exists. Please check again'
        )

    file_content = get_file_structure(upload_dir)
    return file_content


@router.post("/v1/askdoc/recreate_kb")
async def recreate_kb(request: Request):
    params = await request.json()
    knowledge_base_id = params['knowledge_base_id']
    user_id = request.client.host
    logger.info(f'[askdoc - recreate_kb] user id is: {user_id}')
    path_prefix = get_path_prefix(knowledge_base_id, user_id)
    upload_dir = path_prefix + '/upload_dir'
    persist_dir = path_prefix + '/persist_dir'

    # clear persist_dir
    if Path(persist_dir).exists():
        logger.info(f"[askdoc - recreate_kb] deleting persist_dir: {persist_dir}")
        shutil.rmtree(persist_dir)
        # remove_folder_with_ignore(persist_dir)
        logger.info(f"[askdoc - recreate_kb] persist_dir cleared.")
    # create kb
    try:
        logger.info(f"[askdoc - recreate_kb] loading kb with: {upload_dir}")
        instance = plugins['retrieval']["instance"]
        instance.create(input_path=upload_dir, persist_directory=persist_dir)
    except Exception as e:
        logger.info(f"[askdoc - recreate_kb] create kb failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f'Failed to create kb. Exception: {e}'
        )
    return {"status": True}
