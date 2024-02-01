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

import time
import base64
import asyncio
import os
from typing import Optional, Dict, List
from fastapi.routing import APIRouter
from fastapi import APIRouter
from ...cli.log import logger
from ...config import GenerationConfig
from ...utils.database.mysqldb import MysqlDb
from fastapi import Request, BackgroundTasks, status, UploadFile, File
from fastapi.responses import JSONResponse, Response, StreamingResponse
from .photoai_services import *
from .photoai_utils import (
    byte64_to_image,
    image_to_byte64,
    generate_random_name
)
from .voicechat_api import (
    handle_talkingbot_asr as talkingbot_asr,
    create_speaker_embedding as talkingbot_embd
)
from ...plugins import plugins
from intel_extension_for_transformers.neural_chat.prompts import PromptTemplate


class PhotoAIAPIRouter(APIRouter):

    def __init__(self) -> None:
        super().__init__()
        self.chatbot = None

    def set_chatbot(self, chatbot, use_deepspeed, world_size, host, port) -> None:
        self.chatbot = chatbot
        self.use_deepspeed = use_deepspeed
        self.world_size = world_size
        self.host = host
        self.port = port

    def get_chatbot(self):
        if self.chatbot is None:
            logger.error("Chatbot instance is not found.")
            raise RuntimeError("Chatbot instance has not been set.")
        return self.chatbot

    async def handle_voice_chat_request(self, prompt: str, audio_output_path: Optional[str]=None) -> str:
        chatbot = self.get_chatbot()
        try:
            plugins['tts']['enable'] = True
            plugins['retrieval']['enable'] = False

            config = GenerationConfig(audio_output_path=audio_output_path)
            result, link = chatbot.chat_stream(query=prompt, config=config)
            def audio_file_generate(result):
                for path in result:
                    with open(path,mode="rb") as file:
                        bytes = file.read()
                        data = base64.b64encode(bytes)
                    yield f"data: {data}\n\n"
                yield f"data: [DONE]\n\n"
            return StreamingResponse(audio_file_generate(result), media_type="text/event-stream")
        except Exception as e:
            raise Exception(e)


router = PhotoAIAPIRouter()


def get_current_time() -> str:
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai'
    )
    utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
    cur_time = utc_now.astimezone(SHA_TZ).strftime("%Y/%m/%d")
    return cur_time


@router.post("/v1/aiphotos/uploadImages")
async def handle_ai_photos_upload_images(request: Request, background_tasks: BackgroundTasks):
    user_id = request.client.host
    logger.info(f'<uploadImages> user id is: {user_id}')
    res = check_user_ip(user_id)
    logger.info("<uploadImages> "+str(res))

    params = await request.json()
    image_list = params['image_list']

    IMAGE_ROOT_PATH = get_image_root_path()
    image_path = IMAGE_ROOT_PATH+'/user'+str(user_id)
    os.makedirs(image_path, exist_ok=True)
    mysql_db = MysqlDb()

    return_list = []
    image_obj_list = []

    for image in image_list:
        img_b64 = image['imgSrc'].split(',')[1]

        img_obj = byte64_to_image(str.encode(img_b64))
        tmp_name = generate_random_name()
        img_name = tmp_name+'.jpg'
        img_path = image_path+'/'+ img_name
        # save exif info from origin image
        exif = img_obj.info.get('exif', b"")

        # save image info into db
        empty_tags = '{}'
        insert_sql = f"INSERT INTO image_info VALUES(null, '{user_id}', '{img_path}', null, '', \
            'None', 'None', 'None', 'None', true, '{empty_tags}', 'processing', 'active');"
        try:
            with mysql_db.transaction():
                mysql_db.insert(sql=insert_sql, params=None)
        except Exception as e:
            logger.error("<uploadImages> "+str(e))
            return JSONResponse(content=f'Database insert failed for image {img_path}', status_code=500)

        # get image id
        fetch_sql = f"SELECT * FROM image_info WHERE image_path='{img_path}';"
        try:
            result = mysql_db.fetch_one(sql=fetch_sql)
        except Exception as e:
            logger.info("<uploadImages> "+str(e))
            return JSONResponse(content=f'Database select failed for image {img_path}', status_code=500)
        img_id = result['image_id']
        frontend_path = format_image_path(user_id, img_name)
        item = {'img_id': img_id, 'img_path': frontend_path}
        logger.info(f'<uploadImages> Image id is {img_id}, image path is {frontend_path}')
        return_list.append(item)
        obj_item = {"img_obj": img_obj, "exif": exif, "img_path": img_path, "img_id": img_id}
        image_obj_list.append(obj_item)
    mysql_db._close()
    background_tasks.add_task(process_images_in_background, user_id, image_obj_list)
    logger.info('<uploadImages> Finish image uploading and saving')
    return return_list


@router.post("/v1/aiphotos/getAllImages")
def handle_ai_photos_get_all_images(request: Request):
    user_id = request.client.host
    logger.info(f'<getAllImages> user id is: {user_id}')
    check_user_ip(user_id)
    origin = request.headers.get("Origin")
    logger.info(f'<getAllImages> origin: {origin}')

    try:
        result_list = []
        mysql_db = MysqlDb()
        image_list = mysql_db.fetch_all(
            sql=f'''SELECT image_id, image_path FROM image_info
            WHERE user_id="{user_id}" AND exist_status="active";''')
        for image in image_list:
            image_name = image['image_path'].split('/')[-1]
            result_list.append({"image_id": image['image_id'], "image_path": format_image_path(user_id, image_name)})
    except Exception as e:
        return JSONResponse(content=e, status_code=500)
    else:
        mysql_db._close()
        logger.info(f'<getAllImages> all images of user {user_id}: {result_list}')
        return result_list


@router.post("/v1/aiphotos/getTypeList")
def handle_ai_photos_get_type_list(request: Request):
    user_id = request.client.host
    logger.info(f'<getTypeList> user id is: {user_id}')
    check_user_ip(user_id)

    type_result_dict = {"type_list": {}, "prompt_list": {}}

    # address
    address_result = get_type_obj_from_attr('address', user_id)
    type_result_dict['type_list']['address'] = address_result

    # time
    time_result = get_type_obj_from_attr('time', user_id)
    type_result_dict['type_list']['time'] = time_result

    # person
    person_result = get_face_list_by_user_id(user_id)
    type_result_dict['type_list']['person'] = person_result

    # other
    other_time_result = get_images_by_type(user_id, type="time", subtype="None")
    other_add_result = get_images_by_type(user_id, type="address", subtype="default")
    logger.info(f'<getTypeList> other time result: {other_time_result}')
    logger.info(f'<getTypeList> other address result: {other_add_result}')
    for time_res in other_time_result:
        if time_res in other_add_result:
            continue
        other_add_result.append(time_res)
    logger.info(f'<getTypeList> final other result: {other_add_result}')
    type_result_dict['type_list']['other'] = other_add_result

    # prompt list
    address_list = get_address_list(user_id)
    type_result_dict['prompt_list']['address'] = address_list
    type_result_dict['prompt_list']['time'] = list(time_result.keys())
    type_result_dict['prompt_list']['person'] = list(person_result.keys())

    # process status
    type_result_dict["process_status"] = get_process_status(user_id)
    return type_result_dict


@router.post("/v1/aiphotos/getImageByType")
async def handle_ai_photos_get_image_by_type(request: Request):
    user_id = request.client.host
    logger.info(f'<getImageByType> user id is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    type = params['type']
    subtype = params['subtype']

    try:
        result = get_images_by_type(user_id, type, subtype)
    except Exception as e:
        return Response(content=str(e), status_code=status.HTTP_400_BAD_REQUEST)
    return result


@router.post("/v1/aiphotos/getImageDetail")
async def handle_ai_photos_get_image_detail(request: Request):
    user_id = request.client.host
    logger.info(f'<getImageDetail> user id is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    image_id = params['image_id']
    logger.info(f'<getImageDetail> Getting image detail of image {image_id} by user {user_id}')

    try:
        mysql_db = MysqlDb()
        image_info = mysql_db.fetch_one(
            sql=f'''SELECT * FROM image_info WHERE
            image_id={image_id} AND user_id="{user_id}" AND exist_status="active";''',
            params=None)
    except Exception as e:
        logger.error("<getImageDetail> "+str(e))
        return JSONResponse(content=f'Exception {e} occurred when selecting image {image_id} from MySQL.')
    finally:
        mysql_db._close()

    if image_info:
        image_detail = format_image_info(image_info)
        logger.info(f'<getImageDetail> Image detail of image {image_id} is: {image_detail}')
        return image_detail
    else:
        return JSONResponse(
            content=f"No image id: {image_id} for user {user_id}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@router.post("/v1/aiphotos/deleteImage")
async def handel_ai_photos_delete_Image(request: Request):
    params = await request.json()
    image_id = params['image_id']
    logger.info(f'<deleteImage> Getting image detail of image {image_id}')

    user_id = request.client.host
    logger.info(f'<deleteImage> user id is: {user_id}')
    check_user_ip(user_id)

    try:
        delete_single_image(user_id, image_id)
    except Exception as e:
        logger.error("<deleteImage> "+str(e))
        return Response(content=e, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    logger.info(f'<deleteImage> Image {image_id} successfully deleted.')
    return "Succeed"


@router.post("/v1/aiphotos/deleteUser")
def handle_ai_photos_delete_user(request: Request):
    user_id = request.client.host
    logger.info(f'<deleteUser> user ip is: {user_id}')
    check_user_ip(user_id)

    try:
        delete_user_infos(user_id)
    except Exception as e:
        logger.error("<deleteUser> "+str(e))
        raise Exception(e)

    return "Succeed"


@router.post("/v1/aiphotos/updateLabel")
async def handle_ai_photos_update_label(request: Request):
    # check request user
    user_id = request.client.host
    logger.info(f'<updateLabel> user id is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    label_list = params['label_list']

    try:
        mysql_db = MysqlDb()
        for label_obj in label_list:
            label = label_obj['label']
            label_from = label_obj['from']
            label_to = label_obj['to']
            if label == 'person':
                with mysql_db.transaction():
                    mysql_db.update(
                        sql=f'''UPDATE face_info SET face_tag="{label_to}"
                        WHERE face_tag="{label_from}"''',
                        params=None)
                    mysql_db.update(
                        sql=f"""UPDATE image_face SET face_tag='{label_to}'
                        WHERE user_id='{user_id}' and face_tag='{label_from}';""",
                        params=None)
                continue
            if label == 'address':
                update_sql = f"""UPDATE image_info SET address='{label_to}'
                WHERE user_id='{user_id}' and address LIKE '%{label_from}%';"""
            elif label == 'time':
                update_sql = f"""UPDATE image_info SET captured_time='{label_to}'
                WHERE user_id='{user_id}' and DATEDIFF(captured_time, '{label_from}') = 0;"""
            else:
                return JSONResponse(
                    content=f"Illegal label name: {label}",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            with mysql_db.transaction():
                mysql_db.update(sql=update_sql, params=None)
            logger.info(f'<updateLabel> Label {label} updated from {label_from} to {label_to}.')
    except Exception as e:
        return JSONResponse(content=e, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        mysql_db._close()
        logger.info('<updateLabel> Image Labels updated successfully.')

    return "Succeed"


@router.post("/v1/aiphotos/updateTags")
async def handel_ai_photos_update_tags(request: Request):
    # check request user
    user_id = request.client.host
    logger.info(f'<updateTags> user ip is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    image_list = params['image_list']

    try:
        for image in image_list:
            update_image_tags(image)

    except Exception as e:
        logger.error("<updateTags> "+str(e))
        return Response(content=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        logger.info('<updateTags> Image tags updated successfully.')

    return "Succeed"


@router.post("/v1/aiphotos/updateCaption")
async def handel_ai_photos_update_caption(request: Request):
    # check request user
    user_id = request.client.host
    logger.info(f'<updateCaption> user ip is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    image_list = params['image_list']

    for image in image_list:
        try:
            update_image_attr(image, 'caption')
        except Exception as e:
            logger.error("<updateCaption> "+str(e))
            return Response(content=str(e), status_code=status.HTTP_400_BAD_REQUEST)

    return "Succeed"


@router.post("/v1/aiphotos/chatWithImage")
async def handle_ai_photos_chat_to_image(request: Request):
    user_id = request.client.host
    logger.info(f'<chatWithImage> user ip is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    query = params['query']
    logger.info(f'<chatWithImage> generating chat to image for user {user_id} with query: {query}')

    chatbot = router.get_chatbot()
    cur_time = get_current_time()
    pt = PromptTemplate("ner")
    pt.append_message(pt.conv.roles[0], cur_time)
    pt.append_message(pt.conv.roles[1], query)
    prompt = pt.get_prompt()
    response = chatbot.predict(query=prompt)
    response = response.split("[/INST]")[-1]

    try:
        ner_obj = plugins['ner']["instance"]
        result = ner_obj.ner_inference(response)
    except Exception as e:
        logger.error("<chatWithImage> "+str(e))
        raise Exception(e)
    logger.info(f'<chatWithImage> NER result: {result}')

    try:
        result_image_list = get_image_list_by_ner_query(result, user_id, query)
    except Exception as e:
        logger.error("<chatWithImage> "+str(e))
        raise Exception(e)
    return "No query result" if result_image_list==[] else result_image_list


@router.post("/v1/aiphotos/image2Image")
async def handle_image_to_image(request: Request):
    user_id = request.client.host
    logger.info(f'<image2Image> user ip is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    query = params['query']
    image_list = params['ImageList']
    logger.info(f'<image2Image> user: {user_id}, image to image query command: {query}')

    generated_images = []
    steps=25
    strength=0.75
    seed=42
    guidance_scale=7.5
    for img_info in image_list:
        img_id = img_info["imgId"]
        img_path = img_info["imgSrc"]
        userid, img_name = img_path.split('/')[-2], img_path.split('/')[-1]
        IMAGE_ROOT_PATH = get_image_root_path()
        image_path = IMAGE_ROOT_PATH+'/'+userid+'/'+img_name
        logger.info(f'<image2Image> current image id: {img_id}, image path: {image_path}')

        img_b64 = image_to_byte64(image_path)
        data = {"source_img": img_b64.decode(), "prompt": query, "steps": steps,
                "guidance_scale": guidance_scale, "seed": seed, "strength": strength,
                "token": "intel_sd_bf16_112233"}
        start_time = time.time()
        img_str = stable_defusion_func(data)
        logger.info(f"<image2Image> elapsed time: {time.time() - start_time} seconds")
        generated_images.append({"imgId": img_id, "imgSrc": "data:image/jpeg;base64,"+img_str})

    return generated_images


# ================== For streaming ==================
@router.post("/v1/aiphotos/talkingbot/asr")
async def handle_talkingbot_asr(file: UploadFile = File(...)):
    keyword_list = {
        "intel": "Intel",
        " i ": " I ",
        "shanghai": "Shanghai",
        "china": "China",
        "beijing": "Beijing"
    }
    # get asr result from neural chat
    asr_result = talkingbot_asr(file=file)
    res = await asyncio.gather(asr_result)
    res = res[0]['asr_result']
    # substitute keywords manually
    result_list = []
    words = res.split(" ")
    for word in words:
        if word in keyword_list.keys():
            word = keyword_list[word]
        result_list.append(word)
    asr_result = ' '.join(result_list)
    final_result = asr_result[0].upper() + asr_result[1:] + '.'

    return {"asr_result": final_result}


@router.post("/v1/aiphotos/talkingbot/create_embed")
async def handle_talkingbot_create_embedding(file: UploadFile = File(...)):
    result = talkingbot_embd(file=file)
    res = await asyncio.gather(result)
    if isinstance(res, List):
        final_result = res[0]['spk_id']
    elif isinstance(res, Dict):
        final_result = res['spk_id']
    else:
        return "Error occurred."
    return {"voice_id": final_result}


@router.post("/v1/aiphotos/talkingbot/llm_tts")
async def handle_talkingbot_llm_tts(request: Request):
    data = await request.json()
    text = data["text"]
    voice = data["voice"]
    knowledge_id = data["knowledge_id"]
    audio_output_path = data["audio_output_path"] if "audio_output_path" in data else "output_audio"

    logger.info(f'Received prompt: {text}, and use voice: {voice} knowledge_id: {knowledge_id}')

    return await router.handle_voice_chat_request(text, audio_output_path)
