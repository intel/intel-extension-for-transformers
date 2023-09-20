"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import time
from typing import List, Union
import threading
import re

from fastapi import FastAPI, Request, File, UploadFile, Form, status
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
import numpy as np
import requests
import uvicorn
import exifread

from constants import CONTROLLER_HEART_BEAT_EXPIRATION
from utils import build_logger, server_error_msg

from sse_starlette.sse import EventSourceResponse
from starlette.responses import RedirectResponse

# =========================== ADD ==============================
import os
import sys
import base64
import random
import datetime
import ipaddress
import pandas as pd
from io import BytesIO
from PIL import Image
from ner import generate_query_from_prompt
from ner_new import inference as inference_ner
from deepface import DeepFace
from typing import List, Dict
from fastapi import BackgroundTasks
from utils_image import find_GPS_image, get_address_from_gps, generate_caption, image_to_byte64, byte64_to_image, generate_random_name, transfer_xywh
from pydub import AudioSegment

logger = build_logger("controller", "controller.log")


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str


def heart_beat_controller(controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        self.heart_beat_thread.start()

        logger.info("Init controller")

    def register_worker(self, worker_name: str, check_heart_beat: bool,
                        worker_status: dict):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"], worker_status["speed"], worker_status["queue_length"],
            check_heart_beat, time.time())

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            while True:
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            self.worker_info[w_name].queue_length += 1
            logger.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
            return w_name
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def worker_api_generate_stream(self, params):
        print("params=========", params)
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"
        try:
            response = requests.post(worker_addr + "/worker_generate_stream",
                json=params, stream=True, timeout=1000)
            result = ""
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    print("chunk=======", chunk)
                    # yield chunk + b"\0"
                    a = chunk.decode("utf-8")
                    a = re.sub(r'\\u2019', "'", a)
                    a = re.sub(r'\\\\ufffd', '', a)
                    result += a
                    yield f"data: {a}\n\n"
                    # yield f"data: \n\n"
            import sys
            sys.path.append("..")
            from llmcache.cache import put
            put(params["prompt"], result)
            yield f"data: [DONE]\n\n"
        except requests.exceptions.RequestException as e:
            logger.info(f"worker timeout: {worker_addr}")
            ret = {
                "text": server_error_msg,
                "error_code": 3,
            }
            yield json.dumps(ret).encode() + b"\0"


    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5185",
]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_name"], data["check_heart_beat"],
        data.get("worker_status", None))


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()

@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}

@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}

@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    logger.info('Received request: %s', await request.json())
    params = await request.json()
    if "msgData" in params:
        params = params["msgData"]
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.post("/v1/models")
async def list_models():
    models = controller.list_models()
    return {"models": models}

@app.post("/v1/chat/completions")
async def worker_api_generate_stream(request: Request):
    logger.info('Received request: %s', await request.json())
    params = await request.json()
    if "msgData" in params:
        params = params["msgData"]
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.post("/v1/chat/llmcache")
async def get_cache(request: Request):
    logger.info('Received request: %s', await request.json())
    params = await request.json()
    if "msgData" in params:
        params = params["msgData"]
    prompt = params["prompt"]
    import sys
    sys.path.append("..")
    from llmcache.cache import get
    result = get(prompt)
    print(result)
    if(result == None):
        print("cache miss >>>>>>>>>>>>>>>")
        response = RedirectResponse(url="/v1/chat/completions")
        return response
    else:
        print("cache hit >>>>>>>>>>>>>>>>")
        def stream_results():
            yield "data: Response from Cache: {}\n\n".format(result['choices'][0]['text'])
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream")

STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 15000  # milisecond

@app.get('/stream')
async def message_stream(request: Request):
    def new_messages():
        # Add logic here to check for new messages
        yield 'Hello World'
    async def event_generator():
        while True:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break

            # Checks for new messages and return them to client if any
            if new_messages():
                print("11")
                yield {
                        "event": "new_message",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": "message_content"
                }

            await asyncio.sleep(STREAM_DELAY)

    return EventSourceResponse(event_generator())


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return controller.worker_api_get_status()

import random
def get_event_data():
    event_data = generate_random_data()
    return event_data

def generate_random_data():
    random_number = random.randint(1, 100)
    return random_number

@app.post("/api/chat")
async def handle_chat(request: Request):
    logger.info('Received request: %s', await request.json())
    """
    request.headers["Content-Type"] = "text/event-stream"
    request.headers["Cache-Control"] = "no-cache"
    request.headers["Connection"] = "keep-alive"
    """

    async def event_stream():
         while True:
            data = get_event_data()
            if data:
                yield f"data: {data}\n\n"
            await asyncio.sleep(1)

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    logger.info('Sending response: %s', response)
    response.headers["Content-Type"] = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"

    return response

@app.post("/v1/chat/talkingbot")
async def handle_talkingbot(file: UploadFile = File(...), voice: str = Form(...)):
    start = time.time()
    file_name = file.filename
    logger.info(f'Received file: {file_name}, and use voice: {voice}')
    with open("tmp_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file("tmp_audio_bytes")
    # bytes to mp3
    audio.export(f"{file_name}", format="mp3")
    worker_name = controller.get_worker_address("mpt-7b-chat")

    try:
        r = requests.post(worker_name + "/talkingbot", json={"file_name": file_name, "voice": voice}, timeout=20)
    except requests.exceptions.RequestException as e:
        logger.error(f"Talkingbot fails: {worker_name}, {e}")
        return None
    logger.info(f"E2E time: {time.time() - start}")
    return FileResponse(r.content, media_type="video/mp4")

# =================== ADD =======================
UPLOAD_STATUS = {
    "user_id":{
        "total_image_num": 10,
        "current_image_num": 5,
    }
}

################
#  deprecated  #
################
@app.post("/v1/aiphotos/uploadImageList")
async def handle_ai_photos_upload(request: Request):
#   clear UPLOAD_STATUS

    params = await request.json()
    user_id = params['user_id']
    logger.info(f'geting image list by user {user_id}')
    user_ip = request.client.host
    logger.info(f'user id is: {user_ip}')
    image_list = params['image_list']

    res = check_user_ip(user_ip)
    logger.info(res)

    if len(image_list) == 0:
        return "No image list is sent."
    
    # initiate image upload status
    UPLOAD_STATUS[user_id] = {}
    UPLOAD_STATUS[user_id]['total_image_num'] = len(image_list)
    UPLOAD_STATUS[user_id]['current_image_num'] = 0

    image_path = '/home/ubuntu/images/user'+str(user_id)
    os.makedirs(image_path, exist_ok=True)
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    for image in image_list:
        img_b64 = image['imgSrc'].split(',')[1]

        # convert byte64 to Image object
        img_obj = byte64_to_image(str.encode(img_b64))
        # convert and save image to local path
        img_rgb = img_obj.convert('RGB')
        tmp_name = generate_random_name()
        img_name = tmp_name+'.jpg'
        img_path = image_path+'/'+ img_name
        # save image to local path in form of jpg
        img_rgb.save(img_path)
        logger.info(f'Image saved into path {img_path}')

        # generate gps info
        result_gps = find_GPS_image(img_path)
        captured_time = result_gps['date_information']
        gps_info = result_gps['GPS_information']
        latitude, longitude, altitude = None, None, None
        if 'GPSLatitude' in gps_info:
            latitude = gps_info['GPSLatitude']
        if 'GPSLongitude' in gps_info:
            longitude = gps_info['GPSLongitude']
        if 'GPSAltitude' in gps_info:
            altitude = gps_info['GPSAltitude']
        logger.info(f'Image is captured at: {captured_time}, latitude: {latitude}, longitude: {longitude}, altitude: {altitude}')

        # generate address info
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("Please")
        address = get_address_from_gps(latitude, longitude, api_key)
        if address:
            logger.info(f'Image address: {address}')
        else:
            address = None
            logger.info(f'Can not get address from image.')
        
        # generate caption info
        result_caption = generate_caption(img_path)
        if result_caption:
            logger.info(f'Image caption: {result_caption}')
        else:
            result_caption = None
            logger.info(f'Can not generate caption for image.')

        # save image infomations into database
        mysql_db = MysqlDb()
        mysql_db.set_db("ai_photos")
        empty_tags = '{}'
        if not captured_time:
            sql = f"INSERT INTO image_info VALUES(null, '{user_id}', '{img_name}', null, '{result_caption}', \
            '{latitude}', '{longitude}', '{altitude}', '{address}', true, '{empty_tags}', 'ready', 'active');"
        else:
            sql = f"INSERT INTO image_info VALUES(null, '{user_id}', '{img_name}', '{captured_time}', '{result_caption}', \
            '{latitude}', '{longitude}', '{altitude}', '{address}', true, '{empty_tags}', 'ready', 'active');"
        try:
            mysql_db.insert(sql, params=None)
        except:
            raise Exception("Exception occurred when inserting image info into MySQL.")
        else:
            logger.info(f'image_info inserted: {sql}')

        # get image id
        fetch_sql = f"SELECT * FROM image_info WHERE image_path = '{img_name}';"
        try:
            result = mysql_db.fetch_one(sql=fetch_sql)
        except:
            raise Exception("Exception occurred when selecting from image info of MySQL.")
        else:
            img_id = result[0]
            logger.info(f'Image id is {img_id}')

        # use deepface to recognize faces from image db
        dfs = DeepFace.find(img_path=img_path, db_path=image_path, enforce_detection=False, model_name='Facenet')
        logger.info(f'find {len(dfs)} matched images in database.')
        # no face recognized, only the image itself
        if len(dfs) == 1:
            logger.info(f'No same face recognized in image database.')
            # check faces in this image
            try:
                face_objs = DeepFace.represent(img_path=img_path, model_name='Facenet')
                # insert faces into face_info and image_face
                for face_obj in face_objs:
                    face_cnt = mysql_db.fetch_all(sql="SELECT COUNT(*) FROM face_info;")[0][0]
                    tag = 'person'+str(face_cnt+1)
                    face_sql = f"INSERT INTO face_info VALUES(null, '{tag}');"
                    mysql_db.insert(sql=face_sql, params=None)
                    xywh = transfer_xywh(face_obj['facial_area'])
                    logger.info(f"face {tag} inserted into db.")
                    face_id = mysql_db.fetch_one(f"SELECT * FROM face_info WHERE face_tag = '{tag}';")[0]
                    img_face_sql = f"INSERT INTO image_face VALUES(null, {img_id}, '{img_name}', {face_id}, '{xywh}', '{user_id}', '{tag}');"
                    mysql_db.insert(sql=img_face_sql, params=None)
                    logger.info(f"img_face {img_face_sql} inserted into db.")
            except Exception as e:
                logger.error(e)
            else:
                logger.info(f'{len(face_objs)} faces are inserted into database.')
            continue
        
        # process same faces and save into face_info database
        new_face_cnt = 0
        for df in dfs:
            logger.info(f'current df: {df}')
            if df.shape[0] == 0:
                logger.info(f'no match image for current face, save into db.')
                new_face_cnt += 1
                continue
            # get the img path of ref_img
            ref_img_path = df.iloc[0]['identity']
            logger.info(f'ref_img_path is {ref_img_path}')
            # find faces in img2: one or many
            find_face_sql = f"SELECT * FROM image_face WHERE image_path = '{ref_img_path}';"
            try:
                img_face_list = mysql_db.fetch_all(sql=find_face_sql)
            except:
                raise Exception("Exception ocurred while selecting info from image_face.")
            logger.info(f"reference image and faces: {img_face_list}")

            # compare images and save into db
            obj = DeepFace.verify(img1_path=img_path, img2_path=ref_img_path, model_name="Facenet")
            ref_xywh = transfer_xywh(obj['facial_areas']['img2'])
            for img_face in img_face_list:
                if img_face[4] == ref_xywh:
                    face_id = img_face[3]
                    tag = img_face[6]
            xywh = obj['facial_areas']['img1']
            xywh = transfer_xywh(xywh)
            insert_img_face_sql = f"INSERT INTO image_face VALUES(null, {img_id}, '{img_name}', {face_id}, '{xywh}', '{user_id}', '{tag}');"
            try:
                mysql_db.insert(sql=insert_img_face_sql, params=None)
            except:
                raise Exception(f"Exception ocurred while inserting info into image_face.")
            else:
                logger.info(f'image_face data inserted: {insert_img_face_sql}')

        # process new faces of this image
        if new_face_cnt == 0:
            logger.info(f'all faces of {img_path} are processed, go to next image.')
            continue

        # new faces, insert into database
        logger.info(f'add new face of image {img_path}')
        try:
            face_objs = DeepFace.represent(img_path=img_path, model_name='Facenet')
            # insert faces into face_info and image_face
            for face_obj in face_objs:
                # skip knwon faces
                cur_img_face_list = mysql_db.fetch_all(sql=f"SELECT * FROM image_face WHERE image_path = '{img_name}';")
                cur_img_face_df = pd.DataFrame(list(cur_img_face_list))
                xywh = transfer_xywh(face_obj['facial_area'])
                if (cur_img_face_df.iloc[:,4].isin([xywh]).any()):
                    continue
                # add new faces
                face_cnt = mysql_db.fetch_all(sql="SELECT COUNT(*) FROM face_info;")[0][0]
                tag = 'person'+str(face_cnt+1)
                face_sql = f"INSERT INTO face_info VALUES(null, '{tag}');"
                mysql_db.insert(sql=face_sql, params=None)
                logger.info(f"face {tag} inserted into db.")
                face_id = mysql_db.fetch_one(f"SELECT * FROM face_info WHERE face_tag = '{tag}';")[0]
                img_face_sql = f"INSERT INTO image_face VALUES(null, {img_id}, '{img_name}', {face_id}, '{xywh}', '{user_id}', '{tag}');"
                mysql_db.insert(sql=img_face_sql, params=None)
                logger.info(f"img_face {img_face_sql} inserted into db.")
        except Exception as e:
            logger.error(e)
        
        UPLOAD_STATUS[user_id]['current_image_num'] += 1

    return "Succeed"


@app.post("/v1/aiphotos/uploadProgress")
async def handle_ai_photos_upload_progress(request: Request) -> JSONResponse:
    # check request user
    user_id = request.client.host
    logger.info(f'user ip is: {user_id}')
    check_user_ip(user_id)

    upload_status = UPLOAD_STATUS.get(user_id, None)
    if upload_status is None:
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=f'No suche user: {user_id}')

    upload_status['percentage'] = float(upload_status['current_image_num']/upload_status['total_image_num'])
    return upload_status


def get_image_list_by_attr_list(attr_name: str, attr_list: list, user_id: str) -> dict:
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    logger.info(f'Processing attribute: {attr_name}')
    response = {}
    for attr_item in attr_list:
        attr_item = list(attr_item)[0]
        logger.info(f'current attribute is: {attr_item}')
        if attr_name == 'user_id':
            sql = f'SELECT * FROM image_info WHERE {attr_name} = "{attr_item}" and user_id = "{user_id}";'
        elif 'time' in attr_name:
            if attr_item == None:
                sql = f'SELECT * FROM image_info WHERE ISNULL({attr_name}) is TRUE and user_id = "{user_id}"'
            else: 
                sql = f'SELECT * FROM image_info WHERE DATEDIFF({attr_name}, "{attr_item}") = 0 and user_id = "{user_id}";'
        elif 'id' in attr_name:
            sql = f'SELECT * FROM image_info WHERE {attr_name} = {attr_item} and user_id = "{user_id}";'
        else:
            sql = f'SELECT * FROM image_info WHERE {attr_name} = "{attr_item}" and user_id = "{user_id}";'
        result_list = list(mysql_db.fetch_all(sql=sql))
        imageList = []
        for res in result_list:
            logger.info(f'current image info: {res}')
            image_item = format_image_info(res)
            logger.info(f'formated image info: {image_item}')
            imageList.append(image_item)
        response[attr_item] = imageList
    return response


@app.post("/v1/aiphotos/getImageList")
async def handle_ai_photos_get_img_list(request: Request):
    # setup mysql_db
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    user_id = request.client.host
    logger.info(f'user ip is: {user_id}')
    check_user_ip(user_id)

    # result categorized by face
    group_by_face_sql = f'SELECT group_concat(image_id), face_id FROM image_face WHERE user_id = "{user_id}" GROUP BY face_id;'
    query_list = list(mysql_db.fetch_all(sql=group_by_face_sql))
    logger.info(f'get query list: {query_list}')
    response_person = {}
    for item in query_list:
        face_id, face_tag, img_id_list = item[0], item[1].split(',')[0], item[2].split(',')
        logger.info(f'image id list: {img_id_list}')
        imageList = []
        for cur_img_id in img_id_list:
            logger.info(f'current image id: {cur_img_id}')
            cur_img_info = list(mysql_db.fetch_one(sql=f'SELECT * FROM image_info WHERE image_id = {cur_img_id};'))
            logger.info(f'current image info: {cur_img_info}')
            image_item = format_image_info(cur_img_info)
            logger.info(f'formated image info: {image_item}')
            imageList.append(image_item)
        response_person[face_tag] = imageList

    # result categorized by address
    select_by_address_sql = f'SELECT address FROM image_info WHERE user_id = "{user_id}" GROUP BY address;'
    address_list = list(mysql_db.fetch_all(sql=select_by_address_sql))
    response_address = get_image_list_by_attr_list(attr_name='address', attr_list=address_list)

    # result categorized by time
    select_by_time_sql = f'SELECT DATE(captured_time) AS date FROM image_info WHERE user_id = "{user_id}" GROUP BY date ORDER BY date;'
    # tmp_sql = "select DATE_FORMAT(createtime,'%Y-%m'),count(*) from test where user =8 group by DATE_FORMAT(createtime,'%Y-%m');"
    time_list = list(mysql_db.fetch_all(sql=select_by_time_sql))
    logger.info(f'time_list: {time_list}')
    response_time = get_image_list_by_attr_list(attr_name='captured_time', attr_list=time_list, user_id=user_id)
    logger.info(f'time response: {response_time}')

    result = {}
    if response_person:
        result['person'] = response_person
    if response_address:
        result['address'] = response_address
    if response_time:
        result['time'] = response_time

    return result


@app.post("/v1/aiphotos/updateChecked")
async def handel_ai_photos_update_checked(request: Request):
    # check request user
    user_id = request.client.host
    logger.info(f'user ip is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    image_list = params['image_list']

    for image in image_list:
        update_image_attr(image, "checked")

    return "Succeed"


################
#    in use    #
################

IMAGE_ROOT_PATH = "/home/nfs_images"


def check_user_ip(user_ip: str) -> bool:
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    user_list = mysql_db.fetch_one(sql=f'select * from user_info where user_id = "{user_ip}";')
    logger.info(f'[Check IP] user list: {user_list}')
    if user_list == None:
        logger.info(f'[Check IP] no current user, add into db.')
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mysql_db.insert(sql=f"insert into user_info values('{user_ip}', '{cur_time}', null, 1);", params=None)
    return True


def check_image_status(image_id: str):
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    image = mysql_db.fetch_one(sql=f'select * from image_info where image_id="{image_id}" and exist_status="active"',params=None)
    if image==None:
        raise ValueError(f'No image {image_id} saved in MySQL DB.')
    return image


def update_image_tags(image):
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    image_id = image['image_id']
    tags = image['tags']
    image_info = check_image_status(image_id)
    update_sql = 'UPDATE image_info SET '
    update_sql_list = []
    old_tags = eval(image_info[10])
    logger.info(f'[Update Tags] old_tags: {old_tags}')
    tag_name_list = []
    for key, value in tags.items():
        if key == 'time' and value != image_info[3]:
            update_sql_list.append(f' captured_time="{value}" ')
            tag_name_list.append('time')
        elif key == 'latitude' and value != image_info[5]:
            update_sql_list.append(f' latitude="{value}" ')
            tag_name_list.append('latitude')
        elif key == 'longitude' and value != image_info[6]:
            update_sql_list.append(f' longitude="{value}" ')
            tag_name_list.append('longitude')
        elif key == 'altitude' and value != image_info[7]:
            update_sql_list.append(f' altitude="{value}" ')
            tag_name_list.append('altitude')
        elif key == 'location' and value != image_info[8]:
            update_sql_list.append(f' address="{value}" ')
            tag_name_list.append('location')
            
    for tag_name in tag_name_list:
        tags.pop(tag_name)
    old_tags.update(tags)
    new_tags = str(old_tags)
    update_sql_list.append(f' other_tags="{new_tags}" ')
    update_sql_tmp = ','.join(update_sql_list)
    final_sql = update_sql+update_sql_tmp+f' where  image_id={image_id}'
    logger.info(f'[Update Tags] update sql: {final_sql}')
    mysql_db.update(sql=final_sql, params=None)


def update_image_attr(image, attr): 
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    image_id = image['image_id']
    check_image_status(image_id)

    new_attr = image[attr]
    try:
        if attr=='checked':
            new_checked = 1 if new_attr else 0
            mysql_db.update(sql=f"UPDATE image_info SET {attr}={new_checked} WHERE image_id={image_id}", params=None)
        else:
            mysql_db.update(sql=f'UPDATE image_info SET {attr}="{new_attr}" WHERE image_id={image_id}', params=None)
    except Exception as e:
        logger.error(e)
    else:
        logger.info(f'Image {attr} updated successfully.')


def format_image_path(user_id: str, image_name: str) -> str:
    server_ip = os.getenv("IMAGE_SERVER_IP")
    if not server_ip:
        raise Exception("Please configure SERVER IP to environment variables.")
    image_path = "http://"+server_ip+"/ai_photos/user"+user_id+'/'+image_name
    return image_path


def format_image_info(image_info: tuple) -> dict:
    image_item = {}
    image_item['image_id'] = image_info[0]
    image_item['user_id'] = image_info[1]
    image_name = image_info[2].split('/')[-1]
    image_item['image_path'] = format_image_path(image_info[1], image_name)
    image_item['caption'] = image_info[4]
    image_item['checked'] = True if image_info[9] else False
    tag_list = {}
    if image_info[3]:
        tag_list['time'] = datetime.datetime.date(image_info[3])
    if image_info[8] != 'None':
        tag_list['location'] = image_info[8]
    other_tags = eval(image_info[10])
    tag_list.update(other_tags)
    image_item['tag_list'] = tag_list
    return image_item


def delete_single_image(user_id, image_id):
    logger.info(f'[Delete] Deleting image {image_id}')
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    image_path = mysql_db.fetch_one(sql=f'SELECT image_path FROM image_info WHERE image_id={image_id}', params=None)
    if image_path==None:
        info = f'[Delete] Image {image_id} does not exist in MySQL.'
        logger.error(info)
        raise Exception(info)
    image_path = image_path[0]
    
    import shutil
    image_path_dst = IMAGE_ROOT_PATH+'/deleted/user'+str(user_id)
    os.makedirs(image_path_dst, exist_ok=True)
    logger.info(f'[Delete] destination folder created: {image_path_dst}')
    shutil.move(src=image_path, dst=image_path_dst)
    logger.info(f'[Delete] Image {image_path} moved to {image_path_dst}')
    image_name = image_path.split('/')[-1]
    new_image_path = image_path_dst+'/'+image_name
    try:
        mysql_db.update(sql=f"UPDATE image_info SET exist_status='deleted', image_path='{new_image_path}' WHERE image_id={image_id} ;", params=None)
    except Exception as e:
        logger.error(e)
        raise Exception(e)
    logger.info(f'[Delete] Image {image_id} deleted successfully.')


def process_images_in_background( user_id: str, image_obj_list: List[Dict]):
    try:
        logger.info(f'[backgroud] ======= processing image list for user {user_id} in background =======')
        sys.path.append("..")
        from database.mysqldb import MysqlDb
        mysql_db = MysqlDb()
        mysql_db.set_db("ai_photos")

        for i in range(len(image_obj_list)):
            # save image into local path
            image_id = image_obj_list[i]['img_id']
            image_path = image_obj_list[i]['img_path']
            image_obj = image_obj_list[i]['img_obj']
            image_exif = image_obj_list[i]['exif']
            image_obj.save(image_path, exif=image_exif)
            logger.info(f'[backgroud] Image saved into local path {image_path}')
            # process image and generate infos
            try:
                process_single_image(image_id, image_path, user_id)
            except Exception as e:
                logger.error("[backgroud] "+str(e))
                logger.error(f'[backgroud] error occurred, delete image.')
                delete_single_image(user_id, image_id)

    except Exception as e:
        logger.error(e)
        raise ValueError(str(e))
    else:
        logger.info('[backgroud] Background images process finished.')


def process_single_image(img_id, img_path, user_id):
    logger.info(f'[background - single] ----- processing image {img_path} in background -----')
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    # generate gps info
    result_gps = find_GPS_image(img_path)
    captured_time = result_gps['date_information']
    gps_info = result_gps['GPS_information']
    latitude, longitude, altitude = None, None, None
    if 'GPSLatitude' in gps_info:
        latitude = gps_info['GPSLatitude']
    if 'GPSLongitude' in gps_info:
        longitude = gps_info['GPSLongitude']
    if 'GPSAltitude' in gps_info:
        altitude = gps_info['GPSAltitude']
    logger.info(f'[background - single] Image is captured at: {captured_time}, latitude: {latitude}, longitude: {longitude}, altitude: {altitude}')
    if latitude:
        update_image_attr(image={"image_id": img_id, "latitude": latitude}, attr='latitude')
    if longitude:
        update_image_attr(image={"image_id": img_id, "longitude": longitude}, attr='longitude')
    if altitude:
        update_image_attr(image={"image_id": img_id, "altitude": altitude}, attr='altitude')
    if captured_time:
        update_image_attr(image={"image_id": img_id, "captured_time": captured_time}, attr='captured_time')

    # generate address info
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("Please configure environment variable of GOOGLE_API_KEY.")
    address = get_address_from_gps(latitude, longitude, api_key)
    if address:
        logger.info(f'[background - single] Image address: {address}')
        update_image_attr(image={"image_id": img_id, "address": address}, attr='address')
    else:
        address = None
        logger.info(f'[background - single] Can not get address from image.')

    # generate caption info
    logger.info(f'[background - single] Generating caption of image {img_path}')
    try:
        result_caption = generate_caption(img_path)
    except Exception as e:
        logger.error("[background - single] "+str(e))
    if result_caption:
        logger.info(f'[background - single] Image caption: {result_caption}')
        update_image_attr(image={"image_id": img_id, "caption": result_caption}, attr='caption')
    else:
        logger.info(f'[background - single] Can not generate caption for image.')

    # process faces for image
    db_path = IMAGE_ROOT_PATH+"/user"+user_id
    process_face_for_single_image(image_id=img_id, image_path=img_path, db_path=db_path, user_id=user_id)
    logger.info(f'[background - single] Face process done for image {img_id}')

    # update image status
    try:
        mysql_db.update(sql=f"UPDATE image_info SET process_status='ready' WHERE image_id={img_id}", params=None)
    except Exception as e:
        logger.error("[background - single] "+str(e))
    logger.info(f"[background - single] ----- finish image {img_path} processing -----")


def process_face_for_single_image(image_id, image_path, db_path, user_id):
    logger.info(f'[background - face] ### processing face for {image_path} in background ###')
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    # 1. check whether image contains faces
    try:
        face_objs = DeepFace.represent(img_path=image_path, model_name='Facenet')
    except:
        # no face in this image, finish process
        logger.info(f"[background - face] Image {image_id} does not contains faces")
        logger.info(f"[background - face] Image {image_id} face process finished.")
        return None
    face_cnt = len(face_objs)
    logger.info(f'[background - face] Found {face_cnt} faces in image {image_id}')
    face_xywh_list = []
    for face_obj in face_objs:
        xywh = face_obj['facial_area']
        transferred_xywh = transfer_xywh(xywh)
        face_xywh_list.append(transferred_xywh)
    logger.info(f'[background - face] face xywh list of image {image_id} is: {face_xywh_list}')

    # 2. check same faces in db
    import os
    pkl_path = db_path+'/representations_facenet.pkl'
    if os.path.exists(pkl_path):
        logger.info(f'[background - face] pkl file already exists, delete it.')
        os.remove(pkl_path)
    dfs = DeepFace.find(img_path=image_path, db_path=db_path, model_name='Facenet', enforce_detection=False)
    logger.info(f'[background - face] Finding match faces in image database.')
    assert face_cnt == len(dfs)
    logger.info(f'[background - face] dfs: {dfs}')
    for df in dfs:
        # no face matched for current face of image, add new faces later
        if len(df) <= 1:
            logger.info(f'[background - face] length of {df} less than 1, continue')
            continue
        # found ref image
        ref_image_path = df.iloc[0]['identity']
        ref_image_list = df['identity']
        for ref_image_name in ref_image_list:
            logger.info(f'[background - face] current ref_image_name: {ref_image_name}')
            if ref_image_name!=image_path:
                ref_image_path = ref_image_name
                break
        
        # find faces in img2: one or many
        find_face_sql = f"SELECT face_id, face_tag, xywh FROM image_face WHERE image_path='{ref_image_path}' AND user_id='{user_id}';"
        try:
            img_face_list = mysql_db.fetch_all(sql=find_face_sql)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception ocurred while selecting info from image_face: {e}")
        logger.info(f"[background - face] reference image and faces: {img_face_list}")
        # verify face xywh of ref image
        obj = DeepFace.verify(img1_path=image_path, img2_path=ref_image_path, model_name="Facenet")
        ref_xywh = transfer_xywh(obj['facial_areas']['img2'])
        image_xywh = transfer_xywh(obj['facial_areas']['img1'])
        face_id = -1
        face_tag = None
        # find corresponding face_id and face_tag
        for img_face in img_face_list:
            if img_face[2] == ref_xywh:
                face_id = img_face[0]
                face_tag = img_face[1]
        if face_id == -1 and face_tag == None:
            raise Exception(f'Error occurred when verifing faces for reference image: Inconsistent face infomation.')
        # insert into image_face
        insert_img_face_sql = f"INSERT INTO image_face VALUES(null, {image_id}, '{image_path}', {face_id}, '{image_xywh}', '{user_id}', '{face_tag}');"
        try:
            mysql_db.insert(sql=insert_img_face_sql, params=None)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception ocurred while inserting info into image_face: {e}")
        # current face matched and saved into db, delete from face_xywh_list
        logger.info(f'[background - face] image_face data inserted: {insert_img_face_sql}')
        if image_xywh in face_xywh_list:
            face_xywh_list.remove(image_xywh)
        logger.info(f'[background - face] current face_xywh_list: {face_xywh_list}')
    
    # all faces matched in db, no faces left
    if len(face_xywh_list) == 0:
        logger.info(f"[background - face] Image {image_id} face process finished.")
        return None
    
    # 3. add new faces for current image (no reference in db)
    logger.info(f'[background - face] Adding new faces for image {image_id}')
    for cur_xywh in face_xywh_list:
        face_cnt = mysql_db.fetch_all(sql="SELECT COUNT(*) FROM face_info;")[0][0]
        tag = 'person'+str(face_cnt+1)
        face_sql = f"INSERT INTO face_info VALUES(null, '{tag}');"
        try:
            mysql_db.insert(sql=face_sql, params=None)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception ocurred while inserting new face into face_info: {e}")
        logger.info(f"[background - face] face {tag} inserted into db.")
        face_id = mysql_db.fetch_one(f"SELECT * FROM face_info WHERE face_tag='{tag}';")[0]
        logger.info(f"[background - face] new face id is: {face_id}")
        img_face_sql = f"INSERT INTO image_face VALUES(null, {image_id}, '{image_path}', {face_id}, '{cur_xywh}', '{user_id}', '{tag}');"
        try:
            mysql_db.insert(sql=img_face_sql, params=None)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception ocurred while inserting new face into image_face: {e}")
        logger.info(f"[background - face] img_face {img_face_sql} inserted into db.")
    logger.info(f"[background - face] Image {image_id} face process finished.")


def get_type_obj_from_attr(attr, user_id):
    logger.info(f'Geting image type of {attr}')
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    if attr == 'time':
        select_sql = f'SELECT DATE(captured_time) AS date FROM image_info WHERE user_id = "{user_id}" AND exist_status="active" GROUP BY date ORDER BY date;'
    elif attr == 'address':
        select_sql = f'SELECT address FROM image_info WHERE user_id="{user_id}" AND exist_status="active" GROUP BY address;'
    else:
        return {}

    select_list = list(mysql_db.fetch_all(sql=select_sql))
    select_result = {}
    for item in select_list:
        item = item[0]
        logger.info(f"current item of {attr} is: {item}")
        if attr == 'time':
            if item == None:
                continue
                # example_image_path = mysql_db.fetch_one(sql=f'SELECT image_path FROM image_info WHERE ISNULL(captured_time) is TRUE and user_id="{user_id}" and exist_status="active" LIMIT 1;', params=None)[0]
            example_image_path = mysql_db.fetch_one(sql=f'SELECT image_path FROM image_info WHERE DATEDIFF(captured_time, "{item}") = 0 and user_id="{user_id}" and exist_status="active" LIMIT 1;', params=None)[0]
        elif attr == 'address':
            if item == None or item == 'None' or item == 'null':
                continue
            example_image_path = mysql_db.fetch_one(sql=f'SELECT image_path FROM image_info WHERE address="{item}" and user_id="{user_id}" and exist_status="active" LIMIT 1;', params=None)[0]
        
        # if item == None or item == 'None' or item == 'null':
        #     item = 'default'
        image_name = example_image_path.split('/')[-1]
        image_path = format_image_path(user_id, image_name)
        select_result[item] = image_path

    logger.info(f'type list: {select_result}')
    return select_result


def get_process_status(user_id):
    logger.info(f'Geting process status of user {user_id}')
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    total_cnt = mysql_db.fetch_one(sql=f"SELECT COUNT(*) FROM image_info WHERE user_id='{user_id}' AND exist_status='active';")[0]
    processing_cnt = mysql_db.fetch_one(sql=f"SELECT COUNT(*) FROM image_info WHERE user_id='{user_id}' AND exist_status='active' AND process_status='processing';")[0]
    result = {}
    result['total_image'] = total_cnt
    result['processing_image'] = processing_cnt
    result['status'] = "done" if processing_cnt ==0 else 'processing'
    return result


def get_images_by_type(user_id, type, subtype) -> List:
    logger.info(f'Getting image by type {type} - {subtype}')
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    if type == 'address':
        if subtype == 'default':
            subtype = 'None'
        sql=f"SELECT image_id, image_path FROM image_info WHERE user_id='{user_id}' AND exist_status='active' AND address='{subtype}';"

    elif type == 'time':
        if subtype == 'None':
            sql = f'SELECT image_id, image_path FROM image_info WHERE captured_time is null AND user_id="{user_id}" AND exist_status="active";'
        else:
            sql = f'SELECT image_id, image_path FROM image_info WHERE DATE(captured_time)="{subtype}" AND user_id="{user_id}" AND exist_status="active";'

    elif type == 'person':
        sql = f"SELECT image_info.image_id, image_info.image_path FROM image_face INNER JOIN image_info ON image_info.image_id=image_face.image_id WHERE image_info.user_id='{user_id}' AND image_info.exist_status='active' AND image_face.face_tag='{subtype}'"

    logger.info(f'sql: {sql}')
    images = mysql_db.fetch_all(sql=sql, params=None)
    logger.info(f"image list: {images}")
    if len(images) == 0:
        logger.error(f'no label {subtype} in {type}')
        return []
    else:
        images = list(images)
        result = []
        for image in images:
            image_name = image[1].split('/')[-1]
            image_path = format_image_path(user_id, image_name)
            obj = {"image_id": image[0], "image_path": image_path}
            result.append(obj)
        return result


def get_face_list_by_user_id(user_id: str) -> List[Dict]:
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    logger.info(f'getting face list of user {user_id}')
    group_by_face_sql = f'SELECT group_concat(image_face.image_path), group_concat(image_face.face_tag) FROM image_face INNER JOIN image_info ON image_info.image_id=image_face.image_id WHERE image_info.user_id = "{user_id}" AND image_info.exist_status="active" GROUP BY face_id;'
    try:
        query_list = mysql_db.fetch_all(sql=group_by_face_sql)
    except Exception as e:
        logger.error(e)
        raise Exception(e)
    query_list = list(query_list)
    logger.info(f'query result list: {query_list}')
    response_person = {}
    for item in query_list:
        logger.info(f'current item: {item}')
        face_tag, img_path = item[1].split(',')[0], item[0].split(',')[0]
        image_name = img_path.split('/')[-1]
        response_person[face_tag] = format_image_path(user_id, image_name)
    logger.info(f'person list: {response_person}')
    return response_person


def get_image_list_by_ner_query(ner_result: Dict, user_id: str, query: str) -> List[Dict]:
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    logger.info(f'[NER query] start query from ner results')
    query_sql = "SELECT image_info.image_id, image_info.image_path FROM image_info "
    query_flag = False

    # get person name query
    face_list = mysql_db.fetch_all(sql=f"select image_face.face_tag from image_face inner join image_info on image_info.image_id=image_face.image_id where image_info.user_id='{user_id}' AND exist_status='active';", params=None)
    logger.info(f"[NER query] face list is: {face_list}")
    if ner_result['name'] or face_list:
        query_flag = True
        query_sql += "INNER JOIN image_face ON image_info.image_id=image_face.image_id WHERE "
        names = ner_result['name']
        sql_conditions = []
        for name in names:   
            sql_conditions.append(f' image_face.face_tag LIKE "%{name}%" ')
        for face_tag in face_list:
            face_tag = face_tag[0]
            if face_tag in query:
                logger.info(f'[NER query] other face detected in db: [{face_tag}]')
                sql_conditions.append(f' image_face.face_tag LIKE "%{face_tag}%" ')
        if sql_conditions != []:
            sql = 'OR'.join(sql_conditions)
            query_sql += '('+sql+')'
    else:
        logger.info(f'[NER query] no person name in ner query')

    # get location query
    if not ner_result.get('location', None):
        logger.info(f'[NER query] no location in query')
    else:
        if not query_flag:
            query_sql += " WHERE "
        query_flag = True
        locations = ner_result['location']
        sql_conditions = []
        for loc in locations:
            sql_conditions.append(f' image_info.address LIKE "%{loc}%" ')
        sql = 'OR'.join(sql_conditions)
        if query_sql[-1] == ')':
            query_sql += ' AND '
        query_sql += '('+sql+')'
        
    # get time query
    if ner_result['time'] == []:
        logger.info(f'[NER query] no time in query')
    else:
        if not query_flag:
            query_sql += " WHERE "
        query_flag = True
        time_points = ner_result['time']
        sql_conditions = []
        today = datetime.date.today()
        for loc in time_points:
            if today == loc:
                continue
            sql_conditions.append(f' image_info.captured_time LIKE "%{loc}%" ')
        sql = 'OR'.join(sql_conditions)
        if query_sql[-1] == ')':
            query_sql += ' AND '
        query_sql += '('+sql+')'

    # get time period query
    if ner_result['period'] == []:
        logger.info(f'[NER query] no time period in query')
    else:
        if not query_flag:
            query_sql += " WHERE "
        query_flag = True
        periods = ner_result['period']
        logger.info(f'[NER query] periods: {periods}')
        sql_conditions = []
        for period in periods:
            from_time = period['from']
            to_time = period['to']
            sql_conditions.append(f' image_info.captured_time BETWEEN "{from_time}" AND "{to_time}" ')
        sql = 'OR'.join(sql_conditions)
        if query_sql[-1] == ')':
            query_sql += ' AND '
        query_sql += '('+sql+')'
    
    if not query_flag:
        logger.info(f'[NER query] no compatible data for current query')
        return []
    query_sql += f' AND ( image_info.user_id="{user_id}" ) AND ( exist_status="active" ) ;'

    try:
        query_result = mysql_db.fetch_all(sql=query_sql, params=None)
    except Exception as e:
        raise Exception("[NER query] "+str(e))
    result_image_list = []
    for res in query_result:
        image_name = res[1].split('/')[-1]
        image_path = format_image_path(user_id, image_name)
        item = {"image_id": res[0], "imgSrc": image_path}
        result_image_list.append(item)
    logger.info(f'[NER query] result: {result_image_list}')
    return result_image_list


def delete_user_infos(user_id: str):
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    logger.info(f'[delete user] start query from ner results')

    # delete image_face and face_info
    try:
        logger.info(f'[delete user] delete image_face and face_info of user {user_id}.')
        mysql_db.update(sql=f"DELETE image_face, face_info FROM image_face INNER JOIN face_info ON image_face.face_id=face_info.face_id WHERE user_id='{user_id}'", params=None)
    except Exception as e:
        raise Exception(e)
    
    # delete image_info
    try:
        logger.info(f'[delete user] delete image_info of user {user_id}.')
        mysql_db.update(sql=f"DELETE FROM image_info WHERE user_id='{user_id}'", params=None)
    except Exception as e:
        raise Exception(e)
    
    # delete user_info
    try:
        logger.info(f'[delete user] delete user_info of user {user_id}.')
        mysql_db.update(sql=f"DELETE FROM user_info WHERE user_id='{user_id}'", params=None)
    except Exception as e:
        raise Exception(e)

    # delete local images
    try:
        logger.info(f'[delete user] delete local images of user {user_id}.')
        folder_path = IMAGE_ROOT_PATH+'/user'+str(user_id)
        if not os.path.exists(folder_path):
            logger.info(f'[delete user] no image folder for user {user_id}')
            return
        if os.path.isdir(folder_path):
            import shutil
            shutil.rmtree(folder_path)
        else:
            os.remove(folder_path)
        logger.info(f'[delete user] local images of user {user_id} is deleted.')
    except Exception as e:
        raise Exception(e)
    
    logger.info(f'[delete user] user {user_id} infomation all deleted.')


def forward_req_to_sd_inference_runner(inputs):
    resp = requests.post("http://{}:{}".format("198.175.88.27", "80"),
                         data=json.dumps(inputs), timeout=200)
    try:
        img_str = json.loads(resp.text)["img_str"]
        print("compute node: ", json.loads(resp.text)["ip"])
    except:
        print('no inference result. please check server connection')
        return None

    return img_str


def stable_defusion_func(inputs):
    return forward_req_to_sd_inference_runner(inputs)


@app.post("/v1/aiphotos/uploadImages")
async def handle_ai_photos_upload_images(request: Request, background_tasks: BackgroundTasks):
    user_id = request.client.host
    logger.info(f'<uploadImages> user id is: {user_id}')
    res = check_user_ip(user_id)
    logger.info("<uploadImages> "+str(res))

    params = await request.json()
    image_list = params['image_list']

    image_path = IMAGE_ROOT_PATH+'/user'+str(user_id)
    os.makedirs(image_path, exist_ok=True)

    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

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
        img_id = result[0]
        frontend_path = format_image_path(user_id, img_name)
        item = {'img_id': img_id, 'img_path': frontend_path}
        logger.info(f'<uploadImages> Image id is {img_id}, image path is {frontend_path}')
        return_list.append(item)
        obj_item = {"img_obj": img_obj, "exif": exif, "img_path": img_path, "img_id": img_id}
        image_obj_list.append(obj_item)

    background_tasks.add_task(process_images_in_background, user_id, image_obj_list)

    logger.info('<uploadImages> Finish image uploading and saving')
    return return_list


@app.post("/v1/aiphotos/getAllImages")
def handle_ai_photos_get_all_images(request: Request):
    # check request user
    user_id = request.client.host
    logger.info(f'<getAllImages> user id is: {user_id}')
    check_user_ip(user_id)
    origin = request.headers.get("Origin")
    logger.info(f'<getAllImages> origin: {origin}')

    # setup mysql_db
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    try:
        result_list = []
        image_list = list(mysql_db.fetch_all(sql=f'SELECT image_id, image_path FROM image_info WHERE user_id="{user_id}" AND exist_status="active";'))
        for image in image_list:
            image_name = image[1].split('/')[-1]
            result_list.append({"image_id": image[0], "image_path": format_image_path(user_id, image_name)})
    except Exception as e:
        return JSONResponse(content=e, status_code=500)
    else:
        logger.info(f'<getAllImages> all images of user {user_id}: {result_list}')
        return result_list


@app.post("/v1/aiphotos/getTypeList")
def handle_ai_photos_get_type_list(request: Request):
    user_id = request.client.host
    logger.info(f'<getTypeList> user id is: {user_id}')
    check_user_ip(user_id)

    type_result_dict = {"type_list": {}}

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
    # TODO: add other result into return list
    type_result_dict['type_list']['other'] = other_add_result

    type_result_dict["process_status"] = get_process_status(user_id)
    return type_result_dict


@app.post("/v1/aiphotos/getImageByType")
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
        raise Response(content=str(e), status_code=status.HTTP_400_BAD_REQUEST)
    return result


@app.post("/v1/aiphotos/getImageDetail")
async def handle_ai_photos_get_image_detail(request: Request):
    user_id = request.client.host
    logger.info(f'<getImageDetail> user id is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    image_id = params['image_id']
    logger.info(f'<getImageDetail> Getting image detail of image {image_id} by user {user_id}')

    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    try:
        image_info = mysql_db.fetch_one(sql=f'SELECT * FROM image_info WHERE image_id={image_id} AND user_id="{user_id}" AND exist_status="active";', params=None)
    except Exception as e:
        logger.error("<getImageDetail> "+str(e))
        return JSONResponse(content=f'Exception {e} occurred when selecting image {image_id} from MySQL.')
    
    if image_info:
        image_detail = format_image_info(image_info)
        logger.info(f'<getImageDetail> Image detail of image {image_id} is: {image_detail}')
        return image_detail
    else:
        return JSONResponse(content=f"No image id: {image_id} for user {user_id}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/v1/aiphotos/deleteImage")
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


@app.post("/v1/aiphotos/updateLabel")
async def handle_ai_photos_update_label(request: Request):
    # check request user
    user_id = request.client.host
    logger.info(f'<updateLabel> user id is: {user_id}')
    check_user_ip(user_id)

    # setup mysql_db
    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    params = await request.json()
    label_list = params['label_list']

    try: 
        for label_obj in label_list:
            label = label_obj['label']
            label_from = label_obj['from']
            label_to = label_obj['to']
            if label == 'person':
                mysql_db.update(sql=f'UPDATE face_info SET face_tag="{label_to}" WHERE face_tag="{label_from}"', params=None)
                mysql_db.update(sql=f"UPDATE image_face SET face_tag='{label_to}' WHERE user_id='{user_id}' and face_tag='{label_from}';", params=None)
                continue
            if label == 'address':
                update_sql = f"UPDATE image_info SET address='{label_to}' WHERE user_id='{user_id}' and address='{label_from}';"
            elif label == 'time':
                update_sql = f"UPDATE image_info SET captured_time='{label_to}' WHERE user_id='{user_id}' and captured_time='{label_from}';"
            else:
                return JSONResponse(content=f"Illegal label name: {label}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
            mysql_db.update(sql=update_sql, params=None)
            logger.info(f'<updateLabel> Label {label} updated from {label_from} to {label_to}.')
    except Exception as e:
        return JSONResponse(content=e, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    else:
        logger.info('<updateLabel> Image Labels updated successfully.')

    return "Succeed"


@app.post("/v1/aiphotos/updateTags")
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
        return Response(content=str(e), status_code=status.HTTP_400_BAD_REQUEST)
    else:
        logger.info('<updateTags> Image tags updated successfully.')

    return "Succeed"


@app.post("/v1/aiphotos/updateCaption")
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


@app.post("/v1/aiphotos/deleteUser")
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


@app.post("/v1/aiphotos/chatWithImage")
async def handle_ai_photos_chat_to_image(request: Request):
    user_id = request.client.host
    logger.info(f'<chatWithImage> user ip is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    query = params['query']
    logger.info(f'<chatWithImage> generating chat to image for user {user_id} with query: {query}')

    try:
        result = inference_ner(query)
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


@app.post("/v1/aiphotos/image2Image")
async def handle_image_to_image(request: Request):
    user_id = request.client.host
    logger.info(f'<image2Image> user ip is: {user_id}')
    check_user_ip(user_id)

    params = await request.json()
    query = params['query']
    image_list = params['ImageList']
    logger.info(f'<image2Image> user: {user_id}, image to image query command: {query}')

    sys.path.append("..")
    from database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    mysql_db.set_db("ai_photos")

    generated_images = []
    steps=25
    strength=0.75
    seed=42
    guidance_scale=7.5
    for img_info in image_list:
        img_id = img_info["imgId"]
        img_path = img_info["imgSrc"]
        userid, img_name = img_path.split('/')[-2], img_path.split('/')[-1]
        image_path = IMAGE_ROOT_PATH+'/'+userid+'/'+img_name
        logger.info(f'<image2Image> current image id: {img_id}, image path: {image_path}')

        img_b64 = image_to_byte64(image_path)
        data = {"source_img": img_b64.decode(), "prompt": query, "steps": steps,
                "guidance_scale": guidance_scale, "seed": seed, "strength": strength,
                "token": "intel_sd_bf16_112233"}
        start_time = time.time()
        img_str = stable_defusion_func(data)
        logger.info("<image2Image> elapsed time: ", time.time() - start_time)
        generated_images.append({"imgId": img_id, "imgSrc": "data:image/jpeg;base64,"+img_str})

    return generated_images


# ================== For streaming ==================
@app.post("/v1/aiphotos/talkingbot/asr")
async def handle_talkingbot_asr(file: UploadFile = File(...)):
    file_name = file.filename
    logger.info(f'Received file: {file_name}')
    with open("tmp_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file("tmp_audio_bytes")
    audio = audio.set_frame_rate(16000)
    # bytes to mp3
    audio.export(f"{file_name}", format="mp3")
    worker_name = controller.get_worker_address("mpt-7b-chat")

    try:
        r = requests.post(worker_name + "/talkingbot/asr", json={"file_name": file_name}, timeout=1000) # stream=True
    except requests.exceptions.RequestException as e:
        logger.error(f"Talkingbot fails: {worker_name}, {e}")
        return None
    print("+++++++asr+++++++")
    return {"asr_result": r.json()}


@app.post("/v1/aiphotos/talkingbot/create_embed")
async def handle_talkingbot_create_embedding(file: UploadFile = File(...)):
    file_name = file.filename
    logger.info(f'Received file: {file_name}')
    with open("tmp_driven_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file("tmp_driven_audio_bytes")
    audio = audio.set_frame_rate(16000)
    # bytes to mp3
    audio.export(f"{file_name}", format="mp3")
    worker_name = controller.get_worker_address("mpt-7b-chat")
    try:
        r = requests.post(worker_name + "/talkingbot/create_embed", json={"file_name": file_name}, timeout=1000) # stream=True
    except requests.exceptions.RequestException as e:
        logger.error(f"Talkingbot fails: {worker_name}, {e}")
        return None
    return {"voice_id": r.json()}


@app.post("/v1/aiphotos/talkingbot/llm_tts")
async def handle_talkingbot_llm_tts(request: Request):
    data = await request.json()
    text = data["text"]
    voice = data["voice"]
    knowledge_id = data["knowledge_id"]
    print(text)
    print(voice)
    print(knowledge_id)
    worker_name = controller.get_worker_address("mpt-7b-chat")
    try:
        r = requests.post(worker_name + "/talkingbot/llm_tts", json={"text": text, "voice": voice, "knowledge_id": knowledge_id}, timeout=1000, stream=True)
    except requests.exceptions.RequestException as e:
        logger.error(f"Talkingbot fails: {worker_name}, {e}")
        return None
    print("-------llm_tts-----")
    def audio_file_generate(response):
        for f in response:
            print("generate: *************")
            print(f)
            f = f.decode("utf-8")
            path = str(f)
            with open(path,mode="rb") as file:
                bytes = file.read()
                data = base64.b64encode(bytes)
            yield f"data: {data}\n\n"
        yield f"data: [DONE]\n\n"
    return StreamingResponse(audio_file_generate(r), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    parser.add_argument(
        "--cache-chat-config-file", default="llmcache/cache_config.yml", help="the cache config file"
    )
    parser.add_argument(
        "--cache-embedding-model-dir", default="hkunlp/instructor-large", help="the cache embedding model directory"
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    import sys
    sys.path.append("..")
    from llmcache.cache import init_similar_cache_from_config, put
    if args.cache_chat_config_file:
        init_similar_cache_from_config(config_dir=args.cache_chat_config_file,
                                       embedding_model_dir=args.cache_embedding_model_dir)
        put("test","test")

    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", proxy_headers=True, forwarded_allow_ips='*')
