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


import os
import json
import requests
import datetime
from typing import List, Dict
from .photoai_utils import (
    find_GPS_image,
    get_address_from_gps,
    generate_caption,
    transfer_xywh
)
from ...cli.log import logger
from datetime import timedelta, timezone


def get_image_root_path():
    IMAGE_ROOT_PATH = os.getenv("IMAGE_ROOT_PATH")
    return IMAGE_ROOT_PATH


def check_user_ip(user_ip: str) -> bool:
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    user_list = mysql_db.fetch_one(sql=f'select * from user_info where user_id = "{user_ip}";')
    logger.info(f'[Check IP] user list: {user_list}')
    if user_list == None:
        logger.info(f'[Check IP] no current user, add into db.')
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with mysql_db.transaction():
            mysql_db.insert(sql=f"insert into user_info values('{user_ip}', '{cur_time}', null, 1);", params=None)
    mysql_db._close()
    return True


def check_image_status(image_id: str):
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    image = mysql_db.fetch_one(
        sql=f'select * from image_info where image_id="{image_id}" and exist_status="active"',
        params=None
    )
    if image==None:
        raise ValueError(f'No image {image_id} saved in MySQL DB.')
    mysql_db._close()
    return image


def update_image_tags(image):
    image_id = image['image_id']
    tags = image['tags']
    image_info = check_image_status(image_id)
    update_sql = 'UPDATE image_info SET '
    update_sql_list = []
    old_tags = eval(image_info['other_tags'])
    logger.info(f'[Update Tags] old_tags: {old_tags}')
    tag_name_list = []
    for key, value in tags.items():
        if key == 'time' and value != image_info['captured_time']:
            update_sql_list.append(f' captured_time="{value}" ')
            tag_name_list.append('time')
        elif key == 'latitude' and value != image_info['latitude']:
            update_sql_list.append(f' latitude="{value}" ')
            tag_name_list.append('latitude')
        elif key == 'longitude' and value != image_info['longitude']:
            update_sql_list.append(f' longitude="{value}" ')
            tag_name_list.append('longitude')
        elif key == 'altitude' and value != image_info['altitude']:
            update_sql_list.append(f' altitude="{value}" ')
            tag_name_list.append('altitude')
        elif key == 'location' and value != image_info['address']:
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
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    with mysql_db.transaction():
        mysql_db.update(sql=final_sql, params=None)
    mysql_db._close()


def update_image_attr(image, attr):
    from ...utils.database.mysqldb import MysqlDb
    image_id = image['image_id']
    check_image_status(image_id)

    new_attr = image[attr]
    try:
        mysql_db = MysqlDb()
        if attr=='checked':
            new_checked = 1 if new_attr else 0
            with mysql_db.transaction():
                mysql_db.update(
                    sql=f"UPDATE image_info SET {attr}={new_checked} WHERE image_id={image_id}",
                    params=None
                )
        else:
            with mysql_db.transaction():
                mysql_db.update(
                    sql=f'UPDATE image_info SET {attr}="{new_attr}" WHERE image_id={image_id}',
                    params=None
                )
    except Exception as e:
        logger.error(e)
    else:
        mysql_db._close()
        logger.info(f'Image {attr} updated successfully.')


def format_image_path(user_id: str, image_name: str) -> str:
    server_ip = os.getenv("IMAGE_SERVER_IP")
    if not server_ip:
        raise Exception("Please configure SERVER IP to environment variables.")
    image_path = "https://"+server_ip+"/ai_photos/user"+user_id+'/'+image_name
    return image_path


def format_image_info(image_info: dict) -> dict:
    image_item = {}
    image_item['image_id'] = image_info['image_id']
    image_item['user_id'] = image_info['user_id']
    image_name = image_info['image_path'].split('/')[-1]
    image_item['image_path'] = format_image_path(image_info['user_id'], image_name)
    image_item['caption'] = image_info['caption']
    tag_list = {}
    if image_info['captured_time']:
        tag_list['time'] = datetime.datetime.date(image_info['captured_time'])
    if image_info['address'] != 'None':
        tag_list['location'] = image_info['address']
    other_tags = eval(image_info['other_tags'])
    tag_list.update(other_tags)
    image_item['tag_list'] = tag_list
    return image_item


def delete_single_image(user_id, image_id):
    from ...utils.database.mysqldb import MysqlDb
    logger.info(f'[Delete] Deleting image {image_id}')
    mysql_db = MysqlDb()
    image_path = mysql_db.fetch_one(
        sql=f'SELECT image_path FROM image_info WHERE image_id={image_id}',
        params=None
    )
    if image_path==None:
        info = f'[Delete] Image {image_id} does not exist in MySQL.'
        logger.error(info)
        raise Exception(info)
    image_path = image_path['image_path']

    # delete local image
    os.remove(image_path)
    logger.info(f'[Delete] Image {image_path} successfully deleted.')

    # update db info, set image status as 'deleted'
    try:
        with mysql_db.transaction():
            mysql_db.update(
                sql=f"UPDATE image_info SET exist_status='deleted' WHERE image_id={image_id} ;",
                params=None
            )
    except Exception as e:
        logger.error(e)
        raise Exception(e)
    finally:
        mysql_db._close()
    logger.info(f'[Delete] Image {image_id} deleted successfully.')


def process_images_in_background( user_id: str, image_obj_list: List[Dict]):
    try:
        logger.info(f'[background] ======= processing image list for user {user_id} in background =======')
        for i in range(len(image_obj_list)):
            # save image into local path
            image_id = image_obj_list[i]['img_id']
            image_path = image_obj_list[i]['img_path']
            image_obj = image_obj_list[i]['img_obj']
            image_exif = image_obj_list[i]['exif']
            image_obj.save(image_path, exif=image_exif)
            logger.info(f'[background] Image saved into local path {image_path}')
            # process image and generate infos
            try:
                process_single_image(image_id, image_path, user_id)
            except Exception as e:
                logger.error("[background] "+str(e))
                logger.error(f'[background] error occurred, delete image.')
                delete_single_image(user_id, image_id)

    except Exception as e:
        logger.error(e)
        raise ValueError(str(e))
    else:
        logger.info('[background] Background images process finished.')


def process_single_image(img_id, img_path, user_id):
    logger.info(f'[background - single] ----- processing image {img_path} in background -----')

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
    logger.info(f'[background - single] Image is captured at: {captured_time},' +
                'latitude: {latitude}, longitude: {longitude}, altitude: {altitude}')
    if latitude:
        update_image_attr(image={"image_id": img_id, "latitude": latitude}, attr='latitude')
    if longitude:
        update_image_attr(image={"image_id": img_id, "longitude": longitude}, attr='longitude')
    if altitude:
        update_image_attr(image={"image_id": img_id, "altitude": altitude}, attr='altitude')
    if captured_time:
        update_image_attr(image={"image_id": img_id, "captured_time": captured_time}, attr='captured_time')
    else:
        SHA_TZ = timezone(
            timedelta(hours=8),
            name='Asia/Shanghai'
        )
        utc_now = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
        beijing_time = utc_now.astimezone(SHA_TZ)
        update_image_attr(image={"image_id": img_id, "captured_time": beijing_time}, attr='captured_time')

    # generate address info
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("Please configure environment variable of GOOGLE_API_KEY.")
    address = get_address_from_gps(latitude, longitude, api_key)
    if address:
        logger.info(f'[background - single] Image address: {address}')
        address_components = []
        if address.get('country', None):
            address_components.append(address['country'])
        if address.get('administrative_area_level_1', None):
            address_components.append(address['administrative_area_level_1'])
        if address.get('locality', None):
            address_components.append(address['locality'])
        if address.get('sublocality', None):
            address_components.append(address['sublocality'])
        formatted_address = ', '.join(address_components)
        update_image_attr(image={"image_id": img_id, "address": formatted_address}, attr='address')
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
    IMAGE_ROOT_PATH = get_image_root_path()
    db_path = IMAGE_ROOT_PATH+"/user"+user_id
    try:
        process_face_for_single_image(image_id=img_id, image_path=img_path, db_path=db_path, user_id=user_id)
    except Exception as e:
        logger.error(f'[background - single] Error occurred while processing face.')
        raise Exception('[background - single]', str(e))
    logger.info(f'[background - single] Face process done for image {img_id}')

    # update image status
    try:
        from ...utils.database.mysqldb import MysqlDb
        mysql_db = MysqlDb()
        with mysql_db.transaction():
            mysql_db.update(sql=f"UPDATE image_info SET process_status='ready' WHERE image_id={img_id}", params=None)
    except Exception as e:
        logger.error("[background - single] "+str(e))
    finally:
        mysql_db._close()
    logger.info(f"[background - single] ----- finish image {img_path} processing -----")


def process_face_for_single_image(image_id, image_path, db_path, user_id):
    logger.info(f'[background - face] ### processing face for {image_path} in background ###')
    from deepface import DeepFace
    # 1. check whether image contains faces
    try:
        face_objs = DeepFace.represent(img_path=image_path, model_name='Facenet512')
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
    pkl_path = db_path+'/representations_facenet512.pkl'
    if os.path.exists(pkl_path):
        logger.info(f'[background - face] pkl file already exists, delete it.')
        os.remove(pkl_path)
    dfs = DeepFace.find(img_path=image_path, db_path=db_path, model_name='Facenet512', enforce_detection=False)
    logger.info(f'[background - face] Finding match faces in image database.')
    assert face_cnt == len(dfs)
    logger.info(f'[background - face] dfs: {dfs}')
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    for df in dfs:
        logger.info(f'[background - face] current df: {df}')
        # no face matched for current face of image, add new faces later
        if len(df) <= 1:
            logger.info(f'[background - face] length of current df less than 1, continue')
            continue
        # findd ref image
        ref_image_path = None
        ref_image_list = df['identity']
        for ref_image_name in ref_image_list:
            logger.info(f'[background - face] current ref_image_name: {ref_image_name}')
            if ref_image_name!=image_path:
                ref_image_path = ref_image_name
                break
        # no ref image found
        if not ref_image_path:
            logger.info(f'[background - face] no other reference image found, continue')
            continue
        # find faces in img2: one or many
        find_face_sql = f"""
            SELECT face_id, face_tag, xywh FROM image_face WHERE
            image_path='{ref_image_path}' AND user_id='{user_id}';
        """
        try:
            img_face_list = mysql_db.fetch_all(sql=find_face_sql)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception occurred while selecting info from image_face: {e}")
        logger.info(f"[background - face] reference image and faces: {img_face_list}")
        # wrong ref image found
        if img_face_list == ():
            logger.error(f"img_face_list is {img_face_list}, wrong ref image found")
            continue
        # verify face xywh of ref image
        obj = DeepFace.verify(img1_path=image_path, img2_path=ref_image_path, model_name="Facenet512")
        ref_xywh = transfer_xywh(obj['facial_areas']['img2'])
        image_xywh = transfer_xywh(obj['facial_areas']['img1'])
        face_id = -1
        face_tag = None
        # find corresponding face_id and face_tag
        for img_face in img_face_list:
            if img_face['xywh'] == ref_xywh:
                face_id = img_face['face_id']
                face_tag = img_face['face_tag']
        if face_id == -1 and face_tag == None:
            raise Exception(f'Error occurred when verifying faces for reference image: Inconsistent face information.')
        # insert into image_face
        insert_img_face_sql = f"""INSERT INTO image_face
        VALUES(null, {image_id}, '{image_path}', {face_id}, '{image_xywh}', '{user_id}', '{face_tag}');"""
        try:
            with mysql_db.transaction():
               mysql_db.insert(sql=insert_img_face_sql, params=None)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception occurred while inserting info into image_face: {e}")
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
        face_cnt = mysql_db.fetch_one(sql="SELECT COUNT(*) AS cnt FROM face_info;")['cnt']
        tag = 'person'+str(face_cnt+1)
        face_sql = f"INSERT INTO face_info VALUES(null, '{tag}');"
        try:
            with mysql_db.transaction():
                mysql_db.insert(sql=face_sql, params=None)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception occurred while inserting new face into face_info: {e}")
        logger.info(f"[background - face] face {tag} inserted into db.")
        face_id = mysql_db.fetch_one(f"SELECT * FROM face_info WHERE face_tag='{tag}';")['face_id']
        logger.info(f"[background - face] new face id is: {face_id}")
        img_face_sql = f"""INSERT INTO image_face VALUES
        (null, {image_id}, '{image_path}', {face_id}, '{cur_xywh}', '{user_id}', '{tag}');"""
        try:
            with mysql_db.transaction():
                mysql_db.insert(sql=img_face_sql, params=None)
        except Exception as e:
            logger.error("[background - face] "+str(e))
            raise Exception(f"Exception occurred while inserting new face into image_face: {e}")
        logger.info(f"[background - face] img_face {img_face_sql} inserted into db.")
    mysql_db._close()
    logger.info(f"[background - face] Image {image_id} face process finished.")


def get_type_obj_from_attr(attr, user_id):
    logger.info(f'Geting image type of {attr}')

    if attr == 'time':
        select_sql = f'''SELECT DATE(captured_time) AS date FROM image_info
        WHERE user_id = "{user_id}" AND exist_status="active" GROUP BY date ORDER BY date;'''
    elif attr == 'address':
        select_sql = f'''SELECT address FROM image_info
        WHERE user_id="{user_id}" AND exist_status="active" GROUP BY address;'''
    else:
        return {}

    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    select_list = mysql_db.fetch_all(sql=select_sql)
    select_result = {}
    for item in select_list:
        logger.info(f"current item of {attr} is: {item}")
        if attr == 'time':
            item = item['date']
            if item == None:
                continue
            example_image_path = mysql_db.fetch_one(
                sql=f'''SELECT image_path FROM image_info
                WHERE DATEDIFF(captured_time, "{item}") = 0 and user_id="{user_id}"
                and exist_status="active" LIMIT 1;''',
                params=None)['image_path']
        elif attr == 'address':
            item = item['address']
            if item == None or item == 'None' or item == 'null':
                continue
            example_image_path = mysql_db.fetch_one(
                sql=f'''SELECT image_path FROM image_info WHERE
                address="{item}" and user_id="{user_id}" and exist_status="active" LIMIT 1;''',
                params=None)['image_path']

        image_name = example_image_path.split('/')[-1]
        image_path = format_image_path(user_id, image_name)
        select_result[item] = image_path

    mysql_db._close()

    # return time result directly
    if attr == 'time':
        logger.info(f'type list: {select_result}')
        return select_result

    # check whether address simplification is needed
    simplify_flag = True
    cur_country = None
    address_list = list(select_result.keys())
    for address in address_list:
        country = address.split(', ')[0]
        if not cur_country:
            cur_country = country
        else:
            if country != cur_country:
                simplify_flag = False
                break

    # simplify address name dynamically
    if simplify_flag:
        logger.info(f'address need to be simplified')
        new_result = {}
        for key, value in select_result.items():
            new_key = ', '.join(key.split(', ')[1:])
            new_result[new_key] = value
        logger.info(f'type list: {new_result}')
        return new_result
    else:
        logger.info(f'type list: {select_result}')
        return select_result


def get_address_list(user_id) -> list[str]:
    logger.info(f'Getting address list of user {user_id}')
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    select_sql = f'''SELECT address FROM image_info WHERE
    user_id="{user_id}" AND exist_status="active" GROUP BY address;'''
    select_list = mysql_db.fetch_all(sql=select_sql)
    result_list = []
    for item in select_list:
        address = item['address']
        if address == None or address == 'None' or address == 'null':
            continue
        add_list = address.split(', ')
        for add in add_list:
            if add not in result_list:
                result_list.append(add)
    logger.info(f'address list of user {user_id} is {result_list}')
    return result_list


def get_process_status(user_id):
    logger.info(f'Geting process status of user {user_id}')
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    total_cnt = mysql_db.fetch_one(
        sql=f"""SELECT COUNT(*) AS cnt FROM image_info WHERE
        user_id='{user_id}' AND exist_status='active';""")['cnt']
    processing_cnt = mysql_db.fetch_one(
        sql=f"""SELECT COUNT(*) AS cnt FROM image_info WHERE
        user_id='{user_id}' AND exist_status='active' AND process_status='processing';""")['cnt']
    mysql_db._close()
    result = {}
    result['total_image'] = total_cnt
    result['processing_image'] = processing_cnt
    result['status'] = "done" if processing_cnt ==0 else 'processing'
    return result


def get_images_by_type(user_id, type, subtype) -> List:
    logger.info(f'Getting image by type {type} - {subtype}')

    if type == 'address':
        if subtype == 'default':
            subtype = 'None'
        sql=f"""SELECT image_id, image_path FROM image_info WHERE
        user_id='{user_id}' AND exist_status='active' AND address LIKE '%{subtype}%';"""

    elif type == 'time':
        if subtype == 'None':
            sql = f'''SELECT image_id, image_path FROM image_info
            WHERE captured_time is null AND user_id="{user_id}" AND exist_status="active";'''
        else:
            sql = f'''SELECT image_id, image_path FROM image_info
            WHERE DATE(captured_time)="{subtype}" AND user_id="{user_id}" AND exist_status="active";'''

    elif type == 'person':
        sql = f"""SELECT image_info.image_id, image_info.image_path FROM image_face
        INNER JOIN image_info ON image_info.image_id=image_face.image_id
        WHERE image_info.user_id='{user_id}' AND image_info.exist_status='active'
        AND image_face.face_tag='{subtype}'"""

    logger.info(f'sql: {sql}')
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()
    images = mysql_db.fetch_all(sql=sql, params=None)
    mysql_db._close()
    logger.info(f"image list: {images}")
    if len(images) == 0:
        logger.error(f'no label {subtype} in {type}')
        return []
    else:
        result = []
        for image in images:
            image_name = image['image_path'].split('/')[-1]
            image_path = format_image_path(user_id, image_name)
            obj = {"image_id": image['image_id'], "image_path": image_path}
            result.append(obj)
        return result


def get_face_list_by_user_id(user_id: str) -> List[Dict]:
    logger.info(f'getting face list of user {user_id}')
    group_by_face_sql = f'''SELECT group_concat(image_face.image_path) AS image_path,
    group_concat(image_face.face_tag) AS face_tag FROM image_face
    INNER JOIN image_info ON image_info.image_id=image_face.image_id
    WHERE image_info.user_id = "{user_id}" AND image_info.exist_status="active" GROUP BY face_id;'''
    try:
        from ...utils.database.mysqldb import MysqlDb
        mysql_db = MysqlDb()
        query_list = mysql_db.fetch_all(sql=group_by_face_sql)
    except Exception as e:
        logger.error(e)
        raise Exception(e)
    finally:
        mysql_db._close()
    logger.info(f'query result list: {query_list}')
    response_person = {}
    for item in query_list:
        logger.info(f'current item: {item}')
        image_name = item['image_path'].split('/')[-1]
        face_tag = item['face_tag']
        if ',' in face_tag:
            face_tag = face_tag.split(',')[0]
        response_person[face_tag] = format_image_path(user_id, image_name)
    logger.info(f'person list: {response_person}')
    return response_person


def get_image_list_by_ner_query(ner_result: Dict, user_id: str, query: str) -> List[Dict]:

    logger.info(f'[NER query] start query from ner results')
    query_sql = "SELECT image_info.image_id, image_info.image_path FROM image_info "
    query_flag = False
    from ...utils.database.mysqldb import MysqlDb
    mysql_db = MysqlDb()

    # get person name query
    face_list = mysql_db.fetch_all(
        sql=f"""select image_face.face_tag from image_face inner join image_info
        on image_info.image_id=image_face.image_id where
        image_info.user_id='{user_id}' AND exist_status='active';""",
        params=None)
    logger.info(f"[NER query] face list is: {face_list}")
    if face_list:
        sql_conditions = []
        for face_tag in face_list:
            face_tag = face_tag['face_tag']
            if face_tag in query:
                logger.info(f'[NER query] other face detected in db: [{face_tag}]')
                sql_conditions.append(f' image_face.face_tag LIKE "%{face_tag}%" ')
        if sql_conditions != []:
            query_flag = True
            sql = 'OR'.join(sql_conditions)
            query_sql += "INNER JOIN image_face ON image_info.image_id=image_face.image_id WHERE "
            query_sql += '('+sql+')'
        else:
            logger.info(f'[NER query] no person name in ner query')
    else:
        logger.info(f'[NER query] no person name in ner query')

    # get location query
    location_list = get_address_list(user_id)
    logger.info(f"[NER query] location list is: {location_list}")
    if location_list:
        sql_conditions = []
        for db_loc in location_list:
            if db_loc in query:
                sql_conditions.append(f' image_info.address LIKE "%{db_loc}%" ')
        if sql_conditions != []:
            if not query_flag:
                query_sql += " WHERE "
            query_flag = True
            sql = 'OR'.join(sql_conditions)
            if query_sql[-1] == ')':
                query_sql += ' AND '
            query_sql += '('+sql+')'
    else:
        logger.info(f'[NER query] no location in query')

    # get time query
    if ner_result['time']:
        time_points = ner_result['time']
        sql_conditions = []
        for loc in time_points:
            sql_conditions.append(f' image_info.captured_time LIKE "%{loc}%" ')
        if sql_conditions != []:
            if not query_flag:
                query_sql += " WHERE "
            query_flag = True
            sql = 'OR'.join(sql_conditions)
            if query_sql[-1] == ')':
                query_sql += ' AND '
            query_sql += '('+sql+')'
    else:
        logger.info(f'[NER query] no time in query')

    # get time period query
    if ner_result['period']:
        periods = ner_result['period']
        logger.info(f'[NER query] periods: {periods}')
        sql_conditions = []
        for period in periods:
            from_time = period['from']
            to_time = period['to']
            format = "%Y-%m-%d"
            to_time = datetime.datetime.strptime(to_time, format)
            new_to_time = to_time + datetime.timedelta(days=1)
            sql_conditions.append(
                f' image_info.captured_time BETWEEN "{from_time}" AND "{new_to_time.strftime(format)}" '
            )
        if sql_conditions != []:
            if not query_flag:
                query_sql += " WHERE "
            query_flag = True
            sql = 'OR'.join(sql_conditions)
            if query_sql[-1] == ')':
                query_sql += ' AND '
            query_sql += '('+sql+')'
    else:
        logger.info(f'[NER query] no time period in query')

    if not query_flag:
        logger.info(f'[NER query] no compatible data for current query')
        return []
    query_sql += f' AND ( image_info.user_id="{user_id}" ) AND ( exist_status="active" ) ;'
    logger.info(f'[NER query] query sql: {query_sql}')

    try:
        query_result = mysql_db.fetch_all(sql=query_sql, params=None)
    except Exception as e:
        raise Exception("[NER query] "+str(e))
    result_image_list = []
    for res in query_result:
        image_name = res['image_path'].split('/')[-1]
        image_path = format_image_path(user_id, image_name)
        item = {"image_id": res['image_id'], "imgSrc": image_path}
        result_image_list.append(item)
    logger.info(f'[NER query] result: {result_image_list}')
    mysql_db._close()
    return result_image_list


def delete_user_infos(user_id: str):
    logger.info(f'[delete user] start delete user info')

    try:
        from ...utils.database.mysqldb import MysqlDb
        mysql_db = MysqlDb()
        with mysql_db.transaction():
            # delete image_face
            logger.info(f'[delete user] delete image_face of user {user_id}.')
            mysql_db.delete(
                sql=f"""DELETE FROM image_face WHERE user_id='{user_id}'""",
                params=None)

            # delete face_info
            logger.info(f'[delete user] delete face_info of user {user_id}.')
            mysql_db.delete(
                sql=f"""DELETE face_info FROM face_info LEFT JOIN image_face
                ON face_info.face_id = image_face.face_id WHERE image_face.face_id IS NULL""",
                params=None)

            # delete image_info
            logger.info(f'[delete user] delete image_info of user {user_id}.')
            mysql_db.delete(sql=f"DELETE FROM image_info WHERE user_id='{user_id}'", params=None)

            # delete user_info
            logger.info(f'[delete user] delete user_info of user {user_id}.')
            mysql_db.delete(sql=f"DELETE FROM user_info WHERE user_id='{user_id}'", params=None)
    except Exception as e:
        raise Exception(e)
    finally:
        mysql_db._close()

    # delete local images
    try:
        logger.info(f'[delete user] delete local images of user {user_id}.')
        IMAGE_ROOT_PATH = get_image_root_path()
        folder_path = IMAGE_ROOT_PATH+'/user'+str(user_id)
        if not os.path.exists(folder_path):
            logger.info(f'[delete user] no image folder for user {user_id}')
            return
        else:
            if os.path.isdir(folder_path):
                import shutil
                shutil.rmtree(folder_path)
            else:
                os.remove(folder_path)
            logger.info(f'[delete user] local images of user {user_id} is deleted.')
    except Exception as e:
        raise Exception(e)

    logger.info(f'[delete user] user {user_id} information all deleted.')


def forward_req_to_sd_inference_runner(inputs):
    image2image_ip = os.environ.get("IMAGE2IMAGE_IP")
    resp = requests.post("http://{}:{}".format(image2image_ip, "80"),
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
