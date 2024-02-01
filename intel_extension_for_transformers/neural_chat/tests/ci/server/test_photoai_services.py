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
import shutil
import unittest
import datetime
import pandas as pd
from unittest.mock import patch, MagicMock
from intel_extension_for_transformers.neural_chat.server.restful.photoai_services import (
    check_user_ip, check_image_status, update_image_tags, update_image_attr, format_image_info,
    delete_single_image, process_images_in_background, process_single_image,
    process_face_for_single_image, get_type_obj_from_attr, get_address_list,
    get_process_status, get_images_by_type, get_face_list_by_user_id,
    get_image_list_by_ner_query, delete_user_infos
)


MOCK_USER_INFO = {'user_id': '1', 'login_time': None, 'leave_time': None, 'is_active': 1}
MOCK_IMAGE_INFO = {
    'image_id': 1,
    'user_id': '1',
    'image_path': 'image1.jpg',
    'captured_time': datetime.datetime.strptime('2022-02-22', '%Y-%m-%d'),
    'caption': 'mocked caption',
    'latitude': None,
    'longitude': None,
    'altitude': None,
    'address': 'Shanghai',
    'checked': True,
    'other_tags': '{}',
    'process_status': 'ready',
    'exist_status': 'active'
}


@patch('intel_extension_for_transformers.neural_chat.utils.database.mysqldb.MysqlDb')
class UnitTest(unittest.TestCase):

    def setUp(self) -> None:
        os.environ['IMAGE_SERVER_IP'] = 'test_server_ip'
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./MagicMock"):
            shutil.rmtree("./MagicMock")
        if os.path.exists("./test_delete_file.txt"):
            shutil.rmtree("./test_delete_file.txt")

    def test_check_user_ip(self, mock_db):
        mock_db.return_value.fetch_one.return_value = MOCK_USER_INFO
        res = check_user_ip("test_user_ip")
        self.assertTrue(res)
        mock_db.return_value.fetch_one.assert_called_once_with(
            sql='select * from user_info where user_id = "test_user_ip";'
        )


    def test_check_image_status(self, mock_db):
        mock_db.return_value.fetch_one.return_value = MOCK_IMAGE_INFO
        res = check_image_status("test_image_id")
        self.assertTrue(res)
        mock_db.return_value.fetch_one.assert_called_once_with(
            sql='select * from image_info where image_id="test_image_id" and exist_status="active"',
            params=None
        )


    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.check_image_status')
    def test_update_image_tags(self, mock_func, mock_db):
        mock_func.return_value = MOCK_IMAGE_INFO
        mock_db.return_value.transaction.return_value = MagicMock()
        mock_db.return_value.update.return_value = True
        try:
            image = {
                'image_id': 1,
                'tags': {'time': 'new_time'}
            }
            update_image_tags(image)
        except Exception as e:
            raise Exception(e)
        mock_db.return_value.update.assert_called_once_with(
            sql='UPDATE image_info SET  captured_time="new_time" , other_tags="{}"  where  image_id=1',
            params=None
        )


    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.check_image_status')
    def test_update_image_attr(self, mock_func, mock_db):
        mock_func.return_value = MOCK_IMAGE_INFO
        mock_db.return_value.transaction.return_value = MagicMock()
        mock_db.return_value.update.return_value = True
        try:
            image = MOCK_IMAGE_INFO
            update_image_attr(image, "captured_time")
        except Exception as e:
            raise Exception(e)
        mock_db.return_value.update.assert_called_once_with(
            sql='UPDATE image_info SET captured_time="2022-02-22 00:00:00" WHERE image_id=1',
            params=None
        )


    def test_format_image_info(self, mock_db):
        image = MOCK_IMAGE_INFO
        res = format_image_info(image)
        self.assertIn('https://test_server_ip/ai_photos/user1/image1.jpg', res['image_path'])


    def test_delete_single_image(self, mock_db):
        mock_db.return_value.fetch_one.return_value = {'image_path': './test_delete_file.txt'}
        mock_db.return_value.transaction.return_value = MagicMock()
        mock_db.return_value.update.return_value = True
        try:
            with open('./test_delete_file.txt', mode='w') as f:
                pass
            delete_single_image('mock_user_id', 1)
        except Exception as e:
            raise Exception(e)
        mock_db.return_value.update.assert_called_once_with(
            sql="UPDATE image_info SET exist_status='deleted' WHERE image_id=1 ;",
            params=None
        )


    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.process_single_image')
    def test_process_images_in_background(self, mock_func, mock_db):
        mock_func.return_value = True
        try:
            mock_obj = MagicMock()
            mock_obj.save.return_value = True
            image_obj_list = [
                {
                    'img_id': 1,
                    'img_path': 'mocked_path',
                    'img_obj': mock_obj,
                    'exif': "mocked_exif"
                }
            ]
            process_images_in_background(user_id='1', image_obj_list=image_obj_list)
        except Exception as e:
            raise Exception(e)


    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.process_face_for_single_image')
    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.generate_caption')
    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.get_address_from_gps')
    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.update_image_attr')
    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.find_GPS_image')
    def test_process_single_image(self, mock_func1, mock_func2, mock_func3, mock_func4, mock_func5, mock_db):
        mock_func1.return_value = {
            'date_information': None,
            'GPS_information': {
                'GPSLatitude': '',
                'GPSLongitude': '',
                'GPSAltitude': ''
            }
        }
        mock_func2.return_value = True
        mock_func3.return_value = {
            'country': 'China',
            'administrative_area_level_1': 'Shanghai',
            'sublocality': 'Minhangqu'
        }
        mock_func4.return_value = 'mocked caption'
        mock_func5.return_value = True
        os.environ['GOOGLE_API_KEY'] = 'mocked_api_key'
        os.environ['IMAGE_ROOT_PATH'] = './mocked_root_path'
        mock_db.return_value.transaction.return_value = MagicMock()
        mock_db.return_value.update.return_value = True
        try:
            process_single_image(img_id=1, img_path="mocked_path", user_id='1')
        except Exception as e:
            raise Exception(e)
        mock_db.return_value.update.assert_called_once_with(
            sql="UPDATE image_info SET process_status='ready' WHERE image_id=1",
            params=None
        )


    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.transfer_xywh')
    @patch('deepface.DeepFace.verify')
    @patch('deepface.DeepFace.find')
    @patch('deepface.DeepFace.represent')
    def test_process_face_for_single_image(self, mock_represent, mock_find, mock_verify, mock_func1, mock_db):
        mock_represent.return_value = [{'facial_area': 'mocked_xywh'}]
        df = pd.DataFrame(data=['./mocked_identity1'],
                          columns=['identity'])
        mock_find.return_value = [df]
        mock_verify.return_value = {'facial_areas': {
            'img1': 'mocked_xywh1',
            'img2': 'mocked_xywh2'}}
        mock_func1.return_value = 'mocked_transferred_xywh'
        mock_db.return_value.fetch_all.return_value = [{'face_id': 1, 'face_tag': 'mocked_tag'}]
        mock_db.return_value.transaction.return_value = MagicMock()
        mock_db.return_value.insert.return_value = True
        mock_db.return_value.fetch_one.return_value = {'cnt': 0, 'face_id': 1}
        try:
            process_face_for_single_image(
                image_id=1, image_path='./mocked_image_path', db_path='./mocked_db_path', user_id='1')
        except Exception as e:
            raise Exception(e)
        mock_db.return_value.insert.assert_any_call(
            sql="INSERT INTO face_info VALUES(null, 'person1');",
            params=None
        )


    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.format_image_path')
    def test_get_type_obj_from_attr(self, mock_func, mock_db):
        mock_func.return_value = 'Mocked image path'
        mock_db.return_value.fetch_all.return_value = [{'address': 'China, Shanghai'}]
        mock_db.return_value.fetch_one.return_value = {'image_path': "./mocked_image_path"}
        res = get_type_obj_from_attr(attr="address", user_id='1')
        self.assertIn('Mocked image path', res['Shanghai'])


    def test_get_address_list(self, mock_db):
        mock_db.return_value.fetch_all.return_value = [{'address': 'China, Shanghai'}]
        res = get_address_list(user_id='1')
        self.assertIn('Shanghai', res)
        mock_db.return_value.fetch_all.assert_called_once_with(
            sql='''SELECT address FROM image_info WHERE\n    user_id="1" AND exist_status="active" GROUP BY address;''')


    def test_get_process_status(self, mock_db):
        mock_db.return_value.fetch_one.return_value = {'cnt': '2'}
        res = get_process_status(user_id='1')
        self.assertIn('processing', res['status'])


    def test_get_images_by_type(self, mock_db):
        mock_db.return_value.fetch_all.return_value = [{
            'image_path': 'mocked/image.jpg',
            'image_id': 1
        }]
        res = get_images_by_type(user_id='1', type='address', subtype='default')
        self.assertIn('https://test_server_ip/ai_photos/user1/image.jpg', res[0]['image_path'])


    def test_get_face_list_by_user_id(self, mock_db):
        mock_db.return_value.fetch_all.return_value = [{
            'image_path': 'mocked/image.jpg',
            'face_tag': 'person1'
        }]
        res = get_face_list_by_user_id(user_id='1')
        self.assertIn('https://test_server_ip/ai_photos/user1/image.jpg', res['person1'])


    @patch('intel_extension_for_transformers.neural_chat.server.restful.photoai_services.get_address_list')
    def test_get_image_list_by_ner_query(self, mock_func1, mock_db):
        mock_func1.return_value = ['shanghai']
        mock_db.return_value.fetch_all.return_value = [{
            'image_path': 'mocked/image.jpg',
            'face_tag': 'person1',
            'image_id': 1
        }]
        res = get_image_list_by_ner_query(
            ner_result={
                'time': ['2022-02-22'],
                'period': [{'from': '2022-02-02', 'to': '2023-02-02'}]},
            user_id='1',
            query='photos taken in shanghai')
        self.assertIn('https://test_server_ip/ai_photos/user1/image.jpg', res[0]['imgSrc'])


    def test_delete_user_infos(self, mock_db):
        mock_db.return_value.transaction.return_value = MagicMock()
        mock_db.return_value.delete.return_value = True
        try:
            os.environ['IMAGE_ROOT_PATH'] = './mocked_root_path'
            delete_user_infos(user_id='1')
        except Exception as e:
            raise Exception(e)
        mock_db.return_value.delete.assert_any_call(
            sql="DELETE FROM user_info WHERE user_id='1'",
            params=None
        )


if __name__ == "__main__":
    unittest.main()
