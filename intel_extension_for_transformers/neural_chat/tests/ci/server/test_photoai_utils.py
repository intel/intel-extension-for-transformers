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
from PIL import Image
from intel_extension_for_transformers.neural_chat.server.restful.photoai_utils import (
    find_GPS_image, generate_caption, image_to_byte64, byte64_to_image, transfer_xywh
)

class UnitTest(unittest.TestCase):

    def setUp(self) -> None:
        os.environ['IMAGE_SERVER_IP'] = 'test_server_ip'
        return super().setUp()

    def tearDown(self) -> None:
        if os.path.exists("./MagicMock"):
            shutil.rmtree("./MagicMock")

    def test_find_GPS_image(self):
        img_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests" + \
                "/ci/server/test_images/img_bird.JPG"
        if os.path.exists(img_file_path):
            res = find_GPS_image(img_file_path)
        else:
            res = find_GPS_image("./ci/server/test_images/img_bird.JPG")
        self.assertIn('2019:06:18', res['date_information'])
        self.assertEqual(122.63912222222223, res['GPS_information']['GPSLongitude'])

    def test_generate_caption(self):
        img_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests" + \
                "/ci/server/test_images/img_bird.JPG"
        if os.path.exists(img_file_path):
            res = generate_caption(img_file_path)
        else:
            res = generate_caption("./ci/server/test_images/img_bird.JPG")
        self.assertIn('seagulls', res)

    def test_image_byte64(self):
        img_file_path = "/intel-extension-for-transformers/" + \
            "intel_extension_for_transformers/neural_chat/tests" + \
                "/ci/server/test_images/img_bird.JPG"
        if os.path.exists(img_file_path):
            img_b64 = image_to_byte64(img_file_path)
        else:
            img_b64 = image_to_byte64("./ci/server/test_images/img_bird.JPG")
        self.assertIn('79qt3Sr5Y9utCKR//Z', str(img_b64))

        img = byte64_to_image(img_b64)
        self.assertIsInstance(img, Image.Image)

    def test_transfer_xywh(self):
        facial_area = {
            'x': 1,
            'y': 2,
            'w': 3,
            'h': 4
        }
        res = transfer_xywh(facial_area)
        self.assertIn('1_2_3_4_', res)


if __name__ == "__main__":
    unittest.main()
