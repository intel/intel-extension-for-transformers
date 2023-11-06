/*

Copyright (c) 2023 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Create the Neural Chat Database: ai_photos
Create empty Tables: user_info, image_info, face_info, image_face

*/

drop database if exists ai_photos;
create database ai_photos;
use ai_photos;

drop table if exists user_info;
create table user_info
(
  user_id varchar(20) primary key,
  login_time datetime,
  leave_time datetime,
  is_active int,
  unique (user_id)
);

drop table if exists image_info;
create table image_info
(
  image_id int unsigned primary key auto_increment,
  user_id varchar(20),
  image_path varchar(100),
  captured_time datetime,
  caption varchar(200),
  latitude varchar(20),
  longitude varchar(20),
  altitude varchar(20),
  address varchar(200),
  checked boolean,
  other_tags varchar(200),
  process_status varchar(20),
  exist_status varchar(20),
  unique (image_id),
  FOREIGN KEY (user_id) REFERENCES user_info(user_id)
);

drop table if exists face_info;
create table face_info
(
  face_id int unsigned primary key auto_increment,
  face_tag varchar(20),
  unique (face_id)
);

drop table if exists image_face;
create table image_face
(
  image_face_id int unsigned primary key auto_increment,
  image_id int unsigned,
  image_path varchar(100),
  face_id int unsigned,
  xywh varchar(30),
  user_id varchar(20),
  face_tag varchar(20),
  unique (image_face_id),
  FOREIGN KEY (image_id) REFERENCES image_info(image_id),
  FOREIGN KEY (face_id) REFERENCES face_info(face_id)
);

