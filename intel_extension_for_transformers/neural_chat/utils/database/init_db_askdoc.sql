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

Create the Chatbot Database: fastrag
Create empty Tables: feedback
                     '0' represents like, 
                     '1' represents dislike

*/

drop database if exists fastrag;
create database fastrag;
use fastrag;

drop table if exists feedback;
create table feedback
(
    feedback_id tinyint unsigned primary key auto_increment,
    question varchar(1000),
    answer varchar(3000),
    feedback_result tinyint,
    feedback_time datetime,
    comments varchar(2000),
    unique (feedback_id)
);