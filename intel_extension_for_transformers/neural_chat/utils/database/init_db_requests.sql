/*
Create the Chatbot Database: requests
Create empty Tables: requests
*/

drop database if exists requests;
create database requests;
use requests;

drop table if exists record;
create table record
(
    record_id tinyint unsigned primary key auto_increment,
    request_url varchar(500),
    request_body varchar(2000),
    user_id varchar(100),
    captured_time datetime,
    unique (record_id)
);