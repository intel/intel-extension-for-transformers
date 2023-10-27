/*
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
    question varchar(500),
    answer varchar(500),
    feedback_result tinyint,
    feedback_time datetime,
    unique (feedback_id)
);