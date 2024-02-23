# !/usr/bin/env python
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

import unicodedata
import pandas as pd
import re, json
from langchain.document_loaders import UnstructuredMarkdownLoader
from docx import Document as DDocument
from bs4 import BeautifulSoup
import fitz
import easyocr
from PIL import Image
import numpy as np
import io

def uni_pro(text):
    """Check if the character is ASCII or falls in the category of non-spacing marks."""
    normalized_text = unicodedata.normalize('NFKD', text)
    filtered_text = ''
    for char in normalized_text:
        if ord(char) < 128 or unicodedata.category(char) == 'Mn':
            filtered_text += char
    return filtered_text


def read_pdf(pdf_path):
    """Read the pdf file."""
    doc = fitz.open(pdf_path)
    reader = easyocr.Reader(['en'])
    result =''
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pagetext = page.get_text().strip()
        if pagetext:
            if pagetext.endswith('!') or pagetext.endswith('?') or pagetext.endswith('.'):
                result=result+pagetext
            else:
                result=result+pagetext+'.'
        if len(doc.get_page_images(i)) > 0 :
            for img in doc.get_page_images(i):
                if img:
                    pageimg=''
                    xref = img[0]
                    img_data = doc.extract_image(xref)
                    img_bytes = img_data['image']
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    img = np.array(pil_image)
                    img_result = reader.readtext(img, paragraph=True, detail=0)
                    pageimg=pageimg + ', '.join(img_result).strip()
                    if pageimg.endswith('!') or pageimg.endswith('?') or pageimg.endswith('.'):
                        pass
                    else:
                        pageimg=pageimg+'.'
                result=result+pageimg
    return result


def read_html(html_path):
    """Read the html file."""
    with open(html_path, 'r', encoding="utf-8") as file:
        html = file.read()
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=True)
    return text


def read_txt(txt_path):
    """Read txt file."""
    with open(txt_path, 'r') as file:
        text = file.read()
    return text


def read_docx(doc_path):
    """Read docx file."""
    doc = DDocument(doc_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text


def read_md(md_path):
    """Read docx file."""
    loader = UnstructuredMarkdownLoader(md_path)
    text = loader.load()[0].page_content
    return text


def load_json(input, process, max_length, min_length):
    """Load and process json file."""
    data = []
    with open(input, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    new_sens = []
    new_collect = []
    for sub in data:
        sub['content'].replace('#', " ")
        sub['content'] = re.sub(r'\s+', ' ', sub['content'])
        if not process:
            if len(sub['content']) < min_length:
                continue
            new_doc = [sub['content'], sub['link']]
            new_collect.append(new_doc)
        else:
            for sub in data:
                sub['content'].replace('#', " ")
                if len(sub['content'])<min_length:
                    continue
                split_sen = re.split(r'[.?!]', sub['content'])
                for num in range(len(split_sen)):
                    split_sen[num] = re.sub(r'\s+', ' ', split_sen[num])
                    if num +1 < len(split_sen):
                        if len(split_sen[num]) >max_length:
                            new_sens.append(split_sen[num].strip())
                        else:
                            split_sen[num +1] =split_sen[num] +split_sen[num+1]
                    else:
                        new_sens.append(split_sen[num])

            paragraphs = list(set(new_sens))
            for paragraph in paragraphs:
                new_doc = [paragraph, sub['link']]
                new_collect.append(new_doc)
    return new_collect


def load_xlsx(input):
    """Load and process xlsx file."""
    df = pd.read_excel(input)
    header = df.columns.tolist()
    all_data = []
    if 'Questions' in header and 'Answers' in header:
        for index, row in df.iterrows():
            sub = "User Query: " + row['Questions'] + "Answer: " + row["Answers"]
            sub=sub.replace('#', " ")
            sub = sub.replace(r'\t', " ")
            sub = sub.replace('\n', ' ')
            sub = sub.replace('\n\n', ' ')
            sub = re.sub(r'\s+', ' ', sub)
            new_doc = [sub, input]
            all_data.append(new_doc)
    elif 'question' in header and 'answer' in header and 'link' in header:
        for index, row in df.iterrows():
            sub = "Question: " + row['question'] + " Answer: " + row["answer"]
            sub = sub.replace('#', " ")
            sub = sub.replace(r'\t', " ")
            sub = sub.replace('\n', ' ')
            sub = sub.replace('\n\n', ' ')
            sub = re.sub(r'\s+', ' ', sub)
            all_data.append([sub, row['link']])
    elif 'context' in header and 'link' in header:
        for index, row in df.iterrows():
            sub = row['context']
            sub = sub.replace('#', " ")
            sub = sub.replace(r'\t', " ")
            sub = sub.replace('\n', ' ')
            sub = sub.replace('\n\n', ' ')
            sub = re.sub(r'\s+', ' ', sub)
            all_data.append([sub, row['link']])
    return all_data

def load_csv(input):
    """ Load the csv file."""
    df = pd.read_csv(input)
    all_data = []
    documents = []
    for index, row in df.iterrows():
        sub = "User Query: " + row['question'] + "Answer: " + row["correct_answer"]
        all_data.append(sub)

    for data in all_data:
        data.replace('#', " ")
        data = re.sub(r'\s+', ' ', data)
        new_doc = [data, input]
        documents.append(new_doc)
    return documents

def load_structured_data(input, process, max_length, min_length):
    """Load structured context."""
    if input.endswith("jsonl") or input.endswith("json"):
        content = load_json(input, process, max_length, min_length)
    elif input.endswith("xlsx"):
        content = load_xlsx(input)
    elif input.endswith("csv"):
        content = load_csv(input)
    return content

def load_unstructured_data(input):
    """Load unstructured context."""
    if input.endswith("pdf"):
        text = read_pdf(input)
    elif input.endswith("docx"):
        text = read_docx(input)
    elif input.endswith("html"):
        text = read_html(input)
    elif input.endswith("txt"):
        text = read_txt(input)
    elif input.endswith("md"):
        text = read_md(input)

    text = text.replace('\n', ' ')
    text = text.replace('\n\n', ' ')
    text = uni_pro(text)
    text = re.sub(r'\s+', ' ', text)
    return text

def get_chuck_data(content, max_length, min_length, input):
    """Process the context to make it maintain a suitable length for the generation."""
    sentences = re.split('(?<=[!.?])', content)

    paragraphs = []
    current_length = 0
    count = 0
    current_paragraph = ""
    for sub_sen in sentences:
        count +=1
        sentence_length = len(sub_sen)
        if current_length + sentence_length <= max_length:
            current_paragraph += sub_sen
            current_length += sentence_length
            if count == len(sentences) and len(current_paragraph.strip())>min_length:
                paragraphs.append([current_paragraph.strip() ,input])
        else:
            paragraphs.append([current_paragraph.strip() ,input])
            current_paragraph = sub_sen
            current_length = sentence_length

    return paragraphs
