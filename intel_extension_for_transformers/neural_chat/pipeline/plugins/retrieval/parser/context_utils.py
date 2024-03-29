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
from langchain_community.document_loaders import UnstructuredMarkdownLoader
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
        elif '\u4E00' <= char <= '\u9FFF':
            filtered_text += char
        elif ('\u3400' <= char <= '\u4DBF'  # CJK Unified Ideographs Extension A
          or '\u20000' <= char <= '\u2A6DF'  # CJK Unified Ideographs Extension B
          or '\u2A700' <= char <= '\u2B73F'  # CJK Unified Ideographs Extension C
          or '\u2B740' <= char <= '\u2B81F'  # CJK Unified Ideographs Extension D
          or '\u2B820' <= char <= '\u2CEAF'  # CJK Unified Ideographs Extension E
          or '\uF900' <= char <= '\uFAFF'  # CJK Compatibility Ideographs
          or '\u2F800' <= char <= '\u2FA1F'):
            filtered_text += char

    return filtered_text


def read_pdf(pdf_path, table_strategy, table_summary_model_name_or_path):
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
    tables_result = get_tables_result(pdf_path, table_strategy, table_summary_model_name_or_path)
    return result, tables_result


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

def load_unstructured_data(input, table_strategy, table_summary_model_name_or_path):
    """Load unstructured context."""
    tables = None
    if input.endswith("pdf"):
        text, tables = read_pdf(input, table_strategy, table_summary_model_name_or_path)
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
    return text, tables

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


def get_tables_result(pdf_path, table_strategy, table_summary_model_name_or_path):
    """Extract tables information from pdf file."""
    if table_strategy == 'fast':
        return None
    
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import FigureCaption
    from intel_extension_for_transformers.neural_chat.models.model_utils import predict
    from intel_extension_for_transformers.neural_chat.prompts.prompt import TABLESUMMARY_PROMPT

    tables_result = []
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
    )
    tables = [el for el in raw_pdf_elements if el.category == "Table"]
    for table in tables:
        table_coords = table.metadata.coordinates.points
        content = table.metadata.text_as_html
        table_page_number = table.metadata.page_number
        min_distance = float('inf')
        table_summary = None
        if table_strategy == 'hq':
            for element in raw_pdf_elements:
                if isinstance(element, FigureCaption) or element.text.startswith('Tab'):
                    caption_page_number = element.metadata.page_number
                    caption_coords = element.metadata.coordinates.points
                    related, y_distance = get_relation(table_coords, caption_coords, \
                                                        table_page_number, caption_page_number)
                    if related:
                        if y_distance < min_distance:
                            min_distance = y_distance
                            table_summary = element.text
            if table_summary is None:
                parent_id = table.metadata.parent_id
                for element in raw_pdf_elements:
                    if element.id == parent_id:
                        table_summary = element.text
                        break
        elif table_strategy == 'llm':
            prompt = TABLESUMMARY_PROMPT.format(table_content=content)
            params = {}
            params["model_name"] = table_summary_model_name_or_path
            params["prompt"] = prompt
            params["temperature"] = 0.8
            params["top_p"] = 0.9
            params["top_k"] = 40
            params["max_new_tokens"] = 1000
            params["num_beams"] = 2
            params["num_return_sequences"] = 2
            params["use_cache"] = True
            table_summary = predict(**params)
            table_summary = table_summary[table_summary.find('### Generated Summary:\n'):]
            table_summary = re.sub('### Generated Summary:\n', '', table_summary)
        elif table_strategy == None:
            table_summary = None
        if table_summary is None:
            text = f'[Table: {content}]'
        else:
            text = f'|Table: [Summary: {table_summary}], [Content: {content}]|'
        tables_result.append([text, pdf_path])
    return tables_result


def get_relation(table_coords, caption_coords, table_page_number, caption_page_number, threshold=100):
    """Get the relation of a pair of table and caption"""
    same_page = table_page_number == caption_page_number
    x_overlap = (min(table_coords[2][0], caption_coords[2][0]) - max(table_coords[0][0], caption_coords[0][0])) > 0
    if table_coords[0][1] - caption_coords[1][1] >= 0:
        y_distance = table_coords[0][1] - caption_coords[1][1]
    elif caption_coords[0][1] - table_coords[1][1] >= 0:
        y_distance = caption_coords[0][1] - table_coords[1][1]
    else:
        y_distance = 0
    y_close = y_distance < threshold
    return same_page and x_overlap and y_close, y_distance
