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

import requests
from urllib.parse import urlparse, urlunparse
import multiprocessing
import urllib3
import langid
from bs4 import BeautifulSoup
import os
import re
from .context_utils import uni_pro
import logging

logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)
urllib3.disable_warnings()


class Crawler:
    def __init__(self, pool=None):
        if pool:
            assert isinstance(pool, (str, list, tuple)), 'url pool should be str, list or tuple'
        self.pool = pool
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng, \
            */*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, \
            like Gecko) Chrome/113.0.0.0 Safari/537.36'
        }
        self.fetched_pool = set()

    def get_sublinks(self, soup):
        sublinks = []
        for links in soup.find_all('a'):
            sublinks.append(str(links.get('href')))
        return sublinks

    def get_hyperlink(self, soup, base_url):
        sublinks = []
        for links in soup.find_all('a'):
            link = str(links.get('href'))
            if link.startswith('#') or link is None or link == 'None':
                continue
            suffix = link.split('/')[-1]
            if '.' in suffix and suffix.split('.')[-1] not in ['html', 'htmld']:
                continue
            link_parse = urlparse(link)
            base_url_parse = urlparse(base_url)
            if link_parse.path == '':
                continue
            if link_parse.netloc != '':
                # keep crawler works in the same domain
                if link_parse.netloc != base_url_parse.netloc:
                    continue
                sublinks.append(link)
            else:
                sublinks.append(urlunparse((base_url_parse.scheme,
                                            base_url_parse.netloc,
                                            link_parse.path,
                                            link_parse.params,
                                            link_parse.query,
                                            link_parse.fragment)))
        return sublinks

    def fetch(self, url, headers=None, max_times=5):
        if not headers:
            headers = self.headers
        while max_times:
            if not url.startswith('http') or not url.startswith('https'):
                url = 'http://' + url
            logging.info('start fetch %s...', url)
            try:
                response = requests.get(url, headers=headers, verify=True)
                if response.status_code != 200:
                    logging.error('fail to fetch %s, response status code: %s', url, response.status_code)
                else:
                    return response
            except Exception as e:
                logging.error('fail to fetch %s, caused by %s', url, e)
            max_times -= 1
        return None

    def process_work(self, sub_url, work):
        response = self.fetch(sub_url)
        if response is None:
            return []
        self.fetched_pool.add(sub_url)
        soup = self.parse(response.text)
        base_url = self.get_base_url(sub_url)
        sublinks = self.get_hyperlink(soup, base_url)
        if work:
            work(sub_url, soup)
        return sublinks

    def crawl(self, pool, work=None, max_depth=10, workers=10):
        url_pool = set()
        for url in pool:
            base_url = self.get_base_url(url)
            response = self.fetch(url)
            soup = self.parse(response.text)
            sublinks = self.get_hyperlink(soup, base_url)
            self.fetched_pool.add(url)
            url_pool.update(sublinks)
            depth = 0
            while len(url_pool) > 0 and depth < max_depth:
                logging.info('current depth %s...', depth)
                mp = multiprocessing.Pool(processes=workers)
                results = []
                for sub_url in url_pool:
                    if sub_url not in self.fetched_pool:
                        results.append(mp.apply_async(self.process_work, (sub_url, work)))
                mp.close()
                mp.join()
                url_pool = set()
                for result in results:
                    sublinks = result.get()
                    url_pool.update(sublinks)
                depth += 1

    def parse(self, html_doc):
        soup = BeautifulSoup(html_doc, 'lxml')
        return soup

    def download(self, url, file_name):
        logging.info('download %s into %s...', url, file_name)
        try:
            r = requests.get(url, stream=True, headers=self.headers, verify=True)
            f = open(file_name, "wb")
            for chunk in r.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)
        except Exception as e:
            logging.error('fail to download %s, caused by %s', url, e)

    def get_base_url(self, url):
        result = urlparse(url)
        return urlunparse((result.scheme, result.netloc, '', '', '', ''))

    def clean_text(self, text):
        text = text.strip().replace('\r', '\n')
        text = re.sub(' +', ' ', text)
        text = re.sub('\n+', '\n', text)
        text = text.split('\n')
        return '\n'.join([i for i in text if i and i != ' '])


def load_html_data(url):
    crawler = Crawler()
    res = crawler.fetch(url)
    if res == None:
        return None
    soup = crawler.parse(res.text)
    all_text = crawler.clean_text(soup.select_one('body').text)
    main_content = ''
    for element_name in ['main', 'container']:
        main_block = None
        if soup.select(f'.{element_name}'):
            main_block = soup.select(f'.{element_name}')
        elif soup.select(f'#{element_name}'):
            main_block = soup.select(f'#{element_name}')
        if main_block:
            for element in main_block:
                text = crawler.clean_text(element.text)
                if text not in main_content:
                    main_content += f'\n{text}'
            main_content = crawler.clean_text(main_content)

    main_content = main_content.replace('\n', '')
    main_content = main_content.replace('\n\n', '')
    main_content = uni_pro(main_content)
    main_content = re.sub(r'\s+', ' ', main_content)

    # {'text': all_text, 'main_content': main_content}

    return main_content
