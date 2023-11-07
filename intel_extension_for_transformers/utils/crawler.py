import os
import re
import json
import logging
import requests
from urllib.parse import urlparse, urlunparse
import multiprocessing
import urllib3

import langid
import PyPDF2
from bs4 import BeautifulSoup

urllib3.disable_warnings()

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def load_pdf(pdf_path):
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)
    text = ''
    for num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[num]
        text += page.extract_text()
    return text

class Crawler:
    def __init__(self, pool=None):
        if pool:
            assert isinstance(pool, (str, list, tuple)), 'url pool should be str, list or tuple'
        self.pool = pool
        self.headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
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
            logger.info(f'start fetch {url}...')
            try:
                response = requests.get(url, headers=headers, verify=False)
                if response.status_code != 200:
                    logger.error(f'fail to fetch {url}, respose status code: {response.status_code}')
                else:
                    return response
            except Exception as e:
                logger.error(f'fail to fetch {url}, cased by {e}')
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
                logger.info(f'current depth {depth} ...')
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
        logger.info(f'download {url} into {file_name}...')
        try:
            r = requests.get(url, stream=True, headers=self.headers, verify=False)
            f = open(file_name, "wb")
            for chunk in r.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)
        except Exception as e:
            logger.error(f'fail to download {url}, cased by {e}')

    def get_base_url(self, url):
        result = urlparse(url)
        return urlunparse((result.scheme, result.netloc, '', '', '', ''))

    def clean_text(self, text):
        text = text.strip().replace('\r', '\n')
        text = re.sub(' +', ' ', text)
        text = re.sub('\n+', '\n', text)
        text = text.split('\n')
        return '\n'.join([i for i in text if i and i != ' '])


class GithubCrawler(Crawler):
    def __init__(self, *args, output_dir='./crawl_result', **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_md_url(self, soup):
        md_url_pool = set()
        sublinks = self.get_sublinks(soup)
        for link in sublinks:
            if not link.startswith('#') and link.endswith('.md'):
                md_url_pool.add(link)
        return md_url_pool

    def _get_row_url(self, soup):
        elements = soup.select('div[role=rowheader] a')
        url_pools = set()
        for element in elements:
            sub_url = element.get('href')
            if '.' not in sub_url.split('/')[-1] and element.text.split() != ['.'] * 2:
                url_pools.add(sub_url)
        return url_pools

    def _get_new_pool(self, url):
        base_url = self.get_base_url(url)
        response = self.fetch(url)
        soup = self.parse(response.text)
        return set([base_url + i for i in self._get_md_url(soup)]), set([base_url + i for i in self._get_row_url(soup)])

    def _update_pool(self, url, md_url_pool, url_pool):
        base_url = self.get_base_url(url)
        response = self.fetch(url)
        soup = self.parse(response.text)
        md_url_pool.update(set([base_url + i for i in self._get_md_url(soup)]))
        url_pool.update(set([base_url + i for i in self._get_row_url(soup)]))

    def get_md_content(self, url):
        response = self.fetch(url)
        soup = self.parse(response.text)
        element = soup.select_one('article')
        output = []
        if element is None:
            return url, ''
        for line in element.text.split('\n'):
            if line:
                res = element.find(lambda e: e.text == line and e.name.startswith('h'))
                if res:
                    line = res.name + ':' + line
                output.append(line)
        return url, '\n'.join(output)

    def start(self, max_depth=10, workers=10):
        if isinstance(self.pool, str):
            self.pool = [self.pool]
        md_url_pool = set()
        url_pool = set()
        fetched_url = set()

        for url in self.pool:
            self._update_pool(url, md_url_pool, url_pool)
            fetched_url.add(url)
            depth = 0
            while len(url_pool) > 0 and depth < max_depth:
                logger.info(f'current depth {depth} ...')
                result = []
                mp = multiprocessing.Pool(processes=workers)
                for sub_url in url_pool:
                    if sub_url not in fetched_url:
                        result.append(mp.apply_async(self._get_new_pool, (sub_url, )))
                        fetched_url.add(sub_url)
                mp.close()
                mp.join()
                url_pool = set()
                for res in result:
                    mp, up = res.get()
                    url_pool.update(up)
                    md_url_pool.update(mp)
                depth += 1

        result = []
        mp = multiprocessing.Pool(processes=workers)
        for url in md_url_pool:
            result.append(mp.apply_async(self.get_md_content, (url,)))
        mp.close()
        mp.join()
        idx = 0
        idx_dict = {}
        for res in result:
            url, content = res.get()
            idx_dict[idx] = url
            with open(os.path.join(self.output_dir, f'{idx}.txt'), 'w') as f:
                f.write(content)
            idx += 1
        json.dump(idx_dict, open(os.path.join(self.output_dir, 'index.json'), 'w'))


class CircuitCrawler(Crawler):
    def __init__(self, *args, output_dir='./crawl_result', **kwargs):
        super().__init__(*args, **kwargs)
        self.headers.update({
            "Cache-Control": "max-age=0",
            "Cookie": "access_token=z770ZFQEIdGf6czCwsoy2nokcozM4NeHoYwnhg5hGaI; _ga=GA1.2.2073446506.1658713529; IDSID=hengguo; BadgeType=BB; isManager=N; CNCampusCode=SHZ; isMenuVisible=1; ajs_group_id=null; ajs_user_id=%2240696693%22; ajs_anonymous_id=%22cf578c2c-07ec-4ac3-b1eb-a1131cf24ff4%22; s_fid=26C8F0AB24F5B476-298199E6962D1F6F; ELQSTATUS=OK; ELOQUA=GUID=906697F39E824DBF885ED50A398B150B; _cs_c=0; _abck=B51103F18F0989A871324DCB7C6A6228~-1~YAAQdqwsF5kmlAGDAQAA038pCwjsSQiWJnovwNl5wtPLCZzkLadrEmXQKx9j++9Eua1pwXeSsMPz2GOl772NQF+sSQDc+ML6qClNJ1jJTNoZ2NCGE7+90w6B3zWTyn8mLJh+L/Upuj4GCh74hk6lpBHYExXMokSFYbaAKLGw2/4vfCWsZ+XXfUEnymWmb+27volGnYjUDbPFD7Pv8JkmC3BSdmImqbgKCb2w3zkmM9Q3OPQ4J82PLuAzHpMi4uBBficY2apnttO099jggehPhvOqSyQFNopkt6lQx65Cb6Ww7SSGUJE2+bsKXZzoTpQQENAlRxoyNNJIEmpFoTHDXbofGNgRhpF0dWbtUbwUFC2FZ8mr30NdoHcpbg==~-1~-1~-1; AMCV_AD2A1C8B53308E600A490D4D%40AdobeOrg=1585540135%7CMCIDTS%7C19252%7CMCMID%7C75310779783706370332434931749805207698%7CMCAAMLH-1663919541%7C3%7CMCAAMB-1663919541%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1663321941s%7CNONE%7CvVersion%7C4.4.0%7CMCCIDH%7C2131965003; adcloud={%22_les_v%22:%22y%2Cintel.com%2C1663316542%22}; mbox=PC#e0741b84312444dfabf1a6673620ca2c.35_0#1724471070|session#e23cd74f92354c4bb56c59d32b9fb2b3#1663316604; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Mar+28+2023+15%3A07%3A05+GMT%2B0800+(China+Standard+Time)&version=202209.2.0&isIABGlobal=false&hosts=&consentId=3cfefc52-a9fe-4d46-a57f-125e1a84a4cb&interactionCount=0&landingPath=https%3A%2F%2Fwww.intel.com%2Fcontent%2Fwww%2Fus%2Fen%2Fdeveloper%2Ftopic-technology%2Fartificial-intelligence%2Fdeep-learning-boost.html&groups=C0001%3A1%2CC0003%3A0%2CC0004%3A0%2CC0002%3A0; intelresearchUID=9098949724218M1679987234236; _cs_id=81508dff-8c83-a140-f131-314d094ea9f8.1661226277.40.1679989156.1679989156.1589385054.1695390277819; utag_main=v_id:0182c8cd1abd0095db8d333c6c600506f003a06700978$_sn:26$_se:1$_ss:1$_st:1680741698802$wa_ecid:75310779783706370332434931749805207698$wa_erpm_id:12253843$ses_id:1680739898802%3Bexp-session$_pn:1%3Bexp-session; kndctr_AD2A1C8B53308E600A490D4D_AdobeOrg_identity=CiY3NTMxMDc3OTc4MzcwNjM3MDMzMjQzNDkzMTc0OTgwNTIwNzY5OFIPCKPttMasMBgBKgRKUE4z8AHxk5%2Df9TA%3D; BIGipServerlbauto-prdeppubdisplb-443=!roIjSuTSMi+UC5p9e6x3zjaYWx2wLhvVQQPHelzO52Ib4EDhyKb8c4FNX+55yubaT8TzQ9VwSD19ANY=; login-token=97d9b5d9-9341-41c7-a130-17f60d0a1997%3a3820bfd7-dfcd-459a-991d-dca4db9d22e7_25a1903af751648a8f95a05b09902c56%3acrx.default; JSESSIONID=node0ujxl1vh1zity12xk7icbuynp8109692.node0; _gid=GA1.2.349545364.1687825459; _gat=1",
            "Referer": "https://login.microsoftonline.com/",
            "Host": "circuit.intel.com",
            "Sec-Ch-Ua": 'Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "Windows",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        })
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f'{self.output_dir}/text', exist_ok=True)
        os.makedirs(f'{self.output_dir}/pdf', exist_ok=True)
        self.fetched_pool = set()
        self.root = 'circuit.intel.com'

    def _work(self, url, soup):
        # element = soup.find('body')
        useless_domain = ['content/it']
        if self.root not in url:
            return
        for i in useless_domain:
            if i in url:
                return
        element = soup.find('main')
        if not element:
            element = soup.find('body')
        text = element.text
        text = self.clean_text(text)

        file_name = soup.select_one('head > title')
        if file_name:
            file_name = file_name.text
        else:
            file_name = url.split('/')[-1].split('.')[0]
        file_name = file_name.replace('/', '|').replace(' ', '_')
        file_path = f'{self.output_dir}/text/{file_name}.json'
        idx = 0
        while os.path.exists(file_path):
            file_path = f'{self.output_dir}/text/{file_name}_{idx}.json'
            idx += 1
        if langid.classify(text)[0] in ['en', 'zh']:
            json_str = {'content': text, 'link': url}
            json.dump(json_str, open(file_path, 'w'))

        sublinks = self.get_sublinks(soup)
        base_url = self.get_base_url(url)
        for link in sublinks:
            if link.startswith('/'):
                link = base_url + link
            if link.endswith('pdf'):
                if self.root not in link:
                    continue
                file_name = link.split('/')[-1]
                file_name = file_name.replace('/', '|').replace(' ', '_')
                pdf_file = f'{self.output_dir}/pdf/{file_name}'
                self.download(link, pdf_file)
                try:
                    pdf_content = load_pdf(pdf_file)
                    if langid.classify(pdf_content)[0] in ['en', 'zh']:
                        json_str = {'content': pdf_content, 'link': link}
                        file_path = f'{self.output_dir}/text/{file_name}.json'
                        idx = 0
                        while os.path.exists(file_path):
                            file_path = f'{self.output_dir}/text/{file_name}_{idx}.json'
                            idx += 1
                        json.dump(json_str, open(file_path, 'w'))
                except:
                    logger.error(f'fail to load pdf file {pdf_file}...')

    def start(self, max_depth=10, workers=10):
        if isinstance(self.pool, str):
            self.pool = [self.pool]
        self.crawl(self.pool, self._work, max_depth=max_depth, workers=workers)


class LinkedinCrawler(Crawler):
    def __init__(self, pool=None):
        self.host = 'www.linkedin.com'
        self.content_url = 'https://www.linkedin.com/voyager/api/graphql' 
        self.post_query_id = 'voyagerIdentityDashProfileCards.2c28912ec581699a30693e7fd7380517'
        self.info_query_id = 'voyagerIdentityDashProfileCards.dcb183342cf3fe4b94a0575f63d4d3c2'
        super().__init__(pool)
        self.headers = {
            'Host': 'www.linkedin.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cookie': 'lang=v=2&lang=en-us; bcookie="v=2&a9d130b0-0b70-480d-837d-41d9c9d9d166"; bscookie="v=1&20231107015950855a8832-612c-4de6-8046-8da66df823d5AQEWZmtArw944ERRCg521bhJDlz8lSuC"; lidc="b=TB51:s=T:r=T:a=T:p=T:g=3849:u=5:x=1:i=1699322931:t=1699401318:v=2:sig=AQEHJE3022jXDSnVBKwOJAmqk8NDwz-9"; JSESSIONID="ajax:7564268215383886716"; AMCV_14215E3D5995C57C0A495C55%40AdobeOrg=-637568504%7CMCIDTS%7C19669%7CMCMID%7C39475338270328582223082987596741338300%7CMCAAMLH-1699927249%7C9%7CMCAAMB-1699927249%7C6G1ynYcLPuiQxYZrsz_pkqfLG9yMXBpb2zX5dvJdYQJzPXImdj0y%7CMCOPTOUT-1699329649s%7CNONE%7CvVersion%7C5.1.1%7CMCCIDH%7C-1231369551; AMCVS_14215E3D5995C57C0A495C55%40AdobeOrg=1; aam_uuid=38931795198557096053028626623890003831; liap=true; li_at=AQEDAUgZ90MAo0pRAAABi6eCn44AAAGLy48jjk0Af4s1H8RjCTmHZc_OPf1ooOi5jIthoeefjUf1hNz31HlFl0MMiEMQt-r9UAeinhgoVc09lWPabEcUY3tacnPRs7ZzJCK2HEa_t3TVID9oN9xJTlnz; timezone=Asia/Shanghai; li_theme=light; li_theme_set=app; UserMatchHistory=AQJQQxmH85BRgQAAAYuniiINJJfreOs7_-0pkEnPXufpA0JCJVuHlst_pYhfItedDZav_420X8H5r0A-ePVRbze2Id93MNeIz_FgxRHZqxbzkCwMjClWlixPwkdMhR84ulgzDsa00sBmUaVe1NKzLlIC76knqdAQzji3qroBOf0qG9Wio3V-NYiROU_3GQQkY_FoMt3LPwM1wkO2jZ5nZpnaJHmTK3_mSf1cbZooYDs5xxrGNFWoOxCaLcY5t-Ie4tpjzW8wqO5J7RLBXX6nDwbhd5t3MYAmCpIueVLwFhcCaeNEu6ubAh9N7anoNjtAcgw74PY; li_sugr=f231854b-b3ee-4dc4-822b-e7515682ca77; _guid=d49a1ab6-12a8-4324-bfe9-84cdcc43575e; AnalyticsSyncHistory=AQKHFr6wR5q4MgAAAYungsnamCuHsoMvvyeyEMqS3ZXIsZfUG6bydPUG2jWzwnwMAlk1z8Un62Ae9LchAvwtWg; lms_ads=AQGNXoF4n7CifwAAAYungs551K_6w-nDwsMzcB8sJCuieZXTXfB1Z9Z2dD7DSvtoJf7CZFRm57j8IP6oWF4UylxFVfKjIQUq; lms_analytics=AQGNXoF4n7CifwAAAYungs551K_6w-nDwsMzcB8sJCuieZXTXfB1Z9Z2dD7DSvtoJf7CZFRm57j8IP6oWF4UylxFVfKjIQUq; __cf_bm=14mZJG4maLfMFPWADhdn1zK4y3BnExPISXCQrDIaXXM-1699322464-0-AbtOOimHR7OjfQ1uCzuk1TmSWSBTtJosU15M4l7gxMtQRTofKKNqEFpdrHg+RcSy4GmnIlXPsGrrs58hcDZuJxU=',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'TE': 'trailers'
        }

    def start(self):
        res = self.fetch(self.pool)
        pattern = re.compile(',"\\*elements":\\["(.*?)"\\]')
        soup = BeautifulSoup(res.text, 'lxml')
        self.profileUrn = pattern.search(soup.prettify()).group(1)
        print('profileUrn = ', self.profileUrn)

        # fetch infomation
        info_url = f'{self.content_url}?includeWebMetadata=true&variables=(profileUrn:{self.profileUrn})&=&queryId={self.info_query_id}'
        new_headers = {
            'Host': 'www.linkedin.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0',
            'Accept': 'application/vnd.linkedin.normalized+json+2.1',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'x-li-lang': 'en_US',
            'x-li-track': '{"clientVersion":"1.13.6326","mpVersion":"1.13.6326","osName":"web","timezoneOffset":8,"timezone":"Asia/Shanghai","deviceFormFactor":"DESKTOP","mpName":"voyager-web","displayDensity":1,"displayWidth":2560,"displayHeight":1440}',
            'x-li-page-instance': 'urn:li:page:d_flagship3_profile_view_base;v3eYA8WURJKGZtIxSBvt3w==',
            'csrf-token': 'ajax:7564268215383886716',
            'x-restli-protocol-version': '2.0.0',
            'x-li-pem-metadata': 'Voyager - Profile=profile-tab-initial-cards',
            'Connection': 'keep-alive',
            'Referer': self.pool,
            'Cookie': self.headers['Cookie'],
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'TE': 'trailers'
        }
        res = self.fetch(info_url, headers=new_headers, max_times=1)
        breakpoint()

    def login(self, username, password):
        url = 'https://www.linkedin.com'
        querystring = {'trk': 'brandpage_baidu_pc-mainlink'}
        headers = {
            'authority': "www.linkedin.com",
            'cache-control': 'max-age=0,no-cache',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'referer': 'https://www.baidu.com/s?ie=UTF-8&wd=%E9%A2%86%E8%8B%B1',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        res = requests.request('GET', url, headers=headers, params=querystring)
        cookie = ""
        csrf_token = ""
        for c in res.cookies:
            if c.name == 'JSESSIONID':
                csrf_token = c.value
            cookie = cookie + c.name + '=' + c.value + ';'
        soup = BeautifulSoup(res.text, 'lxml')
        # loginCsrfParam
        csrf = soup.select_one('input[name=loginCsrfParam]')['value']
        index_login_data = {
            'cookie': cookie,
            'csrf': csrf,
            'csrf_token': csrf_token
        }

        url = 'https://www.linkedin.com/uas/login-submit'
        querystring = {'loginSubmitSource': 'GUEST_HOME'}
        payload = f"session_key={username}&session_password={password}&isJsEnabled=false&loginCsrfParam={csrf}&fp_data=default&undefined="
        headers.update(
            {
                'referer': 'https://www.linkdin.com/',
                'origin': 'https://www.linkdin.com/',
                'content-type': 'application/x-www-form-urlencoded',
                'cookie': cookie
                })
        requests.request('POST', url, data=payload, headers=headers, params=querystring)
        cookie = ""
        csrf_token = ""
        for c in res.cookies:
            if c.name == 'JSESSIONID':
                csrf_token = c.value
            cookie = cookie + c.name + '=' + c.value + ';'
        return cookie, csrf_token
        


if __name__ == '__main__':
    url = 'https://www.linkedin.com/in/wei-li-sf/'
    c = LinkedinCrawler(url)
    c.start()
   


