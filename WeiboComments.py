import datetime
import requests
import re
import os
from fake_useragent import UserAgent


class WeiboCommentCrawler:

    #默认参数
    # __default_headers={
    #     "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Mobile Safari/537.36",
    #     "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    #     "accept-encoding": "gzip, deflate, br",
    # }

    __default_headers = {
        "User-Agent": UserAgent().random,
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-encoding": "gzip, deflate, br",
    }

    def __init__(self) -> None:
        pass

    # 将新浪微博时间转换作标准格式
    @classmethod
    def __trans_time(self,v_str):
        # 转换GMT到标准格式
        GMT_FORMAT = '%a %b %d %H:%M:%S +0800 %Y'
        timeArray = datetime.datetime.strptime(v_str, GMT_FORMAT)
        ret_time = timeArray.strftime('%Y-%m-%d %H:%M:%S')
        return ret_time


    



    # 获取since_id进行翻页
    @classmethod
    def __get_since_id(self,header, user_id, l_id, contain_id, since_id):
        global min_since_id
        topic_url = 'https://m.weibo.cn/api/container/getIndex?uid={}&luicode=10000011&lfid={}&type=uid&value={}&containerid={}'.format(
            user_id, l_id, user_id, contain_id)
        topic_url += '&since_id=' + str(since_id)
        # print(topic_url)
        result = requests.get(topic_url, headers=header)
        json = result.json()
        items = json.get('data').get('cardlistInfo')
        # print(items)
        min_since_id = items['since_id']
        return min_since_id


    # 获取每一页的内容
    @classmethod
    def __get_page(self,header, user_id, l_id, contain_id, since_id=''):
        li = []
        url = 'https://m.weibo.cn/api/container/getIndex?uid={}&luicode=10000011&lfid={}&type=uid&value={}&containerid={}'.format(
            user_id, l_id, user_id, contain_id)
        url += '&since_id={}'.format(since_id)
        # print(url)
        result = requests.get(url, headers=header)
        try:
            if result.status_code == 200:
                li.append(result.json())
                # print(result.text)
        except requests.ConnectionError as e:
            print('Error', e.args)
        return li


    # 进行数据清洗
    @classmethod
    def __rebuild(self,data):
        dr = re.compile(r'<[^>]+>')
        dt = re.compile(r'转发微博')
        li = []
        text_list = data[0]['data']['cards']
        for j in text_list:
            text_data = j['mblog']['text']
            text_data = dr.sub('', text_data)
            text_data = dt.sub('', text_data)
            if text_data != '':
                li = [self.__trans_time(j['mblog']['created_at']), text_data]
        return li



    #==================================对外接口=========================================
    # 补充内容：
    # 1. 获取用户id
    @classmethod
    def get_user_name(self, user_id, contain_id,header=__default_headers):
        url = 'https://m.weibo.cn/api/container/getIndex?type=uid&value={}&containerid={}'.format(user_id, contain_id)
        result = requests.get(url, headers=header)
        json = result.json()
        user_name = json['data']['cards'][1]['mblog']['user']['screen_name']
        return user_name


    # 2. 保存为文本文件
    @classmethod
    def save_file(self,filename, text):
        folder = os.path.exists('./cache')
        if not folder:
            os.mkdir('./cache')
        with open('./cache/{}.txt'.format(filename), 'w', encoding='utf-8') as file:
            # line = 'time,text'
            # file.write(line + '\n')
            for row in text:
                line = ','.join(str(item) for item in row)
                file.write(line + '\n')
        print('数据已保存')
    # 集成函数
    
    # headers：HTTP请求头，包含了请求的一些元数据，如用户代理、授权信息等。在发送请求时，需要将适当的请求头信息包含在其中，以便与服务器进行通信。
    # uid[i]：微博用户的唯一标识符。每个微博用户都有一个独特的用户ID，用于标识用户的身份。
    # l_fid[i]：微博的唯一标识符。每条微博都有一个独特的微博ID，用于标识微博的内容。
    # container_id[i]：微博容器的唯一标识符。微博容器是一个包含微博及其相关内容的容器，如用户主页、话题页面等。通过容器ID，可以定位到特定的微博容器，从而获取相关的评论信息。
    # min_since_id：最小的评论ID。通过设置最小的评论ID，可以筛选出大于该ID的评论，以获取最新的评论内容。
    @classmethod
    def fetch_file(self,count=5, header=__default_headers, user_id=[], l_id=[], contain_id=[], since_id=''):
        row = self.__rebuild(self.__get_page(header, user_id, l_id, contain_id))
        if len(row) != 0:
            count -= 1
        data_s = [row]
        while count > 0:
            since_id = self.__get_since_id(header, user_id, l_id, contain_id, since_id)
            row = self.__rebuild(self.__get_page(header, user_id, l_id, contain_id, since_id))
            print(row)
            if len(row) != 0:
                data_s.append(row)
                count -= 1
        # print(data_s)
        self.__save_file(self.__get_user_name(header, user_id, contain_id), data_s)

    # 新的数据获取函数
    @classmethod
    def get_data(self,min_id='', user_id=[], l_id=[], contain_id=[],header=__default_headers):
        # 数据记录器
        li = []

        # timer system
        last_time = ''
        last_month = 0
        temp_month = 0  # 临时条件时间（因为微博是按时间顺序分配js的created_at）
        rest_month = 0  # 相隔月份数

        # 尝试事先声明的变量，减少重复回收和声明空间
        text = ''
        timing = ''
        length = 0
        page = []
        while rest_month <= 2:
            min_id = self.__get_since_id(header, user_id, l_id, contain_id, min_id)
            # print(min_since_id)
            page = self.__get_page(header, user_id, l_id, contain_id, min_id)[0]['data']['cards']
            # print(page)
            for sentence in page:
                text = sentence['mblog']['text']
                timing = sentence['mblog']['created_at']

                dr = re.compile(r'<[^>]+>|转发微博|分享图片|\s')
                text = dr.sub('', text)
                length = len(text)
                # print(length)

                # 确保文本长度小于50
                if length <= 50:
                    if last_time == '' and text != '':
                        last_time = self.__trans_time(timing)
                        # print(last_time)
                        last_month = int(last_time[5:7])

                    if text != '':
                        temp_month = int(self.__trans_time(timing)[5:7])
                        # li.append([self.__trans_time(timing), text])#暂时不需要时间
                        li.append([ text])

            # 将时间月份差转换为绝对值
            if temp_month != 0:
                rest_month = abs(last_month - temp_month)
        # print(li)
        # print(timing)
        return li, self.__trans_time(timing)[:11].replace('-', '') + last_time[:11].replace('-', '')


