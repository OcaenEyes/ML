import time
import random
import requests
from requests.exceptions import TooManyRedirects  # 太多的重定向

session = requests.Session()
session.max_redirects = 3


def requester(url, main_url=None, delay=0, cook={}, headers={}, timeout=10, host=None, ninja=False,
              user_agents=[], failed=[], processed=[]):
    """处理请求并且返回响应体"""
    # 标记URL作为已经爬取过
    processed.add(url)
    # 对程序进行特定时间的延迟
    time.sleep(delay)

    def normal(url):
        """默认的请求"""
        finalHeaders = headers or {
            'Host': host,
            # 任意选择一个 user-agent
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip',
            'DNT': '1',
            'Connection': 'close',
        }
        try:
            response = session.get(url, cookies=cook, headers=finalHeaders, verify=False,
                                   timeout=timeout, stream=True)
        except TooManyRedirects:
            return 'dummy'
        if 'text/html' in response.headers['content-type']:
            if response.status_code != '404':
                return response.text
            else:
                response.close()
                failed.add(url)
                return 'dummy'
        else:
            response.close()
            return 'dummy'

    def facebook(url):
        """与 developer.facebook.com API 进行交互"""
        return requests.get('https://developers.facebook.com/tools/debug/echo/?q=' + url,
                            verify=False).text

    def pixlr(url):
        """与 pixlr.com API 进行交互"""
        if url == main_url:
            # 如果 http://example.com 被使用了，pixlr 会抛出错误
            url = main_url + '/'
        return requests.get('https://pixlr.com/proxy/?url=' + url,
                            headers={'Accept-Encoding': 'gzip'}, verify=False).text

    def code_beautify(url):
        """与 codebeautify.org API 进行交互"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0',
            'Accept': 'text/plain, */*; q=0.01',
            'Accept-Encoding': 'gzip',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': 'https://codebeautify.org',
            'Connection': 'close',
        }
        return requests.post('https://codebeautify.com/URLService', headers=headers,
                             data='path=' + url, verify=False).text

    def photopea(url):
        """与 www.photopea.com API 进行交互"""
        return requests.get(
            'https://www.photopea.com/mirror.php?url=' + url, verify=False).text

    if ninja:  # 如果 ninja 模式可以使用
        # 随机选择上述任意的一个请求方式
        response = random.choice(
            [photopea, normal, facebook, pixlr, code_beautify])(url)
        return response or 'dummy'
    else:
        return normal(url)
