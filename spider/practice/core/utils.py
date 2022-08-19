import re
import tld
from urllib.parse import urlparse
import math
from core.config import badTypes


def extract_headers(headers):
    sorted_headers = {}
    # findall(pattern, string)，返回 headers 中所有与前面的正则表达式相匹配的全部字符串，返回形式为列表，
    #   类似于：[('Accept-Language', 'zh-cn'), ('Accept-Encoding', 'gzip, deflate')]
    # r 表示字符串为非转义的原始字符串，其中的字符串不进行转义，
    #   (.*)匹配分组，
    #   .* 表示匹配除换行符之外的所有字符，
    #   \s 匹配任意空白字符，
    #   以冒号作为分割
    mathches = re.findall(r'(.*):\s(.*)', headers)
    for mathch in mathches:
        header = mathch[0]
        value = mathch[1]

        try:
            if value[-1] == ',':
                value = value[:-1]
            sorted_headers[header] = value
        except IndexError:
            pass
        return sorted_headers


def top_level(url):
    """从URL中提取顶级域名"""
    ext = tld.get_tld(url, fix_protocol=True)  # fix_protocol=True，表示忽略协议
    toplevel = '.'.join(urlparse(url).netloc.split('.')[-2:]).split(ext)[0] + ext  # 提取顶级域名
    return toplevel


def regxy(pattern, response, supress_regex, custom):
    """基于用户提供的正则表达式提取字符串"""
    try:
        matches = re.findall(r'%s' % pattern, response)
        for match in matches:
            custom.add(match)
    except:
        supress_regex = True


def is_link(url, processed, files):
    """检查一个链接是否应该被爬取"""
    # 如果文件存在就不应该被爬取
    conclusion = False
    # 如果这个链接并没有被爬取过
    if url not in processed:
        if url.split('.')[-1].lower() in badTypes:
            files.add(url)
        else:
            return True
    return conclusion


def entropy(string):
    """计算字符串的熵"""
    entropy = 0
    for number in range(256):
        result = float(string.encode('utf-8').count(chr(number))) / len(string.encode('utf-8'))
        if result != 0:
            entropy = entropy - result * math.log(result, 2)
    return entropy


def xmlParser(response):
    """从 .xml 文件中提取链接"""
    # 使用正则表达式提取 URLs
    return re.findall(r'<loc>(.*?)</loc>', response)


def remove_regex(urls, regex):
    """
    将非匹配列表解析为正则表达式

    参数:
        urls: 可迭代的 urls
        custom_regex: 要解析的字符串正则表达式

    返回:
        不匹配正则表达式的字符串列表
    """

    if not regex:
        return urls

    # 避免迭代字符串的字符
    if not isinstance(urls, (list, set, tuple)):
        urls = [urls]

    try:
        non_matching_urls = [url for url in urls if not re.search(regex, url)]
    except TypeError:
        return []

    return non_matching_urls


def writer(datasets, dataset_names, output_dir):
    """写入结果"""
    for dataset, dataset_name in zip(datasets, dataset_names):
        if dataset:
            filepath = output_dir + '/' + dataset_name + '.txt'
            with open(filepath, 'w+') as out_file:
                joined = '\n'.join(dataset)
                out_file.write(str(joined.encode('utf-8')))
                out_file.write('\n')


def timer(diff, processed):
    """返回花费的时间"""
    # 把秒转换为分钟和秒
    minutes, seconds = divmod(diff, 60)
    try:
        # 计算请求所耗费的平均时间
        time_per_request = diff / float(len(processed))
    except ZeroDivisionError:
        time_per_request = 0
    return minutes, seconds, time_per_request
