import sys
import threading
from core.colors import info
import concurrent.futures


def threader(function, *urls):
    """为一个功能使用多线程"""
    threads = []
    # 因为 URLs 是一个元组
    urls = urls[0]
    # 迭代 URLs
    for url in urls:
        task = threading.Thread(target=function, args=(url,))
        threads.append(task)
    # 开启多线程
    for thread in threads:
        thread.start()
    # 等待所有的线程完成它们的工作
    for thread in threads:
        thread.join()
    # 删除全部线程
    del threads[:]


def flash(function, links, thread_count):
    """处理 URLs 并使用线程池来执行函数"""
    # 将元组类型的 links 转换为列表
    links = list(links)
    if sys.version_info < (3, 2):
        for begin in range(0, len(links), thread_count):  # 计算每个线程需要处理的链接数量
            end = begin + thread_count
            splitted = links[begin:end]
            threader(function, splitted)
            progress = end
            if progress > len(links):  # 如果溢出进行修复
                progress = len(links)
            print('\r%s Progress: %i/%i' % (info, progress, len(links)),
                  end='\r')
            sys.stdout.flush()
    else:
        threadpool = concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_count)
        futures = (threadpool.submit(function, link) for link in links)
        for i, _ in enumerate(concurrent.futures.as_completed(futures)):
            if i + 1 == len(links) or (i + 1) % thread_count == 0:
                print('%s Progress: %i/%i' % (info, i + 1, len(links)),
                      end='\r')
    print('')
