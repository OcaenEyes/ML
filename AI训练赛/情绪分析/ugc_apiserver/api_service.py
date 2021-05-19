import argparse
import flask
import logging
import json
import os
import re
import sys
import string
import time
import numpy as np

from bert_base.client import BertClient


# 切分句子
def cut_sent(txt):
    # 先预处理空格
    txt = re.sub('([\t]+)', r" ", txt)
    txt = txt.rstrip()
    nlist = txt.split("\n")
    nnlist = [x for x in nlist if x.strip() != '']
    return nnlist


# 对句子进行预测识别
def class_pred(list_text):
    # 文本拆句子
    print("total setance :%d" % (len(list_text)))
    with BertClient(ip='0.0.0.0', port=5575, port_out=5576, show_server_config=False, check_version=False,
                    check_length=False, timeout=10000, mode='CLASS') as bs:
        start_t = time.perf_counter()
        rst = bs.encode(list_text)
        print('result:', rst)
        print('time used:{}'.format(time.perf_counter() - start_t))

    # 抽取标注结果
    pred_label = rst[0]["pred_label"]
    result_txt = [
        [pred_label[i], list_text[i]] for i in range(len(pred_label))
    ]
    return result_txt


# 封装应用服务
def comment_server(args):
    from flask import Flask, request, render_template, jsonify
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template('index.html', version='V 0.1.2')

    @app.route("/api/v0.1/query", methods=['POST'])
    def query():
        res = {}
        txt = request.values['text']
        if not txt:
            res['result'] = "error"
            return jsonify(res)
        lstseg = cut_sent(txt)
        print('-' * 30)
        print('结果，共%d个句子' % (len(lstseg)))
        for x in lstseg:
            print("第 %d个句子：【%s】" % (lstseg.index(x), x))
        print('-' * 30)
        if request.method == 'POST' or 1:
            res['result'] = class_pred(lstseg)
        print('result : %s' % str(res))
        return jsonify(res)

    app.run(
        host=args.ip,
        port=args.port,
        debug=True
    )


# 封装应用
def comment_app():
    parser = argparse.ArgumentParser(description='API DEMO SERVER')
    parser.add_argument('-ip',
                        type=str,
                        default='0.0.0.0',
                        help='bert model serving')
    parser.add_argument('-port',
                        type=int,
                        default=8910,
                        help='listen port,default:8910 ')
    args = parser.parse_args()

    comment_server(args)


if __name__ == '__main__':
    comment_app()
