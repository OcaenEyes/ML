#!/usr/bin/python3.6

import argparse
import json
import jieba
import requests
import tokenization
from flask import Flask
from flask import request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
headers = {
    'content-type': "application/json",
    "cache-control": "no-cache"
}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(max_seq_length, tokenizer, text_a, text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)  # 将中文分字
    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将中文转换成ids
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids  # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数


# 预训练或者自训练的词表文件
vocab_file = "../uncased_L-2_H-128_A-2/vocab.txt"
token = tokenization.FullTokenizer(vocab_file=vocab_file)


def ugc_bert(content):
    input_ids, input_mask, segment_ids = convert_single_example(128, token, content)
    features = {}
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    data_list = []
    data_list.append(features)
    data = json.dumps({"signature_name": "result", "instances": data_list})
    headers = {"content-type": "application/json"}
    # 根据自己的服务ip 端口号和model_name修改
    json_response = requests.post('http://localhost:8501/v1/models/ugc:predict', data=data, headers=headers)
    print(data, json_response.text)
    return jsonify(json.loads(json_response.text))

def email_bert(content):
    input_ids, input_mask, segment_ids = convert_single_example(128, token, content)
    features = {}
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    data_list = []
    data_list.append(features)
    data = json.dumps({"signature_name": "result", "instances": data_list})
    headers = {"content-type": "application/json"}
    # 根据自己的服务ip 端口号和model_name修改
    json_response = requests.post('http://localhost:8501/v1/models/email:predict', data=data, headers=headers)
    print(data, json_response.text)
    return jsonify(json.loads(json_response.text))

@app.route('/api', methods=['GET'])
def detect():
    if 'method' not in request.args.keys():
        raise Exception('method is empty.....')
    method = request.args.get("method")
    content = request.args.get("sen")
    if method == 'ugc':
        return ugc_bert(content)
    if method == 'email':
        return email_bert(content)


if __name__ == '__main__':
    app.run()
