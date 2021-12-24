from poetry_tokenizer import Tokenizer
import tensorflow as tf
import random
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


def token_init():
    tmplist = []
    with open("vocab.txt", 'r', encoding='utf-8') as f:
        for i in f.readlines():
            tmplist.append(i.strip())

    # 创建字典 token -> id 的关系
    token_id_dict = dict(zip(tmplist, range(len(tmplist))))
    # 使用词典重建分词器
    return Tokenizer(token_id_dict)


def generate_poetry(tokenize, model, head=None):
    """
    随机生成一首藏头诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param head: 藏头诗的头
    :return: 一个字符串，表示一首古诗
    """
    # 使用空串初始化token_ids ,加入[CLS]
    token_ids = tokenize.encode("")
    token_ids = token_ids[:-1]

    # 标点符号
    punctuations = ['，', '。']
    punctuations_ids = [tokenize.token_to_id(token) for token in punctuations]

    # 存放生成诗的list
    poetry = []
    # 随机生成诗文
    if (head == "") or (head is None):
        f_words = ['春', '夏', '秋', '冬', '水', '月', '花']
        poetry.append(random.choice(f_words))
        token_id = tokenize.token_to_id(random.choice(f_words))
        token_ids.append(token_id)
        for i in range(4):
            # 生成短句
            while True:
                # 进行预测,只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
                output = model(np.array([token_ids, ], dtype=np.int32))
                _probas = output.numpy()[0, -1, 3:]
                del output

                # 按照出现概率，对所有的token倒序排列
                p_args = _probas.argsort()[::-1][:100]
                # 排列后的概率顺序
                p = _probas[p_args]

                # 对概率归一
                p = p / sum(p)

                # 再按照预测出的概率，随机选择一个词作为预测结果
                target_index = np.random.choice(len(p), p=p)
                target = p_args[target_index] + 3

                # 保存
                token_ids.append(target)
                # 只有不是特殊字符时，才保存到poetry中
                if target > 3:
                    poetry.append(tokenize.id_to_token(target))

                if target in punctuations_ids:
                    break
    elif len(head) == 1:
        # 根据首字生成诗文
        poetry.append(head)
        token_id = tokenize.token_to_id(head)
        token_ids.append(token_id)
        for i in range(4):
            # 生成短句
            while True:
                # 进行预测,只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
                output = model(np.array([token_ids, ], dtype=np.int32))
                _probas = output.numpy()[0, -1, 3:]
                del output

                # 按照出现概率，对所有的token倒序排列
                p_args = _probas.argsort()[::-1][:100]
                # 排列后的概率顺序
                p = _probas[p_args]

                # 对概率归一
                p = p / sum(p)

                # 再按照预测出的概率，随机选择一个词作为预测结果
                target_index = np.random.choice(len(p), p=p)
                target = p_args[target_index] + 3

                # 保存
                token_ids.append(target)
                # 只有不是特殊字符时，才保存到poetry中
                if target > 3:
                    poetry.append(tokenize.id_to_token(target))

                if target in punctuations_ids:
                    break

    else:
        # 对于藏头诗的每一个字。都生成一个短句
        for ch in head:
            # 先记录这个字
            poetry.append(ch)
            # 将藏头字转为id
            token_id = tokenize.token_to_id(ch)
            # 加入进列表
            token_ids.append(token_id)

            # 生成短句
            while True:
                # 进行预测,只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
                output = model(np.array([token_ids, ], dtype=np.int32))
                _probas = output.numpy()[0, -1, 3:]
                del output

                # 按照出现概率，对所有的token倒序排列
                p_args = _probas.argsort()[::-1][:100]
                # 排列后的概率顺序
                p = _probas[p_args]

                # 对概率归一
                p = p / sum(p)

                # 再按照预测出的概率，随机选择一个词作为预测结果
                target_index = np.random.choice(len(p), p=p)
                target = p_args[target_index] + 3

                # 保存
                token_ids.append(target)
                # 只有不是特殊字符时，才保存到poetry中
                if target > 3:
                    poetry.append(tokenize.id_to_token(target))

                if target in punctuations_ids:
                    break
    return ''.join(poetry)


def model_init():
    # model path
    model_path = "./model/best_model.h5"
    # 加载训练好的模型
    model = tf.keras.models.load_model(model_path)
    return model


@app.route("/api/v0.1/poetry", methods=['GET', 'POST'])
def main_poetry():
    mode = request.args.get("mode")
    if mode == "":
        res = {
            "state": "请选定生成模式！♥",
        }
        return jsonify(res=res)
    head = request.args.get("text")
    num = request.args.get("num")
    if mode == '1':
        head = None
    elif mode == '2':
        if len(head) >= 1:
            head = head[0]
    else:
        if len(head) >= 4:
            head = head[0:4]
    if (num is None) or (num == ""):
        num = 5
    model = model_init()
    tokenize = token_init()
    poetrys = {}
    for i in range(int(num)):
        poetry = generate_poetry(tokenize, model, head)
        poetrys["第%s首" % (i + 1)] = poetry
    res = {
        "state": "成功啦！♥",
        "poetry": poetrys
    }
    return jsonify(res=res)


@app.after_request
def cors(environ):
    # environ.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    environ.headers['Access-Control-Allow-Origin'] = 'http://ai.oceangzy.top'
    environ.headers['Access-Control-Allow-Method'] = '*'
    environ.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    environ.headers['Access-Control-Allow-Credentials'] = 'true'
    return environ


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003)
