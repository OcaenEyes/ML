import json
import os
import re
import jieba
from zhon.hanzi import punctuation
from get_config import get_config
import io
import tensorflow as tf

# 加载配置
gConf = {}
gConf = get_config()
conv_path = gConf["resource_data"]
vocab_inp_path = gConf["vocab_inp_path"]
vocab_tar_path = gConf["vocab_tar_path"]
vocab_inp_size = gConf["vocab_inp_size"]
vocab_tar_size = gConf["vocab_tar_size"]
seq_train = gConf["seq_data"]


def predata_util():
    # 判断训语料文件是否存在，如果不存在则提醒
    if not os.path.exists(conv_path):
        print("找不到conv文件")
        exit()

    # 新建一个文件，用于存放处理后的对话语料
    seq_train_file = open(seq_train, "w")
    # 打开要处理的语料，逐条读取并进行数据处理
    with open(conv_path, encoding="utf-8") as f:
        one_conv = ""  # 存储一次完整的对话
        i = 0
        # 开始循环语料
        for line in f:
            line = line.strip("\n")
            line = re.sub(r"[%s]+" % punctuation,"",line) # 去除标点符号
            if line =="":
                continue
            # 判断是否为一段对话的开始，如果是，则把刚处理过的语料保存下来