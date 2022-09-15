import json
import os
import sys
import time
import tensorflow as tf
import seq2seq_model
from get_config import get_config
import io

# 加载配置
gConf = {}
gConf = get_config()
vocab_inp_size = gConf["vocab_inp_size"]
vocab_tar_size = gConf["vocab_tar_size"]
embedding_dim = gConf["embedding_dim"]
units = gConf["layer_size"]
BATCH_SIZE = gConf["batch_size"]

max_length_inp = gConf["max_length"]
max_length_tar = gConf["max_length"]
log_dir = gConf["log_dir"]
writer = tf.summary.create_file_writer(log_dir)


# 对训练语料进行处理，上下文分别加上start end表示
def preprocess_sentence(w):
    w = 'start' + w + 'end'
    return w


# 定义数据读取函数，从训练语料中读取数据并进行 word2number处理， 并生成词典
def read_data(path):
    path = os.getcwd() + "/" + path
    if not os.path.exists(path):
        path = os.path.dirname(os.getcwd()) + "/" + path

    lines = io.open(path, encoding="utf-8").read().strip().split("\n")
    word_pairs = [[preprocess_sentence(w) for w in l.split("\t")] for l in lines]
    input_lang,target_lang = zip(*word_pairs)
    
