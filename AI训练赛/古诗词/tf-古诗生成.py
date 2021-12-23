from collections import Counter
import math
import numpy as np
import tensorflow as tf


class Tokenizer():
    """
        分词器
    """

    def __init__(self, token_dict):
        # 词 -> ID
        self.token_dict = token_dict
        # ID -> 词
        self.token_dict_rev = {value: key for key, value in self.token_dict.items()}
        # 词汇表大小
        self.vocab_size = len(self.token_dict)

    def id_to_token(self, token_id):
        """
        给定一个编号，查找词汇表中对应的词
        :param token_id: 带查找词的编号
        :return: 编号对应的词
        """
        return self.token_dict_rev[token_id]

    def token_to_id(self, token):
        """
        给定一个词，查找它在词汇表中的编号
        未找到则返回低频词[UNK]的编号
        :param token: 带查找编号的词
        :return: 词的编号
        """
        return self.token_dict.get(token, self.token_dict['[UNK]'])

    def encode(self, tokens):
        """
        给定一个字符串s，在头尾分别加上标记开始和结束的特殊字符，并将它转成对应的编号序列
        :param tokens: 待编码字符串
        :return: 编号序列
        """
        # 加上开始标记
        token_ids = [self.token_to_id('[CLS]'), ]
        # 加入字符串编号序列
        for token in tokens:
            token_ids.append(self.token_to_id(token))

        # 加上结束标记
        token_ids.append(self.token_to_id('[SEP]'))
        return token_ids

    def decode(self, token_ids):
        """
        给定一个编号序列，将它解码成字符串
        :param token_ids: 待解码的编号序列
        :return: 解码出的字符串
        """
        # 起止标记字符特殊处理
        spec_tokens = ['[CLS]', '[SEP]']

        # 保存解码出的字符list
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in spec_tokens:
                continue

            tokens.append(token)

        # 拼接字符串
        return ''.join(tokens)


# 禁用词
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 数据集路径
DATASET_PATH = './data/poetry.txt'
# 每个epoch训练完成后，随机生成SHOW_NUM首古诗作为展示
SHOW_NUM = 5
# 最佳模型保存路径
BEST_MODEL_PATH = './out/best_model.h5'
# 句子最大长度
MAX_LEN = 64
# 最小词频
MIN_WORD_FREQUENCY = 8
# 训练的batch_size
BATCH_SIZE = 16
# 共训练多少个epoch
TRAIN_EPOCHS = 2

disallowed_words = DISALLOWED_WORDS
max_len = MAX_LEN
min_word_frequency = MIN_WORD_FREQUENCY
batch_size = BATCH_SIZE

# 加载数据集
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.replace("：", ":") for line in lines]

# 数据集列表
poetry = []
# 逐行处理读取到的数据
for line in lines:
    if line.count(":") != 1:
        continue
    # 分割后半部分
    _, last_part = line.split(":")
    ignore_flag = False
    for dis_word in disallowed_words:
        if dis_word in last_part:
            ignore_flag = True
            break
    if ignore_flag:
        continue
    # 长度不能超过最大长度
    if len(last_part) > max_len - 2:
        continue
    poetry.append(last_part.replace("\n", ""))

# 统计词频
counter = Counter()
for line in poetry:
    counter.update(line)
# 过滤掉低频的词
_tokens = [(token, count) for token, count in counter.items() if count >= min_word_frequency]
# 按词频排序，只保留词列表
_tokens = sorted(_tokens, key=lambda x: -x[1]) 
# 去掉词频 只保留词列表
_tokens = [token for token, count in _tokens]
# 将特殊词和数据集拼接起来
_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + _tokens

# 创建字典 token -> id 的关系
token_id_dict = dict(zip(_tokens, range(len(_tokens))))

# 使用词典重建分词器
tokenizer = Tokenizer(token_id_dict)
# 混洗数据
np.random.shuffle(poetry)


class PoetryDataGenerator():
    """
    训练数据集生成
    """

    def __init__(self, data, random=False):
        # 数据集
        self.data = data
        # batch_size
        self.batch_size = batch_size
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))
        # 每个epoch开始的时候是否随机混洗
        self.random = random

    def sequence_padding(self, data, length=None, padding=None):
        """
        将给定数据填充到相同长度
        :param data: 待填充数据
        :param length: 填充后的长度，不传递此参数则使用data中的最大长度
        :param padding: 用于填充的数据，不传递此参数则使用[PAD]的对应编号
        :return: 填充后的数据
        """

        # 计算填充长度
        if length is None:
            length = max(map(len, data))

        # 计算填充数据
        if padding is None:
            padding = tokenizer.token_to_id('[PAD]')

        # 开始填充
        outputs = []

        for line in data:
            padding_length = length - len(line)
            # 不足就填充
            if padding_length > 0:
                outputs.append(np.concatenate([line, [padding] * padding_length]))
            # 超过就截断
            else:
                outputs.append(line[:length])

        return np.array(outputs)

    def __len__(self):
        return self.steps

    def __iter__(self):
        total = len(self.data)

        # 是否孙吉混洗
        if self.random:
            np.random.shuffle(self.data)

        # 迭代一个epoch，每次yield 一个batch
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_data = []

            # 逐一对古诗进行编码
            for single_data in self.data[start:end]:
                batch_data.append(tokenizer.encode(single_data))

            # 填充为相同长度
            batch_data = self.sequence_padding(batch_data)

            # yield x,y
            yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], tokenizer.vocab_size)

            del batch_data

    def for_fit(self):
        """
        创建一个生成器，用于训练
        """
        while True:
            yield from self.__iter__()


def generate_poetry(tokenizer, model, head):
    """
    随机生成一首藏头诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param head: 藏头诗的头
    :return: 一个字符串，表示一首古诗
    """
    # 使用空串初始化token_ids ,加入[CLS]
    token_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]

    # 标点符号
    punctuations = ['，', '。']
    punctuations_ids = [tokenizer.token_to_id(token) for token in punctuations]

    # 村换生成诗的list
    poetry = []

    # 对于藏头诗的每一个字。都生成一个短句
    for ch in head:
        # 先记录这个字
        poetry.append(ch)
        # 将藏头字转为id
        token_id = tokenizer.token_to_id(ch)
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
                poetry.append(tokenizer.id_to_token(target))

            if target in punctuations_ids:
                break
    return ''.join(poetry)


"""
构建lstm模型
"""
# model = tf.keras.Sequential()
# #不定长度的输入
# model.add(tf.keras.layers.Input(None,))
# # 词嵌入层
# model.add(tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128))
# # 第一个lstm层
# model.add(tf.keras.layers.LSTM(128,dropout=0.5,return_sequence=True))
# model.add(tf.keras.layers.LSTM(128,dropout=0.5,return_sequence=True))

# # 对每一个时间点输出都做softmax, 预测下一个词的概率
# model.add(tf.keras.layers.TimeDistributed(
#     tf.keras.layers.Dense(tokenizer.vocab_size,activation='softmax')
# ))
model = tf.keras.Sequential([
    # 不定长度的输入
    tf.keras.layers.Input((None,)),
    # 词嵌入层
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128),
    # 第一个LSTM层，返回序列作为下一层的输入
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
    # 第二个LSTM层，返回序列作为下一层的输入
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
    # 对每一个时间点的输出都做softmax，预测下一个词的概率
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')),
])
model.summary()

# 配置优化器和损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy)

"""
模型训练
"""


class Evaluate(tf.keras.callbacks.Callback):
    """
    训练过程评估，在每个epoch训练完成后，保留最优权重，并随机生成SHOW_NUM首古诗展示
    """

    def __init__(self):
        super().__init__()
        # 给loss赋一个较大的初始值
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epochi训练完成后调用
        # 如果当前loss更低，则保存当前模型参数
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save(BEST_MODEL_PATH)

        # 随机生成古诗测试，查看效果
        print("生成测试")

        for i in range(SHOW_NUM):
            print(generate_poetry(tokenizer, model, head="春花秋月"))


def train():
    # 创建数据集
    data_generator = PoetryDataGenerator(poetry, random=True)

    model.fit(
        data_generator.for_fit(),
        steps_per_epoch=data_generator.steps,
        workers=-1,
        epochs=TRAIN_EPOCHS,
        callbacks=[Evaluate()]
    )


def predict():
    """
    输入关键字，生成藏头诗
    """
    # 加载训练好的模型
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    keywords = input('输入关键字:\n')

    # 生成藏头诗
    for i in range(SHOW_NUM):
        print(generate_poetry(tokenizer, model, head=("春花秋月")), '\n')


if __name__ == "__main__":
    # train()
    predict()
