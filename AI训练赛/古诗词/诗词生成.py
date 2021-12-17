import random
import os

import keras
import numpy as np
from keras.callbacks import LambdaCallback
from keras.models import Input, Model, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import adam_v2


def preprocess_file():
    # 预料文本内容
    files_content = ''
    protry_file = './data/poetry.txt'
    with open(protry_file, 'r', encoding='utf-8') as f:
        for line in f:
            x = line.strip() + "]"
            x = x.split(":")[1]
            if len(x) <= 5:
                continue
            if x[5] == '，':
                files_content += x

    # print(files_content)
    words = sorted(list(files_content))
    # print(words)
    counted_words = {}

    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1
    # print(counted_words.keys())

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    # print(counted_words.keys())
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*wordPairs)
    # print(words)
    words += (" ",)
    # print(words)
    # word 到id的映射
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, len(words) - 1)

    return word2numF, num2word, words, files_content


class PoetryModel(object):
    def __init__(self):
        self.model = None
        self.do_train = True
        self.loaded_model = True
        self.max_len = 6
        self.batch_size = 32
        self.learning_rate = 0.001

        # 文本预处理
        self.word2numF, self.num2word, self.words, self.files_content = preprocess_file()

        # 诗list
        self.poems = self.files_content.split(']')

        # 诗总数量
        self.poems_num = len(self.poems) - 1
        print(self.poems_num)

        self.model_path = './out/poetry_model.h5'
        self.out_path = './out'
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        # 如果模型文件存在则直接加载模型，否则开始训练
        if os.path.exists(self.model_path) and self.loaded_model:
            self.model = load_model(self.model_path)
        else:
            self.train()

    def build_model(self):
        ''' 建立模型 '''
        print("建立模型")

        input_tensor = Input(shape=(self.max_len, len(self.words)))
        lstm = LSTM(512, return_sequences=True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(len(self.words), activation='softmax')(dropout)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = adam_v2.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        '''
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, 1. / temperature)
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        return int(pro.squeeze())

    def generate_sample_result(self, epoch, logs):
        ''' 训练过程中 每1000个epoch打印当前学习情况 '''
        if epoch % 1000 != 0:
            return
        with open('./out/out.txt', 'a', encoding='utf-8') as f:
            f.write('=================epoch{}================\n'.format(epoch))

        # print('\n=================epoch{}================\n'.format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            # print('=================diversity{}================\n'.format(diversity))
            generate = self.predict_random(temperature=diversity)
            # print(generate)

            # 训练时预测结果 写入txt
            with open('./out/out.txt', 'a', encoding='utf-8') as f:
                f.write(str(generate) + '\n')

    def predict_random(self, temperature=1):
        ''' 随机从库中选取一句开头的诗句 生成五言绝句 '''
        if not self.model:
            return

        index = random.randint(0, self.poems_num)
        # print(index)
        sentence = self.poems[index][:self.max_len]
        generate = self.predict_sen(sentence, temperature=temperature)
        return generate

    def predict_first(self, char, temperature=1):
        ''' 根据给出的首字 生成五言绝句 '''
        if not self.model:
            print('model not loaded')
            return

        index = random.randint(0, self.poems_num)
        # 随机选择一首诗的最后max_len字符+首个文字作为初始输入
        sentence = self.poems[index][1 - self.max_len:] + char
        print('first line', sentence)
        generate = str(char)
        # 直接预测后面的23个字符
        generate += self._preds(sentence, length=23, temperature=temperature)
        return generate

    def predict_sen(self, text, temperature=1):
        ''' 根据给出的前 max_len个字，生成诗句 '''
        if not self.model:
            return

        max_len = self.max_len
        if len(text) < max_len:
            print('输入文案长度不能小于', max_len)
            return

        sentence = text[-max_len:]
        # print("第一行:", sentence)
        generate = str(sentence)
        generate += self._preds(sentence, length=24 - max_len, temperature=temperature)
        return generate

    def predict_hide(self, text, temperature=1):
        ''' 根据给出的4个字，生成藏头诗五言拒绝 '''
        if not self.model:
            return
        if len(text) != 4:
            return

        index = random.randint(0, self.poems_num)
        # 随机选一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1 - self.max_len:] + text[0]
        generate = str(text[0])
        # print("第一行：", sentence)

        for i in range(5):
            next_char = self._pred(sentence, temperature)
            sentence = sentence[1:] + next_char
            generate += next_char

        for i in range(3):
            generate += text[i + 1]
            sentence = sentence[1:] + text[i + 1]
            for i in range(5):
                next_char = self._pred(sentence, temperature)
                sentence = sentence[1:] + next_char
                generate += next_char

        return generate

    def _preds(self, sentence, length=23, temperature=1):
        '''
        sentence:预测输入值
        length:预测出的字符串长度
        '''
        sentence = sentence[:self.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence, temperature)
            generate += pred
            sentence = sentence[1:] + pred
        return generate

    def _pred(self, sentence, temperature=1):
        if len(sentence) < self.max_len:
            return
        sentence = sentence[-self.max_len:]
        x_pred = np.zeros((1, self.max_len, len(self.words)))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.word2numF(char)] = 1.
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, temperature=temperature)
        next_char = self.num2word[next_index]

        return next_char

    def data_generator(self):
        ''' 生成器 生成数据 '''
        i = 0
        while 1:
            x = self.files_content[i:i + self.max_len]
            y = self.files_content[i + self.max_len]

            if ']' in x or ']' in y:
                i += 1
                continue

            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0

            x_vec = np.zeros(
                shape=(1, self.max_len, len(self.words)),
                dtype=np.bool
            )

            for t, char in enumerate(x):
                x_vec[0, t, self.word2numF(char)] = 1.0

            yield x_vec, y_vec
            i += 1

    def train(self):
        ''' 训练模型 '''
        print("training")
        number_of_epoch = len(self.files_content) - (self.max_len + 1) * self.poems_num
        number_of_epoch /= self.batch_size
        number_of_epoch = int(number_of_epoch / 1.5)
        print('epochs=', number_of_epoch)
        print('poems=', self.poems_num)
        print('len(self.files_content)=', len(self.files_content))

        if not self.model:
            self.build_model()

        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.model_path, save_weights_only=False),
                LambdaCallback(on_batch_end=self.generate_sample_result)
            ]
        )


if __name__ == "__main__":
    model = PoetryModel()
    for i in range(3):
        sen = model.predict_hide('争云日夏')
        print(sen)
