import tensorflow as tf
from get_config import get_config

tf.config.run_functions_eagerly(True)

# 初始化超参字典
gConf = {}
gConf = get_config()

# 通过超参字典为vocab_in_size,vocab_tar_size,embedding_dim,units等赋值
vocab_inp_size = gConf["vocab_inp_size"]
vocab_tar_size = gConf["vocab_tar_size"]
embedding_dim = gConf["embedding_dim"]
units = gConf["layer_size"]
BATCH_SIZE = gConf["batch_size"]


# 定义encoder类
class Encoder(tf.keras.Model):
    # 初始化参数，对默认参数进行初始化

    def __init__(self, vocab_size, embedding_dim, encode_units, batch_size):
        """
        @param vocab_size:
        @param emdedding_dim:
        @param encode_units:
        @param batch_size:

        :param vocab_size: 非重复的词汇总数
        :param embedding_dim: 词嵌入的维度
        :enc_units: 编码器中GRU层的隐含节点数
        :batch_sz: 数据批次大小(每次参数更新用到的数据量)
        """
        super(Encoder, self).__init__()
        self.encode_units = encode_units
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        """
               return_sequences：
                       布尔值。是返回输出序列中的最后一个输出还是完整序列。 默认值：False。
                       True代表返回GRU序列模型的每个时间步的输出(每个输出做连接操作)
               return_state：
                       布尔值。 除输出外，是否返回最后一个状态。 默认值：False。
                       True代表除了返回输出外，还需要返回最后一个隐层状态。
               recurrent_initializer：
                       recurrent_kernel权重矩阵的初始化程序，用于对递归状态进行线性转换。 默认值：正交。
                       'glorot_uniform'即循环状态张量的初始化方式为均匀分布。
        """
        # 实例化gru层
        # return_sequences=True代表返回GRU序列模型的每个时间步的输出(每个输出做连接操作)
        # return_state=True代表除了返回输出外，还需要返回最后一个隐层状态
        # recurrent_initializer='glorot_uniform'即循环状态张量的初始化方式为均匀分布
        self.gru = tf.keras.layers.GRU(self.encode_units, return_sequences=True, return_state=True,
                                       recurrent_initializer="glorot_uniform")

    # 定义调用函数
    def call(self, x, hidden):
        """
        @param x:
        @param hidden:
        @return:
        """
        # 对输入进行embedding操作
        x_embedding = self.embedding(x)
        """initial_state：要传递给单元格的第一个调用的初始状态张量的列表（可选，默认为None，这将导致创建零填充的初始状态张量）。"""
        # 通过gru层获得最后一个时间步的输出和隐含状态
        output, state = self.gru(x_embedding, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """ (BATCH_SIZE, 隐藏层中的隐藏神经元数量) """
        # gru层的隐含节点对应的参数张量以零张量初始化
        return tf.zeros((self.batch_size, self.encode_units))


# 定义bahdanauAttention类，bahdanauAttention是常用的attention实现方法之一
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # 注意力网络的初始化
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    """
    传入值：
        features：编码器的输出，(64, 16, 1024) 即 (BATCH_SIZE, 输入序列最大长度句子的长度, 隐藏层中的隐藏神经元数量)
        hidden：解码器的隐层输出状态，(64, 1024) 即 (batch_size, hidden_size) (BATCH_SIZE, 隐藏层中的隐藏神经元数量)
    返回值：
        attention_result：(64, 1024) 即 (batch size, units) (BATCH_SIZE, 隐藏层中的隐藏神经元数量)
        attention_weights：(64, 16, 1) 即 (batch_size, sequence_length, 1) (BATCH_SIZE, 输入序列最大长度句子的长度, 1)
    """

    def call(self, features, hidden):
        """
        description: 具体计算函数
        :param features: 编码器的输出
        :param hidden: 解码器的隐层输出状态
        return: 通过注意力机制处理后的结果和注意力权重attention_weights
        """
        """
        1.hidden_with_time_axis = tf.expand_dims(hidden, 1)
                解码器的隐层输出状态hidden，(64, 1024) 即 (batch_size, hidden_size) (BATCH_SIZE, 隐藏层中的隐藏神经元数量)。
                hidden扩展一个维度从(64, 1024)变成(64, 1,1024)。
        2.score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
                计算注意力得分score。
                features：编码器的输出，(64, 16, 1024)。
                hidden_with_time_axis：解码器的隐层输出状态，(64, 1,1024)
                W1和W2：Dense(隐藏层中的隐藏神经元数量1024)
                tanh(W1(features) + W2(hidden_with_time_axis))：
                ---> tanh(W1((64, 16, 1024)) + W2((64, 1,1024)))
                ---> tanh((64, 16, 1024))
                ---> (64, 16, 1024) 即 (BATCH_SIZE, 输入序列最大长度句子的长度, 隐藏层中的隐藏神经元数量)
        3.attention_weights = tf.nn.softmax(self.V(score), axis=1)
                计算注意力权重attention_weights。
                V：Dense(隐藏层中的隐藏神经元数量1)
                softmax(V(score), axis=1)
                ---> softmax(V((64, 16, 1024)), axis=1)
                ---> softmax((64, 16, 1), axis=1)
                ---> (64, 16, 1) 即 (BATCH_SIZE, 输入序列最大长度句子的长度, 1)
                因为注意力得分score的形状是(BATCH_SIZE, 输入序列最大长度句子的长度, 隐藏层中的隐藏神经元数量)，
                输入序列最大长度句子的长度(max_length)是输入的长度。
                因为我们想为每个输入长度分配一个权重，所以softmax应该用在第一个轴(max_length)上axis=1，
                而softmax默认被应用于最后一个轴axis=-1。
        4.context_vector = tf.reduce_sum(attention_weights * features, axis=1)
                获得注意力机制处理后的结果context_vector。
                reduce_sum(attention_weights * features, axis=1)
                ---> reduce_sum((64, 16, 1) * (64, 16, 1024), axis=1)
                ---> reduce_sum((64, 16, 1024), axis=1)
                ---> (64, 1024) 即 (BATCH_SIZE, 隐藏层中的隐藏神经元数量)
        """

        # 将hidden增加一个维度,(batch_size, hidden_size) --> (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # 根据公式计算注意力得分, 输出score的形状为: (batch_size, 16, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # 根据公式计算注意力权重, 输出attention_weights形状为: (batch_size, 16, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # 最后根据公式获得注意力机制处理后的结果context_vector
        # context_vector的形状为: (batch_size, hidden_size)
        context_vector = attention_weights * features
        # 将乘机后的context_vector按行相加，进行压缩得到最终的context_vector
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


"""
构建RNN解码器：这里RNN是指GRU, 同时在解码器中使用注意力机制.
"""


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decode_units, batch_size):
        super(Decoder, self).__init__()
        # 初始化batch_size、decode_units、embedding 、gru 、fc、attention
        self.batch_size = batch_size
        self.decode_units = decode_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.decode_units, return_sequences=True, return_state=True,
                                       recurrent_initializer="glorot_uniform")

        # 实例化一个Dense层作为输出层
        self.fc = tf.keras.layers.Dense(vocab_size)
        # 在解码器阶段我们将使用注意力机制，这里实例化注意力的类
        self.attention = BahdanauAttention(self.decode_units)

    """
    1.x = self.embedding(x)
            输入：(64, 1) 64行1列，批量大小句子数为64，1列为该行句子的第N列的单词
            输出：(64, 1, 256) (BATCH_SIZE, 输入序列最大长度句子的长度, 嵌入维度)
    2.context_vector, attention_weights = self.attention(hidden, enc_output)
            attention_weights注意力权重：(64, 16, 1) 即 (BATCH_SIZE, 输入序列最大长度句子的长度, 1)
            context_vector注意力机制处理后的结果：(64, 1024) 即 (BATCH_SIZE, 隐藏层中的隐藏神经元数量)
    3.x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
            tf.expand_dims(context_vector, 1)：(64, 1, 1024) 即 (BATCH_SIZE, 1, 隐藏层中的隐藏神经元数量)
            concat([(64, 1, 1024),(64, 1, 256)], axis=-1)：1024+256=1280，最终输出 (64, 1, 1280)
    4.GRU
        1.tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
            return_sequences：
                    布尔值。是返回输出序列中的最后一个输出还是完整序列。 默认值：False。
                    True代表返回GRU序列模型的每个时间步的输出(每个输出做连接操作)
            return_state：
                    布尔值。 除输出外，是否返回最后一个状态。 默认值：False。
                    True代表除了返回输出外，还需要返回最后一个隐层状态。
            recurrent_initializer：
                    recurrent_kernel权重矩阵的初始化程序，用于对递归状态进行线性转换。 默认值：正交。
                    'glorot_uniform'即循环状态张量的初始化方式为均匀分布。
        2.output, state = gru(x)      
            output：
                    (64, 1, 1024) 即 (BATCH_SIZE, 1, 隐藏层中的隐藏神经元数量)
                    (当前批次的样本个数, 当前样本的序列长度(单词个数), 隐藏层中神经元数量 * 1)
            state：
                    (64, 1024) 即 (BATCH_SIZE, 隐藏层中的隐藏神经元数量)
    5.output = tf.reshape(output, (-1, output.shape[2]))
             (-1, output.shape[2])：表示把(64, 1, 1024)转换为(64, 1024) 即 (BATCH_SIZE, 隐藏层中的隐藏神经元数量)
    6.x = self.fc(output)
            x：(64, 4935) 即 (BATCH_SIZE, 目标序列的不重复单词的总数作为目标序列的字典大小)
    """

    def call(self, x, hidden, encode_ouput):
        # print("x.shape",x.shape) #(64, 1)。64行1列，批量大小句子数为64，1列为该行句子的第N列的单词

        # 对decoder的输入通过embedding层
        x = self.embedding(x)
        # print("x1.shape",x.shape) #(64, 1, 256)。(BATCH_SIZE, 输入序列最大长度句子的长度, 嵌入维度)

        # 使用注意力规则计算hidden与enc_output的'相互影响程度(计算attention，输出上下文语境向量)
        context_vector, attention_weights = self.attention(encode_ouput, hidden)
        # print("tf.expand_dims(context_vector, 1).shape",tf.expand_dims(context_vector, 1).shape) #(64, 1, 1024)

        # 将这种'影响程度'与输入x拼接(这个操作也是注意力计算规则的一部分)（拼接上下文语境与decoder的输入embedding，并送入gru中）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # print("x2.shape",x.shape) #(64, 1, 1280)

        # 将新的x输入到gru层中得到输出
        output, state = self.gru(x)
        # print("output1.shape",output.shape) #(64, 1, 1024) 即 (BATCH_SIZE, 1, 隐藏层中的隐藏神经元数量)
        # print("state.shape",state.shape) #(64, 1024) 即 (BATCH_SIZE, 隐藏层中的隐藏神经元数量)

        # 改变输出形状使其适应全连接层的输入形式
        output = tf.reshape(output, (-1, output.shape[2]))
        # print("output2.shape",output.shape) #(64, 1024) 即 (BATCH_SIZE, 隐藏层中的隐藏神经元数量)

        # 使用全连接层作为输出层
        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)
        # print("x3.shape",x.shape) #(64, 4935) 即 (BATCH_SIZE, 目标序列的不重复单词的总数作为目标序列的字典大小)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros(self.batch_size, self.decode_units)


# 定义损失函数
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # mask掉start,去除start对于loss的干扰
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


# 实例化encoder、decoder、optimizer、checkpoint等
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def training_step(inp, targ, target_lang, encode_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        encode_output, encode_hidden = encoder(inp, encode_hidden)
        decode_hidden = encode_hidden

        decode_input = tf.expand_dims([target_lang.word_index["start"]] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, decode_hidden, _ = decoder(decode_input, decode_hidden, encode_output)

            loss += loss_function(targ[:, t], predictions)
            decode_input = tf.expand_dims(targ[:, t], 1)

    step_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return step_loss
