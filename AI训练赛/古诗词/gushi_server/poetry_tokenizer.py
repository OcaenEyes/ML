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
