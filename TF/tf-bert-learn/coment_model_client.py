import sys
import json
import requests
import tokenization

tokenizer = tokenization.FullTokenizer(
    vocab_file='./uncased_L-2_H-128_A-2/vocab.txt',
    do_lower_case=True
)


def text2ids(textList):
    input_ids_list = []
    input_mask_list = []
    for text in textList:
        if len(text) >= 128:
            text = text[:128]
        tokens_a = tokenizer.tokenize(text)
        ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        tokens = []
        segment_ids = []
        tokens.append(0)
        segment_ids.append(0)
        for token in ids_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(0)
        segment_ids.append(0)
        input_ids = [tokens + [0] * (128 - len(tokens))]
        input_mask = [[0] * 128]
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
    return input_ids_list, input_mask_list


def comment_bert_class(textList):
    input_ids_list, input_mask_list = text2ids(textList)
    url = 'http://127.0.0.1:8501/v1/models/bert:predict'
    data = json.dumps({
        "signature_name": "result",
        "inputs": {
            'input_ids': input_ids_list,
            'input_mask': input_mask_list
        }
    })
    result = requests.post(url,data=data).json()
    return result

if __name__=="__main__":
    textList = ["很好的手机，值得拥有","垃圾的手机"]
    result = comment_bert_class(textList)
    print(result)

