import os
from configparser import SafeConfigParser

conf_file = os.getcwd() + "/seq2seq.ini"
if not os.path.exists(conf_file):
    conf_file = os.path.dirname(os.getcwd() + "/seq2seq.ini")


def get_config():
    print(conf_file)
    parser = SafeConfigParser()
    parser.read(conf_file, encoding='utf-8')
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)

# test
# if __name__ == "__main__":
#     # print(get_config())
#     print(conf_file)