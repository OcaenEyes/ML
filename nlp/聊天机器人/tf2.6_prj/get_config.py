import os
from configparser import SafeConfigParser


def get_config():
    conf_file = os.getcwd() + "/seq2seq.ini"
    parser = SafeConfigParser()
    parser.read(conf_file)
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_floats + _conf_strings)


# test
# if __name__ == "__main__":
#     print(get_config())
