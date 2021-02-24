#!/usr/bin/python3.6
'''
BERT模型ckpt文件转为部署tfserving所需的文件
'''
import json
import os
from enum import Enum
import sys
import modeling
from termcolor import colored
import logging
import tensorflow as tf
import argparse
import pickle

tf.app.flags.DEFINE_string()
tf.app.flags.DEFINE_integer()
FLAGS = tf.app.flags.FLAGS

