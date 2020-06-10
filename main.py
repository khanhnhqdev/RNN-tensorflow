import numpy as np
from os import listdir
from os.path import  isfile
import re
from collections import defaultdict
from process_data import * 
from DataReader import *
from RNN import *
import logging
import sys
# Process data from raw text:
# gen_data_and_vocab()
# train_path = './data_set/20news-train-raw.txt'
# test_path  = './data_set/20news-test-raw.txt'
# vocab_path = './data_set/datavocab-raw.txt'
# encode_data(data_path = train_path, vocab_path = vocab_path)
# encode_data(data_path = test_path,  vocab_path = vocab_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./result.log"),
        logging.StreamHandler()
    ]
)


# train and evaluate with RNN

f = open('output.txt','w')
sys.stdout = f
tf.compat.v1.disable_eager_execution()
train_and_evaluate_RNN()



