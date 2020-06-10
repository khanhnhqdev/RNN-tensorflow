import numpy as np
from os import listdir
from os.path import  isfile
import re
from collections import defaultdict
from process_data import * 

# Process data from raw text:
# # gen_data_and_vocab()
# train_path = './data_set/20news-train-raw.txt'
# test_path  = './data_set/20news-test-raw.txt'
# vocab_path = './data_set/datavocab-raw.txt'
# encode_data(data_path = train_path, vocab_path = vocab_path)
# encode_data(data_path = test_path,  vocab_path = vocab_path)

# Train with RNN

# Evaluate
vocab_path = './data_set/datavocab-raw.txt'
with open(vocab_path) as f:
    print(len(f.read()))