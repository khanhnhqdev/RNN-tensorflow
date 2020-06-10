import tensorflow.compat.v1 as tf 
import pandas as pd 
import numpy as np 
import random

class DataReader: 
    def __init__(self, data_path, batch_size, vocab_size):  
        self._batch_size = batch_size 
        with open(data_path) as f: 
            d_lines = f.read().splitlines() 
        
        self._data = [] 
        self._labels = []
        self._sentence_lengths = []
        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>') 
            label, doc_id, sequence_lengths = int(features[0]), int(features[1]), int(features[2])
            tokens = features[3].split() 
            vector = [int(token) for token in tokens]
            self._data.append(vector) 
            self._labels.append(label) 
            self._sentence_lengths.append(sequence_lengths)
        # self._data and self._label: vector and label corresponding to each document
        self._data = np.array(self._data)
        self._labels = np.array(self._labels) 
        self._sentence_lengths = np.array(self._sentence_lengths)
        self._num_epoch = 0 
        self._batch_id = 0 

    def next_batch(self): 
        '''
        get data in batch, after each epoch data is shuffled
        '''
        start = self._batch_id * self._batch_size 
        end = start + self._batch_size
        self._batch_id += 1

        if end > len(self._data): 
            end = len(self._data)
            start = end - self._batch_size
            self._num_epoch += 1
            self._batch_id = 0
            indices = list(range(len(self._data)))
            # random.seed(2020)
            random.shuffle(indices)
            self._data, self._labels, self._sentence_lengths = self._data[indices],   self._labels[indices],  self._sentence_lengths[indices]

        return self._data[start:end], self._labels[start:end], self._sentence_lengths[start:end]

def load_dataset(train_path = './data_set/20news-train-encoded.txt', test_path = './data_set/20news-test-encoded.txt', batch_size = 16, vocab_size = 10): 
    '''
    load train and test dataset
    '''
    train_data_reader = DataReader(data_path = train_path, 
                                batch_size = batch_size, 
                                vocab_size = vocab_size 
                                ) 
    test_data_reader = DataReader(data_path = test_path,
                                batch_size = batch_size,
                                vocab_size = vocab_size 
                                ) 
    return train_data_reader, test_data_reader 
