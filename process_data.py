import numpy as np
from os import listdir
from os.path import  isfile
import re
from collections import defaultdict

MAX_DOC_LENGTH = 500
unknown_ID = 0
padding_ID = 1 

def gen_data_and_vocab():
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        '''
        build a list, each element of the list is: label(groupnews), filename and content(include all words(lower form) in the text of correspond file) 
        '''
        data=[]
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path= parent_path + '/' + newsgroup +'/'
            #files: list of (filename, path to filename) in the newsgroup
            files= [(filename, dir_path+ filename)
                    for filename in listdir(dir_path)
                    if isfile(dir_path+ filename)]
            files.sort()
            label= group_id
            print('processing: {} - {}'.format(group_id, newsgroup))

            for filename, file_path in files:
                with open(file_path,'rb') as f:
                    text= f.read().decode('UTF-8', errors='ignore').lower()
                    words= re.split('\W+', text)
                    if word_count is not None: # only for train data
                        for word in words:
                            word_count[word]+=1
                    content= ' '.join(words)
                    assert  len(content.splitlines())==1
                    data.append(str(label)+ '<fff>'
                                + filename+ '<fff>'+ content)
        return data


    word_count= defaultdict(int)
    path= './data_set/'
    parts=[path + dir_name + '/' for dir_name in  listdir(path)
           if not isfile(path + dir_name)]

    train_path, test_path= (parts[0], parts[1]) if 'train' in parts[0]\
        else (parts[1], parts[0])

    # newsgroups_list: list of group of news after sorted
    newsgroups_list= [newsgroup  for newsgroup in listdir(train_path)]
    newsgroups_list.sort()

    # build vocabulary(from train data) and process data from raw text.
    train_data= collect_data_from(
        parent_path= train_path,
        newsgroup_list= newsgroups_list,
        word_count= word_count
    )
    vocab=[word for word, freq in
           zip(word_count.keys(), word_count.values())
           if freq > 10]
    vocab.sort()
    with open('./data_set/datavocab-raw.txt','w') as f:
        f.write('\n'.join(vocab))

    test_data = collect_data_from(
        parent_path=test_path,
        newsgroup_list=newsgroups_list
    )

    with open('./data_set/20news-train-raw.txt','w') as f:
         f.write('\n'.join(train_data))
    with open('./data_set/20news-test-raw.txt','w') as f:
         f.write('\n'.join(test_data))


def encode_data(data_path, vocab_path):
    '''
    encode from word to ID: each word in vocab is encoded with wordID + 2, 0: unknow ID and 1: padding ID
    write encode data to txt file
    '''
    with open(vocab_path) as f:
        vocab= dict([(word, word_ID + 2)
                     for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents= [(line.split('<fff>')[0],
                     line.split('<fff>')[1],
                     line.split('<fff>')[2])
                    for line in f.read().splitlines()]
    
    encoded_data= []
    for docs in documents:
        label, doc_id, text= docs
        words= text.split()[:MAX_DOC_LENGTH]
        sentence_length= len(words)
        
        encode_text= []
        for word in words:
            if word in vocab:
                encode_text.append(str(vocab[word]))
            else:
                encode_text.append((str(unknown_ID)))
        if len(words) < MAX_DOC_LENGTH:
            num_padding= MAX_DOC_LENGTH - len(words)
            for i in range(num_padding):
                encode_text.append(str(padding_ID))
                
        encoded_data.append(str(label)+ '<fff>' +
                           str(doc_id) +'<fff>'+
                           str(sentence_length)+ '<fff>'+
                           ' '.join(encode_text))

    dir_name= '/'.join(data_path.split('/')[:-1])
    file_name= '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name +'/'+ file_name, 'w') as f:
        f.write('\n'.join(encoded_data))