import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from itertools import chain
from tqdm import tqdm

import sys
sys.path.insert(0, "/DATA/joosung/pytorch_pretrained_BERT_master")
from pytorch_pretrained_bert.tokenization import load_vocab
from pytorch_bert_embedding import *
import torch.optim as optim

bert_model, bert_tokenizer = bert_model_load('bert-base-multilingual-cased')
vocab_size = 6222

vocab = load_vocab('/DATA/joosung/pytorch_pretrained_BERT_master/korea_vocab.txt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_use = True 

from cls_model import BertClassifier
model = BertClassifier(drop_rate=0, gpu=gpu_use)

model_name='nodrop/bert_classifier_50000'
model.load_state_dict(torch.load('models/{}'.format(model_name)))

model = model.to(device)
model.eval()

mb_size = 1
f = open("../../sentiment_data/nsmc-master/ratings_test.txt", 'r')
lines = f.readlines()
data_number = len(lines)-1
n_iter = data_number

line_number = 0
correct = 0
for it in range(n_iter):
    cls_value_list = []
    labels = []

    inputsize = 0  # count batch size
    while inputsize < mb_size:
        line_number = line_number % data_number + 1

        input_line = lines[line_number]

        input_split = re.split('\t', input_line)

        input_sentence = input_split[1]
        input_label = input_split[2].strip()

        condition = True
        try:
            input_label = float(input_label)

            if len(input_sentence) < 1 or (len(bert_tokenizer.tokenize(input_sentence))==1 and bert_tokenizer.tokenize(input_sentence)[0]=='[UNK]'):
                condition = False
            else:
                condition = True
        except:
            condition = False        

        if condition:
            if int(input_label) == 0:
                input_label = [1, 0]
            else:
                input_label = [0, 1]
            labels.append(input_label)

            output_bert = embedding(input_sentence, bert_model, bert_tokenizer)            

            cls_value_emb = np.asarray(output_bert['features'][0]['layers'][0]['values']) # [CLS] token
            cls_value_emb = torch.from_numpy(cls_value_emb) # (768)
            cls_value_emb = cls_value_emb.unsqueeze(0) # (1, 768)    
            cls_value_list.append(cls_value_emb)

            inputsize += 1
#             print(input_sentence)

    gt = torch.from_numpy(np.asarray(labels)).type(torch.FloatTensor).to(device) # (batch, 2)
    cls_input = torch.cat(cls_value_list).type(torch.FloatTensor).to(device) # (batch, 768)

    cls_out = model.classifier(cls_input) # (batch, 2)
    
    if cls_out.argmax(1).item() == input_label.index(1):
        correct+=1
        
    if (it+1) % 5000 == 0:
        print(correct/(it+1) * 100)
    
print(correct/n_iter * 100)