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

from tensorboardX import SummaryWriter
summary = SummaryWriter(logdir='./logs/nodrop')

n_iter = 50000
vocab_size = 6222
mb_size = 32

vocab = load_vocab('/DATA/joosung/pytorch_pretrained_BERT_master/korea_vocab.txt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_use = True 

from cls_model import BertClassifier
model = BertClassifier(drop_rate=0, gpu=gpu_use)

model_name='nodrop/bert_classifier_25000'
model.load_state_dict(torch.load('models/{}'.format(model_name)))

model = model.to(device)
model.train()

def main():
    initial_lr = 0.001
    cls_trainer = optim.Adamax(model.cls_params, lr=initial_lr) # initial 0.001
    max_grad_norm = 10

    f = open("../../sentiment_data/nsmc-master/ratings_train.txt", 'r')
    lines = f.readlines()
    data_number = len(lines)-1

    line_number = 0
    for it in tqdm(range(25001,n_iter)):
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
                if input_label == 0:
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

        gt = torch.from_numpy(np.asarray(labels)).type(torch.FloatTensor).to(device) # (batch, 2)
        cls_input = torch.cat(cls_value_list).type(torch.FloatTensor).to(device) # (batch, 768)

        cls_out = model.classifier(cls_input) # (batch, 2)
        loss = model.cls_loss(gt, cls_out)

        cls_trainer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.cls_params, max_grad_norm)            
        cls_trainer.step()

        summary.add_scalar('cls_loss', loss.item(), it)

        if (it+1) % 5000 == 0:
            save_model(it+1)

def save_model(iter):
    if not os.path.exists('models/nodrop'):
        os.makedirs('models/nodrop')
    torch.save(model.state_dict(), 'models/nodrop/bert_classifier_{}'.format(iter))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()