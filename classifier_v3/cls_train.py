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

# model_name='drop0.4/bert_classifier_20000'
# model.load_state_dict(torch.load('models/{}'.format(model_name)))

model = model.to(device)
model.train()

def main():
    initial_lr = 0.001
    cls_trainer = optim.Adamax(model.cls_params, lr=initial_lr) # initial 0.001
    max_grad_norm = 10
    weight_decay = 5000

    f = open("../../sentiment_data/nsmc-master/ratings_train.txt", 'r')
    lines = f.readlines()
    data_number = len(lines)-1

    line_number = 0
    for it in tqdm(range(n_iter)):
        if (it+1) % weight_decay == 0:
            initial_lr = initial_lr / 2
            cls_trainer = optim.Adamax(model.cls_params, lr=initial_lr) # initial 0.001
        
        cls_value_list = []
        labels = []

        inputsize = 0  # count batch size
        while inputsize < mb_size:
            line_number = line_number % data_number + 1
            
            token_value_list = []

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
                
                for token_order in range(len(output_bert['features'])):
                    token_value = np.asarray(output_bert['features'][token_order]['layers'][0]['values'])
                    token_value = torch.from_numpy(token_value)
                    token_value = token_value.unsqueeze(0)
                    token_value_list.append(token_value)                       
                token_value_emb = torch.cat(token_value_list,0) # (token_len, emb_dim)
                token_value_emb = token_value_emb.unsqueeze(1).type(torch.FloatTensor) # (token_len, 1, emb_dim)
                token_value_emb = token_value_emb[1:-1,:,:] # without [CLS], [SEP]

                cls_value_list.append(token_value_emb) # [(token_len, 1, 768),...]

                inputsize += 1
                
        gt = torch.from_numpy(np.asarray(labels)).type(torch.FloatTensor).to(device) # (batch, 2)
        cls_input = padding_values(cls_value_list).type(torch.FloatTensor).to(device) # (token_len, batch, 768)
        cls_input = cls_input.transpose(0, 1) # (batch, token_len, 768)
        
        cls_out = model.classifier(cls_input) # (batch, 2)
        loss = model.cls_loss(gt, cls_out)

        cls_trainer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.cls_params, max_grad_norm)            
        cls_trainer.step()

        summary.add_scalar('cls_loss', loss.item(), it)

        if (it+1) % 10000 == 0:
            save_model(it+1)


def padding_values(emb_list):
    max_len = 0
    for i in range(len(emb_list)):        
        if max_len < emb_list[i].shape[0]:
            max_len = emb_list[i].shape[0]
    
    padding_list= []
    for i in range(len(emb_list)):
        emb = emb_list[i]
        bert_dim = emb.shape[2]
        
        padding_length = max_len-emb.shape[0]
        padding_zero = np.zeros([padding_length, 1, bert_dim])
        padding_tensor = torch.from_numpy(padding_zero).type(torch.FloatTensor)        
        
        padding_emb = torch.cat([emb, padding_tensor], 0)
        
        padding_list.append(padding_emb)
        
    return torch.cat(padding_list,1)
            
def save_model(iter):
    if not os.path.exists('models/nodrop'):
        os.makedirs('models/nodrop')
    torch.save(model.state_dict(), 'models/nodrop/bert_classifier_{}'.format(iter))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()