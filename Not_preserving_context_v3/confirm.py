import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from tqdm import tqdm
import os

import sys
sys.path.insert(0, "/DATA/joosung/pytorch_pretrained_BERT_master")
from pytorch_pretrained_bert.tokenization import load_vocab
from pytorch_bert_embedding import *
import torch.optim as optim

bert_model, bert_tokenizer = bert_model_load('bert-base-multilingual-cased')

from tensorboardX import SummaryWriter
summary = SummaryWriter(logdir='./logs')

n_iter = 50000
vocab_size = 6222
mb_size = 32

vocab = load_vocab('/DATA/joosung/pytorch_pretrained_BERT_master/korea_vocab.txt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_use = True 


from model import TowardModel
model = TowardModel(gpu=gpu_use)
model_name='simple_model_30000'
model.load_state_dict(torch.load('models/{}'.format(model_name)))

model = model.to(device)
model.train()

def main():
    initial_lr = 0.001
    aed_trainer = optim.Adamax(model.aed_params, lr=initial_lr, betas=(0.5, 0.999)) # initial 0.001
    cls_trainer = optim.Adamax(model.cls_params, lr=initial_lr) # initial 0.001
    gen_cls_trainer = optim.Adamax(model.att_params, lr=initial_lr/2) # att parameters
    
    max_grad_norm = 10
    weight_decay = 5000

    f = open("../../sentiment_data/nsmc-master/ratings_train.txt", 'r')
    lines = f.readlines()
    data_number = len(lines)-1

    line_number = 0
    for it in tqdm(range(30000, n_iter)):
        if (it+1) % weight_decay == 0:
            initial_lr = initial_lr / 2
            aed_trainer = optim.Adamax(model.aed_params, lr=initial_lr, betas=(0.5, 0.999)) # initial 0.001
            cls_trainer = optim.Adamax(model.cls_params, lr=initial_lr) # initial 0.001
            gen_cls_trainer = optim.Adamax(model.att_params, lr=initial_lr/2) # att parameters        
        
        inputs_value = []
        inputs_token = []
        labels = []
        fake_labels = []

        input_line = f.readline()

        inputsize = 0  # count batch size
        while inputsize < mb_size:
            line_number = line_number % data_number + 1

            token_value_list = []
            token_list = []

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
                    print(input_label)
                    input_label = [1, 0]
                    fake_label = [0, 1]
                else:
                    print(input_label)
                    input_label = [0, 1]
                    fake_label = [1, 0]
                inputsize += 1


def padding_tokens(tokens_list):
    """
    tokens: list [[token ... token], ..., [token ... token]] mbsize / seq_len
    ouput_tokens: [max_length, mb_size] 
    """
    max_len = 0
    for i in range(len(tokens_list)):        
        if max_len < len(tokens_list[i]):
            max_len = len(tokens_list[i])
            
    PAD_IDX = 2
    SEP_IDX = 4
    for k in range(len(tokens_list)):
        padding_length = max_len - len(tokens_list[k])
    
        padding_list = []
        for p in range(padding_length):
            padding_list.append(PAD_IDX)
            
        padding_tokens = np.asarray(padding_list)
        padding_tokens = torch.from_numpy(padding_tokens)
        if padding_length > 0:
            tokens_list[k] = torch.cat([tokens_list[k], padding_tokens])
        
        sep_list = np.asarray([SEP_IDX])
        sep_tokens = torch.from_numpy(sep_list)
        tokens_list[k] = torch.cat([tokens_list[k], sep_tokens]).unsqueeze(0)
        
    return torch.cat(tokens_list, 0)

def save_model(iter):
    if not os.path.exists('models/'):
        os.makedirs('models/')
    torch.save(model.state_dict(), 'models/simple_model_{}'.format(iter))  
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()