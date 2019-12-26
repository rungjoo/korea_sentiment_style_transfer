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

from model import TowardModel
model = TowardModel(drop_rate=0, gpu=gpu_use)

model_name='simple_model_classifier_50000'
model.load_state_dict(torch.load('models/{}'.format(model_name)))

model = model.to(device)
model.eval()

def main():
    mb_size = 1
    f = open("../../sentiment_data/nsmc-master/generation_data_v3.txt", 'r')
    lines = f.readlines()
    data_number = len(lines)-1
    n_iter = data_number
    
    real_num = 0
    fake_num = 0
    
    real_correct = 0
    fake_correct = 0

    line_number = 0    
    for it in tqdm(range(n_iter)):
        inputs_value = []
        inputs_token = []
        labels = []

        input_line = f.readline()

        inputsize = 0  # count batch size
        while inputsize < mb_size:
            line_number = line_number % data_number + 1

            token_value_list = []
            token_list = []

            input_line = lines[line_number]

            k=0
            while True:
                if input_line[k] == ' ':
                    break
                k+=1
            input_split = [input_line.strip()[:k], input_line.strip()[k:-1].strip(), input_line.strip()[-1]]             

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
                if it % 2 == 0:
                    real_num += 1
                else:
                    fake_num += 1                 
                
                if input_label == 0:
                    input_label = [1, 0]
                else:
                    input_label = [0, 1]
                labels.append(input_label)

                output_bert = embedding(input_sentence, bert_model, bert_tokenizer)         

                for token_order in range(len(output_bert['features'])):
                    try:
                        token_list.append(vocab[output_bert['features'][token_order]['token']])
                    except:
                        token_list.append(vocab['[UNK]'])  

                tokens = np.asarray(token_list[:-1]) # without [SEP]
                tokens = torch.from_numpy(tokens)
                inputs_token.append(tokens) # [[token_len], ...]

                inputsize += 1

        dec_token = padding_tokens(inputs_token).to(device)

        ## inference
        cls_out = model.discriminator(dec_token[:,1:-1]) # (batch, 2) without [CLS], [SEP]
        if cls_out.argmax(1).item() == input_label.index(1) and it%2 == 0:
            real_correct+=1
        if cls_out.argmax(1).item() == input_label.index(1) and it%2 == 1:
            fake_correct+=1

        if (it+1) % 10000 == 0:
            print("Real accuracy: {}%, Fake accuracy: {}%".format(real_correct/real_num*100, fake_correct/fake_num*100))

    print("Accuracy: {}%, Real accuracy: {}%, Fake accuracy: {}%".format((real_correct+fake_correct)/(real_num+fake_num)*100, real_correct/real_num*100, fake_correct/fake_num*100))

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

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()