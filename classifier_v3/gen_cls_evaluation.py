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
        cls_value_list = []
        labels = []

        inputsize = 0  # count batch size
        while inputsize < mb_size:
            line_number = line_number % data_number + 1
            
            token_value_list = []

            input_line = lines[line_number]
            
            k=0
            while True:
                if input_line[k] == ' ':
                    break
                k+=1
            input_split = [input_line.strip()[:k], input_line.strip()[k:-1].strip(), input_line.strip()[-1]]            
#             input_split = re.split('\t', input_line)
            
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
                
                if int(input_label) == 0: # negative
                    input_label = [1, 0]
                else: # positive
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

        ## inference
        cls_input = padding_values(cls_value_list).type(torch.FloatTensor).to(device) # (token_len, batch, 768)
        cls_input = cls_input.transpose(0, 1) # (batch, token_len, 768)
        
        cls_out = model.classifier(cls_input) # (batch, 2)

        if cls_out.argmax(1).item() == input_label.index(1) and it%2 == 0:
            real_correct+=1
        if cls_out.argmax(1).item() == input_label.index(1) and it%2 == 1:
            fake_correct+=1

        if (it+1) % 10000 == 0:
            print("Real accuracy: {}%, Fake accuracy: {}%".format(real_correct/real_num*100, fake_correct/fake_num*100))

    print("Accuracy: {}%, Real accuracy: {}%, Fake accuracy: {}%".format((real_correct+fake_correct)/(real_num+fake_num)*100, real_correct/real_num*100, fake_correct/fake_num*100))
    print("Test set 구성, positive: {}, negative: {}".format(real_num, fake_num))

def padding_values(emb_list):
    max_len = 5 # original 0 
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

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()