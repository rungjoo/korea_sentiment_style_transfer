import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from tqdm import tqdm

import sys
sys.path.insert(0, "/DATA/joosung/pytorch_pretrained_BERT_master")
from pytorch_pretrained_bert.tokenization import load_vocab
from pytorch_bert_embedding import *
import torch.optim as optim

bert_model, bert_tokenizer = bert_model_load('bert-base-multilingual-cased')

n_iter = 1
vocab_size = 6222
mb_size = 1

vocab = load_vocab('/DATA/joosung/pytorch_pretrained_BERT_master/korea_vocab.txt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_use = True 

from model import TowardModel
model = TowardModel(gpu=gpu_use)
model_name='simple_model_50000'
model.load_state_dict(torch.load('models/{}'.format(model_name)))

model = model.to(device)
model.eval()

def main():
    input_sentence_list = []
    input_sentence_list.append('진짜 진~짜 할 일 없을 때 보시길...')
    input_sentence_list.append('S급 남녀주인공으로 만든 C급 영화')    
    input_sentence_list.append('짱깨 영화에 아까운 시간을 빼앗겼다.')
    input_sentence_list.append('마지막까지 힘을잃지않은 드라마~~')
    input_sentence_list.append('노래와 배우가 매력적인 영화..')  
    input_sentence_list.append('허술한부분이 많지만 10년전임을 감안할때 놀랍습니다!')   
    input_sentence_list.append('한국 애니 화이팅! 우수한 애니메이터들이 외국에 안가도 되도록 한국애니도 많은 지원해줬으면..')    
    input_sentence_list.append('와 보는내내 긴장감이 쩐다 쿠엔텐 타란티노 진짜 영화 잘만드는 거는 인정해야 할 듯')
    input_sentence_list.append('캐스팅된 배우가 아깝다')
    input_sentence_list.append('신선한 로맨스 저절로 웃음이난다')     
    
    input_label_list = []
    input_label_list.append(0)
    input_label_list.append(0)
    input_label_list.append(0)
    input_label_list.append(1)
    input_label_list.append(1)
    input_label_list.append(1)
    input_label_list.append(1)
    input_label_list.append(1)
    input_label_list.append(0)
    input_label_list.append(1)
    
    for k in range(len(input_sentence_list)):        
        inputs_value = []
        inputs_token = []
        labels = []
        token_value_list = []
        
        input_sentence = input_sentence_list[k]    
        input_label = input_label_list[k]

        output_bert = embedding(input_sentence, bert_model, bert_tokenizer)            

        for token_order in range(len(output_bert['features'])):
            token_value = np.asarray(output_bert['features'][token_order]['layers'][0]['values'])
            token_value = torch.from_numpy(token_value)
            token_value = token_value.unsqueeze(0)                
            token_value_list.append(token_value)
                  
        token_value_emb = torch.cat(token_value_list,0)
        token_value_emb = token_value_emb.unsqueeze(1).type(torch.FloatTensor) # [token_len, 1, emb_dim]
        token_value_emb = token_value_emb[1:-1,:,:] # without [CLS], [SEP]

        inputs_value.append(token_value_emb)

        enc_value = padding_values(inputs_value).to(device)

        if input_label == 0:
            start_label = [1, 0]
            gt = 'negative'
        else:
            start_label = [0, 1]
            gt = 'positive'

        print(input_sentence, gt)
        for i in range(11):
            labels = []
            latent_label = []
            if input_label == 0:
                latent_label.append(start_label[0] - i*0.1)
                latent_label.append(start_label[1] + i*0.1)
            else:
                latent_label.append(start_label[0] + i*0.1)
                latent_label.append(start_label[1] - i*0.1)
            labels.append(latent_label)            

            attributes = torch.from_numpy(np.asarray(labels)).type(torch.FloatTensor).to(device)

            ## inference
            recon_gen_tokens = model.generator(enc_value, attributes, train=False)
            real_gen_sentences = model.gen2sentence(recon_gen_tokens)

            cls_1 = model.discriminator(recon_gen_tokens)
            if cls_1.argmax() == 0:
                cls_1 = 'negative'
            else:
                cls_1 = 'positive'
            real_gen_sentences = postprocessing(real_gen_sentences)
            print("latent attributes: ", latent_label, " ", real_gen_sentences, " cls_out: ", cls_1)
            
        print(' ')


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

def postprocessing(token_list):
    sentence = token_list[0]
    
    sentence = sentence.replace('[PAD]', '')
    #     sentence = sentence.replace('[UNK]', '')
    
    sep_pos = sentence.find('[SEP]')
    if sep_pos == -1:
        return sentence.strip()
    else:
        return sentence[:sep_pos].strip() 

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()