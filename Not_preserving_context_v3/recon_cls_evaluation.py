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

vocab_size = 6222
mb_size = 1

vocab = load_vocab('/DATA/joosung/pytorch_pretrained_BERT_master/korea_vocab.txt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_use = True 

def main():
    from model import TowardModel
    model = TowardModel(gpu=gpu_use)
    model_name='simple_model_50000'
    model.load_state_dict(torch.load('models/{}'.format(model_name)))

    model = model.to(device)
    model.eval()


    f = open("../../sentiment_data/nsmc-master/ratings_test.txt", 'r')
    lines = f.readlines()
    data_number = len(lines)-1
    n_iter = data_number

    line_number = 0
    accuracy = 0
    for it in tqdm(range(n_iter)):
        inputs_value = []
        inputs_token = []
        labels = []
        fake_labels = []

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
                    input_label = [1, 0]
                    fake_label = [0, 1]
                else:
                    input_label = [0, 1]
                    fake_label = [1, 0]
                labels.append(input_label)
                fake_labels.append(fake_label)

                output_bert = embedding(input_sentence, bert_model, bert_tokenizer)            

                for token_order in range(len(output_bert['features'])):
                    token_value = np.asarray(output_bert['features'][token_order]['layers'][0]['values'])
                    token_value = torch.from_numpy(token_value)
                    token_value = token_value.unsqueeze(0)                
                    token_value_list.append(token_value)

                    try:
                        token_list.append(vocab[output_bert['features'][token_order]['token']])
                    except:
                        token_list.append(vocab['[UNK]'])                     
                token_value_emb = torch.cat(token_value_list,0)
                token_value_emb = token_value_emb.unsqueeze(1).type(torch.FloatTensor) # [token_len, 1, emb_dim]
                token_value_emb = token_value_emb[1:-1,:,:] # without [CLS], [SEP]


                inputs_value.append(token_value_emb)

                tokens = np.asarray(token_list[:-1]) # without [SEP]
                tokens = torch.from_numpy(tokens)
                inputs_token.append(tokens) # [[token_len], ...]

                inputsize += 1

        enc_value = padding_values(inputs_value).to(device)
        dec_token = padding_tokens(inputs_token).to(device)

        attributes = torch.from_numpy(np.asarray(labels)).type(torch.FloatTensor).to(device)
        fake_attributes = torch.from_numpy(np.asarray(fake_labels)).type(torch.FloatTensor).to(device)

        ## inference
        recon_gen_tokens = model.generator(enc_value, attributes, train=False)

        if attributes[0].cpu().numpy().argmax() == 0:
            gt = 'negative'
        else:
            gt = 'positive'
            
        cls_1 = model.discriminator(recon_gen_tokens)
        if cls_1.argmax() == 0:
            cls_1 = 'negative'
        else:
            cls_1 = 'positive'

        if gt == cls_1:
            accuracy += 1            
        else:
            pass
        
        if it % 10000 == 0:
            print("중간 정확도: {}%".format(accuracy/(it+1)*100))


    print("정확도: {}%".format(accuracy/n_iter*100))


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