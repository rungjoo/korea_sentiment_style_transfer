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
    initial_lr = 0.001 / (2*2*2*2*2*2)
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
                    input_label = [1, 0]
                    fake_label = [0, 1]
                else:
                    input_label = [0, 1]
                    fake_label = [1, 0]
                labels.append(input_label)
                fake_labels.append(fake_label)

                output_bert = embedding(input_sentence, bert_model, bert_tokenizer)            
    #             print(input_sentence, len(output_bert['features']))

                for token_order in range(len(output_bert['features'])):
                    token_value = np.asarray(output_bert['features'][token_order]['layers'][0]['values'])
                    token_value = torch.from_numpy(token_value)
                    token_value = token_value.unsqueeze(0)                
                    token_value_list.append(token_value)

                    token_list.append(vocab[output_bert['features'][token_order]['token']])                       
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

        ## train
        enc_out = model.encoder(enc_value)
        dec_out, dec_out_vocab = model.decoder(enc_value, dec_token, attributes)

        gen_tokens = model.generator(enc_value, fake_attributes)
        gen_sentences = model.gen2sentence(gen_tokens)

        gen_value = []
        for i in range(len(gen_sentences)):
            output_bert = embedding(gen_sentences[i], bert_model, bert_tokenizer)

            token_value_list = []
            for token_order in range(len(output_bert['features'])):
                token_value = np.asarray(output_bert['features'][token_order]['layers'][0]['values'])
                token_value = torch.from_numpy(token_value)
                token_value = token_value.unsqueeze(0)                
                token_value_list.append(token_value)

            token_value_emb = torch.cat(token_value_list,0)
            token_value_emb = token_value_emb.unsqueeze(1).type(torch.FloatTensor) # [token_len, 1, emb_dim]
            token_value_emb = token_value_emb[1:-1,:,:] # without [CLS], [SEP]

            gen_value.append(token_value_emb) 


        gen_enc_value = padding_values(gen_value).to(device)

        gen_enc_out = model.encoder(gen_enc_value)
        gen_dec_out, gen_dec_out_vocab = model.decoder(gen_enc_value, dec_token, attributes)

        cls_out = model.discriminator(dec_token[:,1:-1]) # without [CLS], [SEP]
    #     gen_cls_out = model.discriminator(gen_tokens) # not working train to encoder/decoder
        gen_cls_out = model.gen2cls(enc_value, fake_attributes)

        """
        AE train
        """
        recon_loss, bp_loss = model.AED_loss(dec_token, dec_out_vocab, gen_dec_out_vocab)

        w = it//weight_decay * 0.1

        aed_loss = (1-w)*recon_loss + (1+w)*bp_loss # 30000 steps change to (1+w)

        summary.add_scalar('recon_loss', recon_loss.item(), it)
        summary.add_scalar('bp_loss', bp_loss.item(), it)
        summary.add_scalar('aed_loss', aed_loss.item(), it)

        aed_trainer.zero_grad()
        aed_loss.backward()      
        grad_norm = torch.nn.utils.clip_grad_norm_(model.aed_params, max_grad_norm)            
        aed_trainer.step()


        """
        cls train
        """
        cls_loss = model.cls_loss(attributes, cls_out)

        summary.add_scalar('cls_loss', cls_loss.item(), it)

        cls_trainer.zero_grad()
        cls_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.cls_params, max_grad_norm)            
        cls_trainer.step()

        """
        fianl train
        """
        gen_cls_loss = model.cls_loss(fake_attributes, gen_cls_out)

        summary.add_scalar('gen_cls_loss', gen_cls_loss.item(), it)

        gen_cls_trainer.zero_grad()
        gen_cls_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.att_params, max_grad_norm)            
        gen_cls_trainer.step()

    #     print("recon_loss:{}, bp_loss:{}, aed_loss:{}, cls_loss:{}, gen_cls_loss:{}".format(recon_loss.item(), bp_loss.item(), aed_loss.item(), cls_loss.item(), gen_cls_loss.item()))
    
        if (it+1) % 5000 == 0:
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