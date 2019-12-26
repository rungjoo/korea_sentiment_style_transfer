import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

import sys
sys.path.insert(0, "/DATA/joosung/pytorch_pretrained_BERT_master")
from pytorch_pretrained_bert.tokenization import load_vocab
from pytorch_bert_embedding import *

class TowardModel(nn.Module):
    def __init__(self, drop_rate=0, gpu = True):
        super(TowardModel, self).__init__()
        
        self.UNK_IDX = 1
        self.PAD_IDX = 2
        self.START_IDX = 3
        self.EOS_IDX = 4
        self.MAX_SENT_LEN = 30
        self.gpu = gpu
        
        self.n_vocab = 6222
        self.emb_dim = 768
        
        self.vocab = load_vocab('/DATA/joosung/pytorch_pretrained_BERT_master/korea_vocab.txt')
        self.pos2token = {}
        for k,v in self.vocab.items():
            self.pos2token[v] = k
        
        """
        attribute matrix
        """
        self.hidden_A = 2 # 128
#         self.matrix_A = nn.Linear(2, self.hidden_A) 
        
        """
        Encoder
        """
        self.hidden_GRU_E = self.emb_dim-self.hidden_A # 766=768-2
        
        self.GRU_E = nn.GRU(self.emb_dim, self.hidden_GRU_E)
                
        """
        Decoder
        """
        self.word_dim = self.emb_dim # 768
        self.hidden_GRU_D = self.emb_dim # 768
        
        self.word_emb = nn.Embedding(self.n_vocab, self.word_dim, self.PAD_IDX)
        self.GRU_D = nn.GRU(self.word_dim, self.hidden_GRU_D)
        self.matrix_D = nn.Linear(self.hidden_GRU_D, self.n_vocab)
        
        """
        Discriminator(classifier)
        """
        self.channel_out = 100
        self.conv2d_2 = nn.Conv2d(1,self.channel_out,(2,self.emb_dim))
        self.conv2d_3 = nn.Conv2d(1,self.channel_out,(3,self.emb_dim))
        self.conv2d_4 = nn.Conv2d(1,self.channel_out,(4,self.emb_dim))
        self.conv2d_5 = nn.Conv2d(1,self.channel_out,(5,self.emb_dim))
        self.fc_drop = nn.Dropout(drop_rate)
        self.disc_fc = nn.Linear(4*self.channel_out, 2)
        
        
        ## parameters
        # self.matrix_A.parameters()
        self.aed_params = list(self.GRU_E.parameters())+list(self.word_emb.parameters())+list(self.GRU_D.parameters())+list(self.matrix_D.parameters())
#         self.aed_params = chain(
#             self.GRU_E.parameters(), self.word_emb.parameters(), self.GRU_D.parameters(), self.matrix_D.parameters()
#         )

        self.cls_params = list(self.conv2d_2.parameters())+list(self.conv2d_3.parameters())+list(self.conv2d_4.parameters())+list(self.conv2d_5.parameters())+list(self.disc_fc.parameters())
        
        self.att_params = list(self.GRU_E.parameters())+list(self.GRU_D.parameters())+list(self.matrix_D.parameters())
        
    def encoder(self, enc_input):
        # enc_input: (seq_len, batch, emb_dim)
        _, final_h = self.GRU_E(enc_input) # (1, batch, emb_dim)
        
        if self.gpu == True:
            return final_h.cuda()
        else:
            return final_h  # (1, batch, hidden_GRU_E)
    
    def decoder(self, enc_input, dec_tokens, attributes):
        """
        enc_input: (seq_len, batch, emb_dim)
        dec_tokens: (batch, seq_len+2) with [CLS] [SEP]
        attributes: (batch, 2)
        """
        enc_vec = self.encoder(enc_input) # (1, batch, hidden_GRU_E)
        att = attributes.unsqueeze(0) # (1, batch, 2)
        enc_att = att
#         enc_att = self.matrix_A(att) # (1, batch, hidden_A)
        init_h = torch.cat([enc_vec, enc_att], 2) # (1, batch, hidden_GRU_E+hidden_A)         
        
        dec_input = dec_tokens[:,:-1] # without [SEP]
        dec_1 = self.word_emb(dec_input) # (batch, seq_len, word_dim) with [CLS] / exactly seq_len+1
        dec_1 = dec_1.transpose(0,1) # (seq_len, batch, word_dim)
        
        dec_2, _ = self.GRU_D(dec_1, init_h) # (seq_len, batch, hidden_GRU_D)
        
        dec_out_1 = self.matrix_D(dec_2) # (seq_len, batch, n_vocab)
        
        if self.gpu == True:
            return dec_2.cuda(), dec_out_1.cuda()
        else:
            return dec_2, dec_out_1
    
    def generator(self, enc_input, attributes, train = True): # generate tokens
        batch_size = enc_input.shape[1]
        
        """
        GRU_D initialization
        """
        enc_vec = self.encoder(enc_input) # (1, batch, hidden_E1)
        att = attributes.unsqueeze(0) # (1, batch, 2)
        enc_att = att
#         enc_att = self.matrix_A(att) # (1, batch, hidden_A)
        init_h = torch.cat([enc_vec, enc_att], 2) # (1, batch, hidden_E1+hidden_A)
        
        """
        start token setting
        """
        start_token_list = [self.START_IDX]
        start_token = torch.from_numpy(np.asarray(start_token_list))
        
        start_token = start_token.view(1,-1) # (1, 1)
        if self.gpu == True:
            start_token = start_token.cuda()
            
        dec_in = self.word_emb(start_token) # (1, 1, word_dim)
        dec_in = dec_in.repeat(1,batch_size,1) # (1, batch, word_dim)
        
        """
        generate sentence length
        """
        # consider SEP
        if train==True:
            gen_length = enc_input.shape[0]
            
            gen_token_list = []
            for i in range(gen_length):
                gen_out, init_h = self.GRU_D(dec_in, init_h) # (1, batch, hidden_GRU_D)

                gen_vocab = self.matrix_D(gen_out) # (1, batch, n_vocab)            
                token_prob = F.softmax(gen_vocab, 2) # (1, batch, n_vocab)
                token_pos = token_prob.argmax(2) # (1, batch)

                dec_in = self.word_emb(token_pos) # (1, batch, word_dim)

                gen_token_list.append(token_pos) # [(1,batch),... ]
        else:
            gen_length = self.MAX_SENT_LEN
            
            gen_token_list = []
            for i in range(gen_length):
                gen_out, init_h = self.GRU_D(dec_in, init_h) # (1, batch, hidden_GRU_D)

                gen_vocab = self.matrix_D(gen_out) # (1, batch, n_vocab)            
                token_prob = F.softmax(gen_vocab, 2) # (1, batch, n_vocab)
                token_pos = token_prob.argmax(2) # (1, batch)
                
                if token_pos == self.EOS_IDX:
                    break

                dec_in = self.word_emb(token_pos) # (1, batch, word_dim)

                gen_token_list.append(token_pos) # [(1,batch),... ]        
        
            
        if self.gpu == True:
            return torch.cat(gen_token_list, 0).transpose(0, 1).cuda() # (batch, gen_length)
        else:
            return torch.cat(gen_token_list, 0).transpose(0, 1) # (batch, gen_length)
        
    def discriminator(self, tokens):
        """
        tokens: (batch, seq_len)
        """
        if tokens.size(1) < 5:
            padding_size = 5-tokens.size(1)
            padding_token = []
            for k in range(tokens.size(0)):
                temp = []
                for i in range(padding_size):
                    temp.append(self.PAD_IDX)
                padding_token.append(temp)                
            padding_token=torch.from_numpy(np.array(padding_token))
            if self.gpu == True:
                padding_token = padding_token.cuda()
            tokens=torch.cat([tokens,padding_token], 1) # (batch, seq_len+padding) = (batch, 5)
        
        word_emb = self.word_emb(tokens) # (batch, seq_len, word_dim)
        word_2d = word_emb.unsqueeze(1) # (batch, 1, seq_len, word_dim)

        x2 = F.relu(self.conv2d_2(word_2d)).squeeze(3) # bi-gram, (batch, channel_out, seq_len-1)
        x3 = F.relu(self.conv2d_3(word_2d)).squeeze(3) # 3-gram, (batch, channel_out, seq_len-2)
        x4 = F.relu(self.conv2d_4(word_2d)).squeeze(3) # 4-gram, (batch, channel_out, seq_len-3)
        x5 = F.relu(self.conv2d_5(word_2d)).squeeze(3) # 5-gram, (batch, channel_out, seq_len-4)
        
        # Max-over-time-pool
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2) # (batch, channel_out)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2) # (batch, channel_out)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2) # (batch, channel_out)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2) # (batch, channel_out)
        
        x = torch.cat([x2, x3, x4, x5], dim=1) # (batch, channel_out*4)

        y1 = self.fc_drop(x)
        y2 = self.disc_fc(y1) # (batch, 2)

        if self.gpu == True:
            return y2.cuda()
        else:
            return y2
    
    def gen2cls(self, enc_input, attributes, train = True):
        batch_size = enc_input.shape[1]
        
        """
        GRU_D initialization
        """
        enc_vec = self.encoder(enc_input) # (1, batch, hidden_E1)
        att = attributes.unsqueeze(0) # (1, batch, 2)
        enc_att = att
#         enc_att = self.matrix_A(att) # (1, batch, hidden_A)
        init_h = torch.cat([enc_vec, enc_att], 2) # (1, batch, hidden_E1+hidden_A)
        
        """
        start token setting
        """
        start_token_list = [self.START_IDX]
        start_token = torch.from_numpy(np.asarray(start_token_list))
        
        start_token = start_token.view(1,-1) # (1, 1)
        if self.gpu == True:
            start_token = start_token.cuda()
            
        dec_in = self.word_emb(start_token) # (1, 1, word_dim)
        dec_in = dec_in.repeat(1,batch_size,1) # (1, batch, word_dim)
        
        """
        generate sentence length
        """
        if train==True:
            gen_length = enc_input.shape[0]
        else:
            gen_length = self.MAX_SENT_LEN
        
        gen_token_emb = []
        for i in range(gen_length):
            gen_out, init_h = self.GRU_D(dec_in, init_h) # (1, batch, hidden_GRU_D)
            
            gen_vocab = self.matrix_D(gen_out) # (1, batch, n_vocab)            
            token_prob = F.softmax(gen_vocab, 2) # (1, batch, n_vocab)
            
            ## soft generation sequence because of back-propagation
            dec_in = torch.bmm(token_prob, self.word_emb.weight.unsqueeze(0))  # (1, batch, word_dim) = (1, batch, n_vocab) x (1, n_vocab, emb_dim)
                        
#             token_pos = token_prob.argmax(2) # (1, batch)            
#             dec_in = self.word_emb(token_pos) # (1, batch, word_dim)
            
            gen_token_emb.append(dec_in) # [(1, batch, word_dim),... ]
            
        word_emb = torch.cat(gen_token_emb, 0).transpose(0, 1) # (batch, seq_len, word_dim)        
      
        word_2d = word_emb.unsqueeze(1) # (batch, 1, seq_len, word_dim)

        x2 = F.relu(self.conv2d_2(word_2d)).squeeze(3) # bi-gram, (batch, channel_out, seq_len-1)
        x3 = F.relu(self.conv2d_3(word_2d)).squeeze(3) # 3-gram, (batch, channel_out, seq_len-2)
        x4 = F.relu(self.conv2d_4(word_2d)).squeeze(3) # 4-gram, (batch, channel_out, seq_len-3)
        x5 = F.relu(self.conv2d_5(word_2d)).squeeze(3) # 5-gram, (batch, channel_out, seq_len-4)
        
        # Max-over-time-pool
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2) # (batch, channel_out)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2) # (batch, channel_out)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2) # (batch, channel_out)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2) # (batch, channel_out)
        
        x = torch.cat([x2, x3, x4, x5], dim=1) # (batch, channel_out*4)

        y1 = self.fc_drop(x)
        y2 = self.disc_fc(y1) # (batch, 2)

        if self.gpu == True:
            return y2.cuda()
        else:
            return y2        
        
        
    def gen2sentence(self, gen_tokens):
        """
        gen_tokens: (batch, gen_length)
        """
        
        gen_sentences = []
        for i in range(gen_tokens.shape[0]): # batch
            token_str = ''
            for j in range(gen_tokens.shape[1]): # seq_len
                token = self.pos2token[gen_tokens[i,j].item()]
                token_str += ' '
                token_str += token                
            gen_sentence = self.pos2sentence(token_str)
            gen_sentences.append(gen_sentence)
        return gen_sentences
                
    def pos2sentence(self, token_string):
        token_string = token_string.replace(' ##','')
        token_string = token_string.replace (" ' ","'")
        token_string = token_string.replace (' ?','?')
        token_string = token_string.replace (' !','!')
        token_string = token_string.replace (' .','.')
        token_string = token_string.replace (' ,',',')
        token_string = token_string.strip()
        return token_string
    
    def AED_loss(self, targets, recon_out, gen_out):
        """
        targets: (batch, seq_len+2) with [CLS], [SEP]
        recon_out: (seq_len+1, batch, vocab_size) with [SEP]
        gen_out: (seq_len+1, batch, vocab_size) with [SEP]
        """
        final_targets = targets[:,1:] # (batch, seq_len+1) with only [SEP]
        recon_out = recon_out.permute(1,0,2) # (batch, seq_len+1, vocab_size)
        gen_out = gen_out.permute(1,0,2) # (batch, seq_len+1, vocab_size)
        
        final_targets = final_targets.contiguous()
        recon_out = recon_out.contiguous()
        gen_out = gen_out.contiguous()
                
        final_targets = final_targets.view(-1) # (batch*seq_len) easliy thinking about seq_len
        recon_out = recon_out.view(-1, recon_out.shape[2]) # (batch x seq_len, vocab_size)
        gen_out = gen_out.view(-1, gen_out.shape[2]) # (batch x seq_len, vocab_size)
        
        recon_loss = F.cross_entropy(recon_out, final_targets)
        bp_loss = F.cross_entropy(gen_out, final_targets)        
        
        if self.gpu == True:       
            return recon_loss.cuda(), bp_loss.cuda()
        else:
            return recon_loss, bp_loss
        
    def cls_loss(self, targets, cls_out):
        """
        targets: (batch, 2) / attributes [0,1] or [1,0]
        cls_out: (batch, 2) (logits)        """
        
        final_targets = targets.argmax(1) # (batch)
        cls_loss = F.cross_entropy(cls_out, final_targets)
        
        if self.gpu == True:       
            return cls_loss.cuda()
        else:
            return cls_loss
        
        