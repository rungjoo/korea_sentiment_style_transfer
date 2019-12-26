import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

import sys
sys.path.insert(0, "/DATA/joosung/pytorch_pretrained_BERT_master")
from pytorch_pretrained_bert.tokenization import load_vocab
from pytorch_bert_embedding import *

class BertClassifier(nn.Module):
    def __init__(self, drop_rate=0, gpu = True):
        super().__init__()        
        self.gpu = gpu
        self.emb_dim = 768
        
        """
        evaluation bert classifier
        """
        self.channel_out = 100
        self.conv2d_2 = nn.Conv2d(1,self.channel_out,(2,self.emb_dim))
        self.conv2d_3 = nn.Conv2d(1,self.channel_out,(3,self.emb_dim))
        self.conv2d_4 = nn.Conv2d(1,self.channel_out,(4,self.emb_dim))
        self.conv2d_5 = nn.Conv2d(1,self.channel_out,(5,self.emb_dim))
        self.fc_drop = nn.Dropout(drop_rate)
        self.disc_fc = nn.Linear(4*self.channel_out, 2)       
        
        ## parameters
        self.cls_params = list(self.conv2d_2.parameters())+list(self.conv2d_3.parameters())+list(self.conv2d_4.parameters())+list(self.conv2d_5.parameters())+list(self.disc_fc.parameters())

        
    def classifier(self, cls_token_value):
        """
        cls_token_value: (batch, seq_len, emb_dim)
        """
        word_2d = cls_token_value.unsqueeze(1) # (batch, 1, seq_len, word_dim)

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
        
        