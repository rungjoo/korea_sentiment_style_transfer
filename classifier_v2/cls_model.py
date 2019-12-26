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
        self.hidden_1 = 128
        self.fc1 = nn.Linear(self.emb_dim, self.hidden_1)
        self.fc_drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(self.hidden_1, 2)        
        
        ## parameters
        self.cls_params = list(self.fc1.parameters())+list(self.fc2.parameters())

        
    def classifier(self, cls_token_value):
        """
        cls_token_value: (batch, emb_dim)
        """
        fc1_drop = self.fc_drop(cls_token_value)        
        fc1_out = F.relu(self.fc1(fc1_drop)) # (batch, hidden_1) = (batch, emb_dim) x (emb_dim, hidden_1)
        
        fc2_out = self.fc2(fc1_out) # (batch, 2)
        
        if self.gpu == True:
            return fc2_out.cuda()
        else:
            return fc2_out
        
        
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
        
        