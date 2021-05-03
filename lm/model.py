
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LM(nn.Sequential):
    def __init__(self, num_tokens, embedding_size, num_layers=2, dropout=0.0, tie_weights=True):
        super(LM, self).__init__()
        self.num_tokens = num_tokens 
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        self.encoder = nn.Embedding(self.num_tokens, embedding_size)
        self.rnn = nn.LSTM(embedding_size, self.embedding_size, 
                           num_layers=self.num_layers, 
                           bidirectional=False,
                          batch_first=True)
        self.rnn.flatten_parameters()
        
        self.decoder = nn.Linear(self.embedding_size, self.num_tokens)
        if tie_weights:
            self.decoder.weight = self.encoder.weight
        
    @property
    def num_params(self):
        return np.sum([p.numel() for p in self.parameters() if p.requires_grad])

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
        
        
    def forward(self, input, hidden=None):
        embedded = self.encoder(input)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        output = output.reshape(output.size(0) * output.size(1), output.size(2))
        output = self.decoder(output)
        return output, hidden
    
    def init_hidden(self, bsz):
        zeros = torch.zeros(self.num_layers, bsz, self.embedding_size).type(torch.FloatTensor)
        hidden = zeros, zeros
        return hidden
        

