import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt
import onmt.ModelConstructor
from onmt.Models import NMTModel
from onmt.modules import CopyGenerator
from torch.nn.init import xavier_uniform
from lib import data
import os
from torch.autograd import Variable

class CNNEncoder(nn.Module):
    def __init__(self, input_size, 
                hidden_size, 
                filter_widths, 
                filter_nums,
                dropout,
                embedding):
        super(CNNEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.filter_widths = filter_widths
        self.filter_nums = filter_nums
        self.conv = nn.ModuleList([nn.Conv1d(input_size, filter_nums, w, padding = w-1) for w in filter_widths])
        self.hidden = nn.Linear(len(filter_widths)*filter_nums, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.embedding = embedding

    def forward(self, input):
        emb = self.embedding(input).permute(1, 2, 0) # B x D x L
        xs = [ F.relu(conv(emb)) for conv in self.conv ]
        xs = torch.cat([ self.pool(F.relu(conv(emb))).squeeze(2) for conv in self.conv],1)
        xs = self.dropout(xs)
        output = self.dropout(F.tanh(self.hidden(xs)))
        # B x hidden_size
        return output

class continuousEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, _weight = None):
        super(continuousEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0.)

    @property
    def embedding_size(self):
        return self.embedding_dim

    def forward(self, input):
        self.weight.data[self.padding_idx].fill_(0.)
        discrete = (input.dim() <= 2)
        if discrete:
            return self._backend.Embedding.apply(
                input, self.weight, self.padding_idx,
                None, 2, False, False
            )
        return F.linear(input, torch.t(self.weight))
