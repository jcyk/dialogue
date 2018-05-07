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
from models import CNNEncoder, continuousEmbedding
import math

class Discriminator(nn.Module):
    def __init__(self, q_encoder, r_encoder):
        super(Discriminator, self).__init__()
        self.q_encoder = q_encoder
        self.r_encoder = r_encoder

    def forward(self, qs, rs):
        # qs, rs list of seq_len x batch_size x vocab_size or seq_len x batch_size
        q = [self.q_encoder(q).unsqueeze(1) for q in qs]
        r = [self.r_encoder(r).unsqueeze(2) for r in rs]
        q = torch.cat(q, 1) # batch_size x nums x dims
        r = torch.cat(r, 2) # batch_size x dims x nums
        qr_scores = torch.bmm(q, r) # batch_size x nums_q x nums_r
        qr_size = q.size(2)
        qr_scores = qr_scores / math.sqrt(qr_size)
        return qr_scores

#training methods:
# 1. approximate embedding layer 2. gumbel sample 3 vanilla reinforce 4 REGS

class ADVModel(nn.Module):
    def __init__ (self, model_opt, use_cuda = True, g_checkpoint = None, d_checkpoint = None):
        super(ADVModel, self).__init__()
        self.G = make_G_model(model_opt, use_cuda, g_checkpoint)
        self.D = make_D_model(model_opt, use_cuda, d_checkpoint)
        self.encoder = self.G.encoder
        self.decoder = self.G.decoder
        self.generator = self.G.generator
        self.backprob_mode = model_opt.backprob_mode
        self.model_opt = model_opt

    def sample(self, q, r_prefix, q_lens, temperature = 1., max_len = 50, eps = 1e-20):
        mode = self.backprob_mode
        enc_final, memory_bank = self.G.encoder(q, q_lens)
        dec_state = self.G.decoder.init_decoder_state(q, memory_bank, enc_final)
        inp = r_prefix

        batch_size = q.size(1)
        notyet = torch.ByteTensor(batch_size).fill_(1)
        notyet = notyet.cuda()
        log_prob = 0.
        result = []

        while notyet.any() and len(result)<=max_len:
            decoder_outputs, dec_state, attns = self.G.decoder(inp, memory_bank, dec_state, memory_lengths=q_lens)
            output = decoder_outputs[-1]
            if mode == "approx":
                noise = torch.zeros_like(output)
                noise.data.normal_(0, 0.01)
                cur_log_prob = self.G.generator(output + noise)
                _, next_token = torch.max(cur_log_prob, -1)
                inp = torch.exp(cur_log_prob + eps)
                inp.data.masked_fill_( (1-notyet).view(-1,1), 0.) #batch_size x vocab_size
            if mode == "gumbel":
                seed = torch.zeros_like(output)
                seed.data.uniform_(0, 1)
                gumbel_output = output - torch.log(- torch.log(seed + eps) +eps)
                gumbel_output = gumbel_output / temperature
                cur_log_prob = self.G.generator(gumbel_output)
                _, next_token = torch.max(cur_log_prob, -1)
                inp = torch.exp(cur_log_prob + eps)
                inp.data.masked_fill_( (1- notyet).view(-1, 1), 0.)  #batch_size x vocab_size
            if mode == "reinforce":
                cur_log_prob = self.G.generator(output)
                _, next_token = torch.max(cur_log_prob, -1)
                inp = torch.multinomial(torch.exp(cur_log_prob + eps), 1).squeeze(-1)
                cur_log_prob = torch.gather(cur_log_prob, -1, inp.view(-1, 1)).squeeze(-1)
                cur_log_prob.data.masked_fill_(1-notyet, 0.)
                log_prob = log_prob + cur_log_prob
                inp.data.masked_fill_( 1-notyet, 0) # batch_size
            inp = inp.unsqueeze(0)
            result.append(inp)
            endding = torch.eq(next_token, data.EOT_idx)
            notyet.masked_fill_(endding.data, 0)
        result = torch.cat(result, 0)
        return result, log_prob
        # result: len x batch_size x vocab_size or len x batch_size
        # log_prob: batch_size

    def forward(self, q, r, q_lens, r_lens =None, mode = None):
        if mode == "D":
            return self.D_func(q, r, q_lens, r_lens, r_index = 1)
        if mode == "G":
            return self.G_func(q, r, q_lens, r_lens)
        r = r[:-1]
        enc_final, memory_bank = self.encoder(q, q_lens)
        enc_state = self.decoder.init_decoder_state(q, memory_bank, enc_final)

        decoder_outputs, dec_state, attns = self.decoder(r, memory_bank, enc_state, memory_lengths= q_lens)
        return decoder_outputs, attns, dec_state

    def D_func(self, q, r, q_lens, r_lens, q_index = None, r_index= None):
        r_prefix = torch.LongTensor(1, q.size(1)).fill_(data.EOS_idx)
        r_prefix = Variable(r_prefix).cuda()
        pred, log_prob = self.sample(q, r_prefix, q_lens)
        # batch x nq x nr

        score_matrix = self.D([q],[pred, r[1:]])
        loss, acc = self._compute_D_loss(score_matrix, q_index, r_index)
        return loss, log_prob, acc

    def _compute_D_loss(self, score_matrix, q_index, r_index):
        if r_index is not None:
            batch_size, nq, nr = score_matrix.size()
            tgt = torch.LongTensor(batch_size * nq).fill_(r_index)
            tgt = Variable(tgt).cuda()
            score_matrix = score_matrix.view(-1, nr)
            _, label  = torch.max(score_matrix, -1)
            acc = torch.eq(label, tgt).float().mean().data[0]
            return (F.cross_entropy(score_matrix, tgt, reduce=False)).view(batch_size), acc

    def G_func(self, q, r, q_lens, r_lens):
        if self.backprob_mode == "approx" or self.backprob_mode == "gumbel":
            reward, log_prob, acc = self.D_func(q, r, q_lens, r_lens, r_index = 0)
            return reward
        if self.backprob_mode == "reinforce":
            reward, log_prob, acc= self.D_func(q, r, q_lens, r_lens, r_index = 0)
            reward.detach_()
            return reward  * log_prob

    def save_checkpoint(self, epoch, model_opt, suffix=""):
        if not os.path.exists(model_opt.save_model_path):
            os.makedirs(model_opt.save_model_path)

        torch.save({ 'G':self.G.state_dict(),
                     'D':self.D.state_dict(),
                     'model_opt':self.model_opt,
                     'epoch':epoch
                },
                model_opt.save_model_path+'/epoch%d'%epoch+suffix)

    def load_checkpoint(self, fname):
        ckpt = torch.load(fname)
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])

def make_D_model(model_opt, use_cuda = True, checkpoint= None):
    src_embeddings = continuousEmbedding(model_opt.src_vocab_size, model_opt.embedding_dim, data.PAD_idx)
    tgt_embeddings = continuousEmbedding(model_opt.tgt_vocab_size, model_opt.embedding_dim, data.PAD_idx)
    q_encoder = CNNEncoder(model_opt.embedding_dim, model_opt.qr_size, model_opt.filter_widths, model_opt.filter_nums, model_opt.dropout, src_embeddings)
    r_encoder = CNNEncoder(model_opt.embedding_dim, model_opt.qr_size, model_opt.filter_widths, model_opt.filter_nums, model_opt.dropout, tgt_embeddings)
    model = Discriminator(q_encoder, r_encoder)
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
    else:
        if model_opt.param_init != 0.0:
            print('Initializing D model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
    if use_cuda:
        model.cuda()
    return model


def make_G_model(model_opt, use_cuda = True, checkpoint=None):

    src_embeddings = continuousEmbedding(model_opt.src_vocab_size, model_opt.embedding_dim, data.PAD_idx)
    tgt_embeddings = continuousEmbedding(model_opt.tgt_vocab_size, model_opt.embedding_dim, data.PAD_idx)
    encoder = onmt.ModelConstructor.make_encoder(model_opt, src_embeddings)
    decoder = onmt.ModelConstructor.make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)
    model.model_type = "text"

    # Make Generator.
    generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, model_opt.tgt_vocab_size),
            nn.LogSoftmax(dim=-1))

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Initializing G model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)

    model.generator = generator
    # Make the whole model leverage GPU if indicated to do so.
    if use_cuda:
        model.cuda()

    return model
