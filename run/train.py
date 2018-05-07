from __future__ import division
import sys
sys.path.append('../')
import onmt
from lib import Configurable, Vocab, Data_Loader
from models import ADVModel
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='default.cfg')
    argparser.add_argument('--model', type = str)
    args, extra_args = argparser.parse_known_args()
    opt = Configurable(args.config_file, extra_args)
    model = ADVModel(opt)
    model.load_checkpoint(args.model)
    optimD = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)
    optimD.set_parameters(model.D.named_parameters())

    optimG = onmt.Optim(
              opt.optim, 0.1*opt.learning_rate, opt.max_grad_norm,
              lr_decay=opt.learning_rate_decay,
              start_decay_at=opt.start_decay_at,
              beta1=opt.adam_beta1,
              beta2=opt.adam_beta2,
              adagrad_accum=opt.adagrad_accumulator_init,
              decay_method=opt.decay_method,
              warmup_steps=opt.warmup_steps,
              model_size=opt.rnn_size)
    optimG.set_parameters(model.G.named_parameters())

    tgt_vocab = Vocab(opt.tgt_vocab)
    loss_compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab).cuda()
    train_set = Data_Loader(opt.train_file, opt.batch_size)
    valid_set = Data_Loader(opt.dev_file, opt.batch_size)
    batch_idx = 0
    for epoch in xrange(opt.max_epoch):
        for batch in train_set:
            src, src_len = batch.src
            tgt, tgt_len = batch.tgt
            model.zero_grad()
            G_turn = (batch_idx % 6 == 0)
            if G_turn:
                loss = model(src, tgt, src_len, tgt_len, mode='G')
                loss = loss.mean()
                loss.backward()
                outputs, attns, dec_state = model(src, tgt, src_len)
                loss_compute.sharded_compute_loss(batch, outputs, attns, 0, batch.tgt[0].size(0), 32, batch.batch_size)
                optimG.step()
            else:
                loss, _  = model(src, tgt, src_len, tgt_len, mode="D")
                loss = loss.mean()
                loss.backward()
                optimD.step()
            batch_idx += 1
        model.save_checkpoint(epoch, opt, 'GAN')
