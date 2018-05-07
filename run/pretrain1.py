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
    optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)
    optim.set_parameters(model.D.named_parameters())

    tgt_vocab = Vocab(opt.tgt_vocab)
    train_set = Data_Loader(opt.train_file, opt.batch_size)
    valid_set = Data_Loader(opt.dev_file, opt.batch_size)
    for epoch in xrange(opt.max_epoch):
        model.train()
        for batch in train_set:
            src, src_len = batch.src
            tgt, tgt_len = batch.tgt
            model.zero_grad()
            loss, _, acc  = model(src, tgt, src_len, tgt_len, mode="D")
            loss = loss.mean()
            loss.backward()
            optim.step()
        model.eval()
        tot_acc = 0
        for batch in valid_set:
            src, src_len = batch.src
            tgt, tgt_len = batch.tgt
            _, _, acc  = model(src, tgt, src_len, tgt_len, mode="D")
            tot_acc += acc
        print epoch, tot_acc / len(valid_set)
        model.save_checkpoint(epoch, opt, 'D')
