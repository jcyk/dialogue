import sys
sys.path.append('../')
import onmt
from lib import Configurable, Vocab, Data_Loader
from models import ADVModel
import argparse

def report_func(epoch, batch, num_batches, progress_step, start_time, lr, report_stats):
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = onmt.Statistics()
    return report_stats

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='default.cfg')
    args, extra_args = argparser.parse_known_args()
    opt = Configurable(args.config_file, extra_args)

    model = ADVModel(opt)
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
    optim.set_parameters(model.named_parameters())
    tgt_vocab = Vocab(opt.tgt_vocab)
    loss_compute = onmt.Loss.NMTLossCompute(model.generator, tgt_vocab).cuda()
    trainer = onmt.Trainer(model, loss_compute, loss_compute, optim)
    train_set = Data_Loader(opt.train_file, opt.batch_size)
    valid_set = Data_Loader(opt.dev_file, opt.batch_size)
    for epoch in xrange(opt.max_epoch):
        train_stats = trainer.train(train_set, epoch, report_func)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        valid_stats = trainer.validate(valid_set)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        trainer.epoch_step(valid_stats.ppl(), epoch)
        model.save_checkpoint(epoch, opt)
