import sys
sys.path.append('../')
import onmt
from onmt.translate.Translator import Translator
from lib import Vocab, Data_Loader, Configurable
from models import ADVModel

def translate(model, fname, tgt_vocab, beam_size, n_best, out_file):
    model.eval()
    scorer = onmt.translate.GNMTGlobalScorer(0., -0., "none", "none")
    data_set = Data_Loader(fname, sort = False)
    worker = Translator(model, tgt_vocab, beam_size, n_best = n_best, max_length = 50, gpu = True, global_scorer=scorer)

    with open(out_file,'w') as f:
        for batch in data_set:
            ret = worker.translate_batch(batch)
            preds = ret['predictions']
            for pred in preds:
                sent = ' '.join(str(x) for x in pred[0])
                f.write(sent+'\n')

import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type = str)
    argparser.add_argument('--fname', type = str)
    argparser.add_argument('--beam_size', default = 1)
    argparser.add_argument('--n_best', default = 1)
    argparser.add_argument('--out_file', default = "test_out")
    argparser.add_argument('--config_file', default='default.cfg')
    args, extra_args = argparser.parse_known_args()
    opt = Configurable(args.config_file, extra_args, for_test = True)
    model = ADVModel(opt)
    model.load_checkpoint(args.model)
    vocab = Vocab(opt.tgt_vocab)
    translate(model, args.fname, vocab, args.beam_size, args.n_best, args.out_file)
