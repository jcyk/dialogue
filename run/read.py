import sys
sys.path.append('../')
from lib import Vocab

import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--vocab', default = '../data/movie_25000')
    argparser.add_argument('--fname', type = str)
    args, extra_args = argparser.parse_known_args()
    vocab = Vocab(args.vocab)

    with open(args.fname) as f:
        for line in f.readlines():
            sents = []
            for sent in line.split('|'):
                sent = ' '.join([vocab.i2s(int(word)) for word in sent.split()])
                sents.append(sent)
            new_line = '|'.join(sents)
            print new_line
