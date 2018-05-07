from collections import Counter
import sys
def dist(fname):
    scnt = Counter()
    ucnt = Counter()
    bcnt = Counter()
    with open(fname) as f:
        for line in f.readlines():
            scnt[line] += 1
            words = line.split()
            ucnt.update(words)
            bcnt.update([(i,j) for i,j in zip(words[:-1],words[1:])])
    return len(scnt), len(ucnt), len(bcnt)



if __name__ == "__main__":
    fname = sys.argv[1]
    s, u, b = dist(fname)
    print "sents: %d, unigrams: %d, bigrams: %d"%(s, u, b)
