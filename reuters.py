
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics

from fasttext import FastVector

# --
# Helpers

def featurize(x, d):
    return np.vstack([d[xx] for xx in x.split(' ') if xx in d]).mean(axis=0)

# --
# Params

source_lang = 'en'
target_lang = 'de'

# --
# Load embeddings

dicts = {
    "source" : FastVector(vector_file='./data/wiki.%s.vec' % source_lang),
    "target" : FastVector(vector_file='./data/wiki.%s.vec' % target_lang),
}

dicts['source'].apply_transform('./alignment_matrices/%s.txt' % source_lang)
dicts['target'].apply_transform('./alignment_matrices/%s.txt' % target_lang)

# --
# Load Reuters

paths = {
    "source" : {
        "train" : '/home/bjohnson/projects/reuters-tools/data/%s/train.ft' % source_lang,
        "valid" : '/home/bjohnson/projects/reuters-tools/data/%s/valid.ft' % source_lang,
        "test" : '/home/bjohnson/projects/reuters-tools/data/%s/test.ft' % source_lang,
    },
    "target" : {
        "train" : '/home/bjohnson/projects/reuters-tools/data/%s/train.ft' % target_lang,
        "valid" : '/home/bjohnson/projects/reuters-tools/data/%s/valid.ft' % target_lang,
        "test" : '/home/bjohnson/projects/reuters-tools/data/%s/test.ft' % target_lang,
    }
}

data = {"source" : {}, "target" : {}}
X = {"source" : {}, "target" : {}}
y = {"source" : {}, "target" : {}}
for k in ['source', 'target']:
    print k
    for kk,v in paths[k].iteritems():
        print kk
        data[k][kk] = pd.read_csv(v, sep='\t', header=None)
        data[k][kk].columns = ('lab', 'text')
        
        X[k][kk] = np.vstack(data[k][kk].text.apply(lambda x: featurize(x, dicts[k])))
        y[k][kk] = np.array(data[k][kk].lab)

# --
# Transferring classifiers across languages

svc = LinearSVC().fit(X['source']['train'], y['source']['train'])

metrics.accuracy_score(y['source']['valid'], svc.predict(X['source']['valid']))
metrics.accuracy_score(y['target']['valid'], svc.predict(X['target']['valid']))

# en -> de: Error rate is ~ 4x single language training (0.96 -> 0.80)
# de -> en: Error rate is ~ 4x single language training (0.94 -> 0.74)

# Also, want stronger than normal regularization -- in at least this case,
# it appears that optimizing hyperparameters on the source language yields
# worse performance on the target language.  Makes sense.
