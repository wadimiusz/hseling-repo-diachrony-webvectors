import sys
import gensim
import logging
import pickle
import pandas as pd
from argparse import ArgumentParser
from os import path
from smart_open import open

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = ArgumentParser()
parser.add_argument('--models', '-m', required=True, help='path to the word embeddings models')
parser.add_argument('--corpora', '-c', required=True, help='path to the corpora')
args = parser.parse_args()

corpuses = {}
for year in range(2010, 2020):
    print('Loading corpus:', year, file=sys.stderr)
    df = pd.read_csv(
        path.join(args.corpora, '{year}_contexts.csv.gz'.format(year=year)),
        index_col='ID')
    corpuses.update({year: df})

vocabs = {}
for year in range(2010, 2020):
    model = gensim.models.KeyedVectors.load(
        path.join(args.models, '{year}_rnc_incremental.model'.format(year=year)))
    vocabs[year] = set([w for w in model.index2word if w.endswith('_NOUN')
                        or w.endswith('_PROPN') or w.endswith('_ADJ')])

united_vocab = set.union(*map(set, vocabs.values()))
print('Total words:', len(united_vocab), file=sys.stderr)

for word in united_vocab:
    print('Saving word:', word, file=sys.stderr)
    out_dict = {}
    for year in vocabs:
        if word in vocabs[year]:
            corpus = corpuses.get(year)
            samples = []
            for idx, lemmas, raw in corpus[['LEMMAS', 'RAW']].itertuples():
                lemmas_split = lemmas.split()
                if word in lemmas_split:
                    samples.append([lemmas_split, raw])
            out_dict.update({year: samples})
    with open('pickles/{word}.pickle.gz'.format(word=word), 'wb') as f:
        pickle.dump(out_dict, f)
