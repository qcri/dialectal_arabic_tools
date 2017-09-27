#!/usr/bin/env python
# -*- coding: utf8 -*-
# Generate CoNLL format
import codecs
import numpy as np
from collections import Counter
from itertools import chain
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

__author__ = 'disooqi'


# input_file = r'/home/disooqi/PycharmProjects/dialectal_arabic_tools/data/glf.trg'
# output_file = r'/home/disooqi/PycharmProjects/dialectal_arabic_tools/data/all_02.trg.conll'
#
# with codecs.open(input_file, encoding='utf-8') as orig:
#     with codecs.open(output_file, mode='a', encoding='utf-8') as conll:
#         for line in orig:
#
#             if line.strip() in['EOS', 'EOTWEET']:
#                 conll.write('\n')
#                 continue
#             clitics = line.strip().split('+')
#             for clitic in clitics:
#                 if len(clitic) == 1:
#                     conll.write(clitic + '\tS\n')
#                 elif len(clitic) == 2:
#                     conll.write(clitic[0] + '\tB\n')
#                     conll.write(clitic[1] + '\tE\n')
#                 else:
#                     conll.write(clitic[0] + '\tB\n')
#                     for ch in clitic[1:-1]:
#                         conll.write(ch + '\tM\n')
#                     else:
#                         conll.write(clitic[-1] + '\tE\n')
#             else:
#                 conll.write('WB\tWB\n')
#         else:
#             conll.write('=======================================')

def _fit_term_index(terms, reserved=[], preprocess=lambda x: x):
    # todo I need to know that this is working well
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = reserved + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}


def split_dataset(file_path, test_size=0.2, shuffle=True):
    SEG_TAGS = ('S', 'B', 'E', 'M')
    words, target_of_words = None, None
    with codecs.open(file_path, encoding='utf-8') as conell:
        list_of_lines = [line.strip().split() for line in conell if len(line.strip().split()) == 2]
        chars, targets = list(zip(*list_of_lines))
        words = ''.join(chars).split('WB')
        target_of_words = ''.join(targets).split('WB')

        words = [tuple(word) for word in words]
        target_of_words = [tuple(trg) for trg in target_of_words]

    index2word = _fit_term_index(words, reserved=['<PAD>', '<UNK>'])
    word2index = _invert_index(index2word)

    index2pos = SEG_TAGS
    pos2index = _invert_index(index2pos)

    X_train_idx = np.array([[word2index[w] for w in words] for words in words])
    y_train_idx = np.array([[pos2index[t] for t in pos_tags] for pos_tags in target_of_words])

    X = sequence.pad_sequences(X_train_idx, maxlen=50, padding='post')
    y_train = sequence.pad_sequences(y_train_idx, maxlen=50, padding='post')
    y = np.expand_dims(y_train, -1)

    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)


if __name__ == '__main__':
    split = split_dataset(r'../files/all.trg.conll')
    print(len(split))
