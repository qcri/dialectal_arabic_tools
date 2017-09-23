# -*- coding: utf8 -*-
import argparse
import codecs
from collections import Counter
from itertools import chain
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, ChainCRF, Dense, TimeDistributed

__author__ = 'disooqi'
__created__ = '21 Sep 2017'

parser = argparse.ArgumentParser(description="Segmentation module for the Dialectal Arabic")
group = parser.add_mutually_exclusive_group()
# file related options
parser.add_argument("-d", "--data-dir", help="directory containing train, test and dev file [default: %(default)s]")
parser.add_argument("-g", "--log-file", dest="log_file", help="log file [default: %(default)s]")
parser.add_argument("-p", "--model-dir", dest="model_dir",
                    help="directory to save the best models [default: %(default)s]")
parser.add_argument("-r", "--train-set", dest="train_set",
                    help="maximul sentence length (for fixed size input) [default: %(default)s]")  # input size
parser.add_argument("-v", "--dev-set", dest="validation_set",
                    help="source vocabulary size [default: %(default)s]")  # emb matrix row size
parser.add_argument("-s", "--test-set", dest="test_set",
                    help="target vocabulary size [default: %(default)s]")  # emb matrix row size
parser.add_argument('-i', '--input-file', dest='input_file', help='text file to be segmented [default: %(default)s]')
parser.add_argument('-o', '--output-file', dest='output_file',
                    help='the file used to save the result of segmentation [default: %(default)s]')
parser.add_argument('-f', '--format', help='format of the output-file [default: %(default)s]')
# network related
# input
parser.add_argument("-t", "--max-length", dest="maxlen", type=int,
                    help="maximul sentence length (for fixed size input) [default: %(default)s]")  # input size
parser.add_argument("-e", "--emb-size", dest="word_embedding_dim", type=int,
                    help="dimension of embedding [default: %(default)s]")  # emb matrix col size
# learning related
parser.add_argument("-a", "--learning-algorithm", dest="learn_alg",
                    help="optimization algorithm (adam, sgd, adagrad, rmsprop, adadelta) [default: %(default)s]")
parser.add_argument("-b", "--minibatch-size", dest="batch_size", type=int, help="minibatch size [default: %(default)s]")
parser.add_argument("-n", "--epochs", dest="nb_epoch", type=int, help="nb of epochs [default: %(default)s]")
# others
parser.add_argument("-V", "--verbose-level", dest="verbose_level",
                    help="verbosity level (0 < 1 < 2) [default: %(default)s]")
parser.add_argument('-m', '--mode',
                    help='choose between training or testing mode (segmentation) [default: %(default)s]')

parser.set_defaults(
    # file
    data_dir="./data/",
    model_dir="./model/",
    train_set="./data/all_train_f01.txt",
    validation_set="./data/all_dev_f01.txt",
    test_set="./data/all_test_f01.txt",
    log_file="run.log",
    input_file='./examples/example.in.txt',
    output_file='./examples/example.out.txt',
    format='conll',
    # network related
    maxlen=50,  # cut texts after this number of words (among top max_features most common words)
    word_embedding_dim=200,

    # learning related
    learn_alg="adam",  # sgd, adagrad, rmsprop, adadelta, adam (default)
    # loss = "binary_crossentropy" # hinge, squared_hinge, binary_crossentropy (default)
    batch_size=10,
    nb_epoch=100,
    verbose_level=1,
    lstm_dim=100,
    max_features=108,  # len(index2word)   and   len(_invert_index(index2word))  ## For evaluation mode only
    nb_pos_tags=4,  # len(index2pos)                                             ## For evaluation mode only
)

args = parser.parse_args()

index2pos = ['S', 'B', 'E', 'M']
pos2index = {'S': 0, 'B': 1, 'E': 2, 'M': 3}
# For evaluation mode only
index2word = ['<PAD>', '<UNK>', 'A', 'l', 'y', 'w', 'n', 'm', 'b', 't', 'h', 'r', 'k', 'E', 'd', 'H', 'f', 's', 'c',
              'q', 'p', 'j', 'O', '.', 'S', 'x', 'z', 'T', 'Y', '?', 'g', 'D', 'a', 'I', '!', 'V', ',', '_', '@', '#',
              ':', 'v', 'e', 'o', 'M', '/', 'i', '"', 'Z', 'Q', 'C', ')', '0', '-', '1', 'W', '2', 'u', '(', '4', '9',
              '3', '5', '7', 'N', 'R', '6', 'X', '8', 'J', 'L', 'K', 'B', 'G', 'F', '^', 'P', ';', '😂', 'U', '=', '[',
              '😘', ']', '🍉', '😍', '☺', '😝', '❤', '️', '😜', '•', '\\', 'ٲ', '😋', '🌹', 'ٓ', '😥', '٫', '☹', '😊',
              '😭', '🙈', '😅', '\ue756', '😌', '😔', '♥']

# For evaluation mode only
word2index = {'<PAD>': 0, '<UNK>': 1, 'A': 2, 'l': 3, 'y': 4, 'w': 5, 'n': 6, 'm': 7, 'b': 8, 't': 9, 'h': 10, 'r': 11,
              'k': 12, 'E': 13, 'd': 14, 'H': 15, 'f': 16, 's': 17, 'c': 18, 'q': 19, 'p': 20, 'j': 21, 'O': 22,
              '.': 23, 'S': 24, 'x': 25, 'z': 26, 'T': 27, 'Y': 28, '?': 29, 'g': 30, 'D': 31, 'a': 32, 'I': 33,
              '!': 34, 'V': 35, ',': 36, '_': 37, '@': 38, '#': 39, ':': 40, 'v': 41, 'e': 42, 'o': 43, 'M': 44,
              '/': 45, 'i': 46, '"': 47, 'Z': 48, 'Q': 49, 'C': 50, ')': 51, '0': 52, '-': 53, '1': 54, 'W': 55,
              '2': 56, 'u': 57, '(': 58, '4': 59, '9': 60, '3': 61, '5': 62, '7': 63, 'N': 64, 'R': 65, '6': 66,
              'X': 67, '8': 68, 'J': 69, 'L': 70, 'K': 71, 'B': 72, 'G': 73, 'F': 74, '^': 75, 'P': 76, ';': 77,
              '😂': 78, 'U': 79, '=': 80, '[': 81, '😘': 82, ']': 83, '🍉': 84, '😍': 85, '☺': 86, '😝': 87, '❤': 88,
              '️': 89, '😜': 90, '•': 91, '\\': 92, 'ٲ': 93, '😋': 94, '🌹': 95, 'ٓ': 96, '😥': 97, '٫': 98, '☹': 99,
              '😊': 100, '😭': 101, '🙈': 102, '😅': 103, '\ue756': 104, '😌': 105, '😔': 106, '♥': 107}


def _fit_term_index(terms, reserved=[], preprocess=lambda x: x):
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = reserved + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}


def load_file_as_words(file_path):
    with codecs.open(file_path, encoding='utf-8') as conell:
        list_of_lines = [line.strip().split() for line in conell if len(line.strip().split()) == 2]
        chars, targets = list(zip(*list_of_lines))
        words = ''.join(chars).split('WB')
        target_of_words = ''.join(targets).split('WB')

        words = [tuple(word) for word in words]
        target_of_words = [tuple(trg) for trg in target_of_words]

        return words, target_of_words


def build_embeddings(mxl):
    # Add UNKNOWN and
    targets = np.array([1 for i in range(0, mxl)])
    one_hot_targets = np.eye(args.word_embedding_dim)[targets]
    return one_hot_targets


def segment_file():
    embeddings = build_embeddings(args.max_features)
    #
    print('Loading data...')
    X_chars, y_test = load_file_as_words(args.test_set)
    X_idxs = np.array([[word2index.get(w, word2index['<UNK>']) for w in words] for words in X_chars])
    X_idxs_padded = sequence.pad_sequences(X_idxs, maxlen=args.maxlen, padding='post')

    print('loading model...')
    word_input = Input(shape=(args.maxlen,), dtype='int32', name='word_input')
    word_emb = Embedding(embeddings.shape[0], args.word_embedding_dim, input_length=args.maxlen, name='word_emb',
                         weights=[embeddings])(word_input)
    word_emb_d = Dropout(0.5)(word_emb)
    bilstm = Bidirectional(LSTM(args.lstm_dim, return_sequences=True))(word_emb_d)
    bilstm_d = Dropout(0.5)(bilstm)
    dense = TimeDistributed(Dense(args.nb_pos_tags))(bilstm_d)
    crf = ChainCRF()
    crf_output = crf(dense)
    model = load_model("model/keras_weights_0921.hdf5",
                       custom_objects={'ChainCRF': ChainCRF, 'sparse_loss': crf.sparse_loss}, compile=False)
    model.compile(loss=crf.sparse_loss, optimizer='adam', metrics=['sparse_categorical_accuracy'])

    prediction = model.predict(X_idxs_padded, args.batch_size, verbose=0)

    # TODO - 01. the function should return a segmented string

    with codecs.open(args.output_file, mode='w', encoding='utf-8') as results:
        for pred, word in zip(np.argmax(prediction, axis=2), X_chars):
            assert len(pred) >= len(word)

            for ch, est in zip(word, pred):
                results.write(ch + '\t' + index2pos[est] + '\n')
            else:
                results.write('WB\tWB\n')


if '__name__' == '__main__':
    pass
