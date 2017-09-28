# -*- coding: utf8 -*-
import argparse
import codecs
from collections import Counter
from itertools import chain
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence
from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, ChainCRF, Dense, TimeDistributed
import pkg_resources

# http://setuptools.readthedocs.io/en/latest/pkg_resources.html#resourcemanager-api
keras_model = pkg_resources.resource_filename(__name__, "files/models/segmenter.hdf5")
# input_file = pkg_resources.resource_filename(__name__, "files/example.in.txt")
# output_file = pkg_resources.resource_filename(__name__, "files/example.out.txt")


__author__ = 'disooqi'
__created__ = '21 Sep 2017'

parser = argparse.ArgumentParser(description="Segmentation module for the Dialectal Arabic")
group = parser.add_mutually_exclusive_group()
# file related options
parser.add_argument("-d", "--files-dir", help="directory containing train, test and dev file [default: %(default)s]")
parser.add_argument("-g", "--log-file", dest="log_file", help="log file [default: %(default)s]")
parser.add_argument("-p", "--model-file", dest="model_file",
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
    data_dir="files/",
    model_file=keras_model,
    # train_set="./files/all_train_f01.txt",
    # validation_set="./files/all_dev_f01.txt",
    # test_set="./files/all_test_f01.txt",
    log_file="run.log",
    # input_file=input_file,
    # output_file=output_file,
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
    max_features=217,  # len(index2word)   and   len(_invert_index(index2word))  ## For evaluation mode only
    nb_pos_tags=4,  # len(index2pos)                                             ## For evaluation mode only
)

args = parser.parse_args()

index2pos = ['S', 'B', 'E', 'M']
pos2index = {'S': 0, 'B': 1, 'E': 2, 'M': 3}
# For evaluation mode only
index2word = ['<PAD>', '<UNK>', 'ا', 'ل', 'ي', 'و', 'ن', 'م', 'ب', 'ت', 'ه', 'ر', 'ك', 'ع', 'د', 'ح', 'ف', 'س', 'ش', 'ق', 'ة', 'ج', 'أ', '.', 'ص', 'خ', 'ط', 'ز', 'ى', 'غ', 'a', 'ض', '?', '!', 'إ', 'ذ', '_', '@', '#', 't', 'e', ':', 'o', '،', 'h', 'i', 'ث', '؟', '/', 'l', 'r', 'm', 's', 'n', 'ظ', 'ئ', 'ء', 'آ', 'd', 'y', 'p', '"', 'c', '1', 'b', '0', ')', '-', 'A', 'S', '2', 'u', 'w', '(', ',', '9', 'k', 'D', 'M', 'f', '5', 'ؤ', 'z', '7', 'g', '3', 'T', 'H', '8', '4', 'j', 'R', 'ہ', 'N', 'B', '6', 'v', 'J', 'L', 'E', '*', '^', 'G', 'Y', 'K', 'F', '~', "'", 'Q', 'V', 'I', 'C', 'W', 'گ', 'O', 'P', 'چ', 'ﻻ', 'q', 'Z', '>', 'x', '؛', 'U', 'ھ', '‹', '٤', '…', '😂', '”', '›', 'ﺍ', 'ﻴ', 'ﺎ', '=', '😭', 'ک', 'X', 'ﻛ', 'ﻧ', '<', '[', 'ڤ', 'ﺮ', 'ﺴ', '٧', '😊', '😝', '😘', '😅', '|', 'ﻷ', '🙊', '“', '٢', '\\', 'ﻦ', '١', 'ﺘ', 'ﺪ', 'ﻟ', 'ﻋ', 'ﺸ', 'ﺐ', 'ﺣ', '🍉', '٣', '😍', '☺', '❤', '️', '😜', '•', ']', '»', ';', 'ﻹ', 'ﺳ', 'ﻬ', 'ٲ', 'ﻣ', 'ﻌ', 'ﺑ', 'ﻊ', 'ﻚ', 'ﻃ', 'ﻨ', 'ﻤ', 'ﻭ', 'ﺡ', 'ﺄ', 'ﺟ', 'ﻮ', 'ﺯ', 'ﻞ', 'ﻥ', '٠', '٨', '😋', '🌹', 'ٓ', '😥', '٫', '٥', '☹', '٦', '🙈', '٩', '\ue756', '😌', '😔', '♥', '👍', '😆', '😚', '$', '%']

# For evaluation mode only
word2index = {'<PAD>': 0, '<UNK>': 1, 'ا': 2, 'ل': 3, 'ي': 4, 'و': 5, 'ن': 6, 'م': 7, 'ب': 8, 'ت': 9, 'ه': 10, 'ر': 11, 'ك': 12, 'ع': 13, 'د': 14, 'ح': 15, 'ف': 16, 'س': 17, 'ش': 18, 'ق': 19, 'ة': 20, 'ج': 21, 'أ': 22, '.': 23, 'ص': 24, 'خ': 25, 'ط': 26, 'ز': 27, 'ى': 28, 'غ': 29, 'a': 30, 'ض': 31, '?': 32, '!': 33, 'إ': 34, 'ذ': 35, '_': 36, '@': 37, '#': 38, 't': 39, 'e': 40, ':': 41, 'o': 42, '،': 43, 'h': 44, 'i': 45, 'ث': 46, '؟': 47, '/': 48, 'l': 49, 'r': 50, 'm': 51, 's': 52, 'n': 53, 'ظ': 54, 'ئ': 55, 'ء': 56, 'آ': 57, 'd': 58, 'y': 59, 'p': 60, '"': 61, 'c': 62, '1': 63, 'b': 64, '0': 65, ')': 66, '-': 67, 'A': 68, 'S': 69, '2': 70, 'u': 71, 'w': 72, '(': 73, ',': 74, '9': 75, 'k': 76, 'D': 77, 'M': 78, 'f': 79, '5': 80, 'ؤ': 81, 'z': 82, '7': 83, 'g': 84, '3': 85, 'T': 86, 'H': 87, '8': 88, '4': 89, 'j': 90, 'R': 91, 'ہ': 92, 'N': 93, 'B': 94, '6': 95, 'v': 96, 'J': 97, 'L': 98, 'E': 99, '*': 100, '^': 101, 'G': 102, 'Y': 103, 'K': 104, 'F': 105, '~': 106, "'": 107, 'Q': 108, 'V': 109, 'I': 110, 'C': 111, 'W': 112, 'گ': 113, 'O': 114, 'P': 115, 'چ': 116, 'ﻻ': 117, 'q': 118, 'Z': 119, '>': 120, 'x': 121, '؛': 122, 'U': 123, 'ھ': 124, '‹': 125, '٤': 126, '…': 127, '😂': 128, '”': 129, '›': 130, 'ﺍ': 131, 'ﻴ': 132, 'ﺎ': 133, '=': 134, '😭': 135, 'ک': 136, 'X': 137, 'ﻛ': 138, 'ﻧ': 139, '<': 140, '[': 141, 'ڤ': 142, 'ﺮ': 143, 'ﺴ': 144, '٧': 145, '😊': 146, '😝': 147, '😘': 148, '😅': 149, '|': 150, 'ﻷ': 151, '🙊': 152, '“': 153, '٢': 154, '\\': 155, 'ﻦ': 156, '١': 157, 'ﺘ': 158, 'ﺪ': 159, 'ﻟ': 160, 'ﻋ': 161, 'ﺸ': 162, 'ﺐ': 163, 'ﺣ': 164, '🍉': 165, '٣': 166, '😍': 167, '☺': 168, '❤': 169, '️': 170, '😜': 171, '•': 172, ']': 173, '»': 174, ';': 175, 'ﻹ': 176, 'ﺳ': 177, 'ﻬ': 178, 'ٲ': 179, 'ﻣ': 180, 'ﻌ': 181, 'ﺑ': 182, 'ﻊ': 183, 'ﻚ': 184, 'ﻃ': 185, 'ﻨ': 186, 'ﻤ': 187, 'ﻭ': 188, 'ﺡ': 189, 'ﺄ': 190, 'ﺟ': 191, 'ﻮ': 192, 'ﺯ': 193, 'ﻞ': 194, 'ﻥ': 195, '٠': 196, '٨': 197, '😋': 198, '🌹': 199, 'ٓ': 200, '😥': 201, '٫': 202, '٥': 203, '☹': 204, '٦': 205, '🙈': 206, '٩': 207, '\ue756': 208, '😌': 209, '😔': 210, '♥': 211, '👍': 212, '😆': 213, '😚': 214, '$': 215, '%': 216}

def remove_diacrtics(token):
    # TODO implement 'remove_diacrtics'
    return token

def _fit_term_index(terms, reserved=[], preprocess=lambda x: x):
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = reserved + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}


def load_file_as_words(file_path):
    '''
    :param file_path: path of a fule contains a conll format of segmentation
    :return: a tuple of two lists of lists of characters and tags
    '''
    with codecs.open(file_path, encoding='utf-8') as conell:
        list_of_lines = [line.strip().split() for line in conell if len(line.strip().split()) == 2]
        chars, targets = list(zip(*list_of_lines))
        words = ''.join(chars).split('WB')
        target_of_words = ''.join(targets).split('WB')

        words = [tuple(word) for word in words]
        target_of_words = [tuple(trg) for trg in target_of_words]

        return words, target_of_words

def load_txt_as_ch_list(file_path):
    words = list()
    with codecs.open(file_path, encoding='utf-8') as plan_txt:
        for line in plan_txt:
            tokens = line.strip().split()
            for token in tokens:
                words.append(list(token))

    return words


def build_embeddings(mxl):
    # Add UNKNOWN and
    targets = np.array([1 for i in range(0, mxl)])
    one_hot_targets = np.eye(args.word_embedding_dim)[targets]
    return one_hot_targets


def segment_text(text, dl_model=keras_model):
    X_chars = list()
    tokens = text.strip().split()
    for token in tokens:
        X_chars.append(list(token))

    prediction = __segment__(X_chars, dl_model)

    segmentation = list()
    for pred, word in zip(np.argmax(prediction, axis=2), X_chars):
        assert len(pred) >= len(word)

        for ch, est in zip(word, pred):
            segmentation.append(ch)
            if index2pos[est] in ['S', 'E']:
                segmentation.append('+')
        else:
            segmentation[-1] = ' '

    return ''.join(segmentation)


def segment_file(infile, outfile, dl_model=keras_model):
    X_chars = load_txt_as_ch_list(infile)
    prediction = __segment__(X_chars, dl_model)

    with codecs.open(outfile, mode='w', encoding='utf-8') as results:
        for pred, word in zip(np.argmax(prediction, axis=2), X_chars):
            assert len(pred) >= len(word)

            for ch, est in zip(word, pred):
                results.write(ch + '\t' + index2pos[est] + '\n')
            else:
                results.write('WB\tWB\n')


def __segment__(X_chars, dl_model=keras_model):
    embeddings = build_embeddings(args.max_features)
    X_idxs = np.array([[word2index.get(w, word2index['<UNK>']) for w in words] for words in X_chars])
    X_idxs_padded = sequence.pad_sequences(X_idxs, maxlen=args.maxlen, padding='post')

    word_input = Input(shape=(args.maxlen,), dtype='int32', name='word_input')
    word_emb = Embedding(embeddings.shape[0], args.word_embedding_dim, input_length=args.maxlen, name='word_emb',
                         weights=[embeddings])(word_input)
    word_emb_d = Dropout(0.5)(word_emb)
    bilstm = Bidirectional(LSTM(args.lstm_dim, return_sequences=True))(word_emb_d)
    bilstm_d = Dropout(0.5)(bilstm)
    dense = TimeDistributed(Dense(args.nb_pos_tags))(bilstm_d)
    crf = ChainCRF()
    crf_output = crf(dense)
    model = load_model(dl_model,
                       custom_objects={'ChainCRF': ChainCRF, 'sparse_loss': crf.sparse_loss}, compile=False)
    model.compile(loss=crf.sparse_loss, optimizer='adam', metrics=['sparse_categorical_accuracy'])

    return model.predict(X_idxs_padded, args.batch_size, verbose=0)




if '__name__' == '__main__':
    pass

