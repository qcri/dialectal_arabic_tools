# -*- coding: utf8 -*-
import codecs
import numpy as np
from collections import Counter
from itertools import chain
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Dropout, ChainCRF


class SegmentationModel(object):
    formats = ('conll', 'plain')
    SEG_TAGS = ('S', 'B', 'E', 'M')

    def __init__(self, len_of_longest_word=50):
        self.embeddings = None
        self.lstm_dim = None
        self.maxlen = len_of_longest_word
        self.epoches = None
        self.batch_size = None
        self.embedding_dim = None

        self.index2word = None
        self.word2index = None

        self.X_train, self.X_dev, self.X_test = None, None, None
        self.y_train, self.y_dev, self.y_test = None, None, None
        self.dataset_format = None

        self.train_path, self.dev_path, self.test_path = None, None, None

        self.segmentation_model = None

    def train(self, model_file_name='./files/models/dialectal_segmenter.hdf5'):
        early_stopping = EarlyStopping(patience=10, verbose=1)
        checkpointer = ModelCheckpoint(model_file_name, verbose=1, save_best_only=True)
        self.segmentation_model.fit(self.X_train, self.y_train,
                 batch_size=self.batch_size,
                 epochs=self.epoches,
                 verbose=1,
                 shuffle=True,
                 callbacks=[early_stopping, checkpointer],
                 validation_data=[self.X_dev, self.y_dev]
                 )
        # self.build_model()

    def __str__(self):
        if self.segmentation_model:
            self.segmentation_model.summary()
        else:
            return 'No model has been built'
        return ''

    @classmethod
    def process_files(cls, file_path, dataset_format):
        if dataset_format == 'conll':
            return cls.process_conll_file(file_path)
        elif dataset_format == 'plain':
            pass

    def load_and_split(self, dataset_format, dataset, dev_size=0.2, shuffle=True):
        if self.check_format(dataset_format, dataset):
            self.dataset_format = dataset_format

        X_ch, y_ch = self.process_files(dataset, self.dataset_format)

        self.index2word = self._fit_term_index(X_ch, reserved=['<PAD>', '<UNK>'])
        self.word2index = self._invert_index(self.index2word)

        self.index2pos = self.SEG_TAGS
        pos2index = self._invert_index(self.index2pos)

        X_idx = np.array([[self.word2index[w] for w in words] for words in X_ch])
        y_idx = np.array([[pos2index[t] for t in pos_tags] for pos_tags in y_ch])



        X = sequence.pad_sequences(X_idx, maxlen=50, padding='post')
        y_train = sequence.pad_sequences(y_idx, maxlen=50, padding='post')
        y = np.expand_dims(y_train, -1)

        self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(X, y, test_size=dev_size, shuffle=shuffle)

    def load_dataset_splits(self, train_data_path, dev_data_path, test_data_path, dataset_format='conll'):
        # todo
        if self.check_format(dataset_format, self.train_path, self.dev_path, self.test_path):
            self.train_path = train_data_path
            self.dev_path = dev_data_path
            self.test_path = test_data_path
            self.dataset_format = dataset_format

        X_train_ch, y_train_ch = self.process_files(self.train_path, self.dataset_format)
        X_dev_ch, y_dev_ch = self.process_files(self.dev_path, self.dataset_format)
        X_test_ch, y_test_ch = self.process_files(self.test_path, self.dataset_format)

        self.index2word = self._fit_term_index(X_train_ch, reserved=['<PAD>', '<UNK>'])
        self.word2index = self._invert_index(self.index2word)

        self.index2pos = self.SEG_TAGS
        pos2index = self._invert_index(self.index2pos)

        X_train_idx = np.array([[self.word2index[w] for w in words] for words in X_train_ch])
        y_train_idx = np.array([[pos2index[t] for t in pos_tags] for pos_tags in y_train_ch])

        X_dev_idx = np.array([[self.word2index.get(w, self.word2index['<UNK>']) for w in words] for words in X_dev_ch])
        y_dev_idx = np.array([[pos2index[t] for t in pos_tags] for pos_tags in y_dev_ch])

        X_test_idx = np.array([[self.word2index.get(w, self.word2index['<UNK>']) for w in words] for words in X_test_ch])
        y_test_idx = np.array([[pos2index[t] for t in pos_tags] for pos_tags in y_test_ch])

        self.X_train = sequence.pad_sequences(X_train_idx, maxlen=self.maxlen, padding='post')
        self.X_dev = sequence.pad_sequences(X_dev_idx, maxlen=self.maxlen, padding='post')
        self.X_test = sequence.pad_sequences(X_test_idx, maxlen=self.maxlen, padding='post')

        # todo maxlen should be calculated automatically
        y_train = sequence.pad_sequences(y_train_idx, maxlen=self.maxlen, padding='post')
        self.y_train = np.expand_dims(y_train, -1)
        y_dev = sequence.pad_sequences(y_dev_idx, maxlen=self.maxlen, padding='post')
        self.y_dev = np.expand_dims(y_dev, -1)
        y_test = sequence.pad_sequences(y_test_idx, maxlen=self.maxlen, padding='post')
        self.y_test = np.expand_dims(y_test, -1)

    def build_model(self,  word_embedding_dim=200, lstm_dim=100, batch_size=10, nb_epoch=1, optimizer='adam'):
        self.lstm_dim = lstm_dim
        # cut texts after this number of words (among top max_features most common words)
        self.epoches = nb_epoch
        self.batch_size = batch_size
        self.embedding_dim = word_embedding_dim
        self.embeddings = self.build_embeddings()

        word_input = Input(shape=(self.maxlen,), dtype='int32', name='word_input')
        word_emb = Embedding(self.embeddings.shape[0], self.embedding_dim, input_length=self.maxlen, name='word_emb',
                             weights=[self.embeddings])(word_input)
        word_emb_d = Dropout(0.5)(word_emb)
        bilstm = Bidirectional(LSTM(self.lstm_dim, return_sequences=True))(word_emb_d)
        bilstm_d = Dropout(0.5)(bilstm)
        dense = TimeDistributed(Dense(len(self.index2pos)))(bilstm_d)
        crf = ChainCRF()
        crf_output = crf(dense)
        model = Model(inputs=[word_input], outputs=[crf_output])
        model.compile(loss=crf.sparse_loss,
                      optimizer=optimizer,
                      metrics=['sparse_categorical_accuracy'])

        self.segmentation_model = model

    def _fit_term_index(self, terms, reserved=[], preprocess=lambda x: x):
        # todo I need to know that this is working well
        all_terms = chain(*terms)
        all_terms = map(preprocess, all_terms)
        term_freqs = Counter(all_terms).most_common()
        id2term = reserved + [term for term, tf in term_freqs]
        return id2term

    def _invert_index(self, id2term):
        return {term: i for i, term in enumerate(id2term)}

    @classmethod
    def process_conll_file(cls, file_path):
        with codecs.open(file_path, encoding='utf-8') as conell:
            list_of_lines = [line.strip().split() for line in conell if len(line.strip().split()) == 2]
            chars, targets = list(zip(*list_of_lines))
            words = ''.join(chars).split('WB')
            target_of_words = ''.join(targets).split('WB')

            words = [tuple(word) for word in words]
            target_of_words = [tuple(trg) for trg in target_of_words]

            return words, target_of_words

    def evaluate(self, file_path):
        if self.check_format(file_path, self.dataset_format):
            pass
        # scores_test = model.evaluate(self.X_test, self.y_test,batch_size=100)
        # print("Accuracy for test: ",scores_test[1])
        #
        # scores_dev = model.evaluate(self.X_dev, self.y_dev, batch_size=100)
        # print("Accuracy for dev: ", scores_dev[1])
        # #
        # scores_train = model.evaluate(self.X_train, self.y_train, batch_size=100)
        # print("Accuracy for train: ", scores_train[1])

    @classmethod
    def check_format(cls, dataset_format, *file_path):
        if dataset_format not in cls.formats:
            # TODO create a special exception for check_format func.
            raise Exception

        return True

    def build_embeddings(self):
        mxl = len(self.word2index)
        # Add UNKNOWN and
        # todo fix this
        targets = np.array([1 for i in range(0, mxl)])
        one_hot_targets = np.eye(self.embedding_dim)[targets]
        return one_hot_targets


if __name__ == '__main__':
    model = SegmentationModel()
    model.load_and_split('conll', r'dev_data/all.trg.conll', dev_size=0.05)
    # model.load_dataset_splits(r'dev_data/all_train_f01.txt', r'dev_data/all_dev_f01.txt', r'dev_data/all_test_f01.txt')
    model.build_model()
    model.train()
    # print(model)
