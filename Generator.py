import numpy as np
from gensim.models import word2vec
import pandas as pd

class Generator(object):
    def __init__(self,
                 w2v_model_path,
                 train_data_path,
                 sep='\t',
                 header=None,
                 drop_dup=False,
                 shuffle=False):
        """
        generate data for Recognizer
        :param w2v_model_path: path to word2vec model
        :param train_data_path: path to class-instance pairs file
        :param sep: separator of training data, default is \t
        :param header: header of training data, default is None
        :param drop_dup: if drop duplication of instances in training data, default is False
        :param shuffle: if shuffle training data
        """
        self.w2v_model = word2vec.Word2Vec.load_word2vec_format(w2v_model_path)
        self.vocab = self.w2v_model.index2word
        self.model_dim = int(self.w2v_model.__str__().split(",")[1].split("=")[1])
        print("word2vec model loaded {}".format(self.w2v_model.__str__()))

        self.train_data = pd.read_csv(train_data_path, sep=sep, header=header)
        if drop_dup:
            self.train_data = self.train_data.drop_duplicates(subset=[1])
        if shuffle:
            self.train_data = self.train_data.reindex(
                np.random.permutation(self.train_data.index)).reset_index(drop=True)
        print("is-a data loaded. size {}".format(len(self.train_data)))

        self.cls = np.array(self.train_data.iloc[:, 0])
        self.ins = np.array(self.train_data.iloc[:, 1])

        self.word_dic = {}  # cls -> id
        self.uni_cls = np.unique(self.cls)
        for j in range(len(self.uni_cls)):
            i = self.uni_cls[j]
            self.word_dic[i] = j

    def __onehot(self, w):
        x = np.zeros(len(self.uni_cls), dtype=np.int)
        x[self.word_dic[w]] = 1
        return x

    def train(self, ratio=0.9):
        self.num = len(self.train_data)
        self.num_train = int(self.num * ratio)

        train_X = np.array([self.norm_model(w).reshape(self.model_dim, 1)
                            for w in self.ins[0:self.num_train]])

        train_Y = np.array([self.__onehot(w) for w in self.cls[0:self.num_train]])
        return train_X, train_Y

    def test(self):
        test_X = np.array([self.norm_model(w).reshape(self.model_dim, 1)
                           for w in self.ins[self.num_train + 1:self.num]])

        test_Y = np.array([self.__onehot(w) for w in self.cls[self.num_train + 1:self.num]])
        return test_X, test_Y

    def test_ins(self):
        return self.ins[self.num_train+1:self.num]

    def word_exists(self, word):
        if word in self.vocab:
            return True
        else:
            return False

    def norm_model(self, word):
        x = self.w2v_model[word]
        return x / np.linalg.norm(x)
