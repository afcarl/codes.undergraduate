import datetime
import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


class Estimator(object):
    def __init__(self,
                 dense_layer=[512, 256],
                 lr=5e-4,
                 batch_size=128,
                 epochs=100,
                 data_gen=None,
                 summary=True):
        """
        create an estimator model
        :param dense_layer: hidden layers' dimensions, default is [512, 256]
        :param lr: learning rate of the optimizer, default is 5e-4
        :param batch_size: batch size, default is 128
        :param epochs: epochs, default is 100
        :param data_gen: conjugate Generator object, default is None
        :param summary: if shows summary of created model
        """

        now = datetime.datetime.now()
        self.date = "{}-{}-{}-{}".format(now.month, now.day, now.hour, now.minute)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.dense_layer = dense_layer
        self.data_gen = data_gen
        self.word_dim = self.data_gen.model_dim
        self.loss = "categorical_crossentropy"

        self.layer = Sequential()
        self.layer.add(Flatten(input_shape=(self.word_dim, 1)))
        for i in dense_layer:
            self.layer.add(Dense(i))
            self.layer.add(BatchNormalization())
            self.layer.add(Activation('relu'))
        self.layer.add(Dense(len(data_gen.uni_cls), activation='softmax'))
        adam = Adam(lr=self.lr)
        self.layer.compile(loss=self.loss, optimizer=adam)

        if summary:
            self.layer.summary()

    def __output__(self, labels, topn=1):
        vec = np.array([self.data_gen.norm_model(w).reshape(self.data_gen.model_dim, 1)
                        for w in labels])
        preds = self.layer.predict(vec)
        for i in range(len(vec)):
            cand_ids = np.argsort(-preds[i])
            thewords = np.array([self.data_gen.uni_cls[i] for i in cand_ids[0:topn]])
            yield labels[i], thewords

    def fit(self, x, y, verbose=2):
        """
        trains the created model, you can interrupt this whenever if you are bored.
        :param x: input instances' word vectors
        :param y: input classes' one hot vectors
        :param verbose: default is 2, see keras's documentation
        :return: score, keras's History object.
        """
        try:
            score = self.layer.fit(x, y, batch_size=self.batch_size,
                                   nb_epoch=self.epochs, verbose=verbose)
            return score

        except KeyboardInterrupt:
            print("keyboard interrupted")

    def print(self, labels, topn=3):
        gen = self.__output__(labels, topn)
        for labels, pred in gen:
            print("{}, :{}".format(labels, pred))

    def write(self, labels, output_path='output', filename="output"):
        gen = self.__output__(labels)
        date = "{}-{}-{}-{}".format(self.now.month, self.now.day,
                                    self.now.hour, self.now.minute)
        path = output_path + "/{}-{}.csv".format(filename, date)

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with open(path, 'a') as f:
            f.write("\n Word2Vec model dimension: {}".format(self.word_dim))
            f.write("\n Fully Connected Layers: {}".format(self.dense_layer))
            f.write("\n Fully Connected Layers: {}".format(self.layer.get_config()))
            f.write("\n epochs: {}".format(self.batch_size))
            f.write("\n loss function: {}".format(self.loss))
            f.write("\n{}".format("-" * 10))
            for labels, pred in gen:
                f.write("\n{},{}".format(labels, pred))
        print("{} written".format(path))


    def save(self, directory='save_model'):
        """
        save the model structure and weights
        :param directory: path to the saving directory
        """
        self.layer.save(os.path.join(directory,
                        'weights{}.hdf5'.format(self.date)))

    def predict(self, labels, answer=None, topn=1, verbose=1):
        """
        predicts classes for input instances
        :param labels: labels of instances
        :param answer: answers for type estimation, default is None
        :param topn: number of type candidates, default is 1
        :param verbose: if 2, prints estimated types and its probabilities,
                        if 1, prints estimated types,
                        if 0, prints nothing
        :return: if answer is not None,
        """
        vec = np.array([self.data_gen.norm_model(v).reshape(self.data_gen.model_dim, 1) for v in labels])
        if verbose not in [0, 1, 2]:
            raise Exception("verbose should be in [0,1,2]")

        if answer is None:
            answer = np.zeros_like(labels)
        pred_vecs = self.layer.predict(vec)

        count, score, score_1 = 0, 0, 0
        for i in range(len(vec)):
            count += 1
            cand_ids = np.argsort(-pred_vecs[i])
            pred_words = np.array([self.data_gen.uni_cls[i] for i in cand_ids[0:topn]])
            if pred_words[0] == answer[i]:
               score_1 += 1
            if answer[i] in pred_words:
                score += 1
            elif verbose == 1:
                print("{} prediction{}, answer:{}".format(labels[i], pred_words, answer[i]))
            elif verbose == 2:
                probs = np.array([pred_vecs[i][j] for j in cand_ids])
                w_p = [i for i in zip(pred_words, probs)]
                print("{} prediction{}, answer:{}".format(labels[i], w_p, answer[i]))
        if verbose > 0:
            print("-" * 10)
            print("Categorical")
            print("using top {} candidates".format(topn))
            print("evaluation: {:.2}".format(score / count))
            print("top 1 evaluation: {}".format(score_1 / count))

        if answer is not None:
            return count, score, score_1

