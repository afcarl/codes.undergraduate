import argparse

parser = argparse.ArgumentParser(description="sample program for my thesis")
parser.add_argument('word2vec_path', metavar='W', type=str, nargs=None,
                    help="path to the gensim's word2vec model")
parser.add_argument('train_data_path', metavar='T', type=str, nargs=None,
                    help="path to the training data")
args = parser.parse_args()

if __name__ == '__main__':

    from Generator import Generator
    from Estimator import Estimator
    import numpy as np

    gen = Generator(args.word2vec_path, args.train_data_path, sep=" ")
    train_X, train_Y = gen.train()
    instances = gen.ins
    train, _test = instances[0:gen.num_train], instances[gen.num_train:gen.num]
    train = np.unique(train)
    test = []
    for w in _test:
        if w not in train:
            test.append(w)

    est = Estimator(epochs=10, data_gen=gen)
    est.fit(train_X, train_Y)
    est.predict(test, verbose=2)

