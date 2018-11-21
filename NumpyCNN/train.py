# -*- encoding: utf-8 -*

import NN.Mnist_NN as network
import NN.utils as utils

from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    m = 50000
    X = utils.extract_data('./data/train-images-idx3-ubyte.gz', m, 28)
    y_dash = utils.extract_labels('./data/train-labels-idx1-ubyte.gz', m).reshape(m, 1)
    X -= int(np.mean(X))
    X /= int(np.std(X))
    train_data = np.hstack((X, y_dash))

    # print(np.shape(y_dash))

    save_path = "model"
    #
    cost = network.train(train_data,save_path=save_path)