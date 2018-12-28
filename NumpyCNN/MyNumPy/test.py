import numpy as np
import matplotlib.pyplot as plt
import NN.utils as utils
from Convolution2D import forward
import pickle
import tqdm


def predict(image ,params):
    [f1, f2, w3, w4, b1, b2, b3, b4, conv_stride, pool_filter, pool_stride] = params

    conv1 = forward.Convolution2D(image, f1, b1, conv_stride)  # convolution operation
    conv1[conv1 <= 0] = 0  # relu activation

    conv2 = convolution(conv1, f2, b2, conv_stride)  # second convolution operation
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    pooled = maxpool(conv2, pool_filter, pool_stride)  # maxpooling operation
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

    z = w3.dot(fc) + b3  # first dense layer
    z[z <= 0] = 0  # pass through ReLU non-linearity

    out = w4.dot(z) + b4  # second dense layer
    probs = softmax(out)  # predict class probabilities with the softmax activation function

    return np.argmax(probs), np.max(probs)


if __name__ == '__main__':
    save_path = "model.pkl"
    params, loss = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4, conv_stride, pool_filter, pool_stride] = params


    #显示loss曲线
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()


    #加载测试集数据
    # Get test data
    m = 10000
    X = extract_data('./data/t10k-images-idx3-ubyte.gz', m, 28)
    y_dash = extract_labels('./data/t10k-labels-idx1-ubyte.gz', m).reshape(m, 1)
    # Normalize the data
    X -= int(np.mean(X))  # subtract mean
    X /= int(np.std(X))  # divide by standard deviation
    test_data = np.hstack((X, y_dash))

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]

    t = tqdm(range(len(X)), leave=True)

    #计算精准率和召回率
    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Acc:%0.2f%%" % (float(corr / (i + 1)) * 100))

    print("Overall Accuracy: %.2f" % (float(corr / len(test_data) * 100)))
    x = np.arange(10)
    digit_recall = [x / y for x, y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x, digit_recall)
    plt.show()