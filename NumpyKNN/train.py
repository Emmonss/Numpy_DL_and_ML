import utils
import numpy as np
import KNN as model
import matplotlib.pyplot as plt




def Main():
    train_nums = 40000
    test_nums = 1000
    dataset = utils.extract_data('./data/train-images-idx3-ubyte.gz', train_nums, 28)
    labels = utils.extract_labels('./data/train-labels-idx1-ubyte.gz', train_nums)
    dataset = utils.normalized(dataset)

    testdata = utils.extract_data('./data/t10k-images-idx3-ubyte.gz', test_nums, 28)
    testlabels = utils.extract_labels('./data/t10k-labels-idx1-ubyte.gz', test_nums)
    testdata = utils.normalized(testdata)
    output = model.KNN(testdata, dataset, labels, 10)



    acc,recall = utils.get_Precision_Recall(output,testlabels)
    print(acc)
    print(recall)

    x = np.arange(10)
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x, recall)
    plt.show()


if __name__ == '__main__':
    Main()