import gzip
import numpy as np

def extract_data(filename, num_images, IMAGE_WIDTH):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def normalized(dataset):
    dataset[dataset < 150] = 0.0
    dataset[dataset > 150] = 1.0
    return dataset

def Sort_two(X,Y):
    indice = np.argsort(X)
    return Y[indice]


def get_Precision_Recall(output,testlabels):
    assert len(output) == len(testlabels), "测试结果和标签数目不等！"

    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    for i in range(len(output)):
        if(output[i]==testlabels[i]):
            digit_correct[output[i]] +=1
        digit_count[output[i]]+=1

    # print("digit_count:{}".format(digit_count))
    # print("digit_correct:{}".format(digit_correct))


    acc = float(np.sum(digit_correct))*100 / float(len(output))
    recall = [x / y for x, y in zip(digit_correct, digit_count)]

    return acc,recall

