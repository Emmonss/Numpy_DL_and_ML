import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def make_figure_img(datasets,labels):
    fig = plt.figure()
    x1,x2,x3 = datasets[labels == '1'],datasets[labels == '2'],datasets[labels == '3']
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x1[:, 0], x1[:, 1], marker='*', c='red', label=1, s=40)  # marker定义形状，label与plt.legend画出右上角图例
    plt.scatter(x2[:, 0], x2[:, 1], marker='+', c='green', label=2, s=40)  # * + o  x  s是方块 d是菱形
    plt.scatter(x3[:, 0], x3[:, 1], marker='o', c='blue', label=3, s=40)
    plt.legend(loc='upper right')
    plt.show()


def get_file_matrix(filename):
    with open(filename,'r',encoding="utf-8") as fr:
        arraylines = fr.readlines()
        num = len(arraylines)
        dataset = np.zeros((num,3))
        label = []
        index = 0
        for line in arraylines:
            line = line.strip().split('\t')
            dataset[index,:] = line[0:3]
            label.append(line[-1])
            index+=1
        return np.array(dataset),np.array(label)


def normalized(dataset):
    min = dataset.min(0)
    max = dataset.max(0)
    range = max - min
    ret = np.zeros(dataset.shape)
    m = dataset.shape[0]
    ret = dataset - np.tile(min,(m,1))
    ret = ret/np.tile(range,(m,1))
    return ret


def get_Precision_Recall(output,truelabels):
    assert len(output) == len(truelabels), "测试结果和标签数目不等！"

    digit_count = [0 for i in range(3)]
    digit_correct = [0 for i in range(3)]
    digit_pre = [0 for i in range(3)]

    for i in range(len(output)):
        if(output[i]==truelabels[i]):
            #每类预测==真实值的情况
            digit_correct[int(output[i])-1] +=1
        #预测结果每类的比重
        digit_count[int(output[i])-1]+=1
        #真实情况每类的比重
        digit_pre[int(truelabels[i])-1]+=1


    # print("digit_count:{}".format(digit_count))
    # print("digit_correct:{}".format(digit_correct))


    acc = float(np.sum(digit_correct))*100 / float(len(output))
    precision = [x / y for x, y in zip(digit_correct, digit_count)]
    recall = [x / y for x, y in zip(digit_correct, digit_pre)]

    PjiaR = [x+y for x, y in zip(precision, recall)]
    PR2 = [2*x*y for x, y in zip(precision, recall)]
    F1 = [x / y for x, y in zip(PR2,PjiaR )]

    return acc,precision,recall,F1


if __name__ == '__main__':
    filename = './trainset.txt'
    dataset,label = get_file_matrix(filename)

    dataset = normalized(dataset)
    make_figure_img(dataset,label)
