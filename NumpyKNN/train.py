import utils_datasets as ud
import numpy as np
import KNN as model
import matplotlib.pyplot as plt




def Main():
    horatio = 0.2
    trainset,trainlabel = ud.get_file_matrix('trainset.txt')

    trainset = np.array(ud.normalized(trainset))

    fiter = int(len(trainlabel)*horatio)

    out = model.KNN(trainset[:fiter],trainset[fiter:],trainlabel[fiter:],k = 3)
    acc,recall = ud.get_Precision_Recall(out,trainlabel[:fiter])

    print(acc)
    print(recall)


if __name__ == '__main__':
    Main()