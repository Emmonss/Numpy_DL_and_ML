import utils_datasets as ud
import numpy as np
import KNN as model
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from sklearn.metrics import classification_report



def Main():
    horatio = 0.1
    trainset,trainlabel = ud.get_file_matrix('trainset.txt')

    trainset = np.array(ud.normalized(trainset))

    fiter = int(len(trainlabel)*horatio)

    out = model.KNN(trainset[:fiter],trainset[fiter:],trainlabel[fiter:],k = 3)
    acc,precision,recall,F1 = ud.get_Precision_Recall(out,trainlabel[:fiter])
    hl = hamming_loss(trainlabel[:fiter],out)
    print("hamming_loss:{}".format(hl))
    print("Acc:{}".format(acc))
    print("Precision:{}".format(precision))
    print("Reacall:{}".format(recall))
    print("F1:{}".format(F1))

    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(trainlabel[:fiter], out, target_names=target_names))

if __name__ == '__main__':
    Main()