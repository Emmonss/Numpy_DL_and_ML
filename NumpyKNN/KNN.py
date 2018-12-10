
import numpy as np
from tqdm import tqdm

def get_counts(sequence,k):
    counts = {}
    for i in range(k):
        if sequence[i] in counts:
            counts[sequence[i]] += 1
        else:
            counts[sequence[i]] = 1
    return counts

def Sort_two(X,Y):
    indice = np.argsort(X)
    return Y[indice],X[indice]


def KNN(testdata,dataset,labels,k):
    assert testdata.shape[1] == dataset.shape[1], "测试集和数据集维度要相同！"
    assert k <= dataset.shape[0], "采集数据不能多于数据集数目"

    res_labels = []
    for item in tqdm(testdata):
        temp = np.tile(item, (dataset.shape[0], 1))-dataset
        temp = pow(temp,2)
        temp = np.sqrt(np.sum(temp,axis=1))
        sort_label ,temp  = Sort_two(temp, labels)
        res = get_counts(sort_label,k)
        res = max(zip(res.values(),res.keys()))[1]
        res_labels.append(res)
    return np.array(res_labels)



if __name__ == '__main__':
    x = np.array([1,3,5,7,9,2,4,6,8])

    x2 = np.array([1, 4, 7, 9, 2, 5, 8, 3, 6])
    y = np.array([0,0,1,2,2,0,1,1,2])
    y= Sort_two(x,y)
    print(y)


