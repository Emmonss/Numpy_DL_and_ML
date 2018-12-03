import utils
import numpy as np
from tqdm import  tqdm


def get_counts(sequence,k):
    counts = {}
    for i in range(k):
        if sequence[i] in counts:
            counts[sequence[i]] += 1
        else:
            counts[sequence[i]] = 1
    return counts


def KNN(testdata,dataset,labels,k):
    assert testdata.shape[1] == dataset.shape[1], "测试集和数据集维度要相同！"
    assert k <= dataset.shape[0], "采集数据不能多于数据集数目"

    res_labels = []
    for item in tqdm(testdata):
        temp = np.tile(item, (dataset.shape[0], 1))-dataset
        temp = np.sqrt(np.sum(pow(temp,2),axis=1))
        labels = utils.Sort_two(temp,labels)
        res = get_counts(labels,k)
        res = max(zip(res.values(),res.keys()))[1]
        res_labels.append(res)
    return np.array(res_labels)






