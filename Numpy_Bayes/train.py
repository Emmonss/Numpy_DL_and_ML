import numpy as np
import make_data


def classifyNB(vec2classify, p0Vec,p1Vec,pclass1):
    p1 = sum(vec2classify*p1Vec)+np.log(pclass1)
    p0 = sum(vec2classify*p0Vec)+np.log(1.0 - pclass1)
    if p1>p0:
        return True
    else:
        return False




def train(testEnv):
    list,classer = make_data.MakeData()
    myVocab = make_data.CreateList(list)
    trainMat = []
    for postinDoc in list:
        trainMat.append(make_data.Word2Vec(myVocab,postinDoc))

    p0V,p1V,pAB = make_data.trainNB(np.array(trainMat),np.array(classer))
    print(p0V,p1V,pAB)
    Doc = np.array(make_data.Word2Vec(myVocab,testEnv))
    if classifyNB(Doc,p0V,p1V,pAB):
        print("是有害信息")
    else:
        print("不是有害信息")

if __name__ == '__main__':
    test1 = ['love','my','dalmation']
    test2 = ['stupid','garbage','dog']
    print(train(test2))