import numpy as np

from MakeData import LoadData,CreateList,Word2Vec,trainNB,classifyNB
import MakeData




def train(AbusiveWord,NormalWord,input):
    data,stopword = LoadData(AbusiveWord,NormalWord)
    myVocab,label = CreateList(data,stopword)
    trainMat = []
    for postinDoc in data:
        trainMat.append(Word2Vec(myVocab,postinDoc[0],stopword))

    # print(label)
    p0V,p1V,pAB = trainNB(np.array(trainMat),np.array(label))
    print(p0V,p1V,pAB)

    Doc = np.array(Word2Vec(myVocab, input,stopword))
    if classifyNB(Doc, p0V, p1V, pAB):
        print("是有害信息")
    else:
        print("不是有害信息")



if __name__ == '__main__':
    AbusiveWord = './data/AbusiveWord.txt'
    NormalWord = './data/NormalWord.txt'
    input= "你这个蠢逼"
    train(AbusiveWord,NormalWord,input)