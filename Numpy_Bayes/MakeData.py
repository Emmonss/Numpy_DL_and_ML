import jieba
import numpy as np

StopWords = './data/StopWords.txt'
NewWord = './data/dic.txt'


#加载文本
def LoadData(dfile,zfile):
    data = []
    with open(dfile,'r',encoding='utf-8') as dr:
        for item in dr.readlines():
            item.strip('\n')
            data.append([item,1])

    with open(zfile,'r',encoding='utf-8') as zr:
        for item in  zr.readlines():
            item.strip('\n')
            data.append([item, 0])

    np.random.shuffle(data)

    stopword = set()
    with open(StopWords, 'r', encoding='utf-8') as fsr:
        for line in fsr:
            stopword.add(line.strip('\n'))

    return data,stopword

#分词
def JiebaCut(str,stopword):
    res = []
    jieba.load_userdict(NewWord)
    words = jieba.cut(str.strip('\n'))
    for word in words:
        if word not in stopword:
            res.append(word)

    return res


#制作词典和分类序列
def CreateList(data,stopword):
    vocab = []
    classlabel = []
    for item,label in data:
        words = JiebaCut(item,stopword)
        for word in words:
            if word not in vocab:
                vocab.append(word)
        classlabel.append(label)
    return list(vocab),classlabel

#制作句子于词典的向量
def Word2Vec(vocab,input,stopword):
    Vec = [0]*len(vocab)
    words = JiebaCut(input, stopword)
    for word in words:
        if word in vocab:
            Vec[vocab.index(word)]=1
        else:
            print("{} not in vocabulary".format(word))
    return Vec

#计算概率
def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num= np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom,p1Denom = 2.0,2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = np.log(p1Num / p1Denom)
    p0Vec = np.log(p0Num / p0Denom)
    return p0Vec,p1Vec,pAbusive



def classifyNB(vec2classify, p0Vec,p1Vec,pclass1):
    p1 = sum(vec2classify*p1Vec)+np.log(pclass1)
    p0 = sum(vec2classify*p0Vec)+np.log(1.0 - pclass1)
    if p1>p0:
        return True
    else:
        return False

if __name__ == '__main__':
    AbusiveWord = './data/AbusiveWord.txt'
    NormalWord = './data/NormalWord.txt'
    data,stopword= LoadData(AbusiveWord,NormalWord)
    vocab , classlabel = CreateList(data,stopword)

    vec = Word2Vec(vocab,'你这个人恶心得要死')

    print(vocab)
    print(classlabel)
    print(vec)
