# -*- encoding: utf-8 -*

from Convolution2D import forward,backward
import NN.utils as utils

import numpy as np
import pickle
from tqdm import tqdm
import sys

def Lenet_5_params(nums):
    '''
    网络参数
    :param nums:输出层维度（分类的数目）
    :return:网络各项参数
    '''
    f1 = (8,1,5,5)
    f2 = (8,8,5,5)
    w3 = (128,800)
    w4 = (nums,128)
    conv_stride = 1
    pool_filter = 2
    pool_stride = 2

    f1 = utils.initializeFilter(f1)
    f2 = utils.initializeFilter(f2)
    w3 = utils.initializeWeight(w3)
    w4 = utils.initializeWeight(w4)

    b1 = np.zeros((f1.shape[0], 1))
    b2 = np.zeros((f2.shape[0], 1))
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4,conv_stride,pool_filter,pool_stride]

    return params



def Lenet_5(image , labels, params):
    '''
    手写体识别网络
    :param image:输入图片
    :param labels: 标签
    :param params: 网络参数
    :param conv_stride: 卷积步长
    :param pool_filter: 池化层滤波器大小
    :param pool_s: 池化层滤波器长度
    :return:
    '''

    [f1, f2, w3, w4, b1, b2, b3, b4, conv_stride, pool_filter, pool_stride] = params


    #前向传播层

    #卷积层一
    conv1 = forward.Convolution2D(image = image,
                                  filter=f1,
                                  bias=b1,
                                  stride=conv_stride,
                                  mode = 'VALID')
    #relu
    conv1[conv1 <= 0] = 0


    #卷积层2
    conv2 = forward.Convolution2D(image = conv1,
                                  filter=f2,
                                  bias=b2,
                                  stride=conv_stride,
                                  mode = 'VALID')
    # relu
    conv1[conv1 <= 0] = 0

    #池化层1
    pool2 = forward.MaxPool(conv2,pool_filter,pool_stride)

    #flatten
    (n_conv, dim_y, dim_x) = pool2.shape
    fc1 = pool2.reshape((n_conv*dim_y*dim_x,1))

    # 全连接1
    fc2 = w3.dot(fc1)+b3

    #relu
    fc2[fc2<=0]=0

    # 全连接2
    fc3 = w4.dot(fc2)+b4

    #输出层
    out = forward.SoftMax(fc3)

    #计算loss
    loss = forward.categoricalCrossEntropy(out,labels)

    #反向传播层
    dout = out-labels

    dw4 = dout.dot(fc2.T)
    db4 = np.sum(dout,axis=1).reshape(b4.shape)

    #链式求导
    dfc2 = w4.T.dot(dout)
    dfc2[fc2<=0]=0

    dw3 = dfc2.dot(fc1.T)
    db3 = np.sum(dfc2, axis = 1).reshape(b3.shape)

    dfc1 = w3.T.dot(dfc2)
    dpool2 = dfc1.reshape(pool2.shape)

    dconv2 = backward.MaxPool_Backward(dpool2,conv2,pool_filter,pool_stride)
    dconv2[conv2<=0] = 0

    dconv1 ,df2,db2= backward.Convolution2D_Backward(dconv2, conv1,f2,conv_stride)
    dconv1[conv1<=0] = 0

    dimage ,df1,db1 = backward.Convolution2D_Backward(dconv1,image,f1,conv_stride)

    grads = [df1,df2,dw3,dw4,db1,db2,db3,db4]

    return grads,loss

def Adam(batch,num_classes,lr,cost,params):
    '''
    Adam优化算法
    :param batch: minibathc
    :param num_classes: 分类数目
    :param lr: 学习率
    :param dim: 维度
    :param n_c:通道
    :param cost:损失函数
    :param params:网络参数
    :return:
    '''

    beta1 = 0.9
    beta2 = 0.99

    [f1, f2, w3, w4, b1, b2, b3, b4, conv_stride, pool_filter, pool_stride] = params

    X = batch[:,0:-1]
    X = X.reshape(len(batch),1,28,28)
    Y = batch[:,-1]

    cost_ = 0
    batch_size = len(batch)

    # 初始化梯度和动量参数
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)

    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)

    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)

    for i in range(batch_size):
        x = X[i]
        #对角矩阵？
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)

        grads, loss = Lenet_5(x, y, params)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

        df1 += df1_
        db1 += db1_
        df2 += df2_
        db2 += db2_
        dw3 += dw3_
        db3 += db3_
        dw4 += dw4_
        db4 += db4_

        cost_ += loss

    v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size  # momentum update
    s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2  # RMSProp update
    f1 -= lr * v1 / np.sqrt(s1 + 1e-7)  # combine momentum and RMSProp to perform update with Adam

    bv1 = beta1 * bv1 + (1 - beta1) * db1 / batch_size
    bs1 = beta2 * bs1 + (1 - beta2) * (db1 / batch_size) ** 2
    b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

    v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
    s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
    f2 -= lr * v2 / np.sqrt(s2 + 1e-7)

    bv2 = beta1 * bv2 + (1 - beta1) * db2 / batch_size
    bs2 = beta2 * bs2 + (1 - beta2) * (db2 / batch_size) ** 2
    b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

    v3 = beta1 * v3 + (1 - beta1) * dw3 / batch_size
    s3 = beta2 * s3 + (1 - beta2) * (dw3 / batch_size) ** 2
    w3 -= lr * v3 / np.sqrt(s3 + 1e-7)

    bv3 = beta1 * bv3 + (1 - beta1) * db3 / batch_size
    bs3 = beta2 * bs3 + (1 - beta2) * (db3 / batch_size) ** 2
    b3 -= lr * bv3 / np.sqrt(bs3 + 1e-7)

    v4 = beta1 * v4 + (1 - beta1) * dw4 / batch_size
    s4 = beta2 * s4 + (1 - beta2) * (dw4 / batch_size) ** 2
    w4 -= lr * v4 / np.sqrt(s4 + 1e-7)

    bv4 = beta1 * bv4 + (1 - beta1) * db4 / batch_size
    bs4 = beta2 * bs4 + (1 - beta2) * (db4 / batch_size) ** 2
    b4 -= lr * bv4 / np.sqrt(bs4 + 1e-7)

    cost_ = cost_ / batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4,conv_stride,pool_filter,pool_stride]

    return params, cost


bar_max = 40
metrics = '  '.join([
        '\r[{}]',
        '{:.2f}%',
        '{}/{}',
        'loss={:.3f}'
    ])


def train(train_data,num_classes=10,lr=0.001,batch_size=32,num_epochs=2,save_path = "model.pkl"):
    '''
    训练
    :param train_x: 输出数据集
    :param train_y: 输入标签
    :param num_classes: 分类数目
    :param lr: 学习率
    :param batch_size:batch大小
    :param num_epochs: 训练次数
    :param save_path: 保存模型路径
    :return:
    '''

    params = Lenet_5_params(num_classes)
    cost = []

    print("lr:{};batch_size:{};num_epoch:{};save_path:{}".format(lr,batch_size,num_epochs,save_path))\

    for epoch in range(num_epochs):
        print("epoch:{}".format(epoch+1))
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        for i,batch in enumerate(batches):
            precent = float(i) / float(len(batches))
            bars = int(bar_max * precent)
            params,loss = Adam(batch,num_classes,lr,cost,params)
            sys.stdout.write(metrics.format(
                                    '=' * (bars - 1) + '>' + '-' * (bar_max - bars),
                                    precent,
                                    i+1, len(batches),
                                    loss[-1]
            ))
        # t = tqdm(batches)
        # for x,batch in enumerate(t):
        #     params,cost = Adam(batch,num_classes,lr,cost,params)
        #     t.set_description("loss: %.2f" % (cost[-1]))

    save = [params,cost]
    with open(save_path+'.pkl', 'wb') as file:
        pickle.dump(save, file)

    return cost