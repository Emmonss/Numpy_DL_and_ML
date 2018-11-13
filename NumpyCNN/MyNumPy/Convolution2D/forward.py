# -*- encoding: utf-8 -*

import numpy as np
import utils
import cv2



def Convolution2D(image,filter,bias,stride =1):

    '''
    卷积层前向函数
    :param image:输入的图片矩阵
    :param filter: 卷积核，又称滤波器
    :param bias: 偏转
    :param stride: 卷积步长.默认1*1
    :return: 卷积之后的图像
    '''
    #滤波器参数
    (n_filter, n_con_filter,filt_dim,_) = filter.shape
    #输入图片参数
    n_con,in_dim_y,in_dim_x= image.shape

    #输出参数
    out_dim_y = int((in_dim_y-filt_dim)/stride)+1
    out_dim_x = int((in_dim_x - filt_dim) / stride) + 1
    #判断图片通道和滤波器通道是否相同

    assert n_con == n_con_filter , "滤波器的通道数必须同图片的通道数相等！"
    #输出格式
    out = np.zeros((n_filter,out_dim_y,out_dim_x))

    #卷积计算
    for current_filter in range(n_filter):
        curr_y = out_y = 0
        while curr_y + filt_dim <= in_dim_y:
            curr_x = out_x = 0
            while curr_x + filt_dim <= in_dim_x:
                out[current_filter, out_y, out_x] = np.sum(filter[current_filter] * image[:, curr_y:curr_y + filt_dim, curr_x:curr_x + filt_dim]) + bias[current_filter]
                curr_x += stride
                out_x +=1
            curr_y += stride
            out_y+=1


    return out



def MaxPool(image,filter=2,stride=2):

    '''
    最大池化层
    :param image:输入的图片
    :param filter: 池化层滤波器大小
    :param stride: 池化步长
    :return: 池化后的图片
    '''

    #图片的通道，尺寸
    n_conv,in_dim_y,in_dim_x = image.shape

    #输出的尺寸
    y = int((in_dim_y-filter)/stride)+1
    x = int((in_dim_x - filter) / stride) + 1

    #输出0图片
    out_image = np.zeros((n_conv,y,x))

    for channel in range(n_conv):
        curr_y = out_y = 0
        while curr_y + filter <= in_dim_y:
            curr_x = out_x = 0
            while curr_x + filter <= in_dim_x:
                out_image[channel,out_y,out_x] = np.max(image[channel, curr_y:curr_y+filter, curr_x:curr_x+filter])
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
    return out_image


def SoftMax(X):
    '''
    SoftMax函数
    :param X: 输入向量
    :return: softmax向量（
    '''
    out = np.exp(X)
    return out/np.sum(out)


def categoricalCrossEntropy(probs, label):
    '''
    损失函数
    :param probs: 预测值向量
    :param label: 验证集向量
    :return: loss值
    '''
    return -np.sum(label * np.log(probs))

if __name__ == '__main__':
    filter_1 = (1,1,3,3)
    filter_1 = utils.initializeFilter(filter_1)


    bias_1 = np.zeros((filter_1.shape[0], 1))

    dic = []
    img_path = 'img/1.jpg'
    img = cv2.imread(img_path,0)
    dic.append(img)
    dic = np.asarray(dic,dtype='float32')
    print(dic)
    # out = Convolution2D(dic, filter_1, bias_1)
    out = MaxPool(dic)
    b = np.array(cv2.normalize(out[0], None, 0, 255, cv2.NORM_MINMAX))


    print(out)
    cv2.imwrite('2.jpg',b)

