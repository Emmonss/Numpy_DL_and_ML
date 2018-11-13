# -*- encoding: utf-8 -*

import numpy as np
import utils
import cv2


def Convolution2D_Backward(conv_prev, conv_in, filter, stride):
    '''
    卷积核的反向操作
    :param conv_prev:前卷积图片
    :param conv_in:输入图片
    :param filter:滤波器
    :param stride:步长
    :return:
    '''

    (n_filter,n_c,f_size ,_) = filter.shape
    (_,in_dim_y,in_dim_x) = conv_in.shape

    #输出和输入相同
    dout = np.zeros(conv_in.shape)
    dfilter = np.zeros(filter.shape)
    dbias = np.shape((n_filter,1))

    for current_filter in range(n_filter):
        curr_y = out_y = 0
        while curr_y + f_size <= in_dim_y:
            curr_x = out_x = 0
            while curr_y + f_size <= in_dim_x:
                # 滤波器的梯度下降
                dfilter[current_filter] += conv_prev[current_filter, out_y, out_x] * conv_in[:, curr_y:curr_y + f_size, curr_x:curr_x + f_size]
                # 输入图片的梯度下降
                dout[:, curr_y:curr_y + f_size, curr_x:curr_x + f_size] += conv_prev[current_filter, out_y, out_x] * filter[current_filter]
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1

            dbias[current_filter] = np.sum(conv_prev[current_filter])

    return dout, dfilter, dbias

def MaxPool_Backward(dpool, origin_img, filter, stride):
    '''
    最大池化反向传播层
    :param dpool:
    :param origin_img:
    :param filter:
    :param stride:
    :return:
    '''
    (n_conv, in_dim_y,in_dim_x) =  origin_img.shape

    dout = np.zeros(origin_img.shape)

    for current_conv in range(n_conv):
        curr_y = out_y = 0
        while curr_y + filter <= in_dim_y:
            curr_x = out_x = 0
            while curr_x + filter <= in_dim_x:
                # 当前窗口最大值下标
                (a, b) = utils.nanargmax(origin_img[current_conv, curr_y:curr_y + filter, curr_x:curr_x + filter])
                dout[current_conv, curr_y + a, curr_x + b] = dpool[current_conv, out_y, out_x]

                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1