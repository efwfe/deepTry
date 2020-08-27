# -*- coding:utf-8 -*-
"""
@file name :  ts
@description: 
@author:      张玳辉
@date :       2020/4/10-1:37 下午
"""

import numpy as np
import pandas as pd

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 导入数据
data_train = pd.read_csv('../input/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashion-mnist_test.csv')

# 图像矩阵大小
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# 构造数据集
X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

# 训练集 测试集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
# 测试集
X_test = np.array(data_test.iloc[:, 1])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_val = X_val.astype("float32")

X_train /= 255
X_test /= 255
X_val /= 255

