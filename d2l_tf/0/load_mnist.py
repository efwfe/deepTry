# -*- coding:utf-8 -*-
"""
@file name :  load_mnist
@description: 
@author:      张玳辉
@date :       2020/8/5-3:34 下午
"""
from input_data import read_data_sets

mnist = read_data_sets("/tmp/data", one_hot=True)
print(mnist)
