# -*- coding:utf-8 -*-
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


model = keras.Sequential()
model.add(keras.layers.Embedding(1000, 64, input_length=10))
# 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
# 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
# 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。

input_array = np.random.randint(1000, size=(32, 10))
model.summary()
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
print(output_array.shape)
print(input_array)