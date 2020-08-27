# -*- coding:utf-8 -*-
"""
@file name :  linear_simple
@description: 
@author:      张玳辉
@date :       2020/8/5-4:05 下午
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras

mnist_data = mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist_data
classes = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sanda', 'shirt', 'sneaker', 'bag', 'ankle']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential(
    [keras.layers.Flatten(input_shape=(28, 28)),
     keras.layers.Dense(128, activation='sigmoid'),
     keras.layers.Dense(10)]
)

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_loss, test_acc)