# -*- coding:utf-8 -*-
"""
@file name :  network
@description: 
@author:      张玳辉
@date :       2020/5/28-10:02 上午
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers,optimizers, models
from tensorflow.keras import regularizers

class VGG16(models.Model):

    def __init__(self, input_shape):
        super(VGG16, self).__init__()
        weight_decay = 0.00
        self.num_classes = 10
        model = models.Sequential()

        model.add(layers.Conv2D(64,(3,3), padding='same'), input_shape=input_shape,
                  kernel_regularizer=regularizers.l2(weight_decay))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())

        # pool
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(layers.Activation('relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.MaxPool2D(2,2))
        