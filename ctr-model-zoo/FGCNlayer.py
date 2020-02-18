#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter

Reference:
   [2] Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction.(https://arxiv.org/pdf/1904.04447.pdf)

"""
# date 2020.02.16
import tensorflow as tf
class FGCNlayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size = [7,7,7], pools = [2, 2,2],
        maps = [1, 1,1],filters = [3, 6,12],activation='tanh',embedding_size = None,filed_size = None):
        super().__init__()
        if filed_size == None or embedding_size == None:
            raise ValueError(
                "Unexpected filed_size,filed_size must not None,value must == feature nums in model")
        self.kernel_size = kernel_size
        self.pools = pools
        self.maps = maps
        self.filters = filters
        self.embedding_size = embedding_size
        self.conv = []
        self.pooling = []
        self.dense_fgcn = []
        pool_count = filed_size
        for i in range(len(self.kernel_size)):
            self.conv.append(
                tf.keras.layers.Conv2D(filters=self.filters[i], strides=(1, 1), kernel_size=(self.kernel_size[i], 1),
                                       padding='same', activation = activation,
                                       use_bias=True
                                       ))
            self.pooling.append(tf.keras.layers.MaxPooling2D(pool_size=(self.pools[i], 1)))
            self.dense_fgcn.append(tf.keras.layers.Dense(self.maps[i] * int(pool_count / self.pools[0]) * self.embedding_size))
            pool_count = int(pool_count / self.pools[0])

        self.flatten = tf.keras.layers.Flatten()
    @tf.function
    def call(self, inputs, **kwargs):
        pool_out = tf.expand_dims(inputs, axis=3)
        new_fea = []
        for i in range(len(self.kernel_size)):
            pool_out = self.conv[i](pool_out)
            pool_out = self.pooling[i](pool_out)
            FGCN_out = self.flatten(pool_out)
            FGCN_out = self.dense_fgcn[i](FGCN_out)
            new_fea.append(tf.reshape(FGCN_out, shape=[-1, pool_out.shape[1] * self.maps[i], self.embedding_size]))

        new_fea = tf.concat(new_fea, axis=1)


        return new_fea




