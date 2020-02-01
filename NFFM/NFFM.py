
#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.01.29

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

class NFFM(tf.keras.models.Model):
    def __init__(self, label_dict, label_len_dict,
                 embedding_size, dropout,
                 deep_layers, dropout_deep,
                 deep_layers_activation=tf.nn.relu,
                 embeddings_initializer = tf.keras.initializers.GlorotUniform,
                 kernel_initializer = tf.keras.initializers.GlorotUniform,
                batch_size=256,
                 verbose=False, random_seed=2020,
                 eval_metric=roc_auc_score,
                 l2_reg=0.0, use_bn = True,greater_is_better=True):
        super().__init__()
        self.label_dict = label_dict
        self.label_len_dict = label_len_dict
        self.embedding_size = embedding_size
        self.dropout  = dropout
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.embeddings_initializer = embeddings_initializer
        self.kernel_initializer = kernel_initializer
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_seed = random_seed
        self.eval_metric = eval_metric
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.greater_is_better = greater_is_better
        # lr
        self.emb_lr = {}
        for i in label_len_dict.keys():
            self.emb_lr[i] = tf.keras.layers.Embedding(input_dim=max(self.label_len_dict[i])+1,output_dim=self.embedding_size,embeddings_initializer = self.embeddings_initializer)
        # ffm
        self.emb_ffm = {}
        for i in label_len_dict.keys():
            self.emb_ffm[i] = tf.keras.layers.Embedding(input_dim=max(self.label_len_dict[i]) + 1,
                                                       output_dim=self.embedding_size*len(self.label_dict.keys()),
                                                       embeddings_initializer=self.embeddings_initializer)

        # # deep part
        self.dense = []
        for i in range(len(self.deep_layers)):
            self.dense.append(tf.keras.layers.Dense(self.deep_layers[i],kernel_initializer = self.kernel_initializer,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg)))
        self.dense_out = tf.keras.layers.Dense(1 ,kernel_initializer = self.kernel_initializer,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg))

        a = 1
    def call(self, inputs, training=None):

        x = {}
        # lr
        for i in self.label_len_dict.keys():
            x[i] = self.emb_lr[i](inputs[i])
        for num,key in enumerate(self.label_len_dict.keys()):
            if num == 0:
                out = x[key]
            else:
                out = tf.concat([out,x[key]],axis =1)
        out = tf.reduce_sum(out, axis=1)
        # # DNN
        # for i in range(0, 3):
        #     # y_deep_input = tf.matmul(y_deep_input,self.weights1["layer_%d" % i])
        #     out = self.dense[i](out)
        #     if self.use_bn:
        #         out = tf.keras.layers.BatchNormalization(trainable=training)(out)
        #     out = tf.keras.layers.Activation('relu')(out)
        #     # self.y_deep = tf.keras.layers.Dropout(self.dropout_deep[i], noise_shape=None, seed=None)(self.y_deep,
        #     #                                                                                       training=training)
        #
        # # self.out = tf.add(tf.matmul(self.y_deep, self.weights["concat_projection"]),
        # #                       self.weights["concat_bias"])
        # out = self.dense_out(out)
        out = tf.nn.sigmoid(out)
        # self.out = keras.layers.Activation('sigmoid', name='odm_mbox_conf_softmax')(self.out)
        return out
