
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
import itertools
class NFFM(tf.keras.models.Model):
    def __init__(self, label_dict, label_len_dict,only_lr,
                 embedding_size, dropout,
                 deep_layers,
                 deep_layers_activation=tf.nn.relu,
                 embeddings_initializer = tf.keras.initializers.GlorotUniform,
                 kernel_initializer = tf.keras.initializers.GlorotUniform,
                 verbose=False, random_seed=2020,
                 eval_metric=roc_auc_score,
                 l2_reg=0.0, use_bn = True,greater_is_better=True):
        super().__init__()
        self.label_dict = label_dict
        self.label_len_dict = label_len_dict
        self.only_lr = only_lr
        self.embedding_size = embedding_size
        self.dropout  = dropout
        self.deep_layers = deep_layers
        self.deep_layers_activation = deep_layers_activation
        self.embeddings_initializer = embeddings_initializer
        self.kernel_initializer = kernel_initializer
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

        self.drop_out = []
        for i in self.dropout:
            self.drop_out.append(tf.keras.layers.Dropout(rate = i,seed = 2020))

        # # deep part
        self.dense = []
        for i in range(len(self.deep_layers)):
            self.dense.append(tf.keras.layers.Dense(self.deep_layers[i],kernel_initializer = self.kernel_initializer,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg)))
        self.dense_out = tf.keras.layers.Dense(1 ,kernel_initializer = self.kernel_initializer,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=self.l2_reg))

        a = 1
    def call(self, inputs, training=None):

        lr = {}
        # lr
        for i in self.label_len_dict.keys():
            lr[i] = self.emb_lr[i](inputs[i])

        for num,key in enumerate(self.label_len_dict.keys()):
            if num == 0:
                lr_out = lr[key]
            else:
                lr_out = tf.concat([lr_out,lr[key]],axis =1)
        lr_out = tf.reduce_sum(lr_out, axis=1)
        # logistics
        if self.only_lr:
            return tf.nn.sigmoid(lr_out)

        # ffm
        ffm = {}
        for i in self.label_len_dict.keys():
            ffm[i] = self.emb_ffm[i](inputs[i])
            ffm[i] = tf.reshape(ffm[i],[-1, len(self.label_dict.keys()), self.embedding_size])

        # Interaction
        fea_name = dict(zip(range(0,len(self.label_dict.keys())),self.label_dict.keys()))
        interaction_dict = {}
        for (index1, index2) in itertools.combinations(list(range(0, len(fea_name))), 2):
            fea1, fea2 = fea_name[index1], fea_name[index2]
            interaction_dict.setdefault(fea2, {})[fea1] = ffm[fea2][:, index1, :]
            interaction_dict.setdefault(fea1, {})[fea2] = ffm[fea1][:, index2, :]


        v_i = []
        v_j = []
        input_size = 0
        for (fea1, fea2) in itertools.combinations(interaction_dict.keys(), 2):
            input_size += 1
            v_i.append(interaction_dict[fea1][fea2])
            v_j.append(interaction_dict[fea2][fea1])
        v_i = tf.transpose(v_i, perm=[1, 0, 2])
        v_j = tf.transpose(v_j, perm=[1, 0, 2])
        vi_vj = tf.multiply(v_i, v_j)
        vi_vj = tf.reshape(vi_vj, [-1, input_size * self.embedding_size])

        # DNN
        out = self.drop_out[0](vi_vj,training = training)
        for i in  range(len(self.deep_layers)):
            # y_deep_input = tf.matmul(y_deep_input,self.weights1["layer_%d" % i])
            out = self.dense[i](out)
            if self.use_bn:
                out = tf.keras.layers.BatchNormalization()(out,training = training)
            out = tf.keras.layers.Activation('relu')(out)
            out = self.drop_out[i+1](out,training = training)

        out = self.dense_out(out)
        out = tf.add(out,tf.reshape(lr_out,shape = [-1,1]))
        out = tf.nn.sigmoid(out)
        out = tf.reshape(out,shape=[-1])
        # self.out = keras.layers.Activation('sigmoid', name='odm_mbox_conf_softmax')(self.out)
        return out
