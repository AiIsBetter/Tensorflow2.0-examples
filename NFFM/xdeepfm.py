
#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter

Reference:
    [1] xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems.(https://arxiv.org/pdf/1803.05170.pdf)

"""
# date 2020.02.17

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from FGCNlayer import FGCNlayer
from tensorflow import python
import tqdm
import itertools
class xdeepfm(tf.keras.models.Model):
    def __init__(self, label_dict, label_len_dict,label_col,
                 features,batch_size=256,test_val_batch_size = 256,
                 epochs = 1,
                 train_path = None,early_stop=False,
                 early_stop_round=3,eval_path=None,
                 eval_step = 500, eval_metric = 'auc',
                 patience = 0.001,
                 greater_is_better = True,embedding_size =4,
                 dropout = None,deep_layers = [128,128],
                 deep_layers_activation=tf.nn.relu,
                 embeddings_initializer = tf.keras.initializers.GlorotUniform,
                 kernel_initializer = tf.keras.initializers.GlorotUniform,
                 verbose=1, random_seed=2020,
                 l2_reg=0.0, use_bn = True,cin_layers = [200,200,200],
                 optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)):
        super().__init__()
        self.batch_size = batch_size
        self.test_val_batch_size = test_val_batch_size
        self.epochs = epochs
        self.label_dict = label_dict
        self.label_len_dict = label_len_dict
        self.label_col = label_col
        self.features = features
        self.train_path = train_path
        self.early_stop = early_stop
        self.early_stop_round = early_stop_round
        self.eval_path = eval_path
        self.eval_step = eval_step
        self.eval_metric = eval_metric
        self.patience = patience
        self.greater_is_better = greater_is_better
        self.embedding_size = embedding_size
        self.dropout  = dropout
        self.deep_layers = deep_layers
        self.deep_layers_activation = deep_layers_activation
        self.embeddings_initializer = embeddings_initializer
        self.kernel_initializer = kernel_initializer
        self.verbose = verbose
        self.random_seed = random_seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        self.flatten = tf.keras.layers.Flatten()
        self.fea_nums = len(self.label_dict.keys())

        # cin params
        self.cin_layers = cin_layers


        # init data
        self.dataset()

        # init optimizer
        self.optimizer = optimizer


        #early stop start
        self.round = 0
        if self.greater_is_better:
            self.best_score = 0
        else:
            self.best_score = 99999

        # lr
        self.emb_lr = {}
        for i in label_len_dict.keys():
            self.emb_lr[i] = tf.keras.layers.Embedding(input_dim=max(self.label_len_dict[i]) + 1, output_dim=1,
                                                       embeddings_initializer=self.embeddings_initializer)

        # cin
        self.emb_cin = {}
        for i in label_len_dict.keys():
            self.emb_cin[i] = tf.keras.layers.Embedding(input_dim=max(self.label_len_dict[i])+1,
                                                       output_dim=self.embedding_size,
                                                       embeddings_initializer=self.embeddings_initializer)

        self.cin_conv = []
        for i in range(len(self.cin_layers)):
            self.cin_conv.append(tf.keras.layers.Conv1D(filters=self.cin_layers[i], kernel_size=1,strides=1, padding='VALID',activation='relu',use_bias=False))

        if self.dropout:
            self.drop_out = []
            for i in self.dropout:
                self.drop_out.append(tf.keras.layers.Dropout(rate = i,seed = 2020))


        if self.use_bn:
            self.batch_norm = []
            for i in range(len(self.deep_layers)):
                self.batch_norm.append(tf.keras.layers.BatchNormalization())

        # # deep part
        self.dense = []
        for i in range(len(self.deep_layers)):
            self.dense.append(tf.keras.layers.Dense(self.deep_layers[i], kernel_initializer=self.kernel_initializer))

        self.dense_lr_out = tf.keras.layers.Dense(1, kernel_initializer=self.kernel_initializer)
        self.dense_cin_out = tf.keras.layers.Dense(1, kernel_initializer=self.kernel_initializer)
        self.dense_dnn_out = tf.keras.layers.Dense(1, kernel_initializer=self.kernel_initializer)

    # @tf.function
    def call(self, inputs, training=None):



        lr = []
        cin = []
        for i in self.label_len_dict.keys():
            lr.append(self.emb_lr[i](inputs[i]))
            cin.append(tf.expand_dims(self.emb_cin[i](inputs[i]), axis=1))

        # lr
        lr_out = tf.concat(lr, axis=1)
        lr_out = self.dense_lr_out(lr_out)

        # cin
        cin = tf.concat(cin, axis=1)

        tensor0 = cin
        x_tensor = []
        x_tensor.append(tensor0)
        neurons_num = []
        neurons_num.append(self.fea_nums)
        for index, map_num in enumerate(self.cin_layers):
            xk = self.cinlayer(tensor0,x_tensor[-1],neurons_num[-1],map_num,index)
            x_tensor.append(xk)
            neurons_num.append(map_num)
        pk = [tf.reduce_sum(x, axis=2) for x in x_tensor]
        cin_out = tf.concat(pk, axis=1)
        cin_out = self.dense_cin_out(cin_out)


        # DNN
        dnn_out = self.flatten(cin)
        if self.dropout:
            dnn_out = self.drop_out[0](dnn_out,training = training)

        for i in  range(len(self.deep_layers)):
            dnn_out = self.dense[i](dnn_out)
            if self.use_bn:
                dnn_out = self.batch_norm[i](dnn_out,training = training)
            dnn_out = tf.keras.layers.Activation('relu')(dnn_out)

            if self.dropout:
                dnn_out = self.drop_out[i+1](dnn_out,training = training)
        dnn_out = self.dense_dnn_out(dnn_out)


        out = lr_out+cin_out+dnn_out
        out = tf.nn.sigmoid(out)
        out = tf.reshape(out,shape=[-1])
        return out

    def cinlayer(self,tensor_0,xk,xk_neuron,feature_map_num,index):
        x0 = tf.split(value=tensor_0, num_or_size_splits=self.embedding_size * [1], axis=2)
        xk = tf.split(value=xk, num_or_size_splits=self.embedding_size * [1], axis=2)
        zk = tf.matmul(x0, xk, transpose_b=True)
        zk = tf.reshape(zk, shape=[self.embedding_size, -1, self.fea_nums * xk_neuron])
        zk = tf.transpose(zk, perm=[1, 0, 2])
        xk1 = self.cin_conv[index](zk)  # (batch, D, feature_map_num)
        xk1 = tf.transpose(xk1, perm=[0, 2, 1])  # (batch, feature_map_num, D)
        return xk1



    def eval(self):
        print('evaluting...')
        # writer = tf.summary.create_file_writer('logdir/')
        # tf.summary.trace_on(graph=True, profiler=True)
        for step_val, batch_x in enumerate(self.val_data):
            if step_val%10 == 0 and step_val>0:
                print('evaluting step:{}'.format(step_val))
            y_pred = self.call(batch_x, training=False)
            self.eval_metric.update_state(y_true=batch_x[self.label_col], y_pred=y_pred)
            if step_val>400:
                # with writer.as_default():
                #     tf.summary.trace_export(
                #         name="my_func_trace",
                #         step=0,
                #         profiler_outdir='logdir/')
                break
        # tf.summary.trace_export(
        #     name="my_func_trace",
        #     step=0,
        #     profiler_outdir='logdir/')



    def train(self,checkpoint):
        for epoch in range(self.epochs):
            for step, batch_x in enumerate(self.train_data):
                with tf.GradientTape() as tape:
                    y_pred = self.call(batch_x,training = True)
                    loss = tf.keras.losses.binary_crossentropy(y_true=batch_x['HasDetections'], y_pred=y_pred)
                    if step % self.verbose == 0 and step>0:
                        print("##########################step %d: loss %f ###########################" % (step, loss.numpy()))
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.trainable_variables))
                if self.eval_metric !=None and self.eval_path !=None:
                    if step % self.eval_step == 0 and step > 0:
                        self.eval()
                        if self.early_stop:
                            if self.greater_is_better:
                                if self.eval_metric.result().numpy()<self.best_score:
                                    self.round += 1
                                elif self.eval_metric.result().numpy()-self.best_score > self.patience:
                                    tmp_score = self.eval_metric.result().numpy()
                                    self.best_score = tmp_score
                                    self.round = 0
                                    path = checkpoint.save(checkpoint_number=step) # 保存模型参数到文件
                                    print("model saved to %s" % path)
                            else:
                                if self.eval_metric.result().numpy()>self.best_score:
                                    self.round += 1
                                elif self.best_score - self.eval_metric.result().numpy()>self.patience:
                                    tmp_score = self.eval_metric.result().numpy()
                                    self.best_score = tmp_score
                                    self.round = 0
                                    path = checkpoint.save(checkpoint_number=step)  # 保存模型参数到文件
                                    print("model saved to %s" % path)

                            if self.verbose:
                                print("step %d:  step auc: %f  ,best auc %f . " % (
                                    step, self.eval_metric.result().numpy(), self.best_score))
                            if self.round>self.early_stop_round:
                                break
                # break
            self.eval()
            print("train finish:  step auc: %f  ,best auc %f . " % (
                self.eval_metric.result().numpy(), self.best_score))
            if self.round > self.early_stop_round:
                break


    def dataset(self):
        # options = tf.data.Options()
        # options.experimental_optimization.noop_elimination = True
        # options.experimental_optimization.map_vectorization.enabled = True
        # options.experimental_optimization.apply_default_optimizations = False

        val_filenames = [self.eval_path]
        self.val_data = tf.data.TFRecordDataset(val_filenames)
        self.val_data = self.val_data.repeat()
        self.val_data = self.val_data.shuffle(buffer_size=200000)
        self.val_data = self.val_data.batch(batch_size=self.test_val_batch_size)
        self.val_data = self.val_data.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_data = self.val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        filenames = [self.train_path]
        self.train_data = tf.data.TFRecordDataset(filenames)
        # self.train_data = self.train_data .with_options(options)
        self.train_data = self.train_data.repeat(1)
        self.train_data = self.train_data.shuffle(buffer_size=200000)
        self.train_data = self.train_data.batch(batch_size=self.batch_size)
        self.train_data = self.train_data.map(self.parse_record,
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_data = self.train_data.prefetch(buffer_size=20)

    def parse_record(self,record):
        return tf.io.parse_example(record, features=self.features)

    def infer(self,test_path,target_name = 'target',id_name='id'):
        y_pred = []
        y_id = []
        filenames = [test_path]
        test_data = tf.data.TFRecordDataset(filenames)
        test_data = test_data.repeat(1)
        test_data = test_data.batch(batch_size=self.test_val_batch_size)
        test_data = test_data.map(self.parse_record,
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_data = test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        for step_test, batch_x in enumerate(test_data):
            # if step_test%1000 ==0 and step_test>0:
            print('inferring: ',step_test)
            tmp = self.call(batch_x, training=False)
            y_id = y_id + list(batch_x[id_name].numpy())
            y_pred = y_pred + list(tmp.numpy())
        submission = pd.DataFrame({id_name:y_id,target_name:y_pred})
        return submission