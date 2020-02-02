#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.01.29

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from NFFM import NFFM
import time
import json
from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def main(path = None,test_path = None,debug = True):

    with timer("decode prepare"):
        features = {}
        # with open('../data/decode.txt', 'r') as f:
        #     for line in f.readlines():
        #         features[line[:line.rfind(',')]] = line[line.rfind(',')+1:-1]

        with open(root_path + 'types_dict.json', 'r') as f:
            types_dict = json.loads(f.read())

        with open(root_path + 'label_dict.json', 'r') as f:
            label_dict = json.loads(f.read())

        with open(root_path + 'label_len_dict.json', 'r') as f:
            label_len_dict = json.loads(f.read())

        for i in types_dict.keys():
            if 'int' in types_dict[i]:
                features[i] = tf.io.FixedLenFeature([], tf.int64)
            elif 'float' in types_dict[i]:
                features[i] = tf.io.FixedLenFeature([], tf.float32)
            else:
                features[i] = tf.io.FixedLenFeature([], tf.string)
    with timer("train model"):
        print("model create")
        model = NFFM(label_dict = label_dict,
        label_len_dict = label_len_dict,
        only_lr = False,
        embedding_size = 8,
        dropout  = [0.5,0.5,0.5],#长度为deep layers长度+1
        deep_layers = [128, 128],
        embeddings_initializer = tf.keras.initializers.GlorotUniform,
        kernel_initializer=tf.keras.initializers.GlorotUniform,
        verbose = True,
        random_seed = 2020,
        eval_metric = roc_auc_score,
        l2_reg = 0.0001,
        use_bn = False,
        greater_is_better = True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        metric = tf.keras.metrics.AUC()
        epochs = 1
        batch_size = 256

        filenames = [path]
        data = tf.data.TFRecordDataset(filenames)

        # Parse features, using the above template.
        def parse_record(record):
            return tf.io.parse_single_example(record, features=features)

        # Apply the parsing to each record from the dataset.
        data = data.map(parse_record,num_parallel_calls=10)
        data = data.repeat(1)
        data = data.shuffle(buffer_size = 200000)
        data = data.batch(batch_size = batch_size)
        data = data.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

        val_filenames = [val_path]
        val_data = tf.data.TFRecordDataset(val_filenames)


        # Apply the parsing to each record from the dataset.
        val_data = val_data.map(parse_record,num_parallel_calls=10)
        val_data = val_data.repeat(1)
        val_data = val_data.shuffle(buffer_size=20000)
        val_data = val_data.batch(batch_size = batch_size)
        val_data = val_data.prefetch(buffer_size=20000)
        for epoch in range(epochs):
            for step, batch_x in enumerate(data):
                with tf.GradientTape() as tape:
                    y_pred = model(batch_x,training = True)
                    loss = tf.keras.losses.binary_crossentropy(y_true=batch_x['HasDetections'], y_pred=y_pred)
                    loss = tf.reduce_mean(loss)
                    if step %100==0 and step>0:
                        print("step %d: loss %f" % (step, loss.numpy()))
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
                if step % 500 == 0 and step > 0:
                    for step_val, batch_x in enumerate(val_data):
                        y_pred = model(batch_x, training=False)
                        metric.update_state(y_true=batch_x['HasDetections'], y_pred=y_pred)
                    print("step %d: auc %f" % (step, metric.result().numpy()))
            for step_val, batch_x in enumerate(val_data):
                y_pred = model(batch_x, training=False)
                metric.update_state(y_true=batch_x['HasDetections'], y_pred=y_pred)
            print("step %d: auc %f" % (step, metric.result().numpy()))
            # for step, batch_x in enumerate(val_data):
            #     y_pred = model(batch_x, training=False)
            #     metric.update_state(y_true=batch_x['HasDetections'], y_pred=y_pred)
            # print("epoch %d: auc %f" % (epoch, metric.result().numpy()))

if __name__ == "__main__":
    with timer("train finish"):
        root_path = '../data/'
        path = root_path + 'train_sample.tfrecord'
        val_path = root_path + 'val_sample.tfrecord'
        test_path = root_path + 'test_sample.tfrecord'
        main(path=path,test_path = test_path, debug=True)