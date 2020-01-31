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
        embedding_size = 4,
        dropout  = [0.5,0.5,0.5,0.5],
        deep_layers = [256, 128,64],
        dropout_deep = [0.5,0.5,0.5,0.5],
        embeddings_initializer = tf.keras.initializers.GlorotUniform,
        kernel_initializer=tf.keras.initializers.GlorotUniform,
        epoch = 1000,
        batch_size = 1024,
        verbose = True,
        random_seed = 2020,
        eval_metric = roc_auc_score,
        l2_reg = 0.0001,
        use_bn = True,
        greater_is_better = True)


        filenames = [path]
        data = tf.data.TFRecordDataset(filenames)

        # Parse features, using the above template.
        def parse_record(record):
            return tf.io.parse_single_example(record, features=features)

        # Apply the parsing to each record from the dataset.
        data = data.map(parse_record)
        data = data.repeat()
        data = data.shuffle(buffer_size=1000)
        data = data.batch(batch_size=32)
        data = data.prefetch(buffer_size=1)

        for step, batch_x in enumerate(data):
            with tf.GradientTape() as tape:
                y_pred = model(batch_x,training = True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
            grads = tape.gradient(loss, model.variables)



if __name__ == "__main__":
    with timer("train finish"):
        root_path = '../data/'
        path = root_path + 'train_sample.tfrecord'
        test_path = root_path + 'test_sample.tfrecord'
        main(path=path,test_path = test_path, debug=True)