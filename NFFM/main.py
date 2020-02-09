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

# tf.keras.backend.set_floatx('float16')
# tf.keras.backend.set_epsilon(1e-4)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(path = None,test_path = None,debug = True):

    with timer("decode prepare"):
        features = {}
        # with open('../data/decode.txt', 'r') as f:
        #     for line in f.readlines():
        #         features[line[:line.rfind(',')]] = line[line.rfind(',')+1:-1]

        with open(root_path + 'types_dict_full.json', 'r') as f:
            types_dict = json.loads(f.read())

        with open(root_path + 'label_dict_full.json', 'r') as f:
            label_dict = json.loads(f.read())

        with open(root_path + 'label_len_dict_full.json', 'r') as f:
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
        model = NFFM(
        label_dict = label_dict,
        label_len_dict = label_len_dict,
        label_col = 'HasDetections',
        features = features,
        only_lr = False,
        batch_size=256,
        test_val_batch_size = 4096,
        epochs=1,
        train_path = train_path,
        early_stop=True,
        early_stop_round=1,
        eval_path=val_path,
        eval_step=500,
        eval_metric=tf.keras.metrics.AUC(),
        patience = 0.001,
        greater_is_better=True,
        embedding_size = 8,
        dropout  = None,#长度为deep layers长度+1
        deep_layers = [128, 128],
        embeddings_initializer = tf.keras.initializers.GlorotUniform,
        kernel_initializer=tf.keras.initializers.GlorotUniform,
        verbose = 100,
        random_seed = 2020,
        l2_reg = 0.01,
        use_bn = True,
        )
        # print('model training....')
        checkpoint = tf.train.Checkpoint(myModel=model)
        checkpoint = tf.train.CheckpointManager(checkpoint, directory='model/', max_to_keep=3)

        model.train(checkpoint = checkpoint)

        print('model inferring....')
        # checkpoint = tf.train.Checkpoint(myModel=model)  # 实例化Checkpoint，指定恢复对象为model
        # checkpoint.restore(tf.train.latest_checkpoint('model/'))  # 从文件恢复模型参数
        submission = model.infer(test_path,target_name = 'HasDetections',id_name='MachineIdentifier')
        submission['MachineIdentifier'] = submission['MachineIdentifier'].str.decode("utf-8")
        submission.to_csv('submission_full.csv',index = False)






if __name__ == "__main__":
    with timer("train finish"):
        root_path = '../data/'
        train_path = root_path + 'train_full.tfrecord'
        val_path = root_path + 'val_full.tfrecord'
        test_path = root_path + 'test_full.tfrecord'
        main(path=train_path,test_path = test_path, debug=True)