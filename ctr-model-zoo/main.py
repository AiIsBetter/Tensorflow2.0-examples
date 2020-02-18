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
from PNN_FGCN import PNN_FGCN
from xdeepfm import xdeepfm
from config import load_config
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

def main(debug = True):

    with timer("load config"):
        params = load_config()

    with timer("train model"):
        print("model create")
        # model = NFFM(**params['NFFM'])
        # model = PNN_FGCN(**params['PNN_FGCN'])
        model = xdeepfm(**params['XDEEPFM'])

        # print('model training....')
        checkpoint = tf.train.Checkpoint(myModel=model)
        checkpoint = tf.train.CheckpointManager(checkpoint, directory='model/', max_to_keep=3)

        model.train(checkpoint = checkpoint)

        print('model inferring....')
        # checkpoint = tf.train.Checkpoint(myModel=model)  # 实例化Checkpoint，指定恢复对象为model
        # checkpoint.restore(tf.train.latest_checkpoint('model/'))  # 从文件恢复模型参数
        submission = model.infer(params['test_path'],target_name = 'HasDetections',id_name='MachineIdentifier')
        submission['MachineIdentifier'] = submission['MachineIdentifier'].str.decode("utf-8")
        submission.to_csv('submission_full.csv',index = False)






if __name__ == "__main__":
    with timer("train finish"):

        main(debug=True)