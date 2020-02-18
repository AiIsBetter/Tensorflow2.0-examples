
#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter

"""
# date 2020.02.17
import tensorflow as tf
import json
def load_config():
    root_path = '../data/'
    train_path = root_path + 'train_full.tfrecord'
    val_path = root_path + 'val_full.tfrecord'
    test_path = root_path + 'test_full.tfrecord'

    # load label dict
    features = {}
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

    params = {}

    params['train_path'] = train_path
    params['val_path'] = val_path
    params['test_path'] = test_path

    # NFFM
    params['NFFM'] = {
    'label_dict' : label_dict,
    'label_len_dict' : label_len_dict,
    'label_col' : 'HasDetections',
    'features' : features,
    'only_lr' : False,
    'batch_size' : 256,
    'test_val_batch_size' : 4096,
    'epochs' :1,
    'train_path' : train_path,
    'early_stop':True,
    'early_stop_round':1,
    'eval_path':val_path,
    'eval_step':2000,
    'eval_metric':tf.keras.metrics.AUC(),
    'patience' : 0.001,
    'greater_is_better' : True,
    'embedding_size' : 8,
    'dropout'  : None,#长度为deep layers长度+1
    'deep_layers' : [128,128],
    'embeddings_initializer' : tf.keras.initializers.GlorotUniform,
    'kernel_initializer' : tf.keras.initializers.GlorotUniform,
    'verbose' : 100,
    'random_seed' : 2020,
    'l2_reg' : 1,
    'use_bn' : True,

    }

    # PNN_FGCN
    params['PNN_FGCN'] = {
        'label_dict': label_dict,
        'label_len_dict': label_len_dict,
        'label_col': 'HasDetections',
        'features': features,
        'pnn_model' : 'ipnn',
        'use_fgcn' : True,
        'batch_size': 256,
        'test_val_batch_size': 1024,
        'epochs': 1,
        'train_path': train_path,
        'early_stop': True,
        'early_stop_round': 1,
        'eval_path': val_path,
        'eval_step': 2000,
        'eval_metric': tf.keras.metrics.AUC(),
        'patience': 0.001,
        'greater_is_better': True,
        'embedding_size': 8,
        'dropout': None,  # 长度为deep layers长度+1
        'deep_layers': [4096, 2048, 1024,512],
        'embeddings_initializer': tf.keras.initializers.GlorotUniform,
        'kernel_initializer': tf.keras.initializers.GlorotUniform,
        'verbose': 100,
        'random_seed': 2020,
        'l2_reg': 1,
        'use_bn': True,
    }

    # XDEEPFM
    params['XDEEPFM'] = {
        'label_dict': label_dict,
        'label_len_dict': label_len_dict,
        'label_col': 'HasDetections',
        'features': features,
        'batch_size': 512,
        'test_val_batch_size': 1024,
        'epochs': 1,
        'train_path': train_path,
        'early_stop': True,
        'early_stop_round': 1,
        'eval_path': val_path,
        'eval_step': 1000,
        'eval_metric': tf.keras.metrics.AUC(),
        'patience': 0.001,
        'greater_is_better': True,
        'embedding_size': 10,
        'dropout': None,  # 长度为deep layers长度+1
        'deep_layers': [128, 128],
        'embeddings_initializer': tf.keras.initializers.TruncatedNormal(),
        'kernel_initializer': tf.keras.initializers.TruncatedNormal(),
        'verbose': 100,
        'random_seed': 2020,
        'l2_reg': 1,
        'use_bn': True,
        'cin_layers' : [128,128,128]
    }
    return params