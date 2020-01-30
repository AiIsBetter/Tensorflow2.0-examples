# coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.01.29

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
# sys import
import math
import time
from contextlib import contextmanager


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Generate Integer Features.
def build_int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))

# Generate Float Features.
def build_float_feature(data):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))

# Generate String Features.
def build_string_feature(data):
    data = data.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

# Generate a TF `Example`, parsing all features of the dataset.
def convert_to_tfrecord(types_dict,value):
    features = {}
    fun = {'int64':build_int64_feature,'float64':build_float_feature,'object':build_string_feature}
    for i in value:
        features[i] = fun[types_dict[i]](value[i])
        # if 'int' in types_dict[i]:
        #     features[i] = build_int64_feature(value[i])
        # elif 'float' in types_dict[i]:
        #     features[i] = build_float_feature(value[i])
        # else:
        #     features[i] = build_string_feature(value[i].encode())

    return tf.train.Example(
        features=tf.train.Features(
            feature=features)
    )
def main(path=None, save_path = None,debug=True):
    label_dict = {}
    with timer("train data process"):
            data = pd.read_csv(path,chunksize=1000)
            count = 0
            def hash_encode(x):
                return hash(x)
            with tf.io.TFRecordWriter(save_path) as w:
                for chunk in data:
                    print(count)
                    types_dict = {}
                    for fea in chunk.columns:
                        types= str(chunk[fea].dtype)
                        if types !='object' and types !='category':
                            if fea not in label_dict:
                                label_dict[fea] = list(chunk[fea].unique())
                            else:
                                label_dict[fea] = label_dict[fea]+list(chunk[~chunk[fea].isin(label_dict[fea])][fea])
                            chunk[fea] = chunk[fea].fillna(-1)
                            # if 'float' in types:
                            #     chunk[fea] = chunk[fea].apply(lambda x:int(math.log(1 + x * x))).astype(np.float32)
                            # else:
                            chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                            types_dict[fea] = types
                        else:
                            chunk[fea] = chunk[fea].fillna('-1')
                            types_dict[fea] = types

                    for row in chunk.iterrows():
                        tmp = dict(row[1])
                        # Parse each csv row to TF Example using the above functions.
                        example = convert_to_tfrecord(types_dict,tmp)
                        # Serialize each TF Example to string, and write to TFRecord file.
                        w.write(example.SerializeToString())
                    count +=1
    with open('../data/decode.txt','w') as f:
        for i in types_dict.keys():
            f.writelines(i+','+types_dict[i]+'\n')
    with timer("test data process"):
            data = pd.read_csv(path,chunksize=100000)
            count = 0
            def hash_encode(x):
                return hash(x)
            with tf.io.TFRecordWriter(save_path) as w:
                for chunk in data:
                    print(count)
                    types_dict = {}

                    for fea in chunk.columns:
                        types= str(chunk[fea].dtype)
                        if types !='object' and types !='category':
                            chunk[fea] = chunk[fea].fillna(-1)
                            # if 'float' in types:
                            #     chunk[fea] = chunk[fea].apply(lambda x:int(math.log(1 + x * x))).astype(np.float32)
                            # else:
                            chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                            types_dict[fea] = types
                        else:
                            chunk[fea] = chunk[fea].fillna('-1')
                            types_dict[fea] = types

                    for row in chunk.iterrows():
                        tmp = dict(row[1])
                        # Parse each csv row to TF Example using the above functions.
                        example = convert_to_tfrecord(types_dict,tmp)
                        # Serialize each TF Example to string, and write to TFRecord file.
                        w.write(example.SerializeToString())
                    count +=1



if __name__ == "__main__":
    path = '../data/train_sample.csv'
    save_path = '../data/train_sample.tfrecord'
    with timer("Full feature select run"):
        main(path=path,save_path=save_path, debug=True)