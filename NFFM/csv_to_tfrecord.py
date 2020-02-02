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
import json
import gc
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
    # 该词典只存入要处理的特征，不存入MachineIdentifier id列和HasDetections label列
    label_dict = {}
    label_len_dict = {}
    with timer("train label dict process"):
        # 该模块用来对所有特征列统计特征值，在下个模块用这些获取的字典来进行labelencode
        print("train label dict process start ")
        data = pd.read_csv(path, chunksize=1000000)
        count = 0
        for chunk in data:
            print(count)
            for fea in chunk.columns:
                types = str(chunk[fea].dtype)
                if fea == 'MachineIdentifier' or fea == 'HasDetections':
                    continue
                if types != 'object' and types != 'category':
                    chunk[fea] = chunk[fea].fillna(-1)
                    chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                    if fea not in label_dict:
                        tmp = list(chunk[fea].unique())
                        label_dict[fea] = tmp
                        label_len_dict[fea] = list(range(len(tmp)))
                    else:
                        tmp = list(chunk[~chunk[fea].isin(label_dict[fea])][fea].unique())
                        label_dict[fea] = label_dict[fea] + tmp
                        label_len_dict[fea] = label_len_dict[fea] + list(range(len(label_len_dict[fea]),len(label_len_dict[fea])+len(tmp)))
                else:
                    chunk[fea] = chunk[fea].fillna('-1')
                    if fea not in label_dict:
                        tmp = list(chunk[fea].unique())
                        label_dict[fea] = tmp
                        label_len_dict[fea] = list(range(len(tmp)))
                    else:
                        tmp = list(chunk[~chunk[fea].isin(label_dict[fea])][fea].unique())
                        label_dict[fea] = label_dict[fea] + tmp
                        label_len_dict[fea] = label_len_dict[fea] + list(range(len(label_len_dict[fea]),len(label_len_dict[fea])+len(tmp)))
        del data
        gc.collect()
    with timer("train data process"):
        # 所有特征列做labelencode，使用map函数，没有使用labelencode函数，
        print("train data process start ")
        data = pd.read_csv(path,chunksize=100000)
        count = 0
        with tf.io.TFRecordWriter(save_path) as w:
            for chunk in data:
                print(count)
                # 该词典存入所有列，用来存入所有列到tfrecord里面
                types_dict = {}
                for fea in chunk.columns:
                    if fea == 'MachineIdentifier' or fea == 'HasDetections':
                        types = str(chunk[fea].dtype)
                        types_dict[fea] = types
                        continue
                    types = str(chunk[fea].dtype)
                    if types !='object' and types !='category':
                        chunk[fea] = chunk[fea].fillna(-1)
                        chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                        chunk[fea] = chunk[fea].map(dict(zip(label_dict[fea], label_len_dict[fea])))
                        types = str(chunk[fea].dtype)
                        types_dict[fea] = types
                    else:
                        chunk[fea] = chunk[fea].fillna('-1')
                        chunk[fea] = chunk[fea].map(dict(zip(label_dict[fea],label_len_dict[fea])))
                        types = str(chunk[fea].dtype)
                        types_dict[fea] = types

                for row in chunk.iterrows():
                    tmp = dict(row[1])
                    # Parse each csv row to TF Example using the above functions.
                    example = convert_to_tfrecord(types_dict,tmp)
                    # Serialize each TF Example to string, and write to TFRecord file.
                    w.write(example.SerializeToString())
                count +=1
    # with open('../data/decode.txt','w') as f:
    #     # for i in types_dict.keys():
    #     #     f.writelines(i+','+types_dict[i]+'\n')

    with timer("val label dict process"):
        print("val label dict process start ")
        data = pd.read_csv(valid_path, chunksize=100000)
        count = 0
        # 将valid里面新出现的特征值都归为一类
        for chunk in data:
            print(count)
            for fea in chunk.columns:
                types = str(chunk[fea].dtype)
                if fea == 'MachineIdentifier' or fea == 'HasDetections':
                    continue
                if types != 'object' and types != 'category':
                    chunk[fea] = chunk[fea].fillna(-1)
                    chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                    tmp = list(chunk[~chunk[fea].isin(label_dict[fea])][fea].unique())
                    label_dict[fea] = label_dict[fea] + tmp
                    label_len_dict[fea] = label_len_dict[fea] + [len(label_len_dict[fea])]*len(tmp)
                else:
                    chunk[fea] = chunk[fea].fillna('-1')
                    tmp = list(chunk[~chunk[fea].isin(label_dict[fea])][fea].unique())
                    label_dict[fea] = label_dict[fea] + tmp
                    label_len_dict[fea] = label_len_dict[fea] + [len(label_len_dict[fea])]*len(tmp)

    with timer("val data process"):
        print("val data process start ")
        data = pd.read_csv(valid_path, chunksize=100000)
        count = 0
        with tf.io.TFRecordWriter(save_valid_path) as w:
            for chunk in data:
                print(count)
                for fea in chunk.columns:
                    if fea == 'MachineIdentifier' or fea == 'HasDetections':
                        types = str(chunk[fea].dtype)
                        continue
                    types = str(chunk[fea].dtype)
                    if types != 'object' and types != 'category':
                        chunk[fea] = chunk[fea].fillna(-1)
                        chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                        chunk[fea] = chunk[fea].map(dict(zip(label_dict[fea], label_len_dict[fea])))
                    else:
                        chunk[fea] = chunk[fea].fillna('-1')
                        chunk[fea] = chunk[fea].map(dict(zip(label_dict[fea], label_len_dict[fea])))

                for row in chunk.iterrows():
                    tmp = dict(row[1])
                    example = convert_to_tfrecord(types_dict, tmp)
                    w.write(example.SerializeToString())
                count += 1

    with timer("test label dict process"):
        print("test label dict process start ")
        data = pd.read_csv(test_path, chunksize=100000)
        count = 0
        # 将test里面新出现的特征值都归为一类
        for chunk in data:
            print(count)
            for fea in chunk.columns:
                types = str(chunk[fea].dtype)
                if fea == 'MachineIdentifier' or fea == 'HasDetections':
                    continue
                if types != 'object' and types != 'category':
                    chunk[fea] = chunk[fea].fillna(-1)
                    chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                    tmp = list(chunk[~chunk[fea].isin(label_dict[fea])][fea].unique())
                    label_dict[fea] = label_dict[fea] + tmp
                    label_len_dict[fea] = label_len_dict[fea] + [len(label_len_dict[fea])]*len(tmp)
                else:
                    chunk[fea] = chunk[fea].fillna('-1')
                    tmp = list(chunk[~chunk[fea].isin(label_dict[fea])][fea].unique())
                    label_dict[fea] = label_dict[fea] + tmp
                    label_len_dict[fea] = label_len_dict[fea] + [len(label_len_dict[fea])]*len(tmp)
        # 保存后续需要使用的三个字典
        json_dump = json.dumps(types_dict)
        fileObject = open(root_path + 'types_dict.json', 'w')
        fileObject.write(json_dump)
        fileObject.close()

        for i in label_dict.keys():
            if not isinstance(label_dict[i][0], str):
                label_dict[i] = [int(j) for j in label_dict[i]]
        json_dump = json.dumps(label_dict)
        fileObject = open(root_path + 'label_dict.json', 'w')
        fileObject.write(json_dump)
        fileObject.close()

        for i in label_len_dict.keys():
            if not isinstance(label_len_dict[i][0], str):
                label_len_dict[i] = [int(j) for j in label_len_dict[i]]
        json_dump = json.dumps(label_len_dict)
        fileObject = open(root_path + 'label_len_dict.json', 'w')
        fileObject.write(json_dump)
        fileObject.close()

        del data
        gc.collect()
        with timer("test data process"):
            print("test data process start ")
            data = pd.read_csv(test_path, chunksize=100000)
            count = 0
            with tf.io.TFRecordWriter(save_test_path) as w:
                for chunk in data:
                    print(count)
                    for fea in chunk.columns:
                        if fea == 'MachineIdentifier' or fea == 'HasDetections':
                            types = str(chunk[fea].dtype)
                            continue
                        types = str(chunk[fea].dtype)
                        if types != 'object' and types != 'category':
                            chunk[fea] = chunk[fea].fillna(-1)
                            chunk[fea] = chunk[fea].apply(lambda x: int(math.log(1 + x * x)))
                            chunk[fea] = chunk[fea].map(dict(zip(label_dict[fea], label_len_dict[fea])))
                        else:
                            chunk[fea] = chunk[fea].fillna('-1')
                            chunk[fea] = chunk[fea].map(dict(zip(label_dict[fea], label_len_dict[fea])))

                    for row in chunk.iterrows():
                        tmp = dict(row[1])
                        # Parse each csv row to TF Example using the above functions.
                        example = convert_to_tfrecord(types_dict, tmp)
                        # Serialize each TF Example to string, and write to TFRecord file.
                        w.write(example.SerializeToString())
                    count += 1



if __name__ == "__main__":
    root_path = '../data/'
    path = root_path+'train_full.csv'
    test_path = root_path + 'test_full.csv'
    valid_path = root_path + 'val_full.csv'
    save_path = root_path + 'train_full.tfrecord'
    save_valid_path = root_path + 'val_full.tfrecord'
    save_test_path = root_path + 'test_full.tfrecord'
    with timer("Full feature select run"):
        main(path=path,save_path=save_path, debug=True)