# -*- coding: utf-8 -*-
# !@time: 2021/4/1 下午7:16
# !@author: superMC @email: 18758266469@163.com
# !@fileName: inference.py
import argparse
import os
import numpy as np
from tensorflow import keras
from mmcv import Config
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import MinMaxScaler
from utils.util_dir import generate_dir
from utils.dataset import create_song_dataset
from utils.embeding_sort import embedding_sort
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args




def inference():
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # train_cfg = cfg.get('train_cfg')
    val_cfg = cfg.get('val_cfg')
    model_cfg = cfg.get('model_cfg')
    dataset_cfg = cfg.get('dataset_cfg')
    work_dir = os.path.join('work_dirs', val_cfg.get('work_dir'))
    model_dir = os.path.join(work_dir, 'models', 'embed_model')
    generate_dir(work_dir)
    model = keras.models.load_model(model_dir)
    ## 测试
    # dense_features = np.random.random((1, 10))
    # embedding_features = np.array([[10]])
    # raw_features = np.array([[1]])
    # predict = model((dense_features, embedding_features, raw_features))
    # print(predict)
    # model.summary()
    # --------------------构造embeded特征—---------
    dense_features = np.random.random((100000, 10))
    embedding_features = np.random.random((100000, 1))
    raw_features = np.random.random((100000, 1))
    input_X = (dense_features, embedding_features, raw_features)
    embeded_Y = model(input_X).numpy()

    # ---------------用于测试输入的歌曲-------------
    start = time.time()
    test_X = np.random.random((64,))
    pred, cosine_similarity = embedding_sort(test_X, embeded_Y, 10)  # 输出歌曲idx
    print(cosine_similarity)

    # 根据index找到推荐的歌曲
    for i in pred:
        print(i, cosine_similarity[i])
    print(time.time() - start)


if __name__ == '__main__':
    inference()
