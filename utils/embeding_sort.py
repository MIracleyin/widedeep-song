# todo 实现精排，对embedding后值，计算相似度矩阵，归一化后，返回topk的track_uri
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

def self_cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim == 1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right.".format(a.ndim))
    similarity = np.dot(a, b.T) / (a_norm * b_norm)
    dist = 1. - similarity
    return dist

def partition_arg_topk(array, K, axis=0):
    a_part = np.argpartition(array, -K, axis=axis)[-K: len(array)]
    if axis == 0:
        row_index = np.arange(array.shape[1 - axis])
        a_sec_argsort_K = np.argsort(array[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index][::-1]
    else:
        column_index = np.arange(array.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(array[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K][::-1]

def embedding_sort(input_song, embeded_songs, k):
    cos_similarity = np.zeros(len(embeded_songs))
    for i, embeded_song in enumerate(embeded_songs):
        cos_similarity[i] = self_cosine_distance(input_song, embeded_song)
    mms = MinMaxScaler(feature_range=(0, 1))
    cos_similarity = mms.fit_transform(cos_similarity.reshape(-1, 1))
    test = sorted(cos_similarity)
    return partition_arg_topk(cos_similarity, k), cos_similarity