import warnings

warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def dataset_feature_engineer(data_df, sparse_feature_items, dense_feature_items, embed_dim=8):
    # 复制一个专门的df做特征工程
    data_feature_engineer = data_df.copy()
    # 稠密特征归一化 有些特征已经是0-1之间 有些不是 对那些不是对做归一化
    # normalizer_dense_feature = ["duration_ms_x", "loudness", "tempo"]
    # 这部分放到 deep
    mms = MinMaxScaler(feature_range=(0, 1))
    data_feature_engineer[dense_feature_items] = mms.fit_transform(
        data_feature_engineer[dense_feature_items])  # 测试后，（0-1）基本没有发生变化 不在（0-1）的值归一化了
    dense_features = {feature: np.array(data_feature_engineer[feature]) for feature in dense_feature_items}
    dense_feature_column = [tf.feature_column.numeric_column(feature) for feature in dense_feature_items]
    # dense_features_engineer = tf.compat.v1.feature_column.input_layer(dense_feature_engineer, dense_feature_column)
    # dense_features 尺寸正确
    # 稀疏特征
    # 这部分方放到 wide 还需要做特征交叉
    unique_sparse_feature = {feature: np.array(data_feature_engineer[feature].unique()) for feature in
                             sparse_feature_items}
    sparse_features = {feature: list(data_feature_engineer[feature]) for feature in sparse_feature_items}
    album_uri = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket("album_uri", hash_bucket_size=1000,
                                                              dtype=tf.string), dimension=embed_dim)
    artist_uri = tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket("artist_uri", hash_bucket_size=1000,
                                                              dtype=tf.string), dimension=embed_dim)
    key = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key="key",
                                                                                                       vocabulary_list=
                                                                                                       unique_sparse_feature[
                                                                                                           "key"]))
    mode = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key="mode",
                                                                                                        vocabulary_list=
                                                                                                        unique_sparse_feature[
                                                                                                            "mode"]))

    sparse_feature_column = [album_uri, artist_uri, key, mode]
    # sparse_feature_engineer = tf.compat.v1.feature_column.input_layer(sparse_feature_engineer, sparse_feature_column)
    return sparse_features, sparse_feature_column, dense_features, dense_feature_column # 现在返回的是字典，和特征变换，使用用上一行注释放到tensor里面就好


def create_song_dataset(data, read_part=True, sample_num=10000, test_size=0.2):
    """
    a example about creating song dataset'
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """
    if read_part:
        data_df = pd.read_csv(data, iterator=True)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(data)
    # 离散特征: 专辑链接（字符串）、艺术家链接（字符串）、曲调(0-12)、音符时值（0-5）、所属歌单（整数连续） /曲目链接（字符串）"track_uri" 不使用 "pid"  不使用"time_signature"
    sparse_feature_items = ["album_uri", "artist_uri", "key", "mode"]
    # 连续特征: 歌曲时长、原声程度(0-1)、律动感(0-1)、冲击感(0-1)、歌唱部分占比(0-1)、现场感(0-1)、响度、重复度(0-1)、朗诵比例(0-1)、分钟节拍数、心理感受(0-1)
    dense_feature_items = ["duration_ms_x", "acousticness", "danceability", 'energy', 'instrumentalness', 'liveness',
                           'loudness',
                           'speechiness', 'tempo', 'valence']  # continuous
    # 返回的tensor dense_features 10, sparse_features 30
    sparse_features, sparse_feature_column, dense_features, dense_feature_column = dataset_feature_engineer(data_df,
                                                                                                            sparse_feature_items,
                                                                                                            dense_feature_items)
    #return all_features, (train_x, train_y), (test_x, test_y)
    # tf.dataloader


if __name__ == "__main__":
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_path = '~/workplace/RS/widedeep-song/data/data_sample.csv'
    create_song_dataset(data_path)