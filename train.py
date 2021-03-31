import argparse
import warnings

from mmcv import Config
import numpy as np
from models.widedeep import WideDeep
from utils.dataset import create_song_dataset

warnings.filterwarnings("ignore")
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC


def train():
    # 数据处理
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    cfg = Config.fromfile(args.config)
    train_cfg = cfg.get('train_cfg')
    model_cfg = cfg.get('model_cfg')
    dataset_cfg = cfg.get('dataset_cfg')
    feature_columns, train, test = create_song_dataset(data_path=train_cfg.get('data_path'),
                                                       read_part=train_cfg.get('read_part'),
                                                       sample_num=train_cfg.get('sample_num'),
                                                       test_size=train_cfg.get('test_size'),
                                                       embed_dim=model_cfg.get('embed_dim'))
    train_X, train_y = train
    test_X, test_y = test
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = WideDeep(feature_columns,
                     model_cfg.get('deep_hidden_units'),
                     model_cfg.get('wide_hidden_units'),
                     label_embed_nums=model_cfg.get('label_embed_nums'),
                     dnn_dropout=model_cfg.get('dnn_dropout'),
                     classes=dataset_cfg.get('classes')
                     )
    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=train_cfg.get('lr')),
                  metrics=[AUC()])

    # ===========================Fit==============================
    ## 测试
    # dense_features = np.random.random((1, 10))
    # embedding_features = np.array([[10]])
    # raw_features = np.array([[1]])
    # predict = model((dense_features, embedding_features, raw_features))

    model.fit(
        train_X,
        train_y,
        epochs=train_cfg.get('epochs'),
        #callbacks=[EarlyStopping(monitor='val_auc', patience=2, restore_best_weights=True)],
        # checkpoint
        batch_size=train_cfg.get('batch_size'),
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=train_cfg.get('batch_size'))[1])


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train()
