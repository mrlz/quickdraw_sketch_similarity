from __future__ import division, print_function, absolute_import
import struct
from struct import unpack
import numpy as np
from PIL import Image
import cv2
import random
import os
import matplotlib.pyplot as plt
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from sklearn import datasets
import time
import pickle
import os.path

def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, 128, 128, 1])

        conv1_1 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
        conv1_1_b = tf.layers.batch_normalization(conv1_1, training=is_training)
        conv1_2 = tf.layers.conv2d(conv1_1_b, 64, 3, activation=tf.nn.relu)
        conv1_2_b = tf.layers.batch_normalization(conv1_2, training = is_training)
        conv1_pool = tf.layers.max_pooling2d(conv1_2_b, 3, 2)

        paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
        conv2_1 = tf.layers.conv2d(tf.pad(conv1_pool, paddings, "CONSTANT"), 64, 3, activation=tf.nn.relu)
        conv2_1_b = tf.layers.batch_normalization(conv2_1, training=is_training)
        conv2_2 = tf.layers.conv2d(conv2_1_b, 64, 3, activation=tf.nn.relu)
        conv2_2_b = tf.layers.batch_normalization(conv2_2, training = is_training)
        residual_1 = conv2_2_b + conv1_pool

        conv3_1 = tf.layers.conv2d(tf.pad(residual_1, paddings, "CONSTANT"), 64, 3, activation=tf.nn.relu)
        conv3_1_b = tf.layers.batch_normalization(conv3_1, training=is_training)
        conv3_2 = tf.layers.conv2d(conv3_1_b, 64, 3, activation=tf.nn.relu)
        conv3_2_b = tf.layers.batch_normalization(conv3_2, training = is_training)
        residual_2 = conv3_2_b + residual_1

        conv4_1 = tf.layers.conv2d(residual_2, 128, 3, activation=tf.nn.relu)
        conv4_1_b = tf.layers.batch_normalization(conv4_1, training=is_training)
        conv4_pool = tf.layers.max_pooling2d(conv4_1_b, 3, 2)

        conv5_1 = tf.layers.conv2d(tf.pad(conv4_pool,paddings, "CONSTANT"), 128, 3, activation=tf.nn.relu)
        conv5_1_b = tf.layers.batch_normalization(conv5_1, training=is_training)
        conv5_2 = tf.layers.conv2d(conv5_1_b, 128, 3, activation=tf.nn.relu)
        conv5_2_b = tf.layers.batch_normalization(conv5_2, training = is_training)
        residual_3 = conv5_2_b + conv4_pool

        conv6_1 = tf.layers.conv2d(tf.pad(residual_3, paddings, "CONSTANT"), 128, 3, activation=tf.nn.relu)
        conv6_1_b = tf.layers.batch_normalization(conv6_1, training=is_training)
        conv6_2 = tf.layers.conv2d(conv6_1_b, 128, 3, activation=tf.nn.relu)
        conv6_2_b = tf.layers.batch_normalization(conv6_2, training = is_training)
        residual_4 = conv6_2_b + residual_3

        conv7_1 = tf.layers.conv2d(residual_4, 256, 3, activation=tf.nn.relu)
        conv7_1_b = tf.layers.batch_normalization(conv7_1, training=is_training)
        conv7_pool = tf.layers.max_pooling2d(conv7_1_b, 3, 2)

        conv8_1 = tf.layers.conv2d(tf.pad(conv7_pool, paddings, "CONSTANT"), 256, 3, activation=tf.nn.relu)
        conv8_1_b = tf.layers.batch_normalization(conv8_1, training=is_training)
        conv8_2 = tf.layers.conv2d(conv8_1_b, 256, 3, activation=tf.nn.relu)
        conv8_2_b = tf.layers.batch_normalization(conv8_2, training = is_training)
        residual_5 = conv8_2_b + conv7_pool

        conv9_1 = tf.layers.conv2d(tf.pad(residual_5, paddings, "CONSTANT"), 256, 3, activation=tf.nn.relu)
        conv9_1_b = tf.layers.batch_normalization(conv9_1, training=is_training)
        conv9_2 = tf.layers.conv2d(conv9_1_b, 256, 3, activation=tf.nn.relu)
        conv9_2_b = tf.layers.batch_normalization(conv9_2, training = is_training)
        residual_6 = conv9_2_b + residual_5

        conv10_1 = tf.layers.conv2d(residual_6, 256, 3, activation=tf.nn.relu)
        conv10_1_b = tf.layers.batch_normalization(conv10_1, training=is_training)
        conv10_pool = tf.layers.max_pooling2d(conv10_1_b, 3, 2)
        fc1 = tf.contrib.layers.flatten(conv10_pool)

        fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)

        out = fc1
    return out



# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    learning_rate = 0.001
    num_classes = 100
    dropout = 0.25
    if mode == tf.estimator.ModeKeys.TRAIN:##
        is_training = True##
    else:##
        is_training = False##

    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=is_training)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=is_training)

    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=logits_test)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) ##
    with tf.control_dependencies(update_ops):#
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    tensors_to_log = {'batch_accuracy' : acc_op[1]}##
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 100)##
    estim_specs = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=pred_classes,
      loss=loss_op,
      train_op=train_op,
      eval_metric_ops={'accuracy': acc_op},
      training_hooks =[logging_hook])#

    return estim_specs
