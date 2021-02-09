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
import create_dataset
import n1
import n2

with open('./conversion_labels', 'rb') as fp:
    conversion_labels = pickle.load(fp)

print("Loading data")
start = time.time()
print("Train set")
train_set = create_dataset.load_dataset_if_present('./training_lists', './training', conversion_labels)
print("Test set")
test_set = create_dataset.load_dataset_if_present('./testing_lists', './testing', conversion_labels)
print("Converting to numpy arrays")
images_train = np.array(train_set[0], dtype = np.float32)
labels_train = np.array(train_set[1], dtype = np.float32)
images_test = np.array(test_set[0], dtype = np.float32)
labels_test = np.array(test_set[1], dtype = np.float32)
end = time.time()
print("Loading time:", end-start)


batch_size = 128


net_names = ['n1_10e','n1_15e','n1_20e','n2_10e','n2_15e', 'n2_20e']
for i in range(len(net_names)):

    if(i < 3):
        model = tf.estimator.Estimator(n1.model_fn, model_dir = './checkpoints_'+net_names[i])
    else:
        model = tf.estimator.Estimator(n2.model_fn, model_dir = './checkpoints_'+net_names[i])

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': images_test}, y=labels_test,
        batch_size=batch_size, shuffle=False)

    start = time.time()
    metrics = model.evaluate(input_fn)
    end = time.time()
    print(metrics)
    print("eval time", end-start)
    print("/////////////////////////////////////////")
    print(net_names[i]+"_accuracy_test:", metrics['accuracy'])
    print("/////////////////////////////////////////")


    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': images_train}, y=labels_train,
        batch_size=batch_size, shuffle=False)

    start = time.time()
    metrics = model.evaluate(input_fn)
    end = time.time()
    print(metrics)
    print("eval time", end-start)
    print("////////////////////////////////////////")
    print(net_names[i]+"_accuracy_train:", metrics['accuracy'])
    print("////////////////////////////////////////")
