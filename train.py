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
import n1
import n2
import create_dataset

batch_size = 128

conversion_labels = {}

with open('./conversion_labels', 'rb') as fp:
    conversion_labels = pickle.load(fp)

print("Loading dataset from disk")
start = time.time()
train_set = create_dataset.load_dataset_if_present('./training_lists', './training', conversion_labels)
test_set = create_dataset.load_dataset_if_present('./testing_lists', './testing', conversion_labels)

images_train = np.array(train_set[0], dtype = np.float32)
labels_train = np.array(train_set[1], dtype = np.float32)
images_test = np.array(test_set[0], dtype = np.float32)
labels_test = np.array(test_set[1], dtype = np.float32)
end = time.time()
print("Load time:", end-start)

net_names = ['n1_', 'n2_']
epochs = [10]
for i in range(len(net_names)):
    for j in range(len(epochs)):
        if(i == 0):
            model = tf.estimator.Estimator(n1.model_fn, model_dir = './checkpoints_'+net_names[i] + str(epochs[j]) + "e")
        else:
            model = tf.estimator.Estimator(n2.model_fn, model_dir = './checkpoints_'+net_names[i] + str(epochs[j]) + "e")

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': images_train},
            y=labels_train,
            batch_size=batch_size, num_epochs=epochs[j], shuffle=True)
        print("Starting training")
        start = time.time()
        model.train(input_fn)
        end = time.time()
        print("Train time:", end-start)

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': images_train},
            y=labels_train,
            batch_size=batch_size, shuffle=False)

        print("Evaluating train set")
        start = time.time()
        metrics = model.evaluate(input_fn)
        end = time.time()
        print("eval time", end-start)
        print("accuracy_train:", metrics['accuracy'])

        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': images_test},
            y=labels_test,
            batch_size=batch_size, shuffle=False)

        print("Evaluating test set")
        start = time.time()
        metrics = model.evaluate(input_fn)
        end = time.time()
        print("eval time", end-start)
        print("accuracy_test:", metrics['accuracy'])
