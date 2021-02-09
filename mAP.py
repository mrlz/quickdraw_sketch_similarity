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
import n1_features
import n2_features
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing as mp

def pr(label, list, index):
    equal = 0
    for i in range(index):
        if (label == list[i]):
            equal = equal + 1
    return equal/(index+1)

def AP(label, list, k):
    equal = 0
    amount = 0
    for i in range(k):
        if (label == list[i]):
            equal = equal + 1
            amount = amount + pr(label, list, i)
    if (equal == 0):
        return 0
    return amount/equal

def mAP(labels, lists, k):
    amount = 0
    for i in range(len(labels)):
        amount = amount + AP(labels[i], lists[i][1:], k)
    return amount/len(labels)

def reorder_set(arguments):
    query = arguments[0]
    query_label = arguments[1]
    set = arguments[2]
    distances = []
    for element in set:
        distances.append([np.sum(euclidean_distances([query],[element[0]])), element[1]])
    distances = sorted(distances)
    return [i[1] for i in distances]

def process_queries(features, labels, pool):
    set = [item for item in zip(features,labels)]
    arguments = []
    for i in range(len(labels)):
        arguments.append([features[i], labels[i], set])
    results = pool.map(reorder_set, arguments)
    return results

with open('./conversion_labels', 'rb') as fp:
    conversion_labels = pickle.load(fp)

print("Loading data")
start = time.time()
print("Test set")
test_set = create_dataset.load_dataset_if_present('./testing_lists', './testing', conversion_labels)
print("Converting to numpy arrays")
images_test = np.array(test_set[0], dtype = np.float32)
labels_test = np.array(test_set[1], dtype = np.float32)
end = time.time()
print("Loading time:", end-start)

batch_size = 128

net_names = ['n1_10e', 'n1_15e', 'n1_20e', 'n2_10e', 'n2_15e', 'n2_20e']
pool = mp.Pool(processes=4)
for i in range(len(net_names)):
    if(i < 3):
        model = tf.estimator.Estimator(n1_features.model_fn, model_dir = './checkpoints_'+net_names[i])
    else:
        model = tf.estimator.Estimator(n2_features.model_fn, model_dir = './checkpoints_'+net_names[i])
    distances_path = './'+net_names[i]

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': images_test}, y=labels_test,
        batch_size=batch_size, shuffle=False)


    print("Computing feature vector distances")
    start_t = time.time()
    if os.path.exists(distances_path):
        print(" Found in disk")
        with open(distances_path, 'rb') as fp:
            results = pickle.load(fp)
    else:
        start = time.time()
        print("Computing feature vectors from fc1")
        predictions = list(model.predict(input_fn))
        end = time.time()
        print(" Computing took:", end-start)
        print(" Computing from features")
        results = process_queries(predictions, labels_test, pool)
        with open(distances_path, 'wb') as fp:
            pickle.dump(results, fp)
    end_t = time.time()
    print(" Computing took:", end_t-start_t)
    print("Computing mAP")
    start = time.time()
    k = 4999
    m_AP = mAP(labels_test, results, k)
    end = time.time()
    print(" Computing took:", end-start)
    print("         mAP for " + net_names[i] + ":", m_AP)
