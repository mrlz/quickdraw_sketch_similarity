import struct
from struct import unpack
import numpy as np
from PIL import Image
import cv2
import random
import os
from sklearn import datasets
import time
import pickle
import os.path

def read_dataset(path, conversion_labels):
    images = []
    labels = []
    data = datasets.load_files(path)
    for i in range(len(data['filenames'])):
        type = data['filenames'][i].split('/')[2]
        images.append(list(Image.open(data['filenames'][i]).getdata()))
        labels.append(conversion_labels[type])
    return images, labels

def load_dataset_if_present(path, raw_data_path, conversion_labels):
    if os.path.exists(path):
        print(" Found in disk")
        with open(path, 'rb') as fp:
            lists = pickle.load(fp)
    else:
        print(" Computing from images")
        lists = read_dataset(raw_data_path, conversion_labels)
        with open(path, 'wb') as fp:
            pickle.dump(lists, fp)
    return lists

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break

def convert_to_image(points):
    blank_image = np.zeros((256,256), np.uint8)
    blank_image[:,:] = 255
    for stroke in points:
        start = 0
        point1 = (0,0)
        for point in zip(stroke[0], stroke[1]):
            blank_image[point[0]][point[1]] = 0
            point2 = (point[0], point[1])
            if start == 1:
                cv2.line(blank_image, point1, point2, 0, 3)
            point1 = (point[0], point[1])
            start = 1
    return blank_image

def selectKitems(stream, k):
    selected_items = []
    for i in range(k):
        selected_items.append(stream[i])
    for i in range(k,len(stream)):
        rand = random.randint(0,i+1)
        if(rand < k):
            selected_items[rand] = stream[i]
    return selected_items

def construct_from_scratch():
    counter = 0
    with open('categories.txt') as f:
        categories = f.readlines()
        categories = [x.strip() for x in categories]


    num_classes = 100

    selected_categories = selectKitems(categories,num_classes)
    training_drawings = {}
    testing_drawings = {}
    label_conversion = {}
    index = 0
    print("selected categories")
    print(selected_categories)
    if not os.path.exists('./training'):
        os.makedirs('./training')
    if not os.path.exists('./testing'):
        os.makedirs('./testing')
    for category in selected_categories:
        training_drawings[category] = []
        testing_drawings[category] = []
        label_conversion[category] = index
        if not os.path.exists('./training/' + category):
            os.makedirs('./training/' + category)
        if not os.path.exists('./testing/' + category):
            os.makedirs('./testing/' + category)
        count = 0
        for drawing in selectKitems(list(unpack_drawings('./files/' + category + '.bin')), 1050):
            data = drawing['image']
            id = drawing['key_id']
            img_array = convert_to_image(data)
            img = Image.fromarray(img_array, 'L')
            img.thumbnail([128,128], Image.ANTIALIAS)
            print(id)
            if(count < 1000):
                img.save('./training/' + category + '/' + str(id) + '.png')
                training_drawings[category].append(list(img.getdata()))
            else:
                img.save('./testing/' + category + '/' + str(id) + '.png')
                testing_drawings[category].append(list(img.getdata()))
            count = count + 1
        index = index + 1

    with open('./conversion_labels', 'wb') as fp:
        pickle.dump(label_conversion, fp)
