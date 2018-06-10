import numpy as np
import cv2
import os
import cPickle as pickle

def process_mean(data):
    mean = np.mean(data, axis = 0)
    std = np.std(data - mean, axis = 0)
    preprocess = {'mean':mean, 'std':std}
    pickle.dump(preprocess, open("./data/preprocess_coeffs.pkl", 'wb'))

def process(data):
    preprocess = pickle.load(open("./data/preprocess_coeffs.pkl", 'rb'))
    mean = preprocess['mean']
    std = proprocess['std']
    data -= mean
    data /= std
    return data

def one_hot_encode(labels):
    x = np.zeros([labels.shape[0], labels.shape[1], 3])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] not in (0, 1, 2):
                labels[i][j] = 0
            x[i, j, labels[i][j]] = 1
    return x

def scale(img, height, width):
    return cv2.resize(img, (width, height))

def prep_data(name):
    with open(os.getcwd()+'/Data/'+name+'.txt') as f:
        txt = f.readlines()
        txt = [line.rstrip('\n').split(',') for line in txt]

    data = []
    labels = []
    for i in range(len(txt)):
        data.append(scale(cv2.imread(os.getcwd()+'/Data'+txt[i][0]), 480, 360))
        labels.append(one_hot_encode(scale(cv2.imread(os.getcwd()+'/Data'+txt[i][1]), 480, 360)[:, :, 0]))
    
    data = np.array(data)
    labels = np.array(labels)
    
    if name == 'train':
        process_store(data)

    process(data)
    return data, labels
