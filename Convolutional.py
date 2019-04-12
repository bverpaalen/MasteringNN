import tensorflow as tf

import numpy as np
import imageio as ima
import visvis as vv
import glob
import random

#Load Data
trainDataPath = "./data/seg_train/"
testDataPath = "./data/seg_test/"

labels = ["buildings","forest","glacier","mountain","sea","street"]

def train(data):
    labels = []
    features = []

    random.shuffle(data)
    for dataPoint in data:
        labels.append(dataPoint[0])
        features.append(dataPoint[1])
    #print(key)
    #print(data[key])
    print(labels)

    

def convolutionStep(filterData, weights, bias):
    data = np.dot(filterData, weights.T)
    output = np.sum(data) + float(bias)
    return output

def pads(X, size):
    pad = np.pad(X, ((0,0), (size,size), (size,size)), mode = 'constant', constant_values = (0,0))
    return pads

def forwardPropagation(data, weights):
    w1, w2 = initializeWeights()

def main():
    trainData = []
    testData = {}
    for label in labels:
        trainDatap = trainDataPath + label
        trainFiles = glob.glob(trainDatap + "/*.jpg")

        for trainFile in trainFiles:
            picture = [label,ima.imread(trainFile)/256]
            trainData.append(picture)
        print(label)
        print(len(trainData))

        testData.update({label:[]})
        testDatap = testDataPath + label
        testFiles = glob.glob(testDatap + "/*.jpg")

        for testFile in testFiles:
            testData[label].append(ima.imread(testFile)/256)

    train(trainData)
main()