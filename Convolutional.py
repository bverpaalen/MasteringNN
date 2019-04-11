import tensorflow as tf

import numpy as np
import scipy.ndimage as scn
import glob


#Load Data
trainDataPath = "./data/seg_train/"
testDataPath = "/data/seg_test/"

labels = ["buildings","forest","glacier","mountain","sea","street"]
trainData = {}
testData = {}

for label in labels:
    trainData.update({label:[]})
    trainDatap = trainDataPath + label
    trainFiles = glob.glob(trainDatap + "/*.jpg")
    
    for trainFile in trainFiles:
        trainData[label].append(scn.imread(trainFile, mode = 'L')/256)
    print(label)
    print(len(trainData[label]))
    
    testData.update({label:[]})
    testDatap = testDataPath + label
    testFiles = glob.glob(testDatap + "/*.jpg")    
    
    for testFile in testFiles:
        testData[label].append(scn.imread(testFile, mode = 'L')/256)
    print(label)
    print(len(testData[label]))
    
    
def convolutionStep(filterData, weights, bias):
    data = np.dot(filterData, weights.T)
    output = np.sum(data) + float(bias)
    return output

def pads(X, size):
    pad = np.pad(X, ((0,0), (size,size), (size,size)), mode = 'constant', constant_values = (0,0))
    return pads

def forwardPropagation(data, weights):
    w1, w2 = initializeWeights()