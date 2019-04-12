import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import numpy as np
import imageio as ima
#import visvis as vv
import glob
import random


#Load Data
trainDataPath = "./data/seg_train/"
testDataPath = "./data/seg_test/"

labels = ["buildings","forests","glaciers","mountains","seas","streets"]


def shuffle(data):
    labels = []
    features = []

    random.shuffle(data)
    for dataPoint in data:
        labels.append(dataPoint[0])
        features.append(dataPoint[1])
    
    return labels, features
    

def modelCNN(data, label):
    #CNN Model Architecture:
    model = tf.keras.models.Sequential()
    
    #First Layer:
    conv_1 = Conv2D(
            36, kernel_size = (5, 5),
            strides = (2, 2), padding = 'valid',
            activation = 'relu', bias_initializer = 'zeros',
            input_shape = (150, 150, 1))
    
    pool_1 = MaxPooling2D(2, 2)
    
    model.add(conv_1)
    model.add(pool_1)
    
    #Second Layer:
    conv_2 = Conv2D(
            72, kernel_size = (5, 5),
            strides = (2, 2), padding = 'valid',
            activation = 'relu', bias_initializer = 'zeros')
    
    pool_2 = MaxPooling2D(2, 2)
    
    model.add(conv_2)
    model.add(pool_2)
    
    #Third Layer:
    conv_3 = Conv2D(
            72, kernel_size = (5, 5),
            strides = (2, 2), padding = 'valid',
            activation = 'relu', bias_initializer = 'zeros')
    
    pool_3 = MaxPooling2D(2, 2)
    
    model.add(conv_3)
    model.add(pool_3)
    
    #Fully Connected Layers:
    fConnect = Flatten()
    fConnect_4 = Dense(300, activation = 'relu')
    fConnect_5 = Dense(6, activation = 'softmax')
    
    model.add(fConnect)
    model.add(fConnect_4)
    model.add(fConnect_5)


    #Run Model:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    accuracy = model.fit(data, label, epochs = 5)
    
    return accuracy
    

def main():
    trainData = []
    testData = []
    print("The training data consists of :")
    
    for label in labels:
        trainDatap = trainDataPath + label
        trainFiles = glob.glob(trainDatap + "/*.jpg")
        testDatap = testDataPath + label
        testFiles = glob.glob(testDatap + "/*.jpg")

        for trainFile in trainFiles:
            trainPicture = [ label, ima.imread(trainFile)/256 ]
            trainData.append(trainPicture)
            
        for testFile in testFiles:
            testPicture =  [ label, ima.imread(testFile)/256 ]
            testData.append(testPicture)
        
        print(str(len(trainData)) + ' ' + label)

    inputData, outputData = shuffle(trainData)
    
    modelCNN(inputData, outputData)



#-------------------------------------------------------------------------------------------------------------------------------
main()