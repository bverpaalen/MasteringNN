import tensorflow as tf
import keras

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
    print(trainData[:label])
    
    testData.update({label:[]})
    testDatap = testDataPath + label
    testFiles = glob.glob(testDatap + "/*.jpg")    
    
    for testFile in testFiles:
        testData[label].append(scn.imread(testFile, mode = 'L')/256)
    print(label)
    print(len(testData[label]))
    
for images in trainData:    
    inputLayer = np.array()
    outputs = np.array()
    
for tests in testData:
    imageData = np.array()
    labelData = np.array()

#CNN Model Architecture:
model = tf.keras.models.Sequential()

#First Layer:
conv_1 = keras.layers.Conv2D(
        36, kernel_size = (5, 5),
        strides = (2, 2), padding = 'valid',
        activation = 'relu', bias_initializer = 'zeros',
        input_shape = (150, 150, 1))

pool_1 = keras.layers.MaxPooling2D(2, 2)

model.add([conv_1], [pool_1])

#Second Layer:
conv_2 = keras.layers.Conv2D(
        72, kernel_size = (5, 5),
        strides = (2, 2), padding = 'valid',
        activation = 'relu', bias_initializer = 'zeros')

pool_2 = keras.layers.MaxPool2D(2, 2)

model.add([conv_2], [pool_2])

#Third Layer:
conv_3 = keras.layers.Conv2D(
        72, kernel_size = (5, 5),
        strides = (2, 2), padding = 'valid',
        activation = 'relu', bias_initializer = 'zeros')

pool_3 = keras.layers.MaxPool2D(2, 2)

model.add([conv_3], [pool_3])

#Fully Connected Layers:
fConnect = keras.layers.Flatten()
fConnect_4 = keras.layers.Dense(300, activation = 'relu')
fConnect_5 = keras.layers.Dense(6, activation = 'softmax')

model.add([fConnect], [fConnect_4], [fConnect_5])


#Run Model:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(inputLayer, outputs, epochs = 5)

test_accuracy = model.evaulate(imageData, labelData)