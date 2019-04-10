import numpy as np
import scipy.ndimage as scn
import glob



trainDataPath = "./data/seg_train/"
testDataPath = "/data/seg_test/"

labels = ["buildings","forest","glacier","mountain","sea","street"]
trainData = {}

for label in labels:
    trainData.update({label:[]})
    trainDatap = trainDataPath + label
    trainFiles = glob.glob(trainDatap+"/*.jpg")
    
    for trainFile in trainFiles:
        trainData[label].append(scn.imread(trainFile, mode = 'L'))
    print(label)
    print(len(trainData[label]))
