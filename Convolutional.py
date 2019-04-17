import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import keras as ks
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cv2
import imageio as ima
import visvis as vv
import glob
import random

trainDataPath = "./data/seg_train/"
testDataPath = "./data/seg_test/"

labels = ["buildings"]#,"forest","glacier","mountain","sea","street"]

inputLayers = [{"conv":{"filters":32,"kernel_size":["5","5"],"padding":"same","activation":tf.nn.relu},"pool":{"pool_size":2,"strides":2}},{
"conv":{"filters":64,"kernel_size":["5","5"],"padding":"same","activation":tf.nn.relu},"pool":{"pool_size":2,"strides":2}}
               ]

def cnn_model(features,labels,mode):    
    size = 150
    pool = None

    print(labels)
    try:    
        input_layer = tf.reshape(features["image"],[-1,28,28,1])
    except:
        print()
        print("reshape went wrong")
        #features.print()
        print()
        input_layer = tf.reshape([float(0)]*28*28,[-1,28,28,1])

    #print(features["image"])

    for i in range(len(inputLayers)):
        print(i)
        print()

        layer = inputLayers[i]
        convInputs = layer["conv"]
        poolInputs = layer["pool"]

        if i < 1:
            pool = input_layer

        if pool == None:
            print("EMPTY POOL")
            print(i)
        conv = tf.layers.conv2d(
                 inputs=pool,
                 filters=convInputs["filters"],
                 kernel_size=convInputs["kernel_size"],
                 padding=convInputs["padding"],
                 activation=convInputs["activation"])

        pool = tf.layers.max_pooling2d(
                 inputs=conv,
                 pool_size=poolInputs["pool_size"],
                 strides=poolInputs["strides"])
     #Dense Layer
    tf.shape(pool)
    pool_flat = tf.reshape(pool,[-1,7*7*64])
     
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def randomizeData(data):
    labels = []
    features = []

    random.shuffle(data)
    for dataPoint in data:
        labels.append(dataPoint[0])
        features.append(dataPoint[1])

    return np.array(labels),np.array(features)
    
def LoadData():
    trainData = []
    testData = []
    for label in labels:
        trainDatap = trainDataPath + label
        trainFiles = glob.glob(trainDatap + "/*.jpg")

        for trainFile in trainFiles:
            picture = [label,cv2.cvtColor(ima.imread(trainFile),cv2.COLOR_BGR2GRAY)]
            trainData.append(picture)

        testDatap = testDataPath + label
        testFiles = glob.glob(testDatap + "/*.jpg")

        for testFile in testFiles:
            picture = [label,ima.imread(testFile)]
            testData.append(picture)
    return testData,trainData

def main(argv):
    testData,trainData = LoadData()

    #testLabels,testFeatures = randomizeData(testData)    
    #trainLabels,trainFeatures = randomizeData(trainData)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    trainFeatures = mnist.train.images  # Returns np.array
    trainLabels = np.asarray(mnist.train.labels, dtype=np.int32)
    testFeatures = mnist.test.images  # Returns np.array
    testLabels = np.asarray(mnist.test.labels, dtype=np.int32)    

    classifier = tf.estimator.Estimator(model_fn=cnn_model,model_dir="./cnn_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    #vv.imwrite("test.png",trainFeatures[0])

    print(trainFeatures[0])
    print(len(trainFeatures[0]))
    print(len(trainFeatures))
    train_input = tf.estimator.inputs.numpy_input_fn(x={"image":trainFeatures},y=trainLabels,num_epochs=None,shuffle=True)

    classifier.train(
        input_fn=train_input,
        steps=200000,#2* 10**4,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image":testFeatures}, y=testLabels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()    
