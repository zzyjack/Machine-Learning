# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:21:30 2019

@author: ziyan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget
    
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()



# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx,:], target[randIndx,:]
    return data, target

def conv_layer(X,W,b,stride=1):
    X = tf.nn.conv2d(X,W,strides=[1,stride,stride,1],padding = 'SAME')
    X = tf.nn.bias_add(X,b)
    return tf.nn.relu(X)
    
def max_pooling(X,K=2):
    return tf.nn.max_pool(X,ksize = [1,K,K,1], strides = [1,K,K,1], padding = 'SAME')
    
def batch_norm_flatten(X):
    mean,variance = tf.nn.moments(X,axes=[0,1,2])
    X = tf.nn.batch_normalization(X,mean,variance,offset=None,scale=None,variance_epsilon=1e-3)
    size = X.get_shape().as_list()
    X = tf.reshape(X,[-1,size[1]*size[2]*size[3]])
    return X
    
def fully_c(X,W,b):
    return tf.add(tf.matmul(X,W),b)

def con_net(X,Y,W,b):
    conv = conv_layer(X,W['wc1'],b['bc1'],stride=1)
    pool = max_pooling(conv,K=2)
    flatpool = batch_norm_flatten(pool)
    fc1=tf.nn.relu(fully_c(flatpool,W['wfc1'],b['bfc1']))
    fc2=fully_c(fc1,W['wfc2'],b['bfc2'])
    return fc2

#==============================================================================
# def con_net(X,Y,W,b,keep_prob):
#     conv = conv_layer(X,W['wc1'],b['bc1'],stride=1)
#     pool = max_pooling(conv,K=2)
#     flatpool = batch_norm_flatten(pool)
#     fc1=tf.nn.relu(tf.nn.dropout(fully_c(flatpool,W['wfc1'],b['bfc1']),keep_prob))
#     fc2=fully_c(fc1,W['wfc2'],b['bfc2'])
#     return fc2
#     
#==============================================================================
def reshape_data(X):
    return X.reshape(-1,28,28,1)

trainData = trainData.reshape(-1,28,28,1)
validData = validData.reshape(-1,28,28,1)
testData = testData.reshape(-1,28,28,1)
    
epoch = 50
batch_size = 32
learning_rate = 1e-4
reg=0.5
p= 0.25

tf.reset_default_graph()
weights = {
           'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
           'wfc1': tf.get_variable('W1', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer()),
           'wfc2': tf.get_variable('W2', shape=(784,10), initializer=tf.contrib.layers.xavier_initializer())
}
    
biases = {
          'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
          'bfc1': tf.get_variable('B1', shape=(784), initializer=tf.contrib.layers.xavier_initializer()),
          'bfc2': tf.get_variable('B2', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}


    
X = tf.placeholder("float",[None, 28,28,1])
Y = tf.placeholder("float",[None, 10])
keep_prob = tf.placeholder("float")
pred = con_net(X,Y,weights,biases,keep_prob)
last_layer = tf.nn.softmax(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=Y))
#==============================================================================
# cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=Y))\
#         +reg*tf.nn.l2_loss(weights['wc1'])+reg*tf.nn.l2_loss(weights['wfc1'])\
#         +reg*tf.nn.l2_loss(weights['wfc2']))
#==============================================================================

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(last_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    valid_loss = []
    test_loss = []
    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    iterations = []
    curr_itr = 0
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(epoch):
        trainData, newtrain = shuffle(trainData,newtrain)
        for batch in range(len(trainData)//batch_size):

            batch_x = trainData[batch*batch_size:(batch+1)*batch_size]
            batch_y = newtrain[batch*batch_size:(batch+1)*batch_size]
            opt = sess.run(optimizer, feed_dict={X: batch_x,
                                                              Y: batch_y, keep_prob:p})

        loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                              Y: batch_y, keep_prob:p})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy 
        test_acc,test_cost = sess.run([accuracy,cost], feed_dict={X: testData,Y : newtest, keep_prob:p})
        valid_acc,valid_cost = sess.run([accuracy,cost], feed_dict={X: validData,Y : newvalid, keep_prob:p})
        train_loss.append(loss)
        test_loss.append(test_cost)
        valid_loss.append(valid_cost)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        valid_accuracy.append(valid_acc)
        curr_itr+=1
        iterations.append(curr_itr)
        print("Validation Accuracy:","{:.5f}".format(valid_acc))
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Losses for CNN with 50 epochs 32 batch size p = "+str(1-p))
plt.plot(iterations, train_loss, 'r-', label=r"Training Loss")
plt.plot(iterations, valid_loss, 'b-', label=r"Valid Loss")
plt.plot(iterations, test_loss, 'y-', label=r"Test Loss")
plt.legend(loc = 'best')
plt.savefig(str(1-p)+"tf_units_loss.jpg")   
plt.clf()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy for CNN with 50 epochs 32 batch size p = "+str(1-p))
plt.plot(iterations, train_accuracy, 'r-', label=r"Training Accuracy")
plt.plot(iterations, valid_accuracy, 'b-', label=r"Valid Accuracy")
plt.plot(iterations, test_accuracy, 'y-', label=r"Test Accuracy")
plt.legend(loc = 'best')
plt.savefig(str(1-p)+"tf_units_accuracy.jpg")
plt.clf()

#==============================================================================
# plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
# plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
# plt.title('Training and Test Accuracy')
# plt.xlabel('Epochs ',fontsize=16)
# plt.ylabel('Loss',fontsize=16)
# plt.legend()
# plt.figure()
# plt.show()
#==============================================================================

