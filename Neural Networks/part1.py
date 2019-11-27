import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.io import loadmat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#==============================================================================
# # Load the data
# def loadData():
#     with np.load("notMNIST.npz") as data:
#         Data, Target = data["images"], data["labels"]
#         np.random.seed(521)
#         randIndx = np.arange(len(Data))
#         np.random.shuffle(randIndx)
#         Data = Data[randIndx] / 255.0
#         Target = Target[randIndx]
#         trainData, trainTarget = Data[:10000], Target[:10000]
#         validData, validTarget = Data[10000:16000], Target[10000:16000]
#         testData, testTarget = Data[16000:], Target[16000:]
#     return trainData, validData, testData, trainTarget, validTarget, testTarget
#     
# trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
# #Turning to the desired data format
# trainData = np.transpose(np.reshape(trainData,(10000,28*28)))
# trainTarget = np.transpose(trainTarget)
# validData = np.transpose(np.reshape(validData,(6000,28*28)))
# validTarget = np.transpose(validTarget)
# testData = np.transpose(np.reshape(testData,(2724,28*28)))
# testTarget = np.transpose(testTarget)
# 
#==============================================================================
def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

X_train, y_train = load_data('train_32x32.mat')
X_test, y_test = load_data('test_32x32.mat')
X_train = X_train[:,:,:,0:7000]
X_test = X_test[:,:,:,0:200]
y_test = np.reshape(y_test[0:200],[200,])
y_train = np.reshape(y_train[0:7000],[7000,])
y_test = np.where(y_test == 10, 0, y_test)
y_train = np.where(y_train == 10, 0, y_train)


print("Training Set", X_train.shape, y_train.shape)
print("Test Set", X_test.shape, y_test.shape)

X_train = np.transpose(np.reshape(X_train,(32*32*3,len(y_train))))
X_test = np.transpose(np.reshape(X_test,(32*32*3,len(y_test))))

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
#==============================================================================
#     newvalid = np.zeros((validTarget.shape[0], 10))
#==============================================================================
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
#==============================================================================
#     for item in range(0, validTarget.shape[0]):
#         newvalid[item][validTarget[item]] = 1
#==============================================================================
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newtest
newtrain,newtest = convertOneHot(trainTarget, testTarget)

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData[0,:]))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[:,randIndx], target[randIndx]
    return data, target

def relu(x):
    return x*(x>0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def computeLayer(X, W, b):
    S = np.dot(np.tranpose(W),X) + b
    return S
    
def CE(target, prediction):
    return -np.sum(np.sum(np.transpose(target)*np.log(prediction),axis=0))/len(target)

def gradCE(target, prediction):
    gradientCE = np.transpose(prediction - np.transpose(target))
    return gradientCE
    
def W_init(shape_W,shape_b):
    W = np.random.normal(0,1,shape_W)*np.sqrt(2/shape_W[0])
    b = np.random.normal(0,1,shape_b)*np.sqrt(2/shape_b[0])
    return W,b
    
def forward_pass(X,Y,W1,W2,b1,b2):
    S1 = np.dot(np.transpose(W1),X)+b1
    a1 = relu(S1)
    S2 = np.dot(np.transpose(W2),a1)+b2
    a2 = softmax(S2)
    error = CE(Y,a2)
    return S1,S2,a1,a2,error
    
def grad_weight2(X,Y,S):
    delta = gradCE(Y,S)
    return np.dot(X,delta)/len(Y)
    
def grad_bias2(Y,S):
    b = np.sum((S-np.transpose(Y)),axis=1)
    return np.transpose(np.reshape(b,(len(b),1)))/len(Y)

def d_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
    
def grad_weight1(X,W,delta,dtheta):
    delta0 = np.transpose(np.dot(W,np.transpose(delta))*dtheta)
    return np.dot(X,delta0)/len(X[1,:])

def grad_bias1(W,delta,dtheta):
    b = np.sum(np.transpose(np.dot(W,np.transpose(delta))*dtheta),axis=0)
    return np.transpose(np.reshape(b,(len(W),1)))/len(delta)

def v_update(v_old,gamma,alpha,gradient):
    v_new = gamma*v_old+alpha*gradient
    return v_new

    
def accuracy(target,prediction):
    x=np.argmax(prediction,axis=0)
    y=np.argmax(np.transpose(target),axis=0)
    return np.sum(x==y)/len(target)
    
def grad_descent(Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest,alpha,epoch,units):
    current_itr = 0
    gamma = 0.9
    W1,b1=W_init((3072,units),(units,1))
    W2,b2=W_init((units,10),(10,1))  
    v_W1 = np.full(np.shape(W1),1e-5)
    v_W2 = np.full(np.shape(W2),1e-5)
    v_b1 = np.full(np.shape(b1),1e-5)
    v_b2 = np.full(np.shape(b2),1e-5)
    training_loss = []
    valid_loss = []
    test_loss = []
    training_accu = []
    valid_accu = []
    test_accu =[]
    iterations = []
    while current_itr<epoch:
        S1,S2,a1,a2,error = forward_pass(Xtrain,Ytrain,W1,W2,b1,b2)
        _,_,_,valida2,valid_error = forward_pass(Xvalid,Yvalid,W1,W2,b1,b2)
        _,_,_,testa2,test_error = forward_pass(Xtest,Ytest,W1,W2,b1,b2)
        train_acc = accuracy(Ytrain,a2)
        valid_acc = accuracy(Yvalid,valida2)
        test_acc = accuracy(Ytest,testa2)
        grad_W2 = grad_weight2(a1,Ytrain,a2)
        grad_b2 = np.transpose(grad_bias2(Ytrain,a2))
        grad_W1 = grad_weight1(Xtrain,W2,gradCE(Ytrain,a2),d_relu(S1))
        grad_b1 = np.transpose(grad_bias1(W2,gradCE(Ytrain,a2),d_relu(S1)))
        v_W1 = v_update(v_W1,gamma,alpha,grad_W1)
        v_W2 = v_update(v_W2,gamma,alpha,grad_W2)
        v_b1 = v_update(v_b1,gamma,alpha,grad_b1)
        v_b2 = v_update(v_b2,gamma,alpha,grad_b2)
        W1-=v_W1
        W2-=v_W2
        b1-=v_b1
        b2-=v_b2
        iterations.append(current_itr)
        training_loss.append(error)
        valid_loss.append(valid_error)
        test_loss.append(test_error)
        training_accu.append(train_acc)
        valid_accu.append(valid_acc)
        test_accu.append(test_acc)
        current_itr+=1
        if current_itr > 50 and valid_accu[-1]-valid_accu[-2]<0:
            break
    print("Training loss:", training_loss[-1])
    print("Validation loss:", valid_loss[-1])
    print("Test loss:", test_loss[-1])
    print("Training accuracy:", training_accu[-1])
    print("Validation accuracy:", valid_accu[-1])
    print("Test accuracy:", test_accu[-1])
    return training_loss,valid_loss,test_loss,training_accu,valid_accu,test_accu,iterations,W1,W2,b1,b2
        
training_loss,valid_loss,test_loss,training_accu,valid_accu,test_accu,iterations,W1,W2,b1,b2 = grad_descent(trainData,newtrain,validData,newvalid,testData,newtest,0.05,200,1000)
    
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Cross Entropy Losses for early stopping")
plt.plot(iterations, training_loss, 'r-', label=r"Training Loss")
plt.plot(iterations, valid_loss, 'b-', label=r"Valid Loss")
plt.plot(iterations, test_loss, 'y-', label=r"Test Loss")
plt.legend(loc = 'best')
plt.savefig("early_units_loss.jpg")   
plt.clf()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy for early stopping")
plt.plot(iterations, training_accu, 'r-', label=r"Training Accuracy")
plt.plot(iterations, valid_accu, 'b-', label=r"Valid Accuracy")
plt.plot(iterations, test_accu, 'y-', label=r"Test Accuracy")
plt.legend(loc = 'best')
plt.savefig("early_units_accuracy.jpg")
plt.clf()

#==============================================================================
# def CNNs():
#     tf.set_random_seed(421)
#     
#==============================================================================
    
