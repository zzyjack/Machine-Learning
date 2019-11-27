
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget
# create W array for size 784x1
W=np.zeros(784)
W=np.reshape(W,(784,1))
b=np.array([0])

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

#==============================================================================
# #set the data set shape to match the variables
# trainData = np.transpose(np.reshape(trainData,(3500,784)))
# trainTarget = np.transpose(trainTarget)
# validData = np.transpose(np.reshape(validData,(100,784)))
# validTarget = np.transpose(validTarget)
# testData = np.transpose(np.reshape(testData,(145,784)))
# testTarget = np.transpose(testTarget)
# #W size (784,1)
# def MSE(W, b, x, y, reg):
#     weight_decay = (reg/2)*np.sum(np.transpose(W)**2)
#     error = (np.dot(np.transpose(W),x)+b-y)**2
#     return (1/(2*len(x[1])))*np.sum(error)+weight_decay
# 
# def gradMSE(W, b, x, y, reg):
#     weight_decay = reg*W
#     error = (np.dot(np.transpose(W),x)+b-y)
#     gradient_W = np.dot(x,np.transpose(error))/len(x[1])+weight_decay
#     gradient_b = np.sum(error)/len(x[1])+(reg/2)*np.sum(np.transpose(W)**2)
#     return gradient_W, gradient_b
#     
# def normalMSE(W, b, x, y, reg):
#     error = (np.dot(np.transpose(W),x)+b-y)**2
#     return (1/(2*len(x[1])))*np.sum(error)
# 
# def normalgradMSE(W, b, x, y, reg):
#     error = (np.dot(np.transpose(W),x)+b-y)
#     gradient_W = np.dot(x,np.transpose(error))/len(x[1])
#     gradient_b = np.sum(error)/len(x[1])+(reg/2)*np.sum(np.transpose(W)**2)
#     return gradient_W, gradient_b
# #gw,gb = gradMSE(W,b,trainData,trainTarget,0)
# def sigmoid(x):
#     return (1 / (1 + np.exp(-x)))
#     
# def crossEntropyLoss(W, b, x, y, reg):
#     weight_decay = (reg/2)*np.sum(np.transpose(W)**2)
#     y_hat = sigmoid(np.dot(np.transpose(W),x)+b)
#     error = np.sum((-1)*y*np.log(y_hat)-(1-y)*np.log(1-y_hat))/len(x[1])
#     return error+weight_decay
#     
# def gradCE(W, b, x, y, reg):
#     weight_decay = reg*W
#     gradCE_W = (np.dot(x,np.transpose((-1*y)*(1/(1+np.exp(np.dot(np.transpose(W),trainData)+b)))))+np.dot(x,np.transpose((1-y)*(1/(1+np.exp(-1*np.dot(np.transpose(W),trainData)-b))))))/len(x[1])+weight_decay
#     gradCE_b = np.sum((-1*y/(1+np.exp(np.dot(np.transpose(W),x)+b)))+(1-y)/(1+np.exp(-1*(np.dot(np.transpose(W),x)+b))))/len(x[1])
#     return gradCE_W, gradCE_b
# 
# def grad_descent(W, b, x, y, alpha, iterations, reg, EPS,losstype="None"):
#     new_W = W
#     new_b =b
#     loss_train = np.ones(5000) 
#     loss_valid = np.ones(5000)
#     loss_test = np.ones(5000)
#     epoch = np.arange(5000)
#     if losstype == "gradMSE":
#         grad_loss = gradMSE
#         loss_func = MSE
#     else:
#         grad_loss = gradCE
#         loss_func = crossEntropyLoss
#         
#     for i in range(iterations):
#         gradient_W,gradient_b = grad_loss(new_W,new_b,x,y,reg)
#         new_W = W - (alpha*gradient_W)
#         new_b = b -(alpha*gradient_b)
#         loss_train[i] = (loss_func(new_W,new_b,x,y,reg))
#         loss_valid[i] = (loss_func(new_W,new_b,validData,validTarget,reg))
#         loss_test[i] = (loss_func(new_W,new_b,testData,testTarget,reg))
#         if np.sum(abs(new_W-W))<EPS:
#             print("converged")
#             break
#         W = new_W
#         b = new_b
#     return new_W,new_b,epoch,loss_train,loss_valid,loss_test
#     
# def buildGraph(loss=None):
#     tf.set_random_seed(421)
#     W=tf.Variable(tf.transpose(tf.truncated_normal((784,1),stddev=0.5)),dtype=tf.float32)
#     b=tf.Variable(1.,dtype=tf.float32)
#     x=tf.placeholder(tf.float32,shape=(784,None))
#     y=tf.placeholder(tf.float32,shape=(1,None))
#     lam=tf.placeholder(tf.float32)
#     alpha= tf.placeholder(tf.float32)
#     y_pred=tf.matmul(W,x) + b
#     
#     if loss == "MSE":
# 
#         error = tf.losses.mean_squared_error(y,y_pred)
#         weight_decay = tf.multiply(lam / 2, tf.reduce_sum(tf.square(W)))
#         total_loss = error + weight_decay
#         
#     elif loss == "CE":
#         error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred))
#         weight_decay = tf.multiply(lam / 2, tf.reduce_sum(tf.square(W)))
#         total_loss = error + weight_decay   
#     accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(y_pred, 0.5)), y)))
#     optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(total_loss)
#     optimizer_adam = tf.train.AdamOptimizer(alpha).minimize(total_loss)
#     
#     return x,y,W,b,y_pred,y,total_loss,optimizer,lam,alpha,accuracy,optimizer_adam
# 
# # the function that divide the data set to batches 
# def make_chunks(data, chunk_size):
#     while data.any():
#         chunk, data = data[:,:chunk_size], data[:,chunk_size:]
#         yield chunk
# 
# # the function that reshuffles the dataset 
# def reshuffle(data,label):
#     np.random.seed(1)
#     data = np.transpose(data)
#     label = np.transpose(label)
#     np.random.shuffle(label)
#     np.random.shuffle(data)
#     data = np.transpose(data)
#     label = np.transpose(label)
#     return data, label
# 
# #initilize the tensors    
# x,y,W,b,y_pred,y,total_loss,optimizer,lam,alpha,accuracy,optimizer_adam = buildGraph(loss="MSE")
# 
# def SGD(optimizer,batch_size,epoch):
#     global trainData
#     global trainTarget
#     global validData
#     global validTarget
#     global testData
#     global testTarget
#     init = tf.global_variables_initializer()
#   #build the loss array for train, validation, and testing data set
#     iterations = np.arange(3500/batch_size*epoch)
#     train_loss = []
#   
#     test_loss = []
#   
#     valid_loss = []
# 
# #build the dictionary for tensors
#     train_dict = {
#           x: trainData,
#           y: trainTarget,
#           lam: 0
#               }
#     valid_dict = {
#           x: validData,
#           y: validTarget,
#           lam: 0
#               }
#     test_dict = {
#           x: testData,
#           y: testTarget,
#           lam: 0
#               }
#     with tf.Session() as sess:
#   
#         # Run the initializer
#         sess.run(tf.local_variables_initializer())
#         sess.run(init)
#         for i in range(epoch):
#             trainData,trainTarget = reshuffle(trainData,trainTarget)
#             for(batch_x,batch_y) in zip(make_chunks(trainData,batch_size),make_chunks(trainTarget,batch_size)):
#                 sess.run(optimizer_adam,feed_dict={x:batch_x,y:batch_y,lam:0,alpha:0.001})
#                       #cost = sess.run(total_loss,feed_dict={x:trainData,y:trainTarget,lam:0})
#                 train_loss.append(sess.run(total_loss, feed_dict=train_dict))
#                 valid_loss.append(sess.run(total_loss, feed_dict=valid_dict))
#                 test_loss.append(sess.run(total_loss,feed_dict=test_dict))
#                       
#                       
#     return train_loss,valid_loss,test_loss,iterations
# 
#     
#     
# #create the epoch array for plot   
# train_loss,valid_loss,test_loss,iterations = SGD(optimizer,1750,700)
# plt.plot(iterations, train_loss,label='train loss')
# plt.plot(iterations, valid_loss, label='valid loss')
# plt.plot(iterations, test_loss, label='test loss')
# plt.title('MSE under SGD method')
# plt.xlabel('epoch')
# plt.ylabel('losses')
# plt.legend()
# plt.grid()
# #    
# 
# 
# #plot_alpha(0.005,loss_train)
# #plt.figure(figsize=(20,10))
#     
#     
#==============================================================================

    


