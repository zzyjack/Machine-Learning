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
    

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainData = np.transpose(np.reshape(trainData,(3500,784)))
trainTarget = np.transpose(trainTarget)
validData = np.transpose(np.reshape(validData,(100,784)))
validTarget = np.transpose(validTarget)
testData = np.transpose(np.reshape(testData,(145,784)))
testTarget = np.transpose(testTarget)

def buildGraph(loss=None):
    tf.set_random_seed(421)
    W=tf.Variable(tf.transpose(tf.truncated_normal((784,1),stddev=0.5)),dtype=tf.float32)
    b=tf.Variable(1.,dtype=tf.float32)
    x=tf.placeholder(tf.float32,shape=(784,None))
    y=tf.placeholder(tf.float32,shape=(1,None))
    lam=tf.placeholder(tf.float32)
    alpha= tf.placeholder(tf.float32)
    y_pred=tf.matmul(W,x) + b
    
    if loss == "MSE":

        error = tf.losses.mean_squared_error(y,y_pred)
        weight_decay = tf.multiply(lam / 2, tf.reduce_sum(tf.square(W)))
        total_loss = error + weight_decay
        
    elif loss == "CE":
        error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred))
        weight_decay = tf.multiply(lam / 2, tf.reduce_sum(tf.square(W)))
        total_loss = error + weight_decay   
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(tf.greater(y_pred, 0.5)), y)))
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(total_loss)
    optimizer_adam = tf.train.AdamOptimizer(alpha).minimize(total_loss)
    
    return x,y,W,b,y_pred,y,total_loss,optimizer,lam,alpha,accuracy,optimizer_adam

# the function that divide the data set to batches 
def make_chunks(data, chunk_size):
    while data.any():
        chunk, data = data[:,:chunk_size], data[:,chunk_size:]
        yield chunk
# the function that reshuffles the dataset 
def reshuffle(data,label):
    np.random.seed(1)
    data = np.transpose(data)
    label = np.transpose(label)
    np.random.shuffle(label)
    np.random.shuffle(data)
    data = np.transpose(data)
    label = np.transpose(label)
    return data, label

    
x,y,W,b,y_pred,y,total_loss,optimizer,lam,alpha,accuracy,optimizer_adam = buildGraph(loss="CE")
#  
def SGD(optimizer,batch_size,epoch):
    global trainData
    global trainTarget
    global validData
    global validTarget
    global testData
    global testTarget
    init = tf.global_variables_initializer()
  #build the loss array for train, validation, and testing data set
    iterations = np.arange(epoch)
    train_loss = []
  
    test_loss = []
  
    valid_loss = []

#build the dictionary for tensors
    train_dict = {
          x: trainData,
          y: trainTarget,
          lam: 0
              }
    valid_dict = {
          x: validData,
          y: validTarget,
          lam: 0
              }
    test_dict = {
          x: testData,
          y: testTarget,
          lam: 0
              }
    with tf.Session() as sess:
  
        # Run the initializer
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        for i in range(epoch):
#            trainData,trainTarget = reshuffle(trainData,trainTarget)
            for(batch_x,batch_y) in zip(make_chunks(trainData,batch_size),make_chunks(trainTarget,batch_size)):
                sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,lam:0,alpha:0.001})
                      #cost = sess.run(total_loss,feed_dict={x:trainData,y:trainTarget,lam:0})
            train_loss.append(sess.run(total_loss, feed_dict=train_dict))
            valid_loss.append(sess.run(total_loss, feed_dict=valid_dict))
            test_loss.append(sess.run(total_loss,feed_dict=test_dict))
                      
                      
    return train_loss,valid_loss,test_loss,iterations

    
#train_loss,valid_loss,test_loss,iterations=SGD(optimizer_adam,1750,700)
#create the epoch array for plot   
train_loss1,valid_loss1,test_loss1,iterations = SGD(tf.train.AdamOptimizer(alpha,epsilon=1e-9).minimize(total_loss),500,700)
train_loss2,valid_loss2,test_loss2,iterations = SGD(tf.train.AdamOptimizer(alpha,epsilon=1e-4).minimize(total_loss),500,700)
plt.plot(iterations, train_loss1,label='epsilon=1e-9')
plt.plot(iterations, train_loss2, label='epsilon=1e-4')
#plt.plot(iterations, test_loss, label='test loss')
#plt.plot(iterations, train_loss,label='train loss')
#plt.plot(iterations, valid_loss, label='valid loss')
#plt.plot(iterations, test_loss,label='test loss')
plt.title('fig 3.4.3 MSE under Adam with epsilon 1e-9vs1e-4')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend()
plt.grid()

