#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Nov 12 14:35:13 2018

@author: kxs
'''

from gen_code import newGenerateCaptchaTextandImage,generateCaptchaTextandImage
from gen_code import number
from test_code import getTestData
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

text, image = newGenerateCaptchaTextandImage()
print("Size of the img:", image.shape)  # (60, 160, 3)  
# 图像大小  
imageHeight = image.shape[0]
imageWidth = image.shape[1]
MAX = len(text)
print("length of the code", MAX)  
axis_x_acc = []
axis_y_acc = []
axis_x_loss = []
axis_y_loss = []
'''
batch_x_train = []
batch_y_train = []
'''


def turnToGray(img):
    if len(img.shape) > 2: 
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]  
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        #gray = (int) (0.3 * r + 0.59 * g + 0.11 * b);
        return gray
    else:
        return img





charSet = number   
charSetLen = len(charSet)
#print("charsetlength: ", charSetLen)

def charToPos(c):
    k = ord(c)-ord('0')   #find the position of that char
    return k

# 文本转向量
def textToVector(text):
    vector = np.zeros(MAX * charSetLen)
    for i, c in enumerate(text):
        idx = i * charSetLen + charToPos(c)
        vector[idx] = 1
    return vector




'''
# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % charSetLen
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
'''

# generate a batch
def getBatch(batchSize):
    x = np.zeros([batchSize, imageHeight * imageWidth])
    y = np.zeros([batchSize, MAX * charSetLen])
    for i in range(batchSize):
        text, image = newGenerateCaptchaTextandImage()
        image = turnToGray(image)


        x[i, :] = image.flatten() / 255  
        y[i, :] = textToVector(text)

    return x, y

#getBatch(1)

####################################################################
'''
X = tf.placeholder(tf.float32, [None, imageHeight * imageWidth], name = 'datainput')
Y = tf.placeholder(tf.float32, [None, MAX * charSetLen], name = 'labelInput')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  # dropout
'''

# 定义CNN X, Y
#def crackCode(w_alpha = 0.01, b_alpha = 0.1):
def trainCNN():
    def bias(shape, name = 'bias'):
        init = tf.constant(0.0, shape = shape)
        var = tf.Variable(init, name = name)
        return var
    
    X = tf.placeholder(tf.float32, [None, imageHeight * imageWidth], name = 'datainput')
    Y = tf.placeholder(tf.float32, [None, MAX * charSetLen], name = 'labelInput')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  # dropout
    x = tf.reshape(X, shape=[-1, imageHeight, imageWidth, 1], name = 'xInput')


   
    conv1_W = tf.Variable(tf.truncated_normal([5, 5, 1, 8], stddev = np.sqrt(2/9)), name = 'conv1_W')
    conv1_B = bias([8], 'conv1_B')
    #conv1_B = tf.Variable(tf.constant(0.1, [32]), name = 'conv1_B')
    #conv1_B = tf.Variable(b_alpha * tf.random_normal([32]), name = 'conv1_B')
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME', name = 'conv1'), conv1_B))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'conv1_pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    conv2_W = tf.Variable(tf.truncated_normal([5, 5, 8, 16], stddev = np.sqrt(2/(9*8))), name = 'conv2_W')
    conv2_B = bias([16], 'conv2_B')
    #conv2_B = tf.Variable(tf.constant(0.1, [64]), name = 'conv2_B')
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME', name = 'conv2'), conv2_B))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'conv2_pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    '''
    conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = np.sqrt(2/(9 * 64))), name = 'conv3_W')
    conv3_B = bias([64], 'conv3_B')
    #conv3_B = tf.Variable(tf.constant(0.1, [64]), name = 'conv3_B')
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME', name = 'conv3'), conv3_B))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'conv3_pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    '''
    '''
    conv4_W = tf.Variable(tf.truncated_normal([3, 3, 16, 5], stddev = np.sqrt(2/(9 * 64))), name = 'conv4_W')
    conv4_B = bias([5], 'conv4_B')
    #conv4_B = tf.Variable(tf.random_normal([64]), name = 'conv4_B')
    conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME', name = 'conv4'), conv4_B))
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'conv4_pool')
    conv4 = tf.nn.dropout(conv4, keep_prob)
    
    conv5_W = tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev = np.sqrt(2/(9 * 64))), name = 'conv5_W')
    conv5_B = bias([3], 'conv5_B')
    #conv5_B = tf.Variable(tf.random_normal([32]), name = 'conv5_B')
    conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME', name = 'conv5'), conv5_B))
    #conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = 'conv3_pool')
    conv5 = tf.nn.dropout(conv5, keep_prob)
    '''
    # Fully connected layer
    '''
    w_d = tf.Variable(tf.truncated_normal([8*20*64, 1024]))
    b_d = tf.Variable(tf.truncated_normal([1024])
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    '''
    
    w_d1 = tf.Variable(tf.truncated_normal([15*40*16,1024], stddev = np.sqrt(2/(9*(15*40*16)))), name = 'w_d1')
    b_d1 = bias([1024], 'b_d1')
    #b_d = tf.Variable(tf.constant(0.1, [1024]), name = 'b_d')
    dense1 = tf.reshape(conv2, [-1, 15*40*16])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d1), b_d1))
    dense1 = tf.nn.dropout(dense1, keep_prob)
    
    w_d2 = tf.Variable(tf.truncated_normal([1024,200], stddev = np.sqrt(2/(9*1024))), name = 'w_d')
    b_d2 = bias([200], 'b_d')
    #b_d = tf.Variable(tf.constant(0.1, [1024]), name = 'b_d')
    #dense = tf.reshape(conv4, [-1, 4*10*5])
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d2), b_d2))
    dense2 = tf.nn.dropout(dense2, keep_prob)
    
    
    w_out = tf.Variable(tf.truncated_normal([200, MAX * charSetLen], stddev = np.sqrt(2/(9*200))), name = 'w_out')
    b_out = bias([MAX * charSetLen], 'b_out')
    #b_out = tf.Variable(tf.constant(0.1, [MAX * charSetLen]), name = 'b_out')
    out = tf.add(tf.matmul(dense2, w_out), b_out, name = 'out')
    #out = tf.nn.softmax(out)
    
    #variables_dict = {'conv1_W': conv1_W, 'conv1_B': conv1_B, 'conv2_W': conv2_W, 'conv2_B': conv2_B, 'conv3_W': conv3_W, 'conv3_B': conv3_B, 'w_d': w_d, 'b_d': b_d, 'w_out': w_out, 'b_out': b_out}
    #return out#, variables_dict



# 训练 batch_x_train, batch_y_train, X, Y
#def trainCNN():
    #out = crackCode()
    # loss  
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = Y))
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = out, labels = Y))
    loss = tf.reduce_mean(tf.losses.mean_squared_error(Y,out))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    predict = tf.argmax(tf.reshape(out, [-1, MAX, charSetLen]), axis = 2, name = 'predict')
    correct = tf.argmax(tf.reshape(Y, [-1, MAX, charSetLen]), axis = 2, name = 'correct')
    correctness = tf.equal(predict, correct)
    accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

    saver = tf.train.Saver()
    #tf.reset_default_graph()
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            '''
            batch_x, batch_y = getBatch(100)
            
            for i in range (100):
                batch_x_train.append(batch_x[i])
                batch_y_train.append(batch_y[i])
            '''
            step = 0
           # x,y = getBatch(100)
            while True:
                x, y = getBatch(100)
                _, loss_, acc, out_ = sess.run([optimizer, loss, accuracy, out], feed_dict={X: x, Y: y, keep_prob: 1.0})
                print("steps = %d, loss = %f" % (step, loss_))
                '''
                print(out_[0])
                print(y[0])
                '''
                #saver.save(sess, "./model3/crack_capcha", global_step=step)
              #  batch_x, batch_y = getBatch(100)
              #  acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
              #  print("steps = %d, accuracy = %f" % (step, acc))
                # 每100 step计算一次准确率
                if step % 50 == 0:
                    '''
                    batch_x, batch_y = getBatch(10)
                    acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
                    print("steps = %d, accuracy = %f" % (step, acc))
                    # 如果准确率大于50%,保存模型,完成训练
                    '''
                    
                    axis_x_acc.append(step)
                    axis_y_acc.append(acc)
                    axis_y_loss.append(loss_)
                    
                    print("steps = %d, loss = %f, acc = %f" % (step, loss_, acc))
                    '''
                    print(out_[0])
                    print(y[0])
                    '''
                    
                    if acc > 0.90:
                        saver.save(sess, "./model/crack_capcha.model", global_step=step)
                        break
                    
                    '''
                    if step == 20:
                        break
                    '''
                    '''
                    print("steps = %d, loss = %f, acc = %f" % (step, loss_, acc))
                    print(out_[0])
                    print(y[0])
                    '''
                step += 1
            #return sess

#trainCNN()
'''
plt.figure(figsize=(8,4))
plt.plot(axis_x_acc, axis_y_acc,"b-",linewidth=1)
plt.show()
plt.savefig("acc.jpg")

plt.figure(figsize=(8,4))
plt.plot(axis_x_acc, axis_y_loss, "b-", linewidth = 1)
plt.show()
plt.savefig("loss.jpg")
'''

def crack():
    #out, var_dict = crackCode()

    #saver = tf.train.Saver(var_dict)
    
    saver = tf.train.import_meta_graph("./model/crack_capcha.model-5900.meta")
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("datainput:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    predict = graph.get_tensor_by_name("predict:0")
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.import_meta_graph('./model/crack_capcha.model-20.meta')
        print("============================================")
        saver.restore(sess, tf.train.latest_checkpoint('./model'))
        #sess = trainCNN()
        print("============================================")
        #predict = tf.argmax(tf.reshape(out, [-1, MAX, charSetLen]), 2)
        count = 0
        for i in range(20):
            
            text, image = getTestData()
            
            image = turnToGray(image)
            captchaImage = image.flatten() / 255
            #print(text, captchaImage)
            #print(len(captchaImage))
            print("============================================")
            textList = sess.run(predict, feed_dict={X: [captchaImage], keep_prob: 1.0})
            print("============================================")
            predictText = textList[0].tolist()
            predictText = str(predictText)
            predictText = predictText.replace("[", "").replace("]", "").replace(",", "").replace(" ","")
            #print(predictText, text)
            if text == predictText:
                count += 1
                result = "  Correct"
                print("Correct: {}  Prediction: {}".format(text, predictText) + result)
            else:
                result = "，Incorrect!!!!"
                print("Correct: {}  Prediction: {}".format(text, predictText) + result)

        print("Correctness:" + str(count) + "/20")

crack()
        
def listSplit(low, high, batch_x, batch_y):
    batch_x_train = []
    batch_y_train = []
    batch_test_x = []
    batch_test_y = []
    size1 = len(batch_x)
    for i in range(low, high+1):
        batch_test_x.append(batch_x[i])
        batch_test_y.append(batch_y[i])
    for i in range(0, low):
        batch_x_train.append(batch_x[i])
        batch_y_train.append(batch_y[i])
    for i in range(high+1, size1):
        batch_x_train.append(batch_x[i])
        batch_y_train.append(batch_y[i])
    return batch_x_train, batch_y_train, batch_test_x, batch_test_y
        
        
    
'''        
if __name__ == '__main__':
    text, image = newGenerateCaptchaTextandImage()
    print("Size of the img:", image.shape)  # (60, 160, 3)  
# 图像大小  
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    MAX = len(text)
    print("length of the code", MAX)
    
    charSet = number   
    charSetLen = len(charSet)
    
    X = tf.placeholder(tf.float32, [None, imageHeight * imageWidth], name = 'datainput')
    Y = tf.placeholder(tf.float32, [None, MAX * charSetLen], name = 'labelInput')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  # dropout
    
    batch_x_train = []
    batch_y_train = []
    batch_test_x = []
    batch_test_y = []
    batch_x, batch_y = getBatch(100)
    batch_x_train, batch_y_train, batch_test_x, batch_test_y = listSplit(80, 99, batch_x, batch_y)
    
    #trainCNN(batch_x_train, batch_y_train, X, Y)
    crack(batch_test_x, batch_test_y)
'''    
        
