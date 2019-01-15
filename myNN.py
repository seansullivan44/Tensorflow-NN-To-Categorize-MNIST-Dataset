import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import math




#************************LOAD AND DEAL WITH MNIST DATASET****************************************************
#Load MNIST dataset:
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


#Data statistics:
img_size = 28
img_size_flat = 28 * 28
img_shape = (28, 28) #Length x Width
img_shape_full = (28, 28, 1) #Length x Width x Number of Channels
num_classes = 10 #10 digits (0 through 9)
num_channels = 1 #One channel because dataset is composed of black and white images

x_train = x_train.reshape(-1, img_size_flat)
x_test = x_test.reshape(-1, img_size_flat)

#Normalize data:
x_train = x_train / 255.0
y_train = y_train / 255.0
x_test = x_test / 255.0
y_test = y_test / 255.0

#Store the images corresponding class number as an int:
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)

#Store all class numbers as "one-hot arrays":
y_train = np.eye(10, dtype=float)[y_train]
y_test = np.eye(10, dtype=float)[y_test]





#************************UTILITY FUNCTIONS USED TO HELP DEFINE TENSORFLOW GRAPH**********************************
#HyperParameters:

#Conv Layer #1
filter_size1 = 5
num_filters1 = 16

#Conv Layer #2
filter_size2 = 5
num_filters2 = 36

#FC Layer
fc_size = 128

#Function used to create new weight Tensorflow variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

#Function used to create new bias Tensorflow variables
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#Function used to create a new convolutional layer on the Tensorflow graph
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    
    shape = [filter_size, filter_size, num_input_channels,num_filters]

    weights = new_weights(shape=shape)

    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,filter=weights, strides=[1,1,1,1],padding="SAME")

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")

    layer = tf.nn.relu(layer)

    return layer, weights


#Function used to create a Fully-connected layer on the Tensorflow graph
def new_fc_layer(input, num_inputs,num_outputs,use_relu=True):

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



#Function used to create a flattening layer on the Tensorflow graph
def flatten_layer(layer):
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


#************************DEFINING THE TENSORFLOW GRAPH***********************************************************

#Placeholder variables:

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="X") #Input images

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels]) #Conv layers require a four-dimensional tensor

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="Y") #Labels for the input images

y_true_cls = tf.argmax(y_true, axis=1) #Class number 


#Creation of layers: 

#First Convolutional Layer
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, 
    filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)

#Second Convolutional Layer
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, 
    filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

#Flattening Layer
layer_flat, num_features = flatten_layer(layer_conv2)

#First Fully-Connected Layer
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features,
    num_outputs=fc_size, use_relu=True)

#Second Fully-Connected Layer
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size,
    num_outputs=num_classes, use_relu=True)


y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

#Cost-Function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
    labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#Optimizer (Adam)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#Determine if the prediction is correct:
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




#************************CREATE TENSORFLOW SESSION***********************************************************
session = tf.Session()
session.run(tf.global_variables_initializer())

train_sample_size = 64 #Size of training batch
test_sample_size = 128



total_iterations = 0


#Fuinction optimizes NN:
def optimize(num_iterations):
    global total_iterations

    starting_time = time.time() #Obatin the start time of the optimization 

    for i in range(total_iterations, total_iterations + num_iterations):

        #Get Random group of sample data
        index = np.random.randint(low=0, high=6000, size=train_sample_size) #Random index value for pulling random group of data from training set
        x_train_sample = x_train[index]
        y_train_sample = y_train[index]

        feed_dict_train = {x: x_train_sample, y_true: y_train_sample}

        session.run(optimizer, feed_dict = feed_dict_train)

        if i % 100 == 0:
            current_accuracy = session.run(accuracy, feed_dict=feed_dict_train)
            optimization_message = "Current Iteration: {0:>6}, Current Training Accuracy: {1:>6.1%}"
            print(optimization_message.format(i,current_accuracy))

        total_iterations += num_iterations

        
    end_time = time.time() #Obtain the end time of the operation 
    total_time = end_time - starting_time #Total time taken to optimize

    print("The total optimization time was: " + str(timedelta(seconds=int(round(total_time)))))




#Train the Neural Network:
optimize(num_iterations=1000) 

#session.close()



