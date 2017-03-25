import numpy as np
import math
from math import pi
from numpy import cos, sin, tan
import pickle
import tensorflow as tf


#data retrieval
with open("robot_set.pickle","rb") as f:
	all_data = pickle.load(f)

data_n, position_variables_n, dof_n = all_data[0], all_data[1], all_data[2]


position_data = np.array(all_data[3:3+position_variables_n])	#last index not taken 
joints_data = np.array(all_data[3+position_variables_n:])	#the end in this case is included

del all_data

position_data = position_data.T
joints_data = joints_data.T

test_size = int(0.35*data_n)

test_position = position_data[data_n-test_size:]
test_joints = joints_data[data_n-test_size:]

position_data = np.delete(position_data, list(range(data_n-test_size , position_data.shape[0])), axis=0)
joints_data = np.delete(joints_data, list(range(data_n-test_size , joints_data.shape[0])), axis=0)

#parameters
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000

batch_size = 100

n_epochs = 1100	#number of feed forward and back prop

alpha = 0.0002	#learning rate, default 0.001

#inpts and outputs
ip_len = position_data.shape[1]
x = tf.placeholder('float',shape=[None, ip_len ])
y = tf.placeholder('float')

hidden_1_layer = {'weights':tf.Variable(tf.random_normal([ip_len ,n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))
					  }	

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
				  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))
				  }

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
				  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))
				  }

hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),
				  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))
				  }

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,dof_n])),
				  'biases':tf.Variable(tf.random_normal([dof_n]))
				 }

def neural_network_model(data):
	

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])	
	l1 = tf.nn.tanh(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])	
	l2 = tf.nn.tanh(l2)									  		

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])	
	l3 = tf.nn.sigmoid(l3)		  

	l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])	
	l4 = tf.nn.sigmoid(l4)	

	output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])

	return output	

saver = tf.train.import_meta_graph('my-model.meta')

def use_neural_network(input_data):
	prediction = neural_network_model(x)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, tf.train.latest_checkpoint('./'))
		
		result = prediction.eval(feed_dict={x:input_data})
		return result        
        
ip =  [[ -7.63422434e-02,   3.26458617e-01,   5.23922397e-01 ,  2.70056870e+00]]


print(use_neural_network(ip))

'''
postion [[ -7.63422434e-02   3.26458617e-01   5.23922397e-01   2.70056870e+00]
 [ -1.00316412e-02  -4.09070419e-03   1.41643085e-01   4.14906397e+00]
 [  1.01409350e-01  -1.41532035e-01   1.71564963e-01   1.10718963e+01]
 ..., 
 [  1.65967975e-01  -1.89983552e-01   4.58927648e-01   6.23839935e+00]
 [  2.31635064e-01  -8.30784971e-02   1.59024947e-01   6.20335704e+00]
 [ -1.58212457e-01  -1.74989303e-01  -1.18111470e-02   8.91940960e+00]]
joints [[  1.80051789e+00   1.17917363e-01   9.00050811e-01   1.99676935e-01]
 [  3.87195360e-01   1.40857554e-01   3.76186861e+00   3.64173770e-02]
 [  5.33411084e+00   4.48510477e-02   5.73778550e+00   1.19717572e-03]
 ..., 
 [  5.43042057e+00   5.25689132e-02   8.07978779e-01   1.39016379e-01]
 [  2.79722283e+00   8.97541133e-02   3.40613422e+00   1.97941119e-01]
 [  3.97729874e+00   1.74683150e-01   4.94211085e+00   1.18874545e-01]]
'''