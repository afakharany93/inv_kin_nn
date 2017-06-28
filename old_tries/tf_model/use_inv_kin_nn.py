import numpy as np
import math
from math import pi
from numpy import cos, sin, tan
import pickle
import tensorflow as tf


#data retrieval
with open("../data/robot_set.pickle","rb") as f:
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

ip_len = position_data.shape[1]
'''
file_tf_model = 'nn_model.py'
exec(open(file_tf_model).read())	
'''
hidden_1_layer_w 	= tf.get_collection('hidden_1_layer_w')
hidden_1_layer_b 	= tf.get_collection('hidden_1_layer_b')
hidden_2_layer_w 	= tf.get_collection('hidden_2_layer_w')
hidden_2_layer_b 	= tf.get_collection('hidden_2_layer_b')
hidden_3_layer_w 	= tf.get_collection('hidden_3_layer_w')
hidden_3_layer_b 	= tf.get_collection('hidden_3_layer_b')
hidden_4_layer_w 	= tf.get_collection('hidden_4_layer_w')
hidden_4_layer_b 	= tf.get_collection('hidden_4_layer_b')
output_layer_w 	= tf.get_collection('output_layer_w')
output_layer_b 		= tf.get_collection('output_layer_b')

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000



#inpts and outputs

x = tf.placeholder('float',shape=[None, ip_len ])
y = tf.placeholder('float')

l1 = tf.add(tf.matmul(x,hidden_1_layer_w), hidden_1_layer_b)	
l1 = tf.nn.tanh(l1)

l2 = tf.add(tf.matmul(l1,hidden_2_layer_w), hidden_2_layer_b)	
l2 = tf.nn.tanh(l2)									  		

l3 = tf.add(tf.matmul(l2,hidden_3_layer_w), hidden_3_layer_b)	
l3 = tf.nn.sigmoid(l3)		  

l4 = tf.add(tf.matmul(l3,hidden_4_layer_w), hidden_4_layer_b)	
l4 = tf.nn.sigmoid(l4)	

output = tf.add(tf.matmul(l3,output_layer_w), output_layer_b)


def use_neural_network(input_data):
	prediction = output
	#saver = tf.train.Saver()
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('my-model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))
		#sess.run(tf.global_variables_initializer())
		print(sess.run(hidden_1_layer_b))



		#result = prediction.eval(feed_dict={x:input_data})
		#return result        
        
ip =  [[ -7.63422434e-02,   3.26458617e-01,   5.23922397e-01,   2.70056870e+00],
 [ -1.00316412e-02,  -4.09070419e-03,   1.41643085e-01,   4.14906397e+00],
 [  1.01409350e-01 , -1.41532035e-01   ,1.71564963e-01  , 1.10718963e+01],
 [  1.65967975e-01  ,-1.89983552e-01  , 4.58927648e-01  , 6.23839935e+00],
 [  2.31635064e-01  ,-8.30784971e-02 ,  1.59024947e-01  , 6.20335704e+00],
 [ -1.58212457e-01  ,-1.74989303e-01,  -1.18111470e-02  , 8.91940960e+00]]


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