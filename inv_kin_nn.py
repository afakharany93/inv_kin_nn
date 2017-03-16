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


def neural_network_model(data):
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

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.square(prediction-y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)



	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for i in range(0, ip_len , batch_size):
				batch_x = position_data[i:i+batch_size]
				batch_y = joints_data[i:i+batch_size]
				_,c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of',n_epochs,'loss:',epoch_loss  )
			#print('predection:', prediction.eval({x:batch_x[0:1]}))
			#print('truth', batch_y[0:1])
			print('diff:',prediction.eval({x:batch_x[0:1]})-batch_y[0:1])

		#print('error :',sum(abs(prediction.eval({x:batch_x[0:1]})-batch_y[0:1])))

		print('predection:', prediction.eval({x:batch_x[0:1]}))
		print('truth', batch_y[0:1])
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))

		#print('Accuracy:',accuracy.eval({x:test_position, y:test_joints}))


def eaxmple_nn (X,y):
	print(X)
	print(y)
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.square(prediction-y) )
	with tf.Session() as sess:
		print(prediction.eval(feed_dict={x:X}))
		print(y)

train_neural_network(x)	

#eaxmple_nn (test_position[1:3],test_joints[1:3])



