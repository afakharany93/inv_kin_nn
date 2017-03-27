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

file_tf_model = 'nn_model.py'
exec(open(file_tf_model).read())

tf.add_to_collection('hidden_1_layer_w', hidden_1_layer_w)
tf.add_to_collection('hidden_1_layer_b', hidden_1_layer_b)
tf.add_to_collection('hidden_2_layer_w', hidden_2_layer_w)
tf.add_to_collection('hidden_2_layer_b', hidden_2_layer_b)
tf.add_to_collection('hidden_3_layer_w', hidden_3_layer_w)
tf.add_to_collection('hidden_3_layer_b', hidden_3_layer_b)
tf.add_to_collection('hidden_4_layer_w', hidden_4_layer_w)
tf.add_to_collection('hidden_4_layer_b', hidden_4_layer_b)
tf.add_to_collection('output_layer_w', output_layer_w)
tf.add_to_collection('output_layer_b', output_layer_b)

batch_size = 100

n_epochs = 1100	#number of feed forward and back prop

alpha = 0.0002	#learning rate, default 0.001

saver = tf.train.Saver()

def train_neural_network(x):
	prediction = output
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

		saver.save(sess, 'my-model')

		#print('Accuracy:',accuracy.eval({x:test_position, y:test_joints}))


train_neural_network(x)	


