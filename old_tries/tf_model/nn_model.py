
'''
file_tf_model = 'nn_model.py'
exec(open(file_tf_model).read())
'''



#parameters
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000



#inpts and outputs

x = tf.placeholder('float',shape=[None, ip_len ])
y = tf.placeholder('float')
'''
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
'''

hidden_1_layer_w = tf.Variable(tf.random_normal([ip_len ,n_nodes_hl1]))
hidden_1_layer_b = tf.Variable(tf.random_normal([n_nodes_hl1]))

hidden_2_layer_w = tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2]))
hidden_2_layer_b = tf.Variable(tf.random_normal([n_nodes_hl2]))

hidden_3_layer_w = tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3]))
hidden_3_layer_b = tf.Variable(tf.random_normal([n_nodes_hl3]))

hidden_4_layer_w = tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4]))
hidden_4_layer_b = tf.Variable(tf.random_normal([n_nodes_hl3]))

output_layer_w	 = tf.Variable(tf.random_normal([n_nodes_hl3,dof_n]))
output_layer_b	 = tf.Variable(tf.random_normal([dof_n]))

l1 = tf.add(tf.matmul(x,hidden_1_layer_w), hidden_1_layer_b)	
l1 = tf.nn.tanh(l1)

l2 = tf.add(tf.matmul(l1,hidden_2_layer_w), hidden_2_layer_b)	
l2 = tf.nn.tanh(l2)									  		

l3 = tf.add(tf.matmul(l2,hidden_3_layer_w), hidden_3_layer_b)	
l3 = tf.nn.sigmoid(l3)		  

l4 = tf.add(tf.matmul(l3,hidden_4_layer_w), hidden_4_layer_b)	
l4 = tf.nn.sigmoid(l4)	

output = tf.add(tf.matmul(l3,output_layer_w), output_layer_b)



