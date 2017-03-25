'''
file_model = 'inv_kin_nn_model.py'
exec(open(file_model).read())
'''

#parameters

n_nodes_hl1 = 128
n_nodes_hl2 = 128
n_nodes_hl3 = 128
n_nodes_hl4 = 128
n_nodes_hl5 = 64

alpha = 0.01	#learning rate, default 0.001


#network

#input layer
net = input_data(shape=[None, ip_len], name='input')

#hidden layers
net = fully_connected(net, n_nodes_hl1, activation='sigmoid', bias=True, bias_init='truncated_normal')
net = fully_connected(net, n_nodes_hl2, activation='sigmoid', bias=True, bias_init='truncated_normal')
net = fully_connected(net, n_nodes_hl3, activation='sigmoid', bias=True, bias_init='truncated_normal')
net = fully_connected(net, n_nodes_hl4, activation='sigmoid', bias=True, bias_init='truncated_normal')
net = fully_connected(net, n_nodes_hl5, activation='sigmoid', bias=True, bias_init='truncated_normal')
#net = dropout(net, 0.8)
#output layer
net = fully_connected(net, dof_n)

net = regression(net, optimizer='adam', learning_rate=alpha, loss='mean_square', name='targets') #http://tflearn.org/layers/estimator/
																								 #http://tflearn.org/optimizers/		


