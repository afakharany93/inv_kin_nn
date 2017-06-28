import numpy as np
import math
from math import pi
from numpy import cos, sin, tan
import pickle
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


#data retrieval
with open("../data/robot_set.pickle","rb") as f:
	all_data = pickle.load(f)

data_n, position_variables_n, dof_n = all_data[0], all_data[1], all_data[2]


position_data = np.array(all_data[3:3+position_variables_n])	#last index not taken 
joints_data = np.array(all_data[3+position_variables_n:])	#the end in this case is included

del all_data

position_data = position_data.T
joints_data = joints_data.T

test_size = int(0.3*data_n)

test_position = position_data[data_n-test_size:]
test_joints = joints_data[data_n-test_size:]


position_data = np.delete(position_data, list(range(data_n-test_size , position_data.shape[0])), axis=0)
joints_data = np.delete(joints_data, list(range(data_n-test_size , joints_data.shape[0])), axis=0)
ip_len = position_data.shape[1]

#load model structure
file_model = 'inv_kin_nn_model.py'
exec(open(file_model).read())

#tflearn.config.init_graph (log_device=True, soft_placement=False)	#http://tflearn.org/config/
#parameters
batch_size_t = 256

n_epochs = 1024	#number of feed forward and back prop

model = tflearn.DNN(net)
for i in range(10):
	model.fit({'input': position_data}, {'targets': joints_data}, n_epoch=int(n_epochs/10), 
		validation_set= 0.1,
		snapshot_step = 5000,snapshot_epoch = False,
		batch_size = batch_size_t, validation_batch_size = None,
		show_metric = True,  run_id='inv_kin', shuffle = None)	#http://tflearn.org/models/dnn/

	model.save('inv_kin.model')
