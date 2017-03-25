from numpy import cos, sin, tan
import pickle
import numpy as np
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

test_size = int(0.01*data_n)

test_position = position_data[data_n-test_size:]
test_joints = joints_data[data_n-test_size:]


position_data = np.delete(position_data, list(range(data_n-test_size , position_data.shape[0])), axis=0)
joints_data = np.delete(joints_data, list(range(data_n-test_size , joints_data.shape[0])), axis=0)
ip_len = position_data.shape[1]

#loading the model structure
file_model = 'inv_kin_nn_model.py'
exec(open(file_model).read())

#loading the model values
model = tflearn.DNN(net)
model.load('inv_kin.model')

pred = np.array(model.predict(test_position[1:5]))
print('Prediction: \n', pred)
print('truth: \n', test_joints[1:5])
print('diff: \n', test_joints[1:5]-pred)
print("evaluation: ", model.evaluate(test_position[1:5], test_joints[1:5]))