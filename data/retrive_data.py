import numpy as np
import math
from math import pi
from numpy import cos, sin, tan
import pickle

with open("robot_set.pickle","rb") as f:
	all_data = pickle.load(f)

data_n, position_variables_n, dof_n = all_data[0], all_data[1], all_data[2]

print('data length: ', data_n)
print('no of position variables: ', position_variables_n)
print('no of dof: ', dof_n)

#position_data = np.array(all_data[2:2+position_variables_n])
position_data = np.array(all_data[3:3+position_variables_n])	#last index not taken 
joints_data = np.array(all_data[3+position_variables_n:])	#the end in this case is included




print('position data size', position_data.shape)
print('joint data size', joints_data.shape)


