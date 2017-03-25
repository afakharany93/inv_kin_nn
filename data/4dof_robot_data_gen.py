import numpy as np
import math
from math import pi
from numpy import cos, sin, tan
import pickle

'''
FWD equations:
Px = cos(theta1) * (d2 + l3*cos(theta3) + d4 * cos(theta3))
Py = sin(theta1) * (d2 + l3*cos(theta3) + d4 * cos(theta3))
Pz = l1 + (l3 + d4) * sin(theta3)

'''

#user defined parameters
#data parameters
data_n = 1*(19**4)	#size of vectors of data
dof_n = 4	#number of dof
position_variables_n = 4 	#number of position variables, e.g. 2 => planar, 3=> 3d position, 6=> 3d position + orientation

#robot parametrs
l1 = 0.25	#m
l3 = 0.15	#m

d_max_range = 0.2	#m
d_min_range = 0		#m

theta_max_range = 2*pi 	#rad
theta_min_range = 0	#rad

#calculations
#joint variables
joint1 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint2 = np.random.uniform(low=d_min_range, high=d_max_range, size=(data_n,)) 
joint3 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint4 = np.random.uniform(low=d_min_range, high=d_max_range, size=(data_n,))

#position calculation
Px = cos(joint1) * (joint2 + l3*cos(joint3) + joint4 * cos(joint3))
Py = sin(joint1) * (joint2 + l3*cos(joint3) + joint4 * cos(joint3))
Pz = l1 + (l3 + joint4) * sin(joint3)
orient = joint1 + joint3

#print(position_variables_n,dof_n ,Px, Py, Pz, joint1, joint2, joint3, joint4)

#dumping the data, user must take care of sequence of dumping
with open('robot_set.pickle','wb') as f:
	pickle.dump([data_n, position_variables_n, dof_n ,Px, Py, Pz,orient, joint1, joint2, joint3, joint4], f)
	#			leave first 3 elements reserved, start your changes after them, there are the position data, then the joints data