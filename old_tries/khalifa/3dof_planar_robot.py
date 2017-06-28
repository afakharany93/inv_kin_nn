'''
file_model = '3dof_planar_robot.py'
exec(open(file_model).read())

'''

import numpy as np
import math
from math import pi
from numpy import cos, sin, tan
import pickle
import matplotlib.pyplot as plt

'''
FWD equations:
Px = cos(theta1) * l2*cos(theta1+theta2) + l3*cos(theta1+theta2+theta3))
Py = sin(theta1) * l2*sin(theta1+theta2) + l3*sin(theta1+theta2+theta3))
orientation = theta1+theta2+theta3
'''

#user defined parameters
#data parameters
data_n = 1*(10**6)	#size of vectors of data
dof_n = 3	#number of dof
position_variables_n = 3 	#number of position variables, e.g. 2 => planar, 3=> 3d position, 6=> 3d position + orientation

#robot parametrs
l1 = 0.25	#m
l2 = 0.25	#m
l3 = 0.15	#m

#d_max_range = 0.2	#m
#d_min_range = 0		#m

theta_max_range = pi/2 	#rad
theta_min_range = 0	#rad

#calculations
#joint variables
theta1 = np.linspace(theta_min_range,theta_max_range,19)
theta2 = np.linspace(theta_min_range,theta_max_range,19)
theta3 = np.linspace(theta_min_range,theta_max_range,19)

Px = []
Py = []
orient = []

angle1 = []
angle2 = []
angle3 = []
i = 0
#position calculation
for joint1 in theta1:
	for joint2 in theta2:
		for joint3 in theta3:
			Px = np.append(Px, l1*cos(joint1) + l2*cos(joint1+joint2) + l3*cos(joint1+joint2+joint3)) 
			Py = np.append(Py, l1*sin(joint1) + l2*sin(joint1+joint2) + l3*sin(joint1+joint2+joint3))
			orient = np.append(orient, joint1+joint2+joint3)

			angle1 = np.append(angle1,joint1)
			angle2 = np.append(angle2,joint2)
			angle3 = np.append(angle3,joint3)


#Px = cos(joint1) * (joint2 + l3*cos(joint3) + joint4 * cos(joint3))
#Py = sin(joint1) * (joint2 + l3*cos(joint3) + joint4 * cos(joint3))
#Pz = l1 + (l3 + joint4) * sin(joint3)
#orient = joint1 + joint3

print(Px.size)
print(Py.size)
print(orient.size)

plt.plot(Px,Py,'ro')
plt.show()
#print(position_variables_n,dof_n ,Px, Py, Pz, joint1, joint2, joint3, joint4)

#dumping the data, user must take care of sequence of dumping
with open('robot_set.pickle','wb') as f:
	pickle.dump([data_n, position_variables_n, dof_n ,Px, Py, orient ,angle1, angle2,angle3], f)
	#			leave first 3 elements reserved, start your changes after them, there are the position data, then the joints data