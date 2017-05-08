
# coding: utf-8

# In[17]:

import numpy as np
import math
from math import pi
from math import cos, sin, tan
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import tflearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# # DH Matrix of Robot

# In[18]:

dof_n = 4 #number of dof
pose_variables_n = 16 #number of pose variables, e.g. 2 => planar, 3=> 3d pose, 6=> 3d pose + orientation
def robot_fwd_kin(joints):
    th1 = joints[0]
    th2 = joints[1]  
    th3 = joints[2] 
    th4 = joints[3] 
    l1 = 10
    l2 = 10
    l3 = 10
    l4 = 10
    #first row 
    dh11 = cos(th2 + th3 + th4)*cos(th1)
    dh12 = -1*sin(th2 + th3 + th4)*cos(th1)
    dh13 = sin(th1)
    dh14 = cos(th1)*(l3*cos(th2 + th3) + l2*cos(th2) + l4*cos(th2 + th3 + th4))
    
    #second row
    dh21 = cos(th2 + th3 + th4)*sin(th1)
    dh22 = -1*sin(th2 + th3 + th4)*sin(th1)
    dh23 = -1*cos(th1)
    dh24 = sin(th1)*(l3*cos(th2 + th3) + l2*cos(th2) + l4*cos(th2 + th3 + th4))
    
    #third row
    dh31 = sin(th2 + th3 + th4)
    dh32 = cos(th2 + th3 + th4)
    dh33 = 0
    dh34 = l1 + l3*sin(th2 + th3) + l2*sin(th2) + l4*sin(th2 + th3 + th4)
    
    #forth row
    dh41 = 0
    dh42 = 0
    dh43 = 0
    dh44 = 1
    
    return np.array([[dh11, dh12, dh13, dh14],
                     [dh21, dh22, dh23, dh24],
                     [dh31, dh32, dh33, dh34],
                     [dh41, dh42, dh43, dh44]])


# # Data Generation Segment
# This segment is used to generate the training and testing data then saving them in a pickle file, if the data is already generated and no edits are introduced, the user can skip this part
# 

# In[24]:

#user defined parameters
#data parameters
data_n = int(0.6*(10**5)) #size of vectors of data

#robot parametrs

d_max_range = 0.2 #m
d_min_range = 0   #m

theta_max_range = pi  #rad
theta_min_range = 0   #rad

#calculations
#joint variables

joint1 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint2 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint3 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint4 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))

joints = np.array([joint1, joint2, joint3, joint4]).T

#pose calculation

p = np.empty((data_n,4,4))
for i in range(len(joints)):
    p_1 = robot_fwd_kin(joints[i])
    p[i] = p_1
    

p = p.reshape(-1,16)



#dumping the data, user must take care of sequence of dumping
with open('robot_set.pickle','wb') as f:
    pickle.dump([p, joints], f)
            #leave first 3 elements reserved, start your changes after them, there are the pose data, then the joints data

print('\ndata saved :)')

# # Data Loading Segment

# In[25]:

#data retrieval
with open("robot_set.pickle","rb") as f:
	all_data = pickle.load(f)

pose_data, joints_data = all_data[0], all_data[1]

del all_data

data_n = pose_data.shape[0] #update the size of data sets

test_size = int(0.15*data_n)
validate_size = int(0.15*data_n)

test_pose = pose_data[data_n-test_size:]
test_joints = joints_data[data_n-test_size:]

pose_data = np.delete(pose_data, list(range(data_n-test_size , pose_data.shape[0])), axis=0)
joints_data = np.delete(joints_data, list(range(data_n-test_size , joints_data.shape[0])), axis=0)

op_len = joints_data.shape[1]
ip_len = pose_data.shape[1]

# Data preprocessing

# reshape input to be [samples, time steps, features]
pose_data = np.reshape(pose_data, (pose_data.shape[0], 1, pose_data.shape[1]))
test_pose = np.reshape(test_pose, (test_pose.shape[0], 1, test_pose.shape[1]))



# # Trajectory Data

# In[27]:

points_n_t = 20
traj_joint = np.array([np.linspace(0,pi,points_n_t),np.linspace(0,pi,points_n_t),np.linspace(0,pi,points_n_t), np.linspace(0,0.2,points_n_t)]).T
p_t = np.empty((len(traj_joint),4,4))
for i in range(len(traj_joint)):
    p_1_t = robot_fwd_kin(traj_joint[i])
    p_t[i] = p_1_t
    
p_t_ip = p_t.reshape(-1,16)
p_t_ip = np.reshape(p_t_ip, (p_t_ip.shape[0], 1, p_t_ip.shape[1]))
px = p_t[:,0,3]
py = p_t[:,1,3]
pz = p_t[:,2,3]


# # Neural Network Definition

# Network building
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(7, input_shape=(1, 16)))
model.add(Dense(256, activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(32, activation = 'sigmoid'))
model.add(Dense(16, activation = 'sigmoid'))
model.add(Dense(8, activation = 'sigmoid'))
model.add(Dense(4, activation = 'sigmoid'))
model.add(Dense(4))
model.compile(loss='mean_squared_error', optimizer='adam')

#training
model.fit(pose_data, joints_data, epochs=150, batch_size=150, validation_split=0.15)

model.save('my_model.h5')


print('\ntesting : \n')
scores = model.evaluate(test_pose,test_joints)
print("\n %s: %f" % (model.metrics_names[0], scores)) 


joints_pred = model.predict(p_t_ip)
p_pred = np.empty((len(joints_pred),4,4))
for i in range(len(joints_pred)):
    p_1_pred = robot_fwd_kin(joints_pred[i])
    p_pred[i] = p_1_pred

px_pred = p_pred[:,0,3]
py_pred = p_pred[:,1,3]
pz_pred = p_pred[:,2,3]



plt.figure(1,figsize=(10,8), dpi=80)

plt.subplot(2,2,1)
plt.plot(px_pred, py_pred, color="blue", linewidth=1.0, linestyle="-", label='x,y predicted trajectory')
plt.plot(px, py, color="green", linewidth=1.0, linestyle="-", label='x,y actual trajectory')
plt.legend()

plt.subplot(2,2,2)
plt.plot(px_pred, pz_pred, color="blue", linewidth=1.0, linestyle="-", label='x,z predicted trajectory')
plt.plot(px, pz, color="green", linewidth=1.0, linestyle="-", label='x,z actual trajectory')
plt.legend()

plt.subplot(2,2,3)
plt.plot(py_pred, pz_pred, color="blue", linewidth=1.0, linestyle="-", label='y,z predicted trajectory')
plt.plot(py, pz, color="green", linewidth=1.0, linestyle="-", label='y,z actual trajectory')
plt.legend()

fig = plt.figure(figsize=(7,5), dpi=80)
ax = fig.gca(projection='3d')
ax.plot(px, py, pz, label='actual trajectory')
ax.plot(px_pred, py_pred, pz_pred, label='predicted trajectory')
ax.legend()

plt.show()


# In[15]:



