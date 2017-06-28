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
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


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



# # Trajectory Data

# In[27]:

points_n_t = 200
j1_t = np.linspace(0,pi,points_n_t)
j2_t = np.linspace(0,pi,points_n_t)
j3_t = np.linspace(0,pi/2,points_n_t)
j4_t = np.linspace(0,pi/2,points_n_t)
traj_joint = np.array([j1_t, j2_t, j3_t, j4_t]).T
p_t = np.empty((len(traj_joint),4,4))
for i in range(len(traj_joint)):
    p_1_t = robot_fwd_kin(traj_joint[i])
    p_t[i] = p_1_t
    
p_t_ip = p_t.reshape(-1,16)
p_t_ip = np.reshape(p_t_ip, (p_t_ip.shape[0], 1, p_t_ip.shape[1]))
px = p_t[:,0,3]
py = p_t[:,1,3]
pz = p_t[:,2,3]


model = load_model('my_model.h5')



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