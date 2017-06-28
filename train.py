import numpy as np
import math
from math import pi
from math import cos, sin, tan
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input, Lambda, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import model_build
import sys

model_file = ''
arguments = str(sys.argv).replace("]", "")
if 'retrain' in arguments:
    retrain_flag = True
    print('retraining')
    if '.h5' in arguments:
        for sub_arg in arguments.split(' '):
            if '.h5' in sub_arg: 
                sub_arg = sub_arg.replace("'", "")
                print('loading model file : ', sub_arg)
                model_file = sub_arg
    else:
        model_file = 'model_final.h5'
else:
    retrain_flag = False

dof_n = 5 #number of dof
pose_variables_n = 16 #number of pose variables, e.g. 2 => planar, 3=> 3d pose, 6=> 3d pose + orientation
def robot_fwd_kin(joints_data):
    th1 = joints_data[0]
    th2 = joints_data[1]  
    th3 = joints_data[2] 
    th4 = joints_data[3] 
    th5 = joints_data[4]
    #first row 
    dh11 = sin(pi/2 - th1)*sin(th5) + cos(th2 + th3 + pi/2 + th4)*cos(pi/2 - th1)*cos(th5)
    dh12 =  cos(th5)*sin(pi/2 - th1) - cos(th2 + th3 + pi/2 + th4)*cos(pi/2 - th1)*sin(th5)
    dh13 = sin(th2 + th3 + pi/2 + th4)*cos(pi/2 - th1)
    dh14 =  cos(pi/2 - th1)*(124*sin(th2 + th3 + pi/2 + th4) + 81*cos(th2 + th3) + 80*cos(th2))
    
    #second row
    dh21 = cos(th2 + th3 + pi/2 + th4)*cos(th5)*sin(pi/2 - th1) - cos(pi/2 - th1)*sin(th5)
    dh22 = - cos(pi/2 - th1)*cos(th5) - cos(th2 + th3 + pi/2 + th4)*sin(pi/2 - th1)*sin(th5)
    dh23 = sin(th2 + th3 + pi/2 + th4)*sin(pi/2 - th1)
    dh24 = sin(pi/2 - th1)*(124*sin(th2 + th3 + pi/2 + th4) + 81*cos(th2 + th3) + 80*cos(th2))
    
    #third row
    dh31 = sin(th2 + th3 + pi/2 + th4)*cos(th5)
    dh32 =  -sin(th2 + th3 + pi/2 + th4)*sin(th5)
    dh33 =  -cos(th2 + th3 + pi/2 + th4)
    dh34 = 81*sin(th2 + th3) - 124*cos(th2 + th3 + pi/2 + th4) + 80*sin(th2) + 35
    
    #forth row
    dh41 = 0
    dh42 = 0
    dh43 = 0
    dh44 = 1
    
    return np.array([[dh11, dh12, dh13, dh14],
                     [dh21, dh22, dh23, dh24],
                     [dh31, dh32, dh33, dh34],
                     [dh41, dh42, dh43, dh44]])



#user defined parameters
#data parameters
data_n = int(0.6*(10**5)) #size of vectors of data

#robot parametrs


d_min_range = 0   #m

theta_max_range = pi  #rad
theta_min_range = 0   #rad



#calculations
#joint variables

joint1 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint2 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint3 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint4 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))
joint5 = np.random.uniform(low=theta_min_range, high=theta_max_range, size=(data_n,))

joints_data = np.array([joint1, joint2, joint3, joint4, joint5]).T

#pose calculation

pose_data = np.empty((data_n,4,4))
for i in range(len(joints_data)):
    p_1 = robot_fwd_kin(joints_data[i])
    pose_data[i] = p_1

#p[:,:,3] = p[:,:,3]/381     #normalizing translation part
#print(p)
pose_data = pose_data.reshape(-1,16)

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
points_n_t = 200
traj_joint = np.array([np.linspace(0,pi,points_n_t),np.linspace(0,pi,points_n_t),np.linspace(0,pi,points_n_t), np.linspace(0,0.2,points_n_t), np.linspace(0,0.2,points_n_t)]).T
p_t = np.empty((len(traj_joint),4,4))
for i in range(len(traj_joint)):
    p_1_t = robot_fwd_kin(traj_joint[i])
    p_t[i] = p_1_t
    
p_t_ip = p_t.reshape(-1,16)
p_t_ip = np.reshape(p_t_ip, (p_t_ip.shape[0], 1, p_t_ip.shape[1]))
px = p_t[:,0,3]
py = p_t[:,1,3]
pz = p_t[:,2,3]

model = model_build.build_model(retrain = retrain_flag, model_file = model_file)
model_build.compile_model(model, alpha = 0.0000001, optimzr = Adam )
print(model_build.tb_log_dir)
# model_build.train_model(model,input_data=pose_data, ground_truth=joints_data ,epochs_n=150, batch_Size=150, retrain = retrain_flag, val_split=0.15)

# print('\ntesting : \n')
# scores = model.evaluate(test_pose,test_joints)
# print("\n %s: %f" % (model.metrics_names[0], scores)) 


# joints_pred = model.predict(p_t_ip)
# p_pred = np.empty((len(joints_pred),4,4))
# for i in range(len(joints_pred)):
#     p_1_pred = robot_fwd_kin(joints_pred[i])
#     p_pred[i] = p_1_pred

# px_pred = p_pred[:,0,3]
# py_pred = p_pred[:,1,3]
# pz_pred = p_pred[:,2,3]



# plt.figure(1,figsize=(10,8), dpi=80)

# plt.subplot(2,2,1)
# plt.plot(px_pred, py_pred, color="blue", linewidth=1.0, linestyle="-", label='x,y predicted trajectory')
# plt.plot(px, py, color="green", linewidth=1.0, linestyle="-", label='x,y actual trajectory')
# plt.legend()

# plt.subplot(2,2,2)
# plt.plot(px_pred, pz_pred, color="blue", linewidth=1.0, linestyle="-", label='x,z predicted trajectory')
# plt.plot(px, pz, color="green", linewidth=1.0, linestyle="-", label='x,z actual trajectory')
# plt.legend()

# plt.subplot(2,2,3)
# plt.plot(py_pred, pz_pred, color="blue", linewidth=1.0, linestyle="-", label='y,z predicted trajectory')
# plt.plot(py, pz, color="green", linewidth=1.0, linestyle="-", label='y,z actual trajectory')
# plt.legend()

# fig = plt.figure(figsize=(7,5), dpi=80)
# ax = fig.gca(projection='3d')
# ax.plot(px, py, pz, label='actual trajectory')
# ax.plot(px_pred, py_pred, pz_pred, label='predicted trajectory')
# ax.legend()

# plt.show()