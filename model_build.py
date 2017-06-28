import keras
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
from keras.callbacks import ModelCheckpoint
import pickle
import shutil
import os

tb_log_dir = ''


checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

def build_model(retrain = False, model_file = 'model_final.h5'):
	if retrain:
		model = load_model(model_file) 
	else:
		input_dh = Input(shape=(1,16), name='input')
		lstm_layer = LSTM(7,activation='elu', name='LSTM')(input_dh)
		dense_1 = Dense(300, activation = 'elu', name='dense_1')(lstm_layer)
		dense_2 = Dense(130, activation = 'elu', name='dense_2')(dense_1)
		dense_3 = Dense(65, activation = 'elu', name='dense_3')(dense_2)
		dense_4 = Dense(40, activation = 'elu', name='dense_4')(dense_3)
		dense_5 = Dense(18, activation = 'elu', name='dense_5')(dense_4)
		dense_6 = Dense(9, activation = 'elu', name='dense_6')(dense_5)
		op = Dense(5, name='output')(dense_6)
		model = Model(inputs=input_dh, outputs=op)
	return model


def compile_model(model, loss_metric='mean_squared_error', alpha = 0.001, optimzr = RMSprop ):
	model.compile(loss=loss_metric, optimizer=optimzr(lr=alpha))
	#model.optimizer.lr.assign(alpha)
	global tb_log_dir
	tb_log_dir = './Graph/lr_{},{},param_number_{}'.format(alpha,optimzr, model.count_params())



def train_model(model,input_data, ground_truth ,epochs_n=150, batch_Size=150, retrain = False, val_split=0.15):
	#model.fit(pose_data, joints_data, epochs=5, batch_size=150, validation_split=0.15)

	if retrain == False:
		init_epoch = 0

		# if os.path.isdir('./Graph'):
		# 	shutil.rmtree(path='./Graph')
	else:
		with open("model_extr_param.pickle","rb") as f:
			init_epoch = pickle.load(f)
	print('tensorboard direectory is', tb_log_dir)
	tbCallBack = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1,  write_graph=True, write_images=True)		#to open in terminal   tensorboard --logdir ./Graph 
	model.fit(input_data, ground_truth, epochs=init_epoch+epochs_n, batch_size=batch_Size, validation_split=val_split, callbacks=[tbCallBack], initial_epoch=init_epoch)
	model.save(tb_log_dir+'/model_final.h5')
	init_epoch = init_epoch+epochs_n
	with open("model_extr_param.pickle","wb") as f:
		pickle.dump(init_epoch, f)

