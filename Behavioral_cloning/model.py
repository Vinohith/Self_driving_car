from keras.models import Model 
from keras.layers import Input, Convolution2D, Flatten, Dropout, Dense, ELU, Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K 
import numpy as np 
import cv2
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from os.path import join



CONFIG = {
    'batchsize': 512,
    'input_width': 200,
    'input_height': 66,
    'input_channels': 3,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.2,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.8,
    'crop_height': range(20, 140)
}



def get_model(summary=True):

	init = 'glorot_uniform'

	input_frame = Input(shape=(CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']))

	x = Lambda(lambda z: z / 127.5 - 1.)(input_frame)

	x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
	x = ELU()(x)
	x = Dropout(0.2)(x)

	x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
	x = ELU()(x)
	x = Dropout(0.2)(x)

	x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
	x = ELU()(x)
	x = Dropout(0.2)(x)

	x = Convolution2D(64, 3, 3, border_mode='valid', init=init)(x)
	x = ELU()(x)
	x = Dropout(0.2)(x)

	x = Convolution2D(64, 3, 3, border_mode='valid', init=init)(x)
	x = ELU()(x)
	x = Dropout(0.2)(x)

	x = Flatten()(x)

	x = Dense(100, init=init)(x)
	x = ELU()(x)
	x = Dropout(0.5)(x)

	x = Dense(50, init=init)(x)
	x = ELU()(x)
	x = Dropout(0.5)(x)

	x = Dense(10, init=init)(x)
	x = ELU()(x)
	out = Dense(1, init=init)(x)

	model = Model(input = input_frame, output = out)

	if summary:
		model.summary()

	return model



def split_train_val(csv_driving_data, test_size = 0.2):

	with open(csv_driving_data, 'r') as f:
		reader = csv.reader(f)
		driving_data = [row for row in reader][1:]

	train_data, val_data = train_test_split(driving_data, test_size = test_size, random_state = 1)

	return train_data, val_data



if name = '__main__':

	train_data, val_data = split_train_val('data/driving_log.csv')

	nvidia_net = get_model(summary=True)
	nvidia_net.compile(optimizer='adam', loss='mse')

	with open('logs/model.json', 'w') as f:
		f.write(nvidia_net.to_json)

	checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
	logger = CSVLogger(filename='logs/history.csv')

	nvidia_net.fit_generator(generator=generate_data_batch(train_data, augment_data=True, bias=CONFIG['bias']),
							samples_per_epoch=300*CONFIG['batchsize'],
							nb_epoch=50,
							validation_data=generate_data_batch(val_data, augment_data=False, bias=1.0),
							nb_val_samples=100*CONFIG['batchsize'],
							callbacks=[checkpointer, logger])