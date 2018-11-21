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



def load_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
	"""
	Load a batch of driving data from the "data" list.
	A batch of data is constituted by a batch of frames of the training track as well as the corresponding
	steering directions.
	:param data: list of training data in the format provided by Udacity
	:param batchsize: number of elements in the batch
	:param data_dir: directory in which frames are stored
	:param augment_data: if True, perform data augmentation on training data
	:param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
	:return: X, Y which are the batch of input frames and steering angles respectively
	"""
	# set training images resized shape
	h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

	# prepare output structures
	X = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
	y_steer = np.zeros(shape=(batchsize,), dtype=np.float32)
	y_throttle = np.zeros(shape=(batchsize,), dtype=np.float32)

	# shuffle data
	shuffled_data = shuffle(data)

	loaded_elements = 0
	while loaded_elements < batchsize:

		ct_path, lt_path, rt_path, steer, throttle, brake, speed = shuffled_data.pop()

		# cast strings to float32
		steer = np.float32(steer)
		throttle = np.float32(throttle)

		# randomly choose which camera to use among (central, left, right)
		# in case the chosen camera is not the frontal one, adjust steer accordingly
		delta_correction = CONFIG['delta_correction']
		camera = random.choice(['frontal', 'left', 'right'])
		if camera == 'frontal':
			frame = preprocess(cv2.imread(join(data_dir, ct_path.strip())))
			steer = steer
		elif camera == 'left':
			frame = preprocess(cv2.imread(join(data_dir, lt_path.strip())))
			steer = steer + delta_correction
		elif camera == 'right':
			frame = preprocess(cv2.imread(join(data_dir, rt_path.strip())))
			steer = steer - delta_correction

		if augment_data:

			# mirror images with chance=0.5
			if random.choice([True, False]):
				frame = frame[:, ::-1, :]
				steer *= -1.

			# perturb slightly steering direction
			steer += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

			# if color images, randomly change brightness
			if CONFIG['input_channels'] == 3:
				frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
				frame[:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
				frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
				frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

		# check that each element in the batch meet the condition
		steer_magnitude_thresh = np.random.rand()
		if (abs(steer) + bias) < steer_magnitude_thresh:
			pass  # discard this element
		else:
			X[loaded_elements] = frame
			y_steer[loaded_elements] = steer
			loaded_elements += 1


	return X, y_steer


def generate_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
	"""
	Generator that indefinitely yield batches of training data
	:param data: list of training data in the format provided by Udacity
	:param batchsize: number of elements in the batch
	:param data_dir: directory in which frames are stored
	:param augment_data: if True, perform data augmentation on training data
	:param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
	:return: X, Y which are the batch of input frames and steering angles respectively
	"""
	while True:

		X, y_steer = load_data_batch(data, batchsize, data_dir, augment_data, bias)

		yield X, y_steer



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