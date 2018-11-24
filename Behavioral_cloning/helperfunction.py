import numpy as np 
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import math
from config import *



def data_preperation(data_dir):

	# reading the csv file
	colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
	data = pandas.read_csv(data_dir, skiprows=[0], names=colnames)
	# print(data.head())

	# converting the required columns in to lists
	center = data.center.tolist()
	center_recover = data.center.tolist() 
	# print(center[:5])
	left = data.left.tolist()
	right = data.right.tolist()
	steering = data.steering.tolist()
	steering_recover = data.steering.tolist()

	# spliting data to train and validation 
	center, steering = shuffle(center, steering)
	center, X_val, steering, y_val = train_test_split(center, steering, test_size=0.10, random_state=100)


	# list of image names
	d_straight, d_left, d_right = [], [], []
	# list of angles
	a_straight, a_left, a_right = [], [], []
	for steer in steering:
		# print(steer, steering.index(steer)) 
		index = steering.index(steer)
		if steer > 0.15:
			d_right.append(center[index])
			a_right.append(steer)
		elif steer < -0.15:
			d_left.append(center[index])
			d_left.append(steer)
		else:
			d_straight.append(center[index])
			a_straight.append(steer)

	# number fo straight, left and right samples
	ds_size, dl_size, dr_size = len(d_straight), len(d_left), len(d_right)
	# print(ds_size, dl_size, dr_size)
	main_size = math.ceil(len(center_recover))
	# print(main_size)
	l_xtra = ds_size - dl_size
	r_xtra = ds_size - dr_size
	# generating random samples
	indice_L = random.sample(range(main_size), l_xtra)
	indice_R = random.sample(range(main_size), r_xtra)

	# filter angle less than -0.15 and add right camera images into driving left list, minus an adjustment angle #
	for i in indice_L:
		if steering_recover[i] < -0.15:
			d_left.append(right[i])
			a_left.append(steering_recover[i] - CONFIG['steering_adjustment'])
	# filter angle less than -0.15 and add right camera images into driving left list, minus an adjustment angle #
	for i in indice_R:
		if steering_recover[i] > 0.15:
			d_right.append(left[i])
			a_right.append(steering_recover[i] + CONFIG['steering_adjustment'])

	# combining to create training data
	X_train = d_straight + d_left + d_right
	y_train = a_straight + a_left + a_right
	# print(len(X_train))

	return X_train, y_train, X_val, y_val






if __name__ == '__main__':
	X_train, y_train, X_val, y_val = data_preperation('data/driving_log.csv')
































# def load_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
# 	"""
# 	Load a batch of driving data from the "data" list.
# 	A batch of data is constituted by a batch of frames of the training track as well as the corresponding
# 	steering directions.
# 	:param data: list of training data in the format provided by Udacity
# 	:param batchsize: number of elements in the batch
# 	:param data_dir: directory in which frames are stored
# 	:param augment_data: if True, perform data augmentation on training data
# 	:param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
# 	:return: X, Y which are the batch of input frames and steering angles respectively
# 	"""
# 	# set training images resized shape
# 	h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

# 	# prepare output structures
# 	X = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
# 	y_steer = np.zeros(shape=(batchsize,), dtype=np.float32)
# 	y_throttle = np.zeros(shape=(batchsize,), dtype=np.float32)

# 	# shuffle data
# 	shuffled_data = shuffle(data)

# 	loaded_elements = 0
# 	while loaded_elements < batchsize:

# 		ct_path, lt_path, rt_path, steer, throttle, brake, speed = shuffled_data.pop()

# 		# cast strings to float32
# 		steer = np.float32(steer)
# 		throttle = np.float32(throttle)

# 		# randomly choose which camera to use among (central, left, right)
# 		# in case the chosen camera is not the frontal one, adjust steer accordingly
# 		delta_correction = CONFIG['delta_correction']
# 		camera = random.choice(['frontal', 'left', 'right'])
# 		if camera == 'frontal':
# 			frame = preprocess(cv2.imread(join(data_dir, ct_path.strip())))
# 			steer = steer
# 		elif camera == 'left':
# 			frame = preprocess(cv2.imread(join(data_dir, lt_path.strip())))
# 			steer = steer + delta_correction
# 		elif camera == 'right':
# 			frame = preprocess(cv2.imread(join(data_dir, rt_path.strip())))
# 			steer = steer - delta_correction

# 		if augment_data:

# 			# mirror images with chance=0.5
# 			if random.choice([True, False]):
# 				frame = frame[:, ::-1, :]
# 				steer *= -1.

# 			# perturb slightly steering direction
# 			steer += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

# 			# if color images, randomly change brightness
# 			if CONFIG['input_channels'] == 3:
# 				frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
# 				frame[:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
# 				frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
# 				frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

# 		# check that each element in the batch meet the condition
# 		steer_magnitude_thresh = np.random.rand()
# 		if (abs(steer) + bias) < steer_magnitude_thresh:
# 			pass  # discard this element
# 		else:
# 			X[loaded_elements] = frame
# 			y_steer[loaded_elements] = steer
# 			loaded_elements += 1


# 	return X, y_steer


# def generate_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
# 	"""
# 	Generator that indefinitely yield batches of training data
# 	:param data: list of training data in the format provided by Udacity
# 	:param batchsize: number of elements in the batch
# 	:param data_dir: directory in which frames are stored
# 	:param augment_data: if True, perform data augmentation on training data
# 	:param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
# 	:return: X, Y which are the batch of input frames and steering angles respectively
# 	"""
# 	while True:

# 		X, y_steer = load_data_batch(data, batchsize, data_dir, augment_data, bias)

# 		yield X, y_steer