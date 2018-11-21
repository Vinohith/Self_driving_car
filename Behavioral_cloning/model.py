from keras.models import Model 
from keras.layers import Input, Convolution2D, Flatten, Dropout, Dense, ELU, Lambda
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K 



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


def get_model():

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


if name = '__main__':
	