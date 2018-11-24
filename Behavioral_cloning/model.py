from keras.models import Model 
from keras.layers import Input, Convolution2D, Flatten, Dropout, Dense, ELU, Lambda
import keras.backend as K 
from config import *



def get_nvidiamodel(summary=True):

	init = 'glorot_uniform'

	input_frame = Input(shape=(64,64,3))

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


