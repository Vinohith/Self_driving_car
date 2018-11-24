from helperfunction import data_preperation, generator_data, generator_val
from model import get_nvidiamodel
from keras.callbacks import ModelCheckpoint, CSVLogger
from config import *


if __name__ == '__main__':
	X_train, y_train, X_val, y_val = data_preperation('data/driving_log.csv')
	data_generator = generator_data(CONFIG['batchsize'], X_train, y_train)
	val_generator = generator_val(CONFIG['batchsize'], x_, y_val)

	nvidia_net = get_nvidiamodel(summary = True)
	nvidia_net.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

	with open('logs/model.json', 'w') as f:
		f.write(nvidia_net.to_json)

	checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
	logger = CSVLogger(filename='logs/history.csv')

	model.fit_generator(data_generator, samples_per_epoch = math.ceil(len(X_train)), 
						nb_epoch = 25, validation_data = val_generator, nb_val_samples = len(X_val), 
						callbacks = [checkpointer, logger])
	print('Done training')

	nvidia_net.save_weights('model.h5')
	print('Saved model to disk')





# if __name__ == '__main__':

# 	train_data, val_data = split_train_val('data/driving_log.csv')

# 	nvidia_net = get_nvidiamodel(summary=True)
# 	nvidia_net.compile(optimizer='adam', loss='mse')

# 	with open('logs/model.json', 'w') as f:
# 		f.write(nvidia_net.to_json)

# 	checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
# 	logger = CSVLogger(filename='logs/history.csv')

# 	nvidia_net.fit_generator(generator=generate_data_batch(train_data, augment_data=True, bias=CONFIG['bias']),
# 							samples_per_epoch=300*CONFIG['batchsize'],
# 							nb_epoch=50,
# 							validation_data=generate_data_batch(val_data, augment_data=False, bias=1.0),
# 							nb_val_samples=100*CONFIG['batchsize'],
# 							callbacks=[checkpointer, logger])