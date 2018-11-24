if __name__ == '__main__':

	train_data, val_data = split_train_val('data/driving_log.csv')

	nvidia_net = get_nvidiamodel(summary=True)
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