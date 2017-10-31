sup = True

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    objects.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: scollet <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2017/09/05 22:24:21 by scollet           #+#    #+#              #
#    Updated: 2017/09/05 22:24:22 by scollet          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential

HIDDEN_LAYERS = 2
NEURAL_DENSITY = 32
LEARNING_RATE = 0.001

class Network():
	def __init__(self, d1, d2, d3, out_put):
		self.output_size = out_put
		self.learning_rate = LEARNING_RATE
		self.model = self._build_model(d1, d2, d3, out_put)
		self.target_model = self._build_model(d1, d2, d3, out_put)
	def _build_model(self, d1, d2, d3, out_put):
		model = Sequential()
		model.add(Conv2D(32, (d1, d1), input_shape=(d1, d2, d3), data_format='channels_first'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (d1, d1), input_shape=(d1, d2, d3), data_format='channels_first'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (d1, d1), input_shape=(d1, d2, d3), data_format='channels_first'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(out_put))
		model.add(Activation('relu'))

		model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
		return model
	def pre_gen(self, predict_gen, steps_per_epoch, \
	epochs, validation_data, validation_steps):
		self.model.fit_generator(
			predict_gen,
	        steps_per_epoch=steps_per_epoch,
	        epochs=epochs,
	        validation_data=validation_data,
	        validation_steps=validation_steps)
	def fit_gen(self, train_generator, steps_per_epoch, \
	epochs, validation_data, validation_steps):
		self.model.fit_generator(
	        train_generator,
	        steps_per_epoch=steps_per_epoch,
	        epochs=epochs,
	        validation_data=validation_data,
	        validation_steps=validation_steps)
	def train(self, feed, target):
		self.model.fit(feed, target, epochs=1, verbose=0)
	def predict(self, X):
		self.model.predict(X)
	def load(self, name):
		self.model.load_weights(name)
	def save(self, name):
		self.model.save_weights(name)
