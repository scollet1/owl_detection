import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.utils.np_utils import to_categorical
import sys
import objects
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
import os
from os.path import isfile, join
import cv2
import PIL
from PIL import Image
import shutil
import csv
from keras.utils import plot_model
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from matplotlib import pyplot as plt
import pydot
import graphviz
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from itertools import izip
from random import random
#import kivy
#kivy.require('1.0.6') # replace with your current kivy version !

#from kivy.app import App
#from kivy.uix.label import Label
import argparse

#%matplotlib inline


# Swap softmax with linear


'''we will need to split the image preprocessing into a separate file, I just\
haven't done it yet ;-;  '''

#from preprocess import process

TRAIN = 'train'
VALID = 'validation'
PREDICT = 'predict'

train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

# def custom_test_gen():
#     #loading data
#     (X_train, y_train), (X_test, y_test) = train_datagen.flow_from_directory(
#         'data/train/',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#
#     #some preprocessing
#     y_train = np_utils.to_categorical(y_train,10)
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     X_train /= 255
#     X_test /= 255
#     while 1:
#         for i in range(1875):
#             if i%125==0:
#                 print "i = " + str(i)
#             yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]

# def data_gen(top_dim, bot_dim):
#     """
#     Generator to yield batches of two inputs (per sample) with shapes top_dim and
#     bot_dim along with their labels.
#     """
#     batch_size = 264
#     while True:
#         top_batch = []
#         bot_batch = []
#         batch_labels = []
#         for i in range(batch_size):
#             # Create random arrays
#             rand_pix = np.random.randint(100, 256)
#             top_img = np.full(top_dim, rand_pix)
#             bot_img = np.full(bot_dim, rand_pix)
#
#             # Set a label
#             label = np.random.choice([0, 1])
#             batch_labels.append(label)
#
#             # Pack each input image separately
#             top_batch.append(top_img)
#             bot_batch.append(bot_img)
#
#         yield [np.array(top_batch), np.array(bot_batch)], np.array(batch_labels)

# def custom_valid_gen():
#     #loading data
#     (X_train, y_train), (X_test, y_test) = train_datagen.flow_from_directory(
#         'data/train/',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#
#     #some preprocessing
#     y_train = np_utils.to_categorical(y_train,10)
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     X_train = X_train.astype('float32')
#     X_test = X_test.astype('float32')
#     X_train /= 255
#     X_test /= 255
#     while 1:
#         for i in range(1875):
#             if i%125==0:
#                 print "i = " + str(i)
#             yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]

def printer(model, im):
    #im_batch = np.expand_dims(im,axis=0)
    #conv_im2 = model.predict(im_batch)

    conv_im2 = np.squeeze(conv_im2, axis=0)
    print conv_im2.shape
    conv_im2 = conv_im2.reshape(conv_im2.shape[:2])

    print conv_im2.shape
    plt.imshow(conv_im2)

def train(model):
	batch_size = 8

	#test_datagen = ImageDataGenerator(rescale=1./255)

	params = {'dim_x': 150,
          'dim_y': 150,
          'dim_z': 3,
          'batch_size': batch_size,
          'shuffle': True}

	#thefile = open('test.txt', 'w')
	# Datasets
	if os.path.exists('data/train/npy'):
		shutil.rmtree('data/train/npy')
		os.makedirs('data/train/npy')
	else:
		os.makedirs('data/train/npy')
	if os.path.exists('data/validation/npy'):
		shutil.rmtree('data/validation/npy')
		os.makedirs('data/validation/npy')
	else:
		os.makedirs('data/validation/npy')

	#print "read training"
	basewidth = 150

	# lengtht = len(listdir('data/train/images'))
	# lengthv = len(listdir('data/validation/images'))

	partition = {'train':[], VALID:[]}
	labels = {}
	#for each in range(0, lengtht / 2):
	seed = 9
	np.random.seed(seed)
	X, Y, X_, Y_ = [], [], [], []

	for owls, not_owls in izip(listdir('data/'+TRAIN+'/owls/'), listdir('data/'+TRAIN+'/not_owls/')):
		r = random()
		if os.path.exists('data/'+TRAIN+'/owls/'+owls) and r < 0.50:
			X.append('data/'+TRAIN+'/owls/'+owls)
			Y.append(1)
			Y.append(0)
		elif os.path.exists('data/'+TRAIN+'/not_owls/'+not_owls):
			X.append('data/'+TRAIN+'/not_owls/'+not_owls)
			Y.append(0)
			Y.append(1)

	# print X

	for i in range(len(X)):
		img = Image.open(str(X[i]))
		width, height = img.size
		if width > 150 and height > 150:
			img = img.resize((150, 150), PIL.Image.BICUBIC)
		img.save(str(X[i]))
		this = cv2.imread(X[i])
		np.save('data/'+TRAIN+'/npy/'+TRAIN + str(i) + '.npy', this)

	for owls, not_owls in izip(listdir('data/'+VALID+'/owls/'), listdir('data/'+VALID+'/not_owls/')):
		r = random()
		if os.path.exists('data/'+VALID+'/owls/'+owls) and r < 0.50:
			X_.append('data/'+VALID+'/owls/'+owls)
			Y_.append(1)
			Y_.append(0)
		elif os.path.exists('data/'+VALID+'/not_owls/'+not_owls):
			X_.append('data/'+VALID+'/not_owls/'+not_owls)
			Y_.append(0)
			Y_.append(1)

	for i in range(len(X_)):
		img = Image.open(str(X_[i]))
		width, height = img.size
		if width > 150 and height > 150:
			img = img.resize((150, 150), PIL.Image.BICUBIC)
		img.save(str(X_[i]))
		this = cv2.imread(X_[i])
		np.save('data/'+VALID+'/npy/' + VALID + str(i) + '.npy', this)

	print "partition train"
	each = 0
	for f in listdir('data/'+TRAIN+'/npy'):
		partition['train'].append('data/'+TRAIN+'/npy/' + f)
#		print Y[each]
		labels['data/'+TRAIN+'/npy/' + f] = []
		labels['data/'+TRAIN+'/npy/' + f].append(Y[each])
		each += 1
		labels['data/'+TRAIN+'/npy/' + f].append(Y[each])
		each += 1
		# labels['data/'+TRAIN+'/npy/' + f].append(Y[each])
		# each += 1

	print "partition valid"
	each = 0
	for f in listdir('data/'+VALID+'/npy'):
		partition['validation'].append('data/'+VALID+'/npy/' + f)
#		print Y_[each]
		labels['data/'+VALID+'/npy/' + f] = []
		labels['data/'+VALID+'/npy/' + f].append(Y_[each])
		each += 1
		labels['data/'+VALID+'/npy/' + f].append(Y_[each])
		each += 1
		# labels['data/'+VALID+'/npy/' + f].append(Y_[each])
		# each += 1

	# Generators
	training_generator = objects.DataGenerator(**params).generate(labels, partition['train'])
	validation_generator = objects.DataGenerator(**params).generate(labels, partition['validation'])

	model.fit_gen(
        train_generator = training_generator,
    	steps_per_epoch = len(partition['train'])//batch_size,
        validation_data = validation_generator,
        validation_steps = len(partition['validation'])//batch_size)
	model.save('trained.h5')
	# SVG(model_to_dot(model.model).create(prog='dot', format='svg'))

def predict(model, Path):
	# plt.rcParams['figure.figsize'] = (18, 6)
	#
	# # Utility to search for layer index by name.
	# # Alternatively we can specify this as -1 since it corresponds to the last layer.
	# layer_idx = utils.find_layer_idx(model.model, 'preds')
	# model.model.layers[layer_idx].activation = activations.linear
	# model.model = utils.apply_modifications(model.model)
	#
	# # This is the output node we want to maximize.
	# filter_idx = 0
	# img = visualize_activation(model.model, layer_idx, filter_indices=filter_idx)
	# plt.imshow(img[..., 0])

	img = Image.open(Path)
	width, height = img.size
	if width > 150 and height > 150:
		img = img.resize((150, 150), PIL.Image.BICUBIC)
	img.save(Path)
	this = cv2.imread(Path)
	np.save('data/'+PREDICT+'/npy/'+PREDICT+'.npy', this)
	X = np.empty((1, 150, 150, 3))
	X[0, 0:150, 0:150, 0:3] = np.load('data/'+PREDICT+'/npy/'+PREDICT+'.npy')
	X_ = np.reshape(X, (X.shape[0], 3, 150, 150))
	pred = model.predict(X_)[0]
	# print pred
	#img_batch = np.expand_dims(X_, axis=0)
	#print img_batch
	#conv_img2 = model.predict(X_)

	# conv_img2 = np.squeeze(pred, axis=0)
	# print conv_img2.shape
	# conv_img2 = pred.reshape(pred.shape[:2])

	# print conv_img2.shape
	# plt.imshow(conv_img2)
	#conv_im2 = np.squeeze(pred, axis=0)
	#print conv_im2.shape
	#conv_im2 = conv_im2.reshape(conv_im2.shape[:2])
	#print conv_im2.shape
	#plt.imshow(X_)

	print pred
	#printer(model, X_)
	# if pred[0] > pred[1]:
		# print "owl with certainty : ", pred[0]
	# else:
		# print "not owl with certainty : ", pred[1]

def run(option):
	model = objects.Network(3, 150, 150, 2)
	print "Option selected : " + option
	if os.path.exists('trained.h5'):
		model.load('trained.h5')
	#print os.path.exists(option)
	if option == '--train':
		print "Training ..."
		train(model)
	elif os.path.exists(option):
		print "Predicting ..."
		predict(model, option)
	else:
		print "Error, invalid argument"
		return

if __name__ == "__main__":
#	objects.OwlDetector().run()
	parser = argparse.ArgumentParser()
	parser.add_argument('--predict')
	parser.add_argument('--train')
	option = parser.parse_args()
	if option.predict:
		print "Analyzing image : " + option.predict
		run(option.predict)
	else:
		run('--train')
