import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
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
#import kivy
#kivy.require('1.0.6') # replace with your current kivy version !

#from kivy.app import App
#from kivy.uix.label import Label
import argparse

'''we will need to split the image preprocessing into a separate file, I just\
haven't done it yet ;-;  '''

#from preprocess import process

TRAIN = 'train'
VALID = 'validation'

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

def train(model):
	batch_size = 32

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

	lengtht = len(listdir('data/train/images'))
	lengthv = len(listdir('data/validation/images'))

	partition = {'train':[], VALID:[]}
	labels = {'owl?': []}
	#for each in range(0, lengtht / 2):
	seed = 9
	np.random.seed(seed)
	X, Y, X_, Y_ = [], [], [], []

	with open('train_first.csv') as f:
  		reader = csv.reader(f, delimiter='|')
		for row in reader:
			X.append(str(row[0]))
			Y.append(row[1])

	#dataset = np.loadtxt('train_first.csv', delimiter='|')
	#X = dataset[:,0:1]
	#Y = dataset[:,1:]
	#Path = 'data/'+TRAIN+'/images/' + listdir('data/'+TRAIN+'/images')[each]
	for i in X:
		#print i
		Path = str(i)
		print Path
		img = Image.open(Path)
		img.resize((150, 150), PIL.Image.BICUBIC)
		this = cv2.imread(img)
		np.save('data/'+TRAIN+'/npy/' + 'TRAIN' + str(each) + '.npy', this)

	#dataset = np.loadtxt('valid_first.csv', delimiter='|')
	#X_ = dataset[:,0:1]
	#Y_ = dataset[:,1:]
	with open('valid_first.csv') as f:
  		reader = csv.reader(f, delimiter='|')
		for row in reader:
			X_.append(str(row[0]))
			Y_.append(row[1])

	for i in X_:
		#Path = 'data/'+VALID+'/images/' + listdir('data/'+VALID+'/images')[each]
		img = Image.open(i)
		img = img.resize((150, 150), PIL.Image.BICUBIC)
		this = cv2.imread(img)
		np.save('data/'+VALID+'/npy/' + 'TRAIN' + str(each) + '.npy', this)


	print "partition train"
	for each, f in listdir('data/'+TRAIN+'/npy'):
		partition['train'].append('data/'+TRAIN+'/npy/' + f)
		labels['data/'+TRAIN+'/npy/' + f] = Y[each]

	print "partition valid"
	for each, f in listdir('data/'+VALID+'/npy'):
		partition['validation'].append('data/'+VALID+'/npy/' + f)
		labels['data/'+VALID+'/npy/' + f] = Y_[each]

	# Generators
	training_generator = objects.DataGenerator(**params).generate(labels, partition['train'])
	validation_generator = objects.DataGenerator(**params).generate(labels, partition['validation'])

	model.fit_gen(
        train_generator = training_generator,
    	steps_per_epoch = len(partition['train'])//batch_size,
        validation_data = validation_generator,
        validation_steps = len(partition['validation'])//batch_size)
	model.save('trained.h5')

def predict(model, Path):
	img = Image.open(Path)
	img = img.resize((150, 150), PIL.Image.BICUBIC)
	print img
	this = cv2.imread(img)
	print this
	model.predict(this)

def run(option):
	model = objects.Network(3, 150, 150, 1)
	if os.path.exists('trained.h5'):
		model = model.load('trained.h5')
	if option == '--train':
		train(model)
	elif os.path.exists(option):
		predict(model, option)
	else:
		print "Error, invalid argument"
		os.exit(1)

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
