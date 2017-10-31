import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import sys
import objects
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from preprocess import process

def train(model):
	batch_size = 32

	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='sparse')

	validation_generator = test_datagen.flow_from_directory(
        'data/validation/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='sparse')

	model.fit_gen(
        train_generator,
        2000 // batch_size,
        50,
        validation_generator,
        800 // batch_size)
	model.save_weights('trained.h5')

def predict(model, img):
	batch_size = 32

	predict_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

	predict_gen = predict_datagen.flow_from_directory(
	'data/predict/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='sparse')

	model.pre_gen(
		predict_gen,
        2000 // batch_size,
        50,
        validation_generator,
        800 // batch_size)

def run(args):
	model = objects.Network(3, 150, 150, 1)
	if args[1] == '--train':
		train(model)
	elif args[1] == '--predict':
		predict(model, [args[2]])

if __name__== "__main__":
	run(sys.argv)
