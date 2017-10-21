import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
model = ResNet50(weights='imagenet')

def train():
	

def predict():
	model = objects.Network(3, 150, 150, 1)
	

def run(args):
	if args[1] == '--train':
		train()
	elif args[1] == '--predict':
		predict([args[2]])

if __name__== "__main__":
	run(sys.argv)
