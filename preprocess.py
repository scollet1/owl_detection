from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from subprocess import call
import numpy as np
import csv
from PIL import Image
import cv2
#from matplotlib import pyplot as plt

PATH = "../Desktop/owl_training_data_00/"

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def process():
	with open('./train_data.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		for filen in os.listdir(PATH):
			if filen[0:2] == "EK":
				print filen
				t_p = PATH + filen # total_path
				img = cv2.imread(t_p, 0)
				r_img = cv2.resize(img, (960, 540)) 
				cv2.imshow('image', r_img)
				cv2.waitKey(1)
#				plt.show
				img = load_img(t_p)
				x = img_to_array(img)
				x = x.reshape((1,) + x.shape)
#				test = Image.open(t_p)
#				test.close()
#				call(["open", t_p])
				y = np.array([int(raw_input('How many owls?'))])
				M = x + y
				writer.writerow(M)
				cv2.destroyAllWindows()
			else:
				continue

if __name__ == "__main__":
	process()
