import os
import cv2
import PIL
import scipy
import shutil
import numpy as np
from PIL import Image
from os import listdir
from config import cfg
import tensorflow as tf
from random import random
from itertools import izip

BATCH = 128
TRAIN = 'train'
VALID = 'validation'
PREDICT = 'predict'

def load_data(is_training):

    if is_training:
        return trX, trY
    else:
        return teX / 255., teY


def get_batch_data():
    params = {'dim_x': 150,
          'dim_y': 150,
          'dim_z': 3,
          'batch_size': BATCH,
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
		for i in range(8):
			labels['data/'+TRAIN+'/npy/' + f].append(0)
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

    X_ = np.empty((150, 150, 3))
    Y_ = np.empty((10), dtype = int)
    x = []
    y = []
    indexes = np.arange(len(partition['train']))
    np.random.shuffle(indexes)
    imax = int(len(indexes)/BATCH)
    for i in range(imax):
        # Find list of IDs
        list_IDs = [partition['train'][k] for k in indexes[i*BATCH:(i+1)*BATCH]]
    print "list length : ", len(list_IDs)
    for i, ID in enumerate(list_IDs):
        X_[0:150, 0:150, 0:3] = np.load(ID)
        # X_ = np.reshape(X_, (3, 150, 150))
        x.append(X_)
        Y_[0:10] = labels[ID]
        y.append(Y_)
    x = np.reshape(x, (BATCH, 150, 150, 3))
    y = np.reshape(y, (BATCH, 10))
    # print x
      #   print y[i]
    # return X_, y
	# Generators
	# training_generator = objects.DataGenerator(**params).generate(labels, partition['train'])
	# validation_generator = objects.DataGenerator(**params).generate(labels, partition['validation'])
    # trX, trY = load_data(cfg.is_training)

    # data_queues = tf.train.slice_input_producer([X, Y])
    # X, Y = tf.train.shuffle_batch(data_queues, num_threads=cfg.num_threads,
                                #   batch_size=cfg.batch_size,
                                #   capacity=cfg.batch_size * 64,
                                #   min_after_dequeue=cfg.batch_size * 32,
                                #   allow_smaller_final_batch=False)
    return x, y


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


if __name__ == '__main__':
    X, Y = load_data(cfg.is_training)
    print(X.get_shape())
    print(X.dtype)
