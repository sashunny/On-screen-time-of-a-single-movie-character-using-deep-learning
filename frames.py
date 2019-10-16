

from tqdm import tqdm
import cv2
import os
import numpy as np

img_path = 'data'

class1_data = []
class2_data = []
for classes in os.listdir(img_path):
	fin_path = os.path.join(img_path, classes)
	for fin_classes in tqdm(os.listdir(fin_path)):
		img = cv2.imread(os.path.join(fin_path, fin_classes))
		img = cv2.resize(img, (224,224))
		img = img/255.
		if classes == 'vinod':
			class1_data.append(img)
		else:
			class2_data.append(img)

class1_data = np.array(class1_data)
class2_data = np.array(class2_data)

print ("chk one")

import keras
from keras.applications import VGG16
 
vgg_model = VGG16(include_top=False, weights='imagenet')

vgg_class1 = vgg_model.predict(class1_data)
vgg_class2 = vgg_model.predict(class2_data)
print ("chk two")
from keras.layers import Input, Dense, Dropout
from keras.models import Model
 
inputs = Input(shape=(7*7*512,))
 
dense1 = Dense(1024, activation = 'relu')(inputs)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(512, activation = 'relu')(drop1)
drop2 = Dropout(0.5)(dense2)
outputs = Dense(1, activation = 'sigmoid')(drop2)
 
model = Model(inputs, outputs)
#model.summary()

train_data = np.concatenate((vgg_class1[:100], vgg_class2[:100]), axis = 0)
train_data = train_data.reshape(train_data.shape[0],7*7*512)
 
valid_data = np.concatenate((vgg_class1[100:],  vgg_class2[100:]), axis = 0)
valid_data = valid_data.reshape(valid_data.shape[0],7*7*512)


train_label = np.array([0]*vgg_class1[:100].shape[0] + [1]*vgg_class2[:100].shape[0] )
valid_label = np.array([0]*vgg_class1[100:].shape[0] + [1]*vgg_class2[100:].shape[0])


import tensorflow as tf
from keras.callbacks import ModelCheckpoint
 
tf.logging.set_verbosity(tf.logging.ERROR)
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(train_data, train_label, epochs = 10, batch_size = 64, validation_data = (valid_data, valid_label), verbose = 2, callbacks = callbacks_list)


