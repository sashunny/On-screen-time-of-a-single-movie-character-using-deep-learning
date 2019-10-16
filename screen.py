import  cv2
import face_recognition
import numpy as np
import keras
from keras.applications import VGG16
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model

vgg_model = VGG16(include_top=False, weights='imagenet')

cap= cv2.VideoCapture('clip.mp4')                   #input video
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

model = load_model('best_model.hdf5')
vinod_images = []
no_vinod_images = []
frame_number = 0
i = 0
while True:
	ret, frame = cap.read()
	#frame_number += 1
	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break
	frame = cv2.resize(frame, (224,224))
	#print ("resized")
	test_img = frame/255.
	#print ("executed")
	test_img = np.expand_dims(test_img, 0)
	pred_img = vgg_model.predict(test_img)
	pred_feat = pred_img.reshape(1, 7*7*512)
	out_class = model.predict(pred_feat)
	i = i +1 
	print ( "Processing Frames : ", i)
	if out_class < 0.5:
		vinod_images.append(out_class)
		frame_number += 1
	#else:
	#	no_vinod_images.append(out_class)
 
print ("On-Screen time of Vinod Khanna the movie is :", frame_number/25)
