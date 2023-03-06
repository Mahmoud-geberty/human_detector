import time
import numpy as np 
import cv2,joblib,glob,os
import Sliding as sd
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian

# GOAL (hard negative mining): add false positives to the negative dataset
#       in a separate folder which the training considers after the OGs

hard_neg = False # control whether or not to perform hard negative mining to train the SVM
hard_neg_path = 'DATAIMAGE/hard_negative/'

im_suff = 0

test_path =  'test/'
#test_path = 'DATAIMAGE/positive/'

for filename in glob.glob(os.path.join(test_path, "*")):
	image = cv2.imread(filename)
	image = cv2.resize(image,(400,256))
	size = (64,128)
	step_size = (8,8)
	downscale = 1.10
#List to store the detections
	detections = []
#The current scale of the image 
	scale = 0
	model = joblib.load('models/models.dat')

	first_iter = 1

	for im_scaled in pyramid_gaussian(image, downscale = downscale, channel_axis=-1):
		#The list contains detections at the current scale
		if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
			break
		for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
			if window.shape[0] != size[1] or window.shape[1] != size[0]:
				continue
			window = color.rgb2gray(window)
			if (first_iter): # timestamp start of feature extraction
				st_hog = time.process_time_ns()
			fd=hog(window, block_norm='L2',orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
			if (first_iter): # timestamp end of feature extraction
				et_hog = time.process_time_ns()
			fd = fd.reshape(1, -1)
			if (first_iter): # timestamp start of prediction
				st_svm = time.process_time_ns()
			pred = model.predict(fd)
			if (first_iter): # timestamp where prediction ends
				first_iter = 0
				et_svm = time.process_time_ns()
			if pred == 1:
					
				if model.decision_function(fd) > 0.5:
					# display detected window when hard negative mining is enabled 
					if hard_neg: 
						cv2.imshow('positive window',window)
						key = cv2.waitKey(0)
						if key == 110: # 'n' is pressed
							print("false positive")
							cv2.imwrite(os.path.join(hard_neg_path, "hard_neg{}.jpg".format(im_suff)), 255*window)
							im_suff += 1
						cv2.destroyAllWindows()

					detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd), 
					int(size[0] * (downscale**scale)),
					int(size[1] * (downscale**scale))))
	 
		scale += 1

	clone = image.copy()
	rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
	sc = [score[0] for (x, y, score, w, h) in detections]
#print ("sc: ", sc)
	sc = np.array(sc)
	pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.30)
	for(x1, y1, x2, y2) in pick:
		cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(clone,'Person',(x1-2,y1-2),1,0.75,(121,12,34),1)
	hog_time = et_hog - st_hog
	svm_time = et_svm - st_svm
	print(f"hog cpu time per iteration in ns: {hog_time}")
	print(f"svm cpu time per iteration in ns: {svm_time}")
# justify using hardware for the HOG features
	print(f"hog takes {hog_time/svm_time} more time than svm classification")

	cv2.imshow('Person Detection',clone)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
